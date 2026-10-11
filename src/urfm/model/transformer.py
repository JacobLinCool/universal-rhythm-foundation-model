"""Encoder-decoder Transformer over mel windows and chart tokens.

Encoder: uint8 mel -> dB (on device) -> conv stem (2x time downsampling) + projected
timing features + learned positions -> pre-LN self-attention blocks.
Decoder: factorizable token embedding + learned positions -> pre-LN blocks with causal
self-attention and cross-attention -> tied output projection.

Optional song context (``cfg.song_context``): a small encoder over whole-song units --
pooled mel blocks (~2 s), or beat-synchronous texture units (``cfg.song_feat_dim``) --
plus a marker for the units of the current window, appended to the cross-attention memory
with a key padding mask.

Optional continuous conditions (``cfg.cond_dim``): each normalised value ``z`` is expanded
to ``[z, sin(πz/4), cos(πz/4), sin(πz/8), cos(πz/8)]`` (a linear term for extrapolation and
slow periodic terms whose periods, 8 and 16 standard deviations, never wrap within any
sensible range), masked, and mapped by an MLP to one vector that replaces the embedding of
the ``COND`` token in the prefix. Disabled features leave the original parameters and
behaviour untouched.

Shapes are static for a given (batch, window, decoder bucket), so the model is
``torch.compile`` friendly; all attention goes through ``scaled_dot_product_attention``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from urfm.data.features import MEL, dequantize
from urfm.model.config import ModelConfig


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(d, d, bias=False)
        self.kv = nn.Linear(d, 2 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.dropout = dropout

    def split(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def project_kv(self, source: Tensor) -> tuple[Tensor, Tensor]:
        k, v = self.kv(source).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def attend(self, x: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, causal: bool) -> Tensor:
        q = self.split(self.q(x))
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=causal and mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        b, h, n, dh = y.shape
        return self.out(y.transpose(1, 2).reshape(b, n, h * dh))

    def forward(self, x: Tensor, source: Tensor, mask: Tensor | None = None, causal: bool = False):
        k, v = self.project_kv(source)
        return self.attend(x, k, v, mask, causal)


class FeedForward(nn.Sequential):
    def __init__(self, d: int, mult: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(d, mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(mult * d, d)
        )


class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.n1, self.attn = nn.LayerNorm(d), Attention(d, cfg.n_heads, cfg.dropout)
        self.n2, self.ff = nn.LayerNorm(d), FeedForward(d, cfg.ff_mult, cfg.dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, mask)
        return x + self.ff(self.n2(x))


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.n1, self.self_attn = nn.LayerNorm(d), Attention(d, cfg.n_heads, cfg.dropout)
        self.n2, self.cross = nn.LayerNorm(d), Attention(d, cfg.n_heads, cfg.dropout)
        self.n3, self.ff = nn.LayerNorm(d), FeedForward(d, cfg.ff_mult, cfg.dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Tensor | None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        h = self.n1(x)
        x = x + self.self_attn(h, h, self_mask, causal=True)
        x = x + self.cross(self.n2(x), memory, memory_mask)
        return x + self.ff(self.n3(x))

    def prefill(self, x: Tensor, cache: KVCache, layer: int, mask: Tensor) -> Tensor:
        """Whole-prompt pass that also fills the self-attention cache."""
        h = self.n1(x)
        k, v = self.self_attn.project_kv(h)
        p = x.shape[1]
        cache.k[layer][:, :, :p] = k
        cache.v[layer][:, :, :p] = v
        x = x + self.self_attn.attend(h, k, v, mask, causal=False)
        x = x + self.cross.attend(self.n2(x), cache.mk[layer], cache.mv[layer], cache.mmask, False)
        return x + self.ff(self.n3(x))

    def step(self, x: Tensor, cache: KVCache, layer: int, t: Tensor, key_mask: Tensor) -> Tensor:
        """One-token decoding: ``x`` is ``[B, 1, d]``; ``t`` the write index (scalar tensor)."""
        h = self.n1(x)
        k, v = self.self_attn.project_kv(h)
        cache.write(layer, t, k, v)
        x = x + self.self_attn.attend(h, cache.k[layer], cache.v[layer], key_mask, causal=False)
        x = x + self.cross.attend(self.n2(x), cache.mk[layer], cache.mv[layer], cache.mmask, False)
        return x + self.ff(self.n3(x))


class KVCache:
    """Preallocated self-attention keys/values plus per-layer cross-attention projections."""

    def __init__(
        self, model: ChartModel, memory: Tensor, length: int, memory_mask: Tensor | None = None
    ) -> None:
        cfg = model.cfg
        b = memory.shape[0]
        dh = cfg.d_model // cfg.n_heads
        shape = (b, cfg.n_heads, length, dh)
        # under autocast the projections are low precision; store the cache the same way
        dtype = (
            torch.get_autocast_dtype(memory.device.type)
            if torch.is_autocast_enabled(memory.device.type)
            else memory.dtype
        )
        self.k = [memory.new_zeros(shape, dtype=dtype) for _ in model.decoder]
        self.v = [memory.new_zeros(shape, dtype=dtype) for _ in model.decoder]
        self.mk, self.mv = zip(
            *(blk.cross.project_kv(memory) for blk in model.blocks()), strict=True
        )
        self.mmask = _key_mask(memory_mask)

    def write(self, layer: int, t: Tensor, k: Tensor, v: Tensor) -> None:
        self.k[layer].index_copy_(2, t.view(1), k.to(self.k[layer].dtype))
        self.v[layer].index_copy_(2, t.view(1), v.to(self.v[layer].dtype))


COND_FREQS = (torch.pi / 4, torch.pi / 8)


class CondEmbed(nn.Module):
    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.n = n
        width = n * (1 + 2 * len(COND_FREQS)) + n
        self.mlp = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        """``values`` ``[B, n]`` z-scores, ``mask`` ``[B, n]`` -> ``[B, d]``."""
        z = values.float() * mask
        parts = [z]
        for w in COND_FREQS:
            parts += [torch.sin(w * z) * mask, torch.cos(w * z) * mask]
        x = torch.cat([*parts, mask.float()], dim=-1)
        first: nn.Linear = self.mlp[0]  # type: ignore[assignment]
        return self.mlp(x.to(first.weight.dtype))


class ChartModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stem = nn.Sequential(
            nn.Conv1d(cfg.n_mels, d, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d, d, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.frame_proj = nn.Linear(2 * cfg.frame_feat_dim, d)
        self.enc_pos = nn.Parameter(torch.zeros(cfg.max_enc_len, d))
        self.encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.enc_layers))
        self.enc_norm = nn.LayerNorm(d)
        self.memory = nn.Parameter(torch.zeros(cfg.memory_tokens, d)) if cfg.memory_tokens else None
        if cfg.song_context:
            self.song_in = nn.Linear(cfg.song_feat_dim or cfg.n_mels, d)
            self.song_pos = nn.Parameter(torch.zeros(cfg.song_max, d))
            self.song_marker = nn.Embedding(2, d)
            self.song_encoder = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.song_layers))
            self.song_norm = nn.LayerNorm(d)
            self.source = nn.Parameter(torch.zeros(2, d))
            """Added to window (0) and song (1) memory so cross-attention can tell them apart."""

        self.cond = CondEmbed(cfg.cond_dim, d) if cfg.cond_dim else None
        e = cfg.emb_dim or d
        self.embed = nn.Embedding(cfg.vocab_size, e)
        self.emb_up = nn.Linear(e, d, bias=False) if e != d else nn.Identity()
        self.emb_down = nn.Linear(d, e, bias=False) if e != d else nn.Identity()
        self.dec_pos = nn.Parameter(torch.zeros(cfg.max_dec_len, d))
        self.decoder = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.dec_layers))
        self.dec_norm = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def blocks(self) -> list[DecoderBlock]:
        return [blk for blk in self.decoder if isinstance(blk, DecoderBlock)]

    def num_params(self) -> dict[str, int]:
        def count(mods) -> int:
            return sum(p.numel() for m in mods for p in m.parameters())

        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "encoder_layers": count(self.encoder),
            "decoder_layers": count(self.decoder),
            "embedding": self.embed.weight.numel(),
        }

    # -- encoder ---------------------------------------------------------------

    def encode(self, mel: Tensor, feats: Tensor) -> Tensor:
        """``mel`` uint8 ``[B, T, n_mels]``; ``feats`` float ``[B, T, F]``; ``T`` even."""
        x = dequantize(mel, MEL) / (MEL.db_range / 2) + 1.0  # roughly [-1, 1]
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)
        b, t2, _ = x.shape
        f = feats[:, : 2 * t2].reshape(b, t2, -1).to(x.dtype)
        x = x + self.frame_proj(f) + self.enc_pos[:t2]
        for blk in self.encoder:
            x = blk(x)
        x = self.enc_norm(x)
        if self.memory is not None:
            x = torch.cat([x, self.memory.expand(b, -1, -1)], dim=1)
        return x

    def encode_song(self, song: Tensor, marker: Tensor, length: Tensor) -> tuple[Tensor, Tensor]:
        """``song`` uint8 ``[B, S, n_mels]`` pooled blocks, ``marker`` ``[B, S]`` (1 = blocks
        of the current window), ``length`` ``[B]`` valid blocks. Returns states and mask."""
        s = song.shape[1]
        valid = torch.arange(s, device=song.device)[None] < length[:, None]
        if self.cfg.song_feat_dim:
            x = song.to(self.song_in.weight.dtype)
        else:
            x = dequantize(song, MEL) / (MEL.db_range / 2) + 1.0
        x = self.song_in(x) + self.song_pos[:s] + self.song_marker(marker.long())
        attn = _key_mask(valid)
        for blk in self.song_encoder:
            x = blk(x, attn)
        return self.song_norm(x), valid

    def memory_for(
        self,
        mel: Tensor,
        feats: Tensor,
        song: Tensor | None = None,
        song_marker: Tensor | None = None,
        song_len: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Cross-attention memory and its key mask (``None`` = all valid)."""
        window = self.encode(mel, feats)
        if not self.cfg.song_context:
            return window, None
        if song is None or song_marker is None or song_len is None:
            raise ValueError("song_context models need song, song_marker and song_len")
        states, valid = self.encode_song(song, song_marker, song_len)
        memory = torch.cat([window + self.source[0], states + self.source[1]], dim=1)
        ones = torch.ones(window.shape[:2], dtype=torch.bool, device=window.device)
        return memory, torch.cat([ones, valid], dim=1)

    # -- decoder ---------------------------------------------------------------

    def embed_tokens(self, tokens: Tensor, positions: Tensor, cond: Tensor | None = None) -> Tensor:
        """``cond`` ``[B, d]`` (from :meth:`condition`) replaces the ``COND`` token embedding."""
        x = self.emb_up(self.embed(tokens))
        if cond is not None:
            slot = (tokens == self.cfg.cond_token)[..., None]
            x = torch.where(slot, cond[:, None, :].to(x.dtype), x)
        return x + self.dec_pos[positions]

    def condition(self, values: Tensor | None, mask: Tensor | None) -> Tensor | None:
        if self.cond is None or values is None or mask is None:
            return None
        return self.cond(values, mask)

    def logits(self, h: Tensor) -> Tensor:
        return self.emb_down(self.dec_norm(h)) @ self.embed.weight.T

    def decode(
        self,
        tokens: Tensor,
        memory: Tensor,
        memory_mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        n = tokens.shape[1]
        x = self.embed_tokens(tokens, torch.arange(n, device=tokens.device), cond)
        mask = _key_mask(memory_mask)
        for blk in self.decoder:
            x = blk(x, memory, None, mask)
        return self.logits(x)

    def forward(
        self,
        mel: Tensor,
        feats: Tensor,
        tokens: Tensor,
        song: Tensor | None = None,
        song_marker: Tensor | None = None,
        song_len: Tensor | None = None,
        cond: Tensor | None = None,
        cond_mask: Tensor | None = None,
    ) -> Tensor:
        """Teacher-forced logits ``[B, L, V]`` for next-token prediction of ``tokens``."""
        memory, mask = self.memory_for(mel, feats, song, song_marker, song_len)
        return self.decode(tokens, memory, mask, self.condition(cond, cond_mask))

    def loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Mean cross-entropy over ``loss_mask`` targets; returns (loss, n_correct, n_tokens)."""
        tokens, mask = batch["tokens"], batch["loss_mask"]
        song = (
            (batch["song"], batch["song_marker"], batch["song_len"])
            if self.cfg.song_context
            else (None, None, None)
        )
        logits = self(
            batch["mel"],
            batch["feats"],
            tokens[:, :-1],
            *song,
            batch.get("cond"),
            batch.get("cond_mask"),
        )
        target = tokens[:, 1:]
        m = mask[:, 1:]
        ce = F.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
        n = m.sum().clamp_min(1)
        loss = (ce * m).sum() / n
        correct = ((logits.argmax(-1) == target) & m).sum()
        return loss, correct, m.sum()

    # -- incremental decoding ------------------------------------------------------

    def prefill(
        self,
        tokens: Tensor,
        pad_mask: Tensor,
        memory: Tensor,
        length: int,
        memory_mask: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> tuple[KVCache, Tensor]:
        """Run a left-padded prompt ``[B, P]`` (``pad_mask`` True = real token); ``cond`` is
        the output of :meth:`condition`.

        Returns the cache and logits for the next token ``[B, V]``.
        """
        p = tokens.shape[1]
        cache = KVCache(self, memory, length, memory_mask)
        positions = (pad_mask.long().cumsum(1) - 1).clamp_min(0)
        x = self.embed_tokens(tokens, positions, cond)
        causal = torch.ones(p, p, dtype=torch.bool, device=tokens.device).tril()
        mask = (causal[None] & pad_mask[:, None, :])[:, None]
        # padded query rows attend to themselves so softmax stays finite
        mask = mask | torch.eye(p, dtype=torch.bool, device=tokens.device)[None, None]
        for i, blk in enumerate(self.blocks()):
            x = blk.prefill(x, cache, i, mask)
        return cache, self.logits(x[:, -1:])[:, 0]

    def step(
        self, token: Tensor, position: Tensor, cache: KVCache, t: int, key_mask: Tensor
    ) -> Tensor:
        """Decode one token per row. ``key_mask`` ``[B, length]`` marks valid cache slots."""
        x = self.embed_tokens(token[:, None], position[:, None])
        tt = torch.tensor(t, device=token.device)
        mask = key_mask[:, None, None, :]
        for i, blk in enumerate(self.blocks()):
            x = blk.step(x, cache, i, tt, mask)
        return self.logits(x)[:, 0]


def _key_mask(valid: Tensor | None) -> Tensor | None:
    """``[B, M]`` key validity -> broadcastable SDPA boolean mask ``[B, 1, 1, M]``."""
    return None if valid is None else valid[:, None, None, :]
