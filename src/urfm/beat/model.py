"""Frame-level beat and downbeat model.

uint8 log-mel ``[B, T, n_mels]`` -> a small 2D conv frontend that shrinks the frequency
axis (time resolution is kept: beats need ~10 ms precision) -> pre-LN Transformer blocks
with rotary positions (shift-invariant, any length) -> two logits per frame (beat,
downbeat). All attention goes through ``scaled_dot_product_attention``.

Loss (:func:`beat_loss`): class-balanced BCE (positives and negatives averaged separately),
taking for each chunk the best of four metrical levels of the beat labels (as labelled,
double, and the two halves). Charts may write the same music at 120 or 240 BPM; the model
is not punished for choosing either, and the loss does not favour sparser levels.
"""

from __future__ import annotations

import msgspec
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from urfm.data.features import MEL, dequantize


class BeatConfig(msgspec.Struct, frozen=True, kw_only=True):
    n_mels: int = 128
    d_model: int = 256
    layers: int = 6
    heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    channels: tuple[int, int, int] = (32, 64, 128)
    downbeat_weight: float = 0.5


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: BeatConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.heads = cfg.heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.ff_mult * d),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_mult * d, d),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        q, k, v = (y.transpose(1, 2) for y in (q, k, v))
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v)
        x = x + self.drop(self.out(y.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.ff(self.n2(x)))


class BeatModel(nn.Module):
    def __init__(self, cfg: BeatConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.channels
        # frequency 128 -> 64 -> 16 -> 4; time untouched
        self.frontend = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1, stride=(1, 2)),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d((1, 4)),
        )
        self.proj = nn.Linear(c3 * (cfg.n_mels // 32), cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, mel: Tensor) -> Tensor:
        """``mel`` uint8 ``[B, T, n_mels]`` -> logits ``[B, T, 2]`` (beat, downbeat)."""
        x = dequantize(mel, MEL) / (MEL.db_range / 2) + 1.0
        x = self.frontend(x[:, None])  # [B, C, T, F']
        b, c, t, f = x.shape
        x = self.proj(x.permute(0, 2, 1, 3).reshape(b, t, c * f))
        dh = self.cfg.d_model // self.cfg.heads
        pos = torch.arange(t, device=x.device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
        ang = pos[:, None] * inv[None]
        cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.head(self.norm(x))


def _balanced_bce(logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Per-chunk mean BCE over positive frames plus mean over negative frames: ``[B]``."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = (target > 0) & mask
    neg = (target == 0) & mask
    lp = (bce * pos).sum(-1) / pos.sum(-1).clamp_min(1)
    ln = (bce * neg).sum(-1) / neg.sum(-1).clamp_min(1)
    return lp + ln


def beat_loss(
    logits: Tensor, beats: Tensor, downbeats: Tensor, mask: Tensor, downbeat_weight: float
) -> tuple[Tensor, Tensor]:
    """``logits [B, T, 2]``; ``beats [B, 4, T]`` (label levels), ``downbeats [B, T]``,
    ``mask [B, T]``. Returns (loss, index of the chosen level per chunk)."""
    lb = logits[..., 0].float()
    per_level = torch.stack(
        [_balanced_bce(lb, beats[:, i].float(), mask) for i in range(beats.shape[1])], 1
    )
    best, level = per_level.min(1)
    ld = _balanced_bce(logits[..., 1].float(), downbeats.float(), mask)
    has = mask.any(-1).float()
    loss = ((best + downbeat_weight * ld) * has).sum() / has.sum().clamp_min(1)
    return loss, level
