"""Song planner (R6): one bidirectional Transformer over all sections of a song.

Input per section: audio descriptors (``urfm.data.sections``), plus the game and the chart
conditions shared by every section. Output per section: categorical section-plan values
(``urfm.tokenize.sections``). Section contrast is decided here, with the whole song in view;
the chart decoder only realises the plan.
"""

from __future__ import annotations

import msgspec
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from urfm.model.transformer import CondEmbed

PLAN_HEADS = ("dens", "fine", "rest", "ka", "sub")
EXTRA_HEADS = ("peak", "rep", "var", "sw", "long")
"""R6.1 heads, present when ``PlannerConfig.head_sizes`` has ten entries."""


class PlannerConfig(msgspec.Struct, frozen=True, kw_only=True):
    in_dim: int
    games: tuple[str, ...]
    head_sizes: tuple[int, ...]
    """Classes per output, in ``PLAN_HEADS`` order."""
    d_model: int = 256
    layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    max_sections: int = 384
    cond_dim: int = 0
    dropout: float = 0.1


class Planner(nn.Module):
    def __init__(self, cfg: PlannerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.inp = nn.Sequential(nn.LayerNorm(cfg.in_dim), nn.Linear(cfg.in_dim, d))
        self.pos = nn.Parameter(torch.zeros(cfg.max_sections, d))
        nn.init.normal_(self.pos, std=0.02)
        self.game = nn.Embedding(len(cfg.games), d)
        self.cond = CondEmbed(cfg.cond_dim, d) if cfg.cond_dim else None
        layer = nn.TransformerEncoderLayer(
            d,
            cfg.n_heads,
            d * cfg.ff_mult,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, sum(cfg.head_sizes))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        feats: Tensor,
        mask: Tensor,
        game: Tensor,
        cond: Tensor | None = None,
        cond_mask: Tensor | None = None,
    ) -> list[Tensor]:
        """``feats [B, S, in_dim]``, ``mask [B, S]`` (True = section) -> logits per head
        ``[B, S, classes]``."""
        s = feats.shape[1]
        x = self.inp(feats) + self.pos[:s] + self.game(game)[:, None]
        if self.cond is not None and cond is not None and cond_mask is not None:
            x = x + self.cond(cond, cond_mask)[:, None].to(x.dtype)
        x = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        return list(self.out(x).split(list(self.cfg.head_sizes), dim=-1))

    def loss(
        self,
        feats: Tensor,
        mask: Tensor,
        game: Tensor,
        targets: Tensor,
        cond: Tensor | None = None,
        cond_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``targets [B, S, heads]`` (-100 = ignore) -> (mean loss over heads, per-head
        accuracy ``[heads]``)."""
        logits = self(feats, mask, game, cond, cond_mask)
        losses, accs = [], []
        for i, lg in enumerate(logits):
            t = targets[..., i].masked_fill(~mask, -100)
            losses.append(F.cross_entropy(lg.flatten(0, 1).float(), t.flatten(), ignore_index=-100))
            valid = t != -100
            hit = (lg.argmax(-1) == t) & valid
            accs.append(hit.sum() / valid.sum().clamp_min(1))
        return torch.stack(losses).mean(), torch.stack(accs)
