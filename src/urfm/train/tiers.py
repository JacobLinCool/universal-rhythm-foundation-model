"""Per-game difficulty tiers from onset-density percentiles.

A cross-game ``TIER`` guessed from difficulty names does not mean the same thing in
every game (osu! "Normal" is not Taiko "normal"). Instead, ``TIER_k`` means "the k-th
quintile of onset density among this game's training charts": a relative, per-game
notion of difficulty that is computable for every chart. Absolute density stays in
``DENS``. The table is built from the training index and stored in checkpoints so
generation and evaluation use the same definition.
"""

from __future__ import annotations

import bisect

import msgspec
import numpy as np

TIERS = 5


class TierTable(msgspec.Struct, frozen=True):
    edges: dict[str, list[float]]
    """mechanics -> 4 quintile boundaries of onsets/s."""
    centers: dict[str, list[float]]
    """mechanics -> 5 typical densities (10/30/50/70/90th percentiles), for generation."""

    @classmethod
    def build(cls, rates: dict[str, list[float]]) -> TierTable:
        edges, centers = {}, {}
        for mech, values in rates.items():
            xs = np.asarray([v for v in values if v > 0], dtype=np.float64)
            if len(xs) == 0:
                continue
            edges[mech] = np.quantile(xs, [0.2, 0.4, 0.6, 0.8]).round(4).tolist()
            centers[mech] = np.quantile(xs, [0.1, 0.3, 0.5, 0.7, 0.9]).round(4).tolist()
        return cls(edges=edges, centers=centers)

    def tier(self, mechanics: str, onsets_per_s: float) -> int | None:
        edges = self.edges.get(mechanics)
        return None if edges is None else bisect.bisect_right(edges, onsets_per_s)

    def density(self, mechanics: str, tier: int) -> float | None:
        """Typical onsets/s of ``tier`` in ``mechanics`` (to condition generation)."""
        centers = self.centers.get(mechanics)
        return None if centers is None else centers[max(0, min(tier, TIERS - 1))]


def chart_prefix(tok, tiers: TierTable | None, chartset, chart) -> list[int]:
    """Conditioning prefix for a reference chart: per-game tier when a table is given,
    otherwise the tokenizer's name-based tier (old checkpoints)."""
    if tiers is None:
        return tok.prefix(chartset, chart)
    from urfm.tokenize.tokenizer import chart_lanes, onset_rate

    rate = onset_rate(chartset, chart)
    return tok.conditioning(
        chart.mechanics,
        tier=tiers.tier(chart.mechanics, rate),
        onsets_per_s=rate,
        lanes=chart_lanes(chart),
    )


def load_tiers(meta: dict) -> TierTable | None:
    raw = meta.get("tiers")
    return None if raw is None else msgspec.convert(raw, TierTable)
