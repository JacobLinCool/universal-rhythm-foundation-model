"""Game-agnostic projections of charts.

The :func:`interaction` view reduces any chart to the sequence of moments where
the player is asked to act, described only with concepts every rhythm game
shares: onset, duration, interaction primitive, simultaneity, accent and
whether the spatial target moved. It is the data interface between the lossless
IR (layer 1) and learned shared representations (layer 2), and the basis for
cross-game statistics.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from itertools import groupby, pairwise
from statistics import fmean, median

import msgspec

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event, Primitive
from urfm.mechanics import registry


class Interaction(msgspec.Struct, frozen=True, kw_only=True):
    onset_us: int
    beat: Fraction
    duration_us: int
    primitive: Primitive
    simultaneity: int
    """Number of independent actions sharing this onset (events of one ``group`` count once)."""
    accent: bool
    moved: bool | None
    """Whether the target differs from the previous onset's; ``None`` without spatial input."""


def active_events(chart: Chart, route: str | None = None) -> list[Event]:
    """Interactive events on one path through the chart's branches.

    ``route`` selects a branch route by name; ``None`` follows each segment's default.
    """
    spec = registry.get(chart.mechanics)
    defaults = {b.id: route if route in b.routes else b.default for b in chart.branches}
    out = []
    for ev in chart.events:
        role = spec.role(ev.role)
        if role is not None and not role.interactive:
            continue
        if ev.branch is not None and defaults.get(ev.branch.segment) != ev.branch.route:
            continue
        out.append(ev)
    return out


def interaction(chartset: ChartSet, chart: Chart, route: str | None = None) -> list[Interaction]:
    spec = registry.get(chart.mechanics)
    accents = set(spec.accent_modifiers)
    timing = chartset.timing_of(chart)
    events = active_events(chart, route)
    out: list[Interaction] = []
    previous_targets: set | None = None

    def onset(ev: Event) -> int:
        return timing.us(ev.time)

    for us, group in groupby(sorted(events, key=onset), key=onset):
        members = list(group)
        targets = {ev.target for ev in members if ev.target is not None}
        head = max(members, key=lambda ev: _rank(ev.primitive))
        end = max((timing.us(ev.end) for ev in members if ev.end is not None), default=us)
        out.append(
            Interaction(
                onset_us=us,
                beat=head.time.beat,
                duration_us=end - us,
                primitive=head.primitive,
                simultaneity=len({ev.group if ev.group is not None else id(ev) for ev in members}),
                accent=any(accents & set(ev.modifiers) for ev in members),
                moved=targets != previous_targets if targets and previous_targets else None,
            )
        )
        if targets:
            previous_targets = targets
    return out


def _rank(primitive: Primitive) -> int:
    return [Primitive.HIT, Primitive.REPEAT, Primitive.HOLD, Primitive.PATH].index(primitive)


# -- statistics ---------------------------------------------------------------

IOI_BEAT_BINS = (
    Fraction(1, 8),
    Fraction(1, 6),
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
    Fraction(1),
    Fraction(2),
)


class ChartStats(msgspec.Struct, frozen=True, kw_only=True):
    mechanics: str
    family: str
    difficulty: str | None
    level: float | None
    onsets: int
    duration_s: float
    onsets_per_s: float
    peak_onsets_per_s: float
    """Maximum onsets in any 2-second window, divided by 2."""
    chord_ratio: float
    accent_ratio: float
    move_ratio: float | None
    primitive_ratio: dict[str, float]
    ioi_median_ms: float | None
    ioi_beat_hist: dict[str, float]
    """Share of inter-onset intervals (in beats) closest to each bin; ``other`` otherwise."""


def chart_stats(chartset: ChartSet, chart: Chart) -> ChartStats:
    spec = registry.get(chart.mechanics)
    seq = interaction(chartset, chart)
    n = len(seq)
    onsets = [x.onset_us for x in seq]
    duration = (onsets[-1] - onsets[0]) / 1e6 if n > 1 else 0.0
    iois_us = [b - a for a, b in pairwise(onsets)]
    iois_beat = [b.beat - a.beat for a, b in pairwise(seq)]
    hist = Counter(_bin(x) for x in iois_beat)
    moves = [x.moved for x in seq if x.moved is not None]
    return ChartStats(
        mechanics=chart.mechanics,
        family=spec.family,
        difficulty=chart.difficulty.label,
        level=chart.difficulty.level,
        onsets=n,
        duration_s=duration,
        onsets_per_s=n / duration if duration else 0.0,
        peak_onsets_per_s=_peak_density(onsets, 2_000_000) / 2,
        chord_ratio=sum(x.simultaneity > 1 for x in seq) / n if n else 0.0,
        accent_ratio=sum(x.accent for x in seq) / n if n else 0.0,
        move_ratio=fmean(moves) if moves else None,
        primitive_ratio={p.value: sum(x.primitive is p for x in seq) / n for p in Primitive}
        if n
        else {},
        ioi_median_ms=median(iois_us) / 1000 if iois_us else None,
        ioi_beat_hist={k: v / len(iois_beat) for k, v in sorted(hist.items())} if iois_beat else {},
    )


def _bin(ioi: Fraction) -> str:
    for b in IOI_BEAT_BINS:
        if abs(ioi - b) <= Fraction(1, 96):
            return str(b)
    return "other"


def _peak_density(onsets: list[int], window_us: int) -> int:
    best, lo = 0, 0
    for hi, t in enumerate(onsets):
        while t - onsets[lo] >= window_us:
            lo += 1
        best = max(best, hi - lo + 1)
    return best


def summarize(rows: list[ChartStats], by: str = "family") -> dict[str, dict[str, float]]:
    """Mean of the scalar statistics per ``family`` or per ``mechanics``."""
    out: dict[str, dict[str, float]] = {}

    def key(row: ChartStats) -> str:
        return getattr(row, by)

    for name, group in groupby(sorted(rows, key=key), key=key):
        items = list(group)
        out[name] = {
            "charts": len(items),
            "onsets_per_s": fmean(r.onsets_per_s for r in items),
            "peak_onsets_per_s": fmean(r.peak_onsets_per_s for r in items),
            "chord_ratio": fmean(r.chord_ratio for r in items),
            "accent_ratio": fmean(r.accent_ratio for r in items),
            "hold_like_ratio": fmean(
                sum(r.primitive_ratio.get(p, 0.0) for p in ("hold", "repeat", "path"))
                for r in items
            ),
            "ioi_median_ms": _nanmean(r.ioi_median_ms for r in items),
            "move_ratio": _nanmean(r.move_ratio for r in items),
        }
    return out


def _nanmean(values) -> float:
    xs = [v for v in values if v is not None]
    return fmean(xs) if xs else math.nan
