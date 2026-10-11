"""Continuous, unbounded chart conditions.

Each condition is a number measured on a chart, on a scale chosen from how the quantity
behaves rather than tuned by experiment:

* rates (onsets per second, peak rate, onsets per beat) and event frequencies are
  **logarithmic** -- density is perceived and grows multiplicatively (4 -> 8/s feels like
  8 -> 16/s), and doubling the tempo doubles every rate;
* counts per minute use ``log2(1 + x)`` (log for large counts, 0 for none);
* an official level is **linear** (levels are designed as an even ladder);
* shares (e.g. of notes off the 16th grid) are **linear** in ``[0, 1]``.

Values are centred and scaled by typical values (``CENTER``/``SCALE``, z-scores) and passed
to the model unclipped: asking for "20% denser than anything seen" is a larger number,
not a saturated bin. A missing value (no official level for an osu! map) is masked.
"""

from __future__ import annotations

import math

import numpy as np

from urfm.ir.chart import Chart, ChartSet
from urfm.tokenize.tokenizer import playable_events

NAMES = (
    "log2_rate",
    "log2_peak_rate",
    "log2_per_beat",
    "level",
    "log2p_scroll_per_min",
    "log2p_meter_per_min",
    "fine_share",
)
CENTER = np.array([1.5, 2.6, -0.2, 5.0, 0.5, 0.2, 0.08], np.float32)
SCALE = np.array([1.0, 1.0, 1.0, 3.0, 1.5, 1.0, 0.15], np.float32)
NOTATION = frozenset({"log2_per_beat", "fine_share"})
"""Conditions that depend on how the chart is notated (its beat grid), not on how it sounds:
unknown when the grid comes from audio, so requests on a predicted grid leave them out."""
PEAK_WINDOW_S = 2.0
PEAK_QUANTILE = 0.9
LEVEL_GAMES = frozenset({"taiko"})
"""Games whose ``Difficulty.level`` is an official, comparable level."""
SCROLL_KINDS = frozenset({"scroll", "osu:sv"})


def measure(chartset: ChartSet, chart: Chart) -> tuple[np.ndarray, np.ndarray]:
    """Raw condition values (on their own scale, see :data:`NAMES`) and a validity mask."""
    timing = chartset.timing_of(chart)
    events = playable_events(chart)
    values = np.zeros(len(NAMES), np.float32)
    mask = np.zeros(len(NAMES), bool)
    if len(events) < 2:
        return values, mask
    beats = np.unique(np.asarray([float(e.time.beat) for e in events]))
    us = np.unique(
        np.round(
            timing.us_array([e.time.beat for e in events], [e.time.residual_us for e in events])
        )
    )
    span_s = max((us[-1] - us[0]) / 1e6, 1.0)
    rate = len(us) / span_s
    span_beats = max(float(beats[-1] - beats[0]), 1.0)
    # peak: the 90th percentile of onsets per 2 s over the charted span
    edges = np.arange(us[0], us[-1] + PEAK_WINDOW_S * 1e6, PEAK_WINDOW_S * 1e6)
    counts = np.histogram(us, edges)[0] if len(edges) > 1 else np.array([len(us)])
    peak = float(np.quantile(counts, PEAK_QUANTILE)) / PEAK_WINDOW_S
    minutes = span_s / 60
    scrolls = sum(1 for c in chart.controls if c.kind in SCROLL_KINDS)
    meters = len(timing.meters)
    quarter = beats * 4
    fine = float(np.mean(np.abs(quarter - np.round(quarter)) > 1e-6))
    values[:] = [
        math.log2(max(rate, 1e-3)),
        math.log2(max(peak, 0.25)),
        math.log2(max(len(beats) / span_beats, 1e-3)),
        chart.difficulty.level or 0.0,
        math.log2(1 + scrolls / minutes),
        math.log2(1 + max(meters - 1, 0) / minutes),
        fine,
    ]
    mask[:] = True
    level = chart.difficulty.level
    mask[NAMES.index("level")] = chart.mechanics in LEVEL_GAMES and level is not None and level > 0
    return values, mask


def without_notation(cond: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values, mask = cond[0].copy(), cond[1].copy()
    for i, name in enumerate(NAMES):
        if name in NOTATION:
            values[i], mask[i] = 0.0, False
    return values, mask


def normalise(values: np.ndarray) -> np.ndarray:
    return ((values - CENTER) / SCALE).astype(np.float32)


def chart_conditions(chartset: ChartSet, chart: Chart) -> tuple[np.ndarray, np.ndarray]:
    """Normalised conditions and mask for a reference chart (training and evaluation)."""
    values, mask = measure(chartset, chart)
    return normalise(values), mask


def conditions(
    *,
    rate: float | None = None,
    peak_rate: float | None = None,
    per_beat: float | None = None,
    level: float | None = None,
    scroll_per_min: float | None = None,
    meter_per_min: float | None = None,
    fine_share: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised conditions from explicit targets (generation); ``None`` = unspecified."""
    raw = [
        None if rate is None else math.log2(rate),
        None if peak_rate is None else math.log2(peak_rate),
        None if per_beat is None else math.log2(per_beat),
        level,
        None if scroll_per_min is None else math.log2(1 + scroll_per_min),
        None if meter_per_min is None else math.log2(1 + meter_per_min),
        fine_share,
    ]
    values = np.array([0.0 if x is None else x for x in raw], np.float32)
    mask = np.array([x is not None for x in raw], bool)
    return np.where(mask, normalise(values), 0.0).astype(np.float32), mask


def cond_prefix(tok, mechanics: str, lanes: int | None = None) -> list[int]:
    """Prefix for condition-embedding models: ``GAME [LANES] COND``."""
    from urfm.mechanics import registry

    v = tok.vocab
    spec = registry.get(mechanics)
    out = [v.id("GAME", mechanics)]
    lanes = spec.space.lanes or lanes
    if spec.space.kind.value == "lanes" and lanes:
        out.append(v.id("LANES", min(lanes, 18)))
    out.append(v.id("COND"))
    return out
