"""Beat and downbeat labels from charts.

Every chart's timing map says where beats and measures fall. For a song we pick one
representative chart (official Taiko first, then osu! modes), fold visual tempo tricks into
scroll (:func:`urfm.data.canonical.canonicalize`) so the labels follow the music, and list
beat and downbeat times over the charted span (the grid is extrapolated, and unreliable,
far before the first or after the last note).
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from urfm.data.canonical import canonicalize
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.timing import TimingMap
from urfm.tokenize.tokenizer import playable_events

PRIORITY = ("taiko", "osu_taiko", "osu_mania", "osu_catch", "osu_std")
TEMPO_RANGE = (30, 1000)
"""Charts with BPM outside this range (red-line gimmicks) give no beat labels."""
MARGIN_MEASURES = 1


def representative(chartset: ChartSet) -> Chart | None:
    """The chart whose timing labels the song: highest-priority game, most events."""
    charts = [c for c in chartset.charts if c.events and c.mechanics in PRIORITY]
    if not charts:
        return None
    return min(charts, key=lambda c: (PRIORITY.index(c.mechanics), -len(c.events)))


def song_labels(
    chartset: ChartSet, duration_s: float
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | None:
    """``(beats_s, downbeats_s, (lo_s, hi_s))`` for a song, or ``None`` if unusable.

    ``lo_s``/``hi_s`` bound the labelled span: one measure around the charted notes,
    clipped to the audio.
    """
    chart = representative(chartset)
    if chart is None:
        return None
    raw = chartset.timing_of(chart)
    if raw.warps or any(not TEMPO_RANGE[0] <= t.bpm <= TEMPO_RANGE[1] for t in raw.tempos):
        return None
    if any(m.measure_beats <= 0 for m in raw.meters):
        return None
    cs, chart = canonicalize(chartset, chart)
    timing = cs.timing_of(chart)
    events = playable_events(chart)
    if len(events) < 8:
        return None
    first = min(e.time.beat for e in events)
    last = max((e.end or e.time).beat for e in events)
    if last - first > 20_000:  # beat numbering is not musical (osu! re-anchoring tricks)
        return None
    lo_b = first - MARGIN_MEASURES * timing.meter_at(first).measure_beats
    hi_b = last + MARGIN_MEASURES * timing.meter_at(last).measure_beats
    beats, downbeats = grid(timing, lo_b, hi_b)
    if len(beats) < 8:
        return None
    lo = max(beats[0], 0.0)
    hi = min(beats[-1], duration_s)
    keep = (beats >= lo) & (beats <= hi)
    keep_d = (downbeats >= lo) & (downbeats <= hi)
    return beats[keep], downbeats[keep_d], (lo, hi)


def grid(timing: TimingMap, lo: Fraction, hi: Fraction) -> tuple[np.ndarray, np.ndarray]:
    """Times (s) of whole beats and of measure starts in ``[lo, hi]`` beats.

    Times must increase: with negative stops (osu! red lines re-anchoring the grid) the
    tail of the previous grid that overlaps the new one is dropped.
    """
    b0, b1 = math.ceil(lo), math.floor(hi)
    whole = [Fraction(b) for b in range(b0, b1 + 1)]
    beats = _increasing(timing.us_array(whole) / 1e6)
    floor = timing.meter_at(lo)
    start = floor.beat + math.floor((lo - floor.beat) / floor.measure_beats) * floor.measure_beats
    starts = [m for m in _measures(timing, start, hi) if m >= lo]
    downbeats = _increasing(timing.us_array(starts) / 1e6) if starts else np.zeros(0)
    return beats, downbeats


def _measures(timing: TimingMap, start: Fraction, hi: Fraction) -> list[Fraction]:
    out, beat = [], start
    while beat <= hi and len(out) < 100_000:
        out.append(beat)
        beat += max(timing.meter_at(beat).measure_beats, Fraction(1, 64))
    return out


def _increasing(times: np.ndarray, min_gap: float = 0.03) -> np.ndarray:
    """Strictly increasing times, keeping the later grid where grids overlap (an osu! red
    line re-anchors the beat; the old grid's tail is what gets cut)."""
    if len(times) == 0:
        return times
    keep = [len(times) - 1]
    for i in range(len(times) - 2, -1, -1):
        if times[i] < times[keep[-1]] - min_gap:
            keep.append(i)
    return times[keep[::-1]]
