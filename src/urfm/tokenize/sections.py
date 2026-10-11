"""Sections: the unit of the song-level plan (R6).

A section is ``k`` beats of the timing map, ``k`` the power of two nearest ``SECTION_S``
seconds at the median tempo (8 beats at 150 BPM). The same music written at half or double
BPM therefore gets sections of the same length in seconds. Section ``i`` spans beats
``[i * k, (i + 1) * k)``.

Each section's plan is five categorical values: onset density (the ``DENS`` bins), share of
fine intervals (a 16th of a beat or less), share of the section in rests of at least one beat,
share of ``ka`` hits (Taiko family; 0 elsewhere), and the finest subdivision per beat that holds
90% of the onsets (``GRID_VALUES``; it is relative to the grid, so the plan says "32nds" on a
half-BPM grid where the human wrote 16ths).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple

import numpy as np

from urfm.ir.timing import TimingMap
from urfm.tokenize.vocab import GRID_VALUES, SHARE_BINS, dens_bin

SECTION_S = 3.2
MAX_SECTION_BEATS = 16
SHARE_EDGES = (0.05, 0.2, 0.4, 0.7)
"""Upper edges of the share bins: [0, .05), [.05, .2), [.2, .4), [.4, .7), [.7, 1]."""
assert len(SHARE_EDGES) + 1 == SHARE_BINS
FINE_GAP = 0.25 + 1 / 48
GRID_COVER = 0.9


class SectionPlan(NamedTuple):
    section: int
    dens: int
    fine: int
    rest: int
    ka: int
    sub: int
    """Index into ``GRID_VALUES``."""
    # R6.1 extension (``None`` for plans without it)
    peak: int | None = None
    """Most onsets in any 2 s span, per second (``DENS`` bins): bursts that a mean hides."""
    rep: int | None = None
    """Repetition of an earlier section (``REP_EDGES`` bins: new .. exact repeat)."""
    var: int | None = None
    """Distinct inter-onset intervals in the section (0 .. ``VAR_BINS - 1``, capped)."""
    sw: int | None = None
    """Share of consecutive onsets that change colour (share bins; 0 outside Taiko)."""
    long: int | None = None
    """Share of the section covered by holds, rolls and balloons (share bins)."""


def section_beats(timing: TimingMap) -> int:
    bpms = [float(t.bpm) for t in timing.tempos] or [120.0]
    beat_s = 60.0 / float(np.median(bpms))
    k = 2 ** round(math.log2(max(SECTION_S / beat_s, 1e-6)))
    return int(min(max(k, 1), MAX_SECTION_BEATS))


def share_bin(x: float) -> int:
    return int(np.searchsorted(SHARE_EDGES, x, side="right"))


PEAK_S = 2.0
REP_EDGES = (0.45, 0.7, 0.95)
"""Upper edges of the repetition bins over the best cosine similarity of the section's onset
grid to an earlier one (at least ``REP_LAG`` back): new, loose, variant, repeat."""
REP_BINS = len(REP_EDGES) + 1
assert REP_BINS == 4  # vocabulary: SREP_0..3
REP_LAG = 2
REP_SUBS = 12
VAR_BINS = 6  # vocabulary: SVAR_0..5

GRID_TOL = 1 / 96
"""Beats an onset may sit off a grid point and still count as on it: osu! charts store
milliseconds and charts re-notated on an audio-predicted grid are a few ms off, so exact
matching put nearly all of them in the finest class (v1 of the plan targets, used by r6-plan
step 9000; fixed for ``r6-plan2``)."""


def finest_grid(beats: np.ndarray) -> int:
    """Index in ``GRID_VALUES`` of the coarsest grid holding ``GRID_COVER`` of the onsets
    (within ``GRID_TOL`` beats)."""
    if not len(beats):
        return 0
    x = np.asarray(beats, np.float64) % 1.0
    for i, g in enumerate(GRID_VALUES[1:], start=1):
        off = np.abs(x * g - np.round(x * g)) / g
        if (off < GRID_TOL + 1e-9).mean() >= GRID_COVER - 1e-9:
            return i
    return len(GRID_VALUES) - 1


def plan_extras(
    timing: TimingMap,
    beats: np.ndarray,
    ka: np.ndarray,
    long_spans: list[tuple[float, float]],
    first: int,
    last: int,
    k: int,
) -> dict[int, tuple[int, int, int, int, int]]:
    """``(peak, rep, var, sw, long)`` bins of sections ``first..last``; repetition is measured
    against the earlier sections of the same range, so pass the whole song."""
    n = last - first + 1
    if n <= 0:
        return {}
    us = np.asarray(timing.us_array(beats), np.float64) if len(beats) else np.zeros(0)
    grid = np.zeros((n, 2, k * REP_SUBS), np.float32)
    raw = []
    for i in range(n):
        a, b = (first + i) * k, (first + i + 1) * k
        lo, hi = np.searchsorted(beats, a - 1e-9), np.searchsorted(beats, b - 1e-9)
        sb, su, sk = beats[lo:hi], us[lo:hi], ka[lo:hi]
        secs = float(timing.beat_to_us_exact(Fraction(b)) - timing.beat_to_us_exact(Fraction(a)))
        peak = 0.0
        if len(su):
            j = np.searchsorted(su, su + PEAK_S * 1e6)
            peak = float((j - np.arange(len(su))).max()) / min(PEAK_S, max(secs / 1e6, 1e-3))
        var = len(np.unique(np.round(np.diff(sb) * 48))) if len(sb) > 1 else 0
        sw = float((sk[1:] != sk[:-1]).mean()) if len(sk) > 1 else 0.0
        covered = sum(max(0.0, min(e, b) - max(s, a)) for s, e in long_spans)
        pos = np.rint((sb - a) * REP_SUBS).astype(np.int64)
        ok = (pos >= 0) & (pos < k * REP_SUBS)
        grid[i, sk[ok].astype(np.int64), pos[ok]] = 1.0
        raw.append((peak, var, sw, covered / k))
    g = grid.reshape(n, -1)
    norm = np.linalg.norm(g, axis=1)
    sim = (g @ g.T) / np.maximum(norm[:, None] * norm[None], 1e-12)
    out = {}
    for i, (peak, var, sw, long) in enumerate(raw):
        best = float(sim[i, : i - REP_LAG + 1].max()) if i >= REP_LAG and norm[i] > 0 else 0.0
        out[first + i] = (
            dens_bin(peak),
            int(np.searchsorted(REP_EDGES, best, side="right")),
            min(var, VAR_BINS - 1),
            share_bin(sw),
            share_bin(min(long, 1.0)),
        )
    return out


def plan_sections(
    timing: TimingMap,
    beats: np.ndarray,
    ka: np.ndarray,
    first: int,
    last: int,
    k: int,
    extras: dict[int, tuple[int, int, int, int, int]] | None = None,
) -> list[SectionPlan]:
    """Plans of sections ``first..last`` (inclusive) from onset ``beats`` (sorted, unique,
    float) and a parallel boolean ``ka`` array; ``extras`` (from :func:`plan_extras`) fills
    the R6.1 values."""
    out = []
    for i in range(first, last + 1):
        a, b = i * k, (i + 1) * k
        lo, hi = np.searchsorted(beats, a - 1e-9), np.searchsorted(beats, b - 1e-9)
        sel = beats[lo:hi]
        secs = (timing.beat_to_us_exact(Fraction(b)) - timing.beat_to_us_exact(Fraction(a))) / 1e6
        inner = np.diff(sel)
        gaps = np.diff(np.concatenate([[a], sel, [b]]))
        out.append(
            SectionPlan(
                section=i,
                dens=dens_bin(len(sel) / max(float(secs), 1e-3)),
                fine=share_bin(float((inner <= FINE_GAP).mean()) if len(inner) else 0.0),
                rest=share_bin(float(gaps[gaps >= 1 - 1e-9].sum() / k)),
                ka=share_bin(float(ka[lo:hi].mean()) if hi > lo else 0.0),
                sub=finest_grid(sel),
                **(
                    dict(zip(EXTRA_FIELDS, extras[i], strict=True))
                    if extras and i in extras
                    else {}
                ),
            )
        )
    return out


EXTRA_FIELDS = ("peak", "rep", "var", "sw", "long")


def window_sections(timing: TimingMap, start_us: int, end_us: int, k: int) -> tuple[int, int]:
    """First and last section overlapping the window ``[start_us, end_us)``."""
    a = timing.us_to_beat(start_us)
    b = timing.us_to_beat(max(end_us - 1, start_us))
    return math.floor(a / k), math.floor(b / k)
