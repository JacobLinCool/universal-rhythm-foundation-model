"""Timing evaluation: beat F-measure (strict and metrical-level tolerant), downbeat F, and
how well a predicted grid fits the notes of a human chart."""

from __future__ import annotations

import numpy as np

from urfm.beat.data import levels
from urfm.ir.timing import TimingMap
from urfm.tokenize.tokenizer import beats_at_us


def matched(ref: np.ndarray, est: np.ndarray, tol: float) -> int:
    """One-to-one matches within ``tol`` seconds (greedy on sorted times)."""
    i = j = n = 0
    ref, est = np.sort(ref), np.sort(est)
    while i < len(ref) and j < len(est):
        d = est[j] - ref[i]
        if abs(d) <= tol:
            n += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return n


def f_measure(ref: np.ndarray, est: np.ndarray, tol: float = 0.07) -> float:
    if len(ref) == 0 and len(est) == 0:
        return 1.0
    return 2 * matched(ref, est, tol) / max(len(ref) + len(est), 1)


def beat_scores(
    ref: np.ndarray, est: np.ndarray, span: tuple[float, float] | None = None
) -> dict[str, float]:
    """F@70ms, F@25ms, and the best F@70ms over the reference's metrical levels."""
    if span is not None:
        est = est[(est >= span[0] - 0.035) & (est <= span[1] + 0.035)]
    return {
        "beat_f70": f_measure(ref, est, 0.07),
        "beat_f25": f_measure(ref, est, 0.025),
        "beat_f70_any_level": max(f_measure(r, est, 0.07) for r in levels(ref)),
    }


GRID = (4, 3)
"""Note positions count as on the grid when they fall on 1/4 or 1/3 beats."""


def grid_fit(timing: TimingMap, onsets_s: np.ndarray, tol: float = 0.015) -> float:
    """Fraction of note onsets within ``tol`` seconds of a 1/4- or 1/3-beat position of
    ``timing`` (the grid a chart would be written on). Tolerant of doubled or halved tempo
    up to these subdivisions."""
    if len(onsets_s) == 0:
        return 1.0
    us = np.asarray(onsets_s, np.float64) * 1e6
    beats = beats_at_us(timing, us)
    best = np.full(len(us), np.inf)
    for div in GRID:
        target = np.round(beats * div) / div
        t_us = timing.us_array(list(target))
        best = np.minimum(best, np.abs(t_us - us))
    return float(np.mean(best <= tol * 1e6))
