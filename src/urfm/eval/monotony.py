"""Monotony: stretches where a chart repeats one short pattern.

**Main metric** (``excess_monotone_time``): seconds per minute of monotone stretches longer than
humans write at the same density,

    E_time = 60 / T * sum_s max(0, L_s - q95_human(L | log2 d_s))

where a stretch ``s`` is a maximal run of 10 s windows (1 s hop) with rhythm repetition
``R(w) >= 0.9``, ``L_s`` its duration, ``d_s`` its median notes per second, and

    R(w) = max_{1 <= p <= 16} mean_i [ s_i == s_{i+p} ]

over the window's onset symbols (interval to the next onset in beats; colours ignored, so a
constant stream with shuffled colours counts). Difficulty enters only through density: the
allowance ``q95_human`` comes from Taiko-1000 training charts of every course. Held-out human
test charts score 0.1-0.5 s/min at every course. ``excess_monotony`` is the window-level variant
(share of windows with ``R`` above the human 95th percentile at that density); it cannot flag
very sparse or very dense windows, where human repetition already reaches ``R = 1``.

The original, uncalibrated form follows.

A player notices a pattern that goes on unchanged for a long time (``don don don don ...``, or
``don don ka`` over and over) far more than any single note. Each onset becomes a symbol --
its action (don, ka, big, roll, balloon...) and the interval to the next onset in beats --
and a window of ``window_s`` seconds is **monotone** when its symbol sequence is periodic with
a short period: some ``p <= max_period`` with ``s[i] == s[i + p]`` for at least ``match`` of
the positions. Windows with fewer than ``min_notes`` onsets (rests, sparse intros) are not
judged.

Per chart:

* ``monotone_share``: share of judged windows that are monotone;
* ``longest_monotone_s``: longest stretch of time covered by consecutive monotone windows;
* ``period_hist``: how often each period occurs among monotone windows.

Intervals are in the chart's own beats, so periodicity does not depend on tempo or on the
notation octave.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import cache

import numpy as np

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.views import active_events

IOI_CLASSES = tuple(
    Fraction(n, d)
    for n, d in (
        (1, 12),
        (1, 8),
        (1, 6),
        (1, 4),
        (1, 3),
        (3, 8),
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 1),
        (3, 2),
        (2, 1),
    )
)


def symbols(chartset: ChartSet, chart: Chart) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Onset times (s) and one symbol ``(action, interval class)`` per onset."""
    timing = chartset.timing_of(chart)
    events = sorted(active_events(chart), key=lambda e: e.time)
    by_onset: dict[int, list] = {}
    for e in events:
        by_onset.setdefault(timing.us(e.time), []).append(e)
    onsets = sorted(by_onset)
    out: list[tuple[str, int]] = []
    beats = [by_onset[u][0].time.beat for u in onsets]
    for i, u in enumerate(onsets):
        e = by_onset[u][0]
        action = e.role.split(":")[-1] + ("+" if e.modifiers else "")
        if i + 1 < len(onsets):
            gap = beats[i + 1] - beats[i]
            cls = min(range(len(IOI_CLASSES)), key=lambda k: abs(IOI_CLASSES[k] - gap))
            if abs(IOI_CLASSES[cls] - gap) > Fraction(1, 48):
                cls = len(IOI_CLASSES)  # longer or irregular gap
        else:
            cls = len(IOI_CLASSES)
        out.append((action, cls))
    return np.asarray(onsets, dtype=np.float64) / 1e6, out


def period(seq: list, max_period: int, match: float) -> int | None:
    """Smallest period ``p <= max_period`` that ``seq`` follows at ``match`` of positions."""
    n = len(seq)
    for p in range(1, min(max_period, n // 2) + 1):
        hits = sum(seq[i] == seq[i + p] for i in range(n - p))
        if hits >= match * (n - p):
            return p
    return None


def monotony(
    chartset: ChartSet,
    chart: Chart,
    window_s: float = 10.0,
    hop_s: float = 1.0,
    max_period: int = 8,
    match: float = 0.9,
    min_notes: int = 8,
    rhythm_only: bool = False,
) -> dict:
    """See the module docstring; ``rhythm_only`` ignores the action (don/ka...) so a constant
    rhythm with shuffled colours also counts as monotone."""
    times, syms = symbols(chartset, chart)
    if rhythm_only:
        syms = [(None, cls) for _, cls in syms]
    if len(times) < min_notes:
        return {"windows": 0, "monotone_share": None, "longest_monotone_s": 0.0}
    starts = np.arange(times[0], max(times[-1] - window_s, times[0]) + hop_s / 2, hop_s)
    judged, mono = 0, []
    periods: Counter[int] = Counter()
    for s in starts:
        a, b = np.searchsorted(times, [s, s + window_s])
        seq = syms[a:b]
        if len(seq) < min_notes:
            mono.append(False)
            continue
        judged += 1
        p = period(seq, max_period, match)
        mono.append(p is not None)
        if p is not None:
            periods[p] += 1
    # longest time covered by consecutive monotone windows
    longest, run = 0.0, 0
    for flag in mono:
        run = run + 1 if flag else 0
        if run:
            longest = max(longest, window_s + (run - 1) * hop_s)
    return {
        "windows": judged,
        "monotone_share": sum(mono) / judged if judged else None,
        "longest_monotone_s": longest,
        "period_hist": dict(sorted(periods.items())),
    }


# -- continuous, density-calibrated form -------------------------------------------------


def repetition(seq: list, max_period: int = 16) -> float:
    """``R = max_p mean_i [s_i == s_{i+p}]`` over periods ``p <= max_period`` (``1`` = one
    pattern repeated exactly)."""
    n = len(seq)
    best = 0.0
    for p in range(1, min(max_period, n // 2) + 1):
        hits = sum(seq[i] == seq[i + p] for i in range(n - p))
        best = max(best, hits / (n - p))
    return best


def window_scores(
    chartset: ChartSet,
    chart: Chart,
    window_s: float = 10.0,
    hop_s: float = 1.0,
    min_notes: int = 8,
    max_period: int = 16,
) -> list[tuple[float, float, float, float]]:
    """``(start s, notes per second, R rhythm and colour, R rhythm only)`` per judged window."""
    times, syms = symbols(chartset, chart)
    if len(times) < min_notes:
        return []
    rhythm = [(None, c) for _, c in syms]
    out = []
    for s in np.arange(times[0], max(times[-1] - window_s, times[0]) + hop_s / 2, hop_s):
        a, b = (int(x) for x in np.searchsorted(times, [s, s + window_s]))
        if b - a < min_notes:
            continue
        out.append(
            (
                float(s),
                (b - a) / window_s,
                repetition(syms[a:b], max_period),
                repetition(rhythm[a:b], max_period),
            )
        )
    return out


class Reference:
    """Human ``R`` quantiles by local density (log2 notes per second, ``BIN`` wide bins),
    built from training charts; densities outside the table use the nearest bin."""

    BIN = 0.25

    def __init__(self, table: dict) -> None:
        self.table = table  # {"full": {bin: q}, "rhythm": {bin: q}}

    @classmethod
    def build(cls, windows: list[tuple[float, float, float]], q: float = 0.95, min_count=30):
        """``windows``: ``(notes per second, R full, R rhythm)`` of human charts."""
        by: dict[int, list] = {}
        for d, rf, rr in windows:
            by.setdefault(int(np.floor(np.log2(d) / cls.BIN)), []).append((rf, rr))
        table: dict[str, dict[str, float]] = {"full": {}, "rhythm": {}, "count": {}}
        for k, v in sorted(by.items()):
            if len(v) < min_count:
                continue
            arr = np.asarray(v)
            table["full"][str(k)] = float(np.quantile(arr[:, 0], q))
            table["rhythm"][str(k)] = float(np.quantile(arr[:, 1], q))
            table["count"][str(k)] = len(v)
        return cls(table)

    def threshold(self, kind: str, density: float) -> float:
        bins = sorted(int(k) for k in self.table[kind])
        k = int(np.floor(np.log2(max(density, 1e-3)) / self.BIN))
        k = min(max(k, bins[0]), bins[-1])
        if str(k) not in self.table[kind]:  # gap: nearest available bin
            k = min(bins, key=lambda b: abs(b - k))
        return self.table[kind][str(k)]


def excess_monotony(
    chartset: ChartSet,
    chart: Chart,
    ref: Reference,
    kind: str = "rhythm",
    hop_s: float = 1.0,
    window_s: float = 10.0,
) -> dict:
    """Share of 10 s windows more repetitive than 95% of human windows of the same density
    (humans: about 5% at every density and difficulty), and the longest such stretch."""
    ws = window_scores(chartset, chart, window_s, hop_s)
    if not ws:
        return {"windows": 0, "excess_share": None, "excess_longest_s": 0.0}
    col = 2 if kind == "full" else 3
    flags = [w[col] > ref.threshold(kind, w[1]) for w in ws]
    longest, run, prev = 0.0, 0, None
    for (start, *_), flag in zip(ws, flags, strict=True):
        contiguous = prev is not None and start - prev <= hop_s + 1e-6
        run = (run + 1 if contiguous else 1) if flag else 0
        if flag:
            longest = max(longest, window_s + (run - 1) * hop_s)
        prev = start
    return {
        "windows": len(ws),
        "excess_share": float(np.mean(flags)),
        "excess_longest_s": longest,
        "mean_R": float(np.mean([w[col] for w in ws])),
    }


def stretches(
    chartset: ChartSet,
    chart: Chart,
    tau: float = 0.9,
    kind: str = "rhythm",
    hop_s: float = 1.0,
    window_s: float = 10.0,
) -> tuple[list[tuple[float, float]], float]:
    """Monotone stretches: maximal runs of consecutive windows with ``R >= tau``, as
    ``(duration s, median notes per second)``, and the chart's judged duration (s)."""
    ws = window_scores(chartset, chart, window_s, hop_s)
    if not ws:
        return [], 0.0
    col = 2 if kind == "full" else 3
    out: list[tuple[float, float]] = []
    run: list[tuple[float, float]] = []
    prev = None
    for start, d, *rs in ws:
        hit = rs[col - 2] >= tau
        contiguous = prev is not None and start - prev <= hop_s + 1e-6
        if hit and (contiguous or not run):
            run.append((start, d))
        else:
            if run:
                out.append(
                    (window_s + (len(run) - 1) * hop_s, float(np.median([x for _, x in run])))
                )
            run = [(start, d)] if hit else []
        prev = start
    if run:
        out.append((window_s + (len(run) - 1) * hop_s, float(np.median([x for _, x in run]))))
    duration = ws[-1][0] - ws[0][0] + window_s
    return out, duration


class StretchReference:
    """Human 95th-percentile monotone-stretch duration by density (``BIN`` = 0.5 log2)."""

    BIN = 0.5

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table

    @classmethod
    def build(cls, items: list[tuple[float, float]], q: float = 0.95, min_count: int = 30):
        by: dict[int, list[float]] = {}
        for length, d in items:
            by.setdefault(int(np.floor(np.log2(d) / cls.BIN)), []).append(length)
        return cls(
            {str(k): float(np.quantile(v, q)) for k, v in sorted(by.items()) if len(v) >= min_count}
        )

    def allowance(self, density: float) -> float:
        bins = sorted(int(k) for k in self.table)
        k = int(np.floor(np.log2(max(density, 1e-3)) / self.BIN))
        k = min(max(k, bins[0]), bins[-1])
        k = min(bins, key=lambda b: abs(b - k))
        return self.table[str(k)]


def excess_monotone_time(
    chartset: ChartSet, chart: Chart, ref: StretchReference, tau: float = 0.9
) -> dict:
    """``E_time``: seconds per minute of monotone stretches beyond what humans write at the
    same density (sum of ``max(0, L - q95(d))`` over stretches, per minute of chart), and the
    largest single excess."""
    items, duration = stretches(chartset, chart, tau)
    if duration <= 0:
        return {"excess_s_per_min": None, "max_excess_s": 0.0}
    excess = [max(0.0, length - ref.allowance(d)) for length, d in items]
    return {
        "excess_s_per_min": 60 * sum(excess) / duration,
        "max_excess_s": max(excess, default=0.0),
        "stretches": len(items),
    }


@cache
def default_references() -> tuple[Reference, StretchReference]:
    """References built from Taiko-1000 training charts (``data/monotony_reference.json``)."""
    import json
    from pathlib import Path

    data = json.loads((Path(__file__).parent / "data" / "monotony_reference.json").read_text())
    return Reference(data["window_q95"]), StretchReference(data["stretch_q95"])
