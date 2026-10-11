"""Exact transformations of charts and timing maps (augmentation and normalisation).

Two kinds of change keep a chart's meaning while altering how it is written or played:

* **Re-timing** (:func:`warp_timing`): the audio is played faster or slower, globally or in
  short beat spans. Beats do not move; only the timing map changes, so every event keeps
  its beat position and the new map tells where it now sounds.
* **Re-notation** (:func:`rescale_beats`, :func:`map_chart_times`): the same sound written
  on a different beat grid (e.g. at double BPM, where 16th notes become 8th notes).
  Absolute times do not change; every beat position is rewritten.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from fractions import Fraction

import msgspec

from urfm.ir.chart import BranchSegment, Chart, ChartSet
from urfm.ir.controls import Control
from urfm.ir.events import Event, PathSegment
from urfm.ir.time import Time
from urfm.ir.timing import MeterChange, Stop, TempoChange, TimingMap, Warp

TimeFn = Callable[[Time], Time]


# -- re-notation -------------------------------------------------------------------------


def map_chart_times(chart: Chart, fn: TimeFn) -> Chart:
    """``chart`` with every :class:`Time` (events, ends, paths, controls, branches) mapped."""

    def opt(t: Time | None) -> Time | None:
        return None if t is None else fn(t)

    def path(seg: PathSegment) -> PathSegment:
        if seg.start is None and seg.end is None:
            return seg
        return msgspec.structs.replace(seg, start=opt(seg.start), end=opt(seg.end))

    def event(ev: Event) -> Event:
        return msgspec.structs.replace(
            ev, time=fn(ev.time), end=opt(ev.end), path=tuple(path(s) for s in ev.path)
        )

    def control(c: Control) -> Control:
        return msgspec.structs.replace(c, time=fn(c.time))

    def branch(b: BranchSegment) -> BranchSegment:
        return msgspec.structs.replace(b, start=fn(b.start), end=opt(b.end))

    return msgspec.structs.replace(
        chart,
        events=tuple(event(e) for e in chart.events),
        controls=tuple(control(c) for c in chart.controls),
        branches=tuple(branch(b) for b in chart.branches),
    )


def rescale_timing(timing: TimingMap, factor: Fraction) -> TimingMap:
    """The same tempo map written with ``factor`` times as many beats (``factor`` 2 = double
    BPM notation). Absolute time of every rescaled beat is unchanged; measures keep their
    duration (a 4/4 measure becomes 8/4 at ``factor`` 2, or 4/8 at ``factor`` 1/2)."""
    return msgspec.structs.replace(
        timing,
        tempos=tuple(TempoChange(t.beat * factor, t.bpm * factor) for t in timing.tempos),
        stops=tuple(msgspec.structs.replace(s, beat=s.beat * factor) for s in timing.stops),
        warps=tuple(Warp(w.beat * factor, w.length * factor) for w in timing.warps),
        meters=tuple(_scaled_meter(m, factor) for m in timing.meters),
    )


def _scaled_meter(m: MeterChange, factor: Fraction) -> MeterChange:
    """Meter at ``m.beat * factor`` whose measures last ``factor`` times as many beats; the
    numerator grows or the denominator shrinks, whichever stays an integer."""
    beats = m.measure_beats * factor  # = 4 * num / den
    num, den = m.numerator, m.denominator
    scaled = Fraction(num) * factor
    if scaled.denominator == 1:
        num = int(scaled)
    else:
        den_f = Fraction(den) / factor
        if den_f.denominator == 1:
            den = int(den_f)
        else:
            num, den = beats.numerator, 4 * beats.denominator
    return MeterChange(m.beat * factor, num, den)


def rescale_beats(chartset: ChartSet, factor: Fraction) -> ChartSet:
    """Re-notate a whole chart set on a ``factor`` times finer beat grid (same sound)."""

    def fn(t: Time) -> Time:
        return Time(t.beat * factor, t.residual_us)

    return msgspec.structs.replace(
        chartset,
        timing=rescale_timing(chartset.timing, factor),
        charts=tuple(
            msgspec.structs.replace(
                map_chart_times(c, fn),
                timing=None if c.timing is None else rescale_timing(c.timing, factor),
            )
            for c in chartset.charts
        ),
    )


# -- re-timing ---------------------------------------------------------------------------


class TimeWarp(msgspec.Struct, frozen=True):
    """A monotone piecewise-linear map of audio time ``t -> t'`` (µs, float).

    ``src``/``dst`` are matching knot times; outside the knots the first and last slopes
    extend. :meth:`inverse` gives, for output times, the source time to read audio from.
    """

    src: tuple[float, ...]
    dst: tuple[float, ...]
    first_slope: float
    last_slope: float

    def forward(self, t):
        import numpy as np

        return _piecewise(np.asarray(t, np.float64), self.src, self.dst, self)

    def inverse(self, t):
        import numpy as np

        inv = TimeWarp(self.dst, self.src, 1 / self.first_slope, 1 / self.last_slope)
        return _piecewise(np.asarray(t, np.float64), self.dst, self.src, inv)


def _piecewise(t, xs, ys, w: TimeWarp):
    import numpy as np

    if not xs:
        return t * w.first_slope
    out = np.interp(t, xs, ys)
    out = np.where(t < xs[0], ys[0] + (t - xs[0]) * w.first_slope, out)
    return np.where(t > xs[-1], ys[-1] + (t - xs[-1]) * w.last_slope, out)


def warp_timing(
    timing: TimingMap, knots: Sequence[Fraction], speeds: Sequence[Fraction]
) -> tuple[TimingMap, TimeWarp]:
    """Play beat spans faster or slower: ``speeds[i]`` applies to beats in
    ``[knots[i-1], knots[i])`` (``speeds[0]`` before the first knot, ``speeds[-1]`` after the
    last). With no knots, ``speeds[0]`` is a global tempo change.

    Returns the new timing map (same beats, tempos multiplied, stops divided by the local
    speed) and the audio-time map ``t -> t'`` (anchored at ``t = 0``) used to resample the
    audio. Knots must not coincide with stops.
    """
    knots = sorted(Fraction(k) for k in knots)
    speeds = [Fraction(s) for s in speeds]
    if len(speeds) != len(knots) + 1 or any(s <= 0 for s in speeds):
        raise ValueError("need len(knots) + 1 positive speeds")

    def speed_at(beat: Fraction) -> Fraction:
        return speeds[bisect.bisect_right(knots, beat)]

    # audio-time knots: source time of each beat knot and its image under the warp
    src = [timing.beat_to_us_exact(k) for k in knots]
    dst: list[Fraction] = []
    for i, t in enumerate(src):
        if i == 0:
            dst.append(t / speeds[0])
        else:
            dst.append(dst[-1] + (t - src[i - 1]) / speeds[i])
    warp = TimeWarp(
        tuple(float(x) for x in src),
        tuple(float(x) for x in dst),
        float(1 / speeds[0]),
        float(1 / speeds[-1]),
    )

    def image(t: Fraction) -> Fraction:
        i = bisect.bisect_right(src, t)
        if i == 0:
            return t / speeds[0]
        return dst[i - 1] + (t - src[i - 1]) / speeds[i]

    tempo_beats = sorted({t.beat for t in timing.tempos} | set(knots))
    tempos = tuple(TempoChange(b, timing.bpm_at(b) * speed_at(b)) for b in tempo_beats)
    stops = tuple(Stop(s.beat, s.duration_us / speed_at(s.beat), s.before) for s in timing.stops)
    new = msgspec.structs.replace(
        timing, offset_us=image(timing.offset_us), tempos=_merge_tempos(tempos), stops=stops
    )
    return new, warp


def _merge_tempos(tempos: Sequence[TempoChange]) -> tuple[TempoChange, ...]:
    out: list[TempoChange] = []
    for t in tempos:
        if out and out[-1].bpm == t.bpm:
            continue
        out.append(t)
    return tuple(out)


def stretch_timing(timing: TimingMap, speed: Fraction) -> tuple[TimingMap, TimeWarp]:
    """The whole song ``speed`` times faster (``speed`` 1.25 = 25% faster)."""
    return warp_timing(timing, (), (Fraction(speed),))


def warp_chartset(
    chartset: ChartSet, knots: Sequence[Fraction], speeds: Sequence[Fraction]
) -> tuple[ChartSet, TimeWarp]:
    """:func:`warp_timing` applied to the shared map and every chart override. Residuals
    (osu! timestamps off the beat grid) are scaled by the local speed."""
    timing, warp = warp_timing(chartset.timing, knots, speeds)
    ks = sorted(Fraction(k) for k in knots)
    sp = [Fraction(s) for s in speeds]

    def fn(t: Time) -> Time:
        if not t.residual_us:
            return t
        s = sp[bisect.bisect_right(ks, t.beat)]
        return Time(t.beat, round(t.residual_us / s))

    charts = []
    for c in chartset.charts:
        own = None if c.timing is None else warp_timing(c.timing, knots, speeds)[0]
        charts.append(msgspec.structs.replace(map_chart_times(c, fn), timing=own))
    return msgspec.structs.replace(chartset, timing=timing, charts=tuple(charts)), warp


def retime_chart(chart: Chart, old: TimingMap, new: TimingMap, resolution: int = 192) -> Chart:
    """Re-notate ``chart`` (timed by ``old``) on another timing map: every position becomes
    the nearest ``1/resolution`` beat of ``new`` plus the exact residual, so absolute times
    are unchanged (e.g. a human chart on an audio-predicted grid)."""

    def fn(t: Time) -> Time:
        return new.time_at_us(old.us(t), resolution)

    return msgspec.structs.replace(map_chart_times(chart, fn), timing=new)
