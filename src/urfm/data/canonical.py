"""Separate musical tempo from visual tempo tricks (time-exact).

In Taiko (and osu!taiko) the scroll speed of a note is proportional to ``BPM x scroll``, so
charters often change the BPM only to change how notes move: a few beats at BPM 32 to make
notes crawl, then back to 128. Such a "visual" tempo segment is not audible. A timing model
cannot predict it from audio, and a note model conditioned on audio-derived timing never
sees it.

:func:`canonicalize` rewrites visual segments as the surrounding musical tempo plus an
equivalent scroll multiplier (``old BPM / musical BPM``). Every event keeps its exact
absolute time and its exact on-screen speed; only the notation changes.

A segment is visual when its notes (and its end) sit on a *simpler* grid of the previous
musical tempo than of its own tempo. Notes in a genuine 150 -> 180 BPM change fall on
1/4 beats of 180 but on 5/24 beats of 150, so it stays; notes of a BPM-32 crawl inside a
128 BPM song fall on 1/16 beats of 32 but 1/4 beats of 128, so it merges. Octave ratios
(x2, x1/2) with equally simple grids keep whichever tempo is closer to the song's dominant one.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

import msgspec

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.controls import Control
from urfm.ir.time import Time
from urfm.ir.timing import MeterChange, Stop, TempoChange, TimingMap
from urfm.ir.transform import map_chart_times

SCROLL_FOLLOWS_BPM = frozenset({"taiko", "osu_taiko"})
"""Games whose note speed is proportional to BPM x scroll (visual tempo is meaningful)."""
SCROLL_KIND = {"taiko": "scroll", "osu_taiko": "osu:sv"}
TEMPO_RANGE = (10, 2000)
"""Maps with BPM outside this range (osu! red-line gimmicks at millions of BPM) are left
alone: their beat numbers are astronomically large and measures cannot be enumerated."""
MAX_DEN = 48
"""Positions on the musical grid must be multiples of 1/48 beat to count as simple."""


@dataclass
class CanonicalStats:
    segments: int = 0
    merged: int = 0
    kept: int = 0
    reasons: Counter[str] = field(default_factory=Counter)


def _den(xs: list[Fraction]) -> int:
    d = 1
    for x in xs:
        d = max(d, x.denominator)
    return d


def _is_octave(r: Fraction) -> bool:
    while r.numerator % 2 == 0 and r.numerator > 1:
        r = Fraction(r.numerator // 2, r.denominator)
    while r.denominator % 2 == 0 and r.denominator > 1:
        r = Fraction(r.numerator, r.denominator // 2)
    return r == 1


def _merges(
    xs: list[Fraction], b0: Fraction, n0: Fraction, own: Fraction, cand: Fraction, dominant
) -> tuple[bool, str]:
    """Whether a segment at ``own`` BPM whose positions ``xs`` start at old beat ``b0`` (new
    beat ``n0``) reads more simply at ``cand`` BPM."""
    if own == cand:
        return True, "same_bpm"
    if not xs:
        return True, "empty_tail"
    s = cand / own
    old_den = _den(xs)
    new_den = _den([n0 + (x - b0) * s for x in xs])
    if new_den > MAX_DEN:
        return False, "off_grid"
    if new_den < old_den:
        return True, "simpler_grid"
    if new_den == old_den and _is_octave(own / cand):
        closer = abs(math.log(cand / dominant)) <= abs(math.log(own / dominant))
        return closer, "octave" if closer else "octave_kept"
    return False, "musical"


def canonicalize(
    chartset: ChartSet, chart: Chart, stats: CanonicalStats | None = None
) -> tuple[ChartSet, Chart]:
    """``(chartset, chart)`` with visual tempo segments folded into scroll controls.

    Charts of games without BPM-dependent scrolling, with warps, or with a single tempo are
    returned unchanged. The returned chart carries its own timing override.
    """
    stats = stats if stats is not None else CanonicalStats()
    timing = chartset.timing_of(chart)
    tempos = sorted(timing.tempos, key=lambda t: t.beat)
    if chart.mechanics not in SCROLL_FOLLOWS_BPM or timing.warps or len(tempos) < 2:
        return chartset, chart
    if any(not TEMPO_RANGE[0] <= t.bpm <= TEMPO_RANGE[1] for t in tempos):
        stats.reasons["extreme_tempo"] += 1
        return chartset, chart
    if any(m.measure_beats <= 0 for m in timing.meters):
        stats.reasons["bad_meter"] += 1
        return chartset, chart
    # drop tempo changes at or before the first one (only the last at a beat counts)
    dedup: list[TempoChange] = []
    for t in tempos:
        if dedup and dedup[-1].beat == t.beat:
            dedup[-1] = t
        else:
            dedup.append(t)
    tempos = dedup
    starts = [t.beat for t in tempos]
    onsets = sorted({e.time.beat for e in chart.events} | {c.time.beat for c in chart.controls})
    if onsets and max(onsets[-1], starts[-1]) - min(onsets[0], starts[0]) > 50_000:
        stats.reasons["huge_span"] += 1
        return chartset, chart

    # the tempo that lasts longest breaks octave ties (x2 / x1/2 notation of the same grid)
    spans: dict[Fraction, Fraction] = {}
    last = max(onsets[-1] if onsets else starts[-1], starts[-1])
    for i, t in enumerate(tempos):
        end = starts[i + 1] if i + 1 < len(tempos) else last
        spans[t.bpm] = spans.get(t.bpm, Fraction(0)) + max(end - t.beat, 0) / t.bpm
    dominant = max(spans, key=lambda bpm: spans[bpm])

    # walk segments, deciding musical tempo and new start beat for each
    new_start = [starts[0]]
    musical = [tempos[0].bpm]
    scale = [Fraction(1)]
    for i in range(1, len(tempos)):
        stats.segments += 1
        b0 = starts[i]
        b1 = starts[i + 1] if i + 1 < len(tempos) else None
        prev_scale = scale[-1]
        n0 = new_start[-1] + (b0 - starts[i - 1]) * prev_scale
        own = tempos[i].bpm
        lo = bisect.bisect_left(onsets, b0)
        hi = bisect.bisect_left(onsets, b1) if b1 is not None else len(onsets)
        xs = onsets[lo:hi] + ([b1] if b1 is not None else [])
        # the previous musical tempo, or the song's dominant one (a crawl at 1/8 of the
        # base tempo right after a real tempo change still folds back into the base)
        merge, why, cand = False, "musical", own
        for c in dict.fromkeys((musical[-1], dominant)):
            ok, reason = _merges(xs, b0, n0, own, c, dominant)
            if ok:
                merge, why, cand = True, reason, c
                break
            if why == "musical":
                why = reason
        s = cand / own  # new beats per old beat if merged
        stats.reasons[why] += 1
        if merge:
            stats.merged += 1
            musical.append(cand)
            scale.append(s)
        else:
            stats.kept += 1
            musical.append(own)
            scale.append(Fraction(1))
        new_start.append(n0)

    if all(x == 1 for x in scale):
        return chartset, chart

    def new_beat(beat: Fraction) -> Fraction:
        i = max(bisect.bisect_right(starts, beat) - 1, 0)
        if beat < starts[0]:
            return beat
        return new_start[i] + (beat - starts[i]) * scale[i]

    def fn(t: Time) -> Time:
        return Time(new_beat(t.beat), t.residual_us)

    new_tempos: list[TempoChange] = []
    for b, bpm in zip(new_start, musical, strict=True):
        if new_tempos and new_tempos[-1].bpm == bpm:
            continue
        new_tempos.append(TempoChange(b, bpm))
    new_timing = msgspec.structs.replace(
        timing,
        tempos=tuple(new_tempos),
        stops=tuple(Stop(new_beat(s.beat), s.duration_us, s.before) for s in timing.stops),
        meters=_meters(timing, new_beat, onsets[-1] if onsets else starts[-1]),
    )
    mapped = map_chart_times(chart, fn)
    visual = [tempos[i].bpm / musical[i] for i in range(len(tempos))]
    controls = _scroll_controls(chart, mapped, starts, visual, new_beat)
    out = msgspec.structs.replace(mapped, controls=controls, timing=new_timing)
    return chartset, out


def _meters(timing: TimingMap, new_beat, last_beat: Fraction) -> tuple[MeterChange, ...]:
    """Meters whose measures start exactly where the old ones did (lengths re-expressed in
    new beats; unusual lengths get non-power-of-two denominators)."""
    until = last_beat + 4 * timing.meter_at(last_beat).measure_beats
    starts = timing.measure_starts(until)
    if len(starts) < 2:
        return tuple(
            MeterChange(new_beat(m.beat), m.numerator, m.denominator) for m in timing.meters
        )
    out: list[MeterChange] = []
    for a, b in itertools.pairwise(starts):
        length = new_beat(b) - new_beat(a)
        old = timing.meter_at(a)
        if length == old.measure_beats:
            num, den = old.numerator, old.denominator
        else:
            q = Fraction(length) / 4  # num / den
            num, den = q.numerator, q.denominator
            while den < 4 and num % 2 == 0:  # prefer x/4 style when exact
                num, den = num * 2, den * 2
        if not out or (out[-1].numerator, out[-1].denominator) != (num, den):
            out.append(MeterChange(new_beat(a), num, den))
    # meters after the last enumerated measure keep their own definition
    tail = [
        MeterChange(new_beat(m.beat), m.numerator, m.denominator)
        for m in timing.meters
        if m.beat > starts[-1]
    ]
    return tuple(out + tail)


def _scroll_controls(
    original: Chart,
    mapped: Chart,
    starts: list[Fraction],
    visual: list[Fraction],
    new_beat,
) -> tuple[Control, ...]:
    """Scroll values multiplied by each segment's visual factor; a control is added wherever
    the factor changes. Branch-specific scroll controls are scaled in place."""
    kind = SCROLL_KIND[original.mechanics]
    scrolls = sorted(
        (c for c in original.controls if c.kind == kind and c.branch is None),
        key=lambda c: c.time,
    )
    sbeats = [c.time.beat for c in scrolls]

    def base_at(beat: Fraction) -> float:
        i = bisect.bisect_right(sbeats, beat) - 1
        if i < 0:
            return 1.0
        v = scrolls[i].value
        return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else 1.0

    def factor_at(beat: Fraction) -> Fraction:
        return visual[max(bisect.bisect_right(starts, beat) - 1, 0)]

    out: list[Control] = []
    for old, new in zip(original.controls, mapped.controls, strict=True):
        if old.kind == kind and isinstance(old.value, int | float) and old.value is not True:
            value = round(float(old.value) * float(factor_at(old.time.beat)), 6)
            out.append(msgspec.structs.replace(new, value=value))
        else:
            out.append(new)
    have = {c.time.beat for c in out if c.kind == kind and c.branch is None}
    for i, b in enumerate(starts):
        if i and visual[i] != visual[i - 1] and new_beat(b) not in have:
            value = round(base_at(b) * float(visual[i]), 6)
            out.append(Control(time=Time(new_beat(b)), kind=kind, value=value))
    return tuple(sorted(out, key=lambda c: c.time))


def rescroll(old_chart: Chart, old: TimingMap, new_chart: Chart, new: TimingMap) -> Chart:
    """Scroll controls for ``new_chart`` (``old_chart`` re-notated on timing ``new``) that keep
    every note's on-screen speed (``BPM x scroll``) where BPM-following games are concerned.

    Values are snapped to the tokenizer's scroll grid and only changes are kept, so the small
    tempo wobbles of a predicted map do not turn into streams of near-identical controls.
    """
    from urfm.tokenize.vocab import SCROLL_GRID, scroll_bin

    kind = SCROLL_KIND.get(old_chart.mechanics)
    if kind is None:
        return new_chart
    scrolls = sorted(
        (c for c in old_chart.controls if c.kind == kind and c.branch is None),
        key=lambda c: old.us(c.time),
    )
    s_us = [old.us(c.time) for c in scrolls]
    times = sorted(
        {0, *s_us}
        | {round(old.beat_to_us_exact(t.beat)) for t in old.tempos}
        | {round(new.beat_to_us_exact(t.beat)) for t in new.tempos}
    )

    def scroll_at(us: int) -> float:
        i = bisect.bisect_right(s_us, us) - 1
        v = scrolls[i].value if i >= 0 else 1.0
        return float(v) if isinstance(v, int | float) and not isinstance(v, bool) else 1.0

    out = [c for c in new_chart.controls if c.kind != kind or c.branch is not None]
    previous = 1.0
    for us in times:
        speed = float(old.bpm_at(old.us_to_beat(us))) * scroll_at(us)
        value = speed / float(new.bpm_at(new.us_to_beat(us)))
        if value <= 0:
            continue
        snapped = SCROLL_GRID[scroll_bin(value)]
        if snapped != previous:
            out.append(Control(time=new.time_at_us(max(us, 0)), kind=kind, value=snapped))
            previous = snapped
    return msgspec.structs.replace(new_chart, controls=tuple(sorted(out, key=lambda c: c.time)))
