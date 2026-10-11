"""simai (``maidata.txt``) writer.

Ticks are placed on exact beats: every event, tempo change and hi-speed change
becomes a tick, and the writer chooses ``{div}`` values so that each gap between
ticks is a whole number of commas. Durations are written in their original
notation when the reader preserved it (``native["dur"]``), otherwise derived
from the event times.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from math import lcm

from urfm.codecs.base import CodecError
from urfm.codecs.simai import syntax as sx
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event, Primitive
from urfm.ir.targets import RingButton, TouchRegion
from urfm.ir.timing import TimingMap

MAX_RUN_DIV = 64


@dataclass
class _Tick:
    tempo: Fraction | None = None
    speed: float | None = None
    parts: dict[int, list[str]] = field(default_factory=lambda: defaultdict(list))


def write(chartset: ChartSet) -> bytes:
    song = chartset.song
    lines = [f"&title={song.title}", f"&artist={song.artist}"]
    if song.creator:
        lines.append(f"&des={song.creator}")
    if song.genre:
        lines.append(f"&genre={song.genre}")
    if chartset.timing.offset_us:
        lines.append(f"&first={sx.format_number(chartset.timing.offset_us / sx.US_PER_SECOND)}")
    lines.extend(f"&{key}={value}" for key, value in song.extra.items())

    used: set[int] = set()
    for chart in chartset.charts:
        slot = _slot(chart, used)
        used.add(slot)
        timing = chartset.timing_of(chart)
        if timing.offset_us != chartset.timing.offset_us:
            raise CodecError("simai shares &first between difficulties")
        if timing.stops or timing.warps:
            raise CodecError("simai cannot express stops or warps")
        if chart.meta.get("level"):
            lines.append(f"&lv_{slot}={chart.meta['level']}")
        if chart.meta.get("designer"):
            lines.append(f"&des_{slot}={chart.meta['designer']}")
        lines.append(f"&inote_{slot}=")
        lines.append(write_fumen(chart, timing))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _slot(chart: Chart, used: set[int]) -> int:
    slot = chart.native.get("slot")
    if not isinstance(slot, int):
        slot = sx.LABEL_SLOTS.get(chart.difficulty.label or "")
    if slot is None or slot in used:
        slot = next(n for n in range(1, 100) if n not in used)
    return slot


# -- note stream ----------------------------------------------------------------


def write_fumen(chart: Chart, timing: TimingMap) -> str:
    ticks: dict[Fraction, _Tick] = defaultdict(_Tick)
    for tempo in timing.tempos:
        ticks[tempo.beat].tempo = tempo.bpm
    for control in chart.controls:
        if control.kind != "scroll" or not isinstance(control.value, int | float):
            raise CodecError(f"simai cannot express control {control.kind!r}")
        if control.time.residual_us:
            raise CodecError("simai controls must lie on beats")
        ticks[control.time.beat].speed = float(control.value)

    by_group: dict[int, list[Event]] = defaultdict(list)
    for ev in chart.events:
        if ev.group is not None:
            by_group[ev.group].append(ev)
    done: set[int] = set()
    for ev in chart.events:
        if ev.time.residual_us:
            raise CodecError("simai events must lie on exact beats")
        if ev.group is not None and ev.role in ("maimai:star", "maimai:slide"):
            if ev.group in done:
                continue
            done.add(ev.group)
            text = _slide_note(by_group[ev.group], timing)
        else:
            text = _note(ev, timing)
        k = int(ev.native.get("pseudo", 0))
        ticks[ev.time.beat - k * sx.PSEUDO_EACH_STEP].parts[k].append(text)

    beats = sorted(ticks)
    if not beats or beats[0] < 0:
        raise CodecError("simai charts start at beat 0")
    divs = _choose_divs([b - a for a, b in pairwise(beats)])
    out: list[str] = []
    current_div: Fraction | None = None
    for i, beat in enumerate(beats):
        tick = ticks[beat]
        if tick.tempo is not None:
            out.append(f"({sx.format_number(tick.tempo)})")
        if tick.speed is not None:
            out.append(f"<HS*{_float(tick.speed)}>")
        div = divs[i] if i < len(divs) else current_div or Fraction(4)
        if div != current_div:
            out.append(f"{{{sx.format_number(div)}}}")
            current_div = div
        if tick.parts:
            out.append(
                "`".join("/".join(tick.parts.get(k, [])) for k in range(max(tick.parts) + 1))
            )
        if i + 1 < len(beats):
            commas = (beats[i + 1] - beat) * div / 4
            assert commas.denominator == 1
            out.append("," * commas.numerator)
            if beats[i + 1] // 4 > beat // 4:
                out.append("\n")
        else:
            out.append(",\n")
    out.append("E")
    return "".join(out)


def _need(gap: Fraction) -> int:
    """Smallest ``{div}`` that makes ``gap`` beats a whole number of commas."""
    return (gap / 4).denominator


def _choose_divs(gaps: list[Fraction]) -> list[Fraction]:
    """One ``{div}`` per gap, reusing a div over runs of compatible gaps."""
    divs: list[Fraction] = []
    i = 0
    while i < len(gaps):
        div = _need(gaps[i])
        j = i + 1
        while j < len(gaps):
            merged = lcm(div, _need(gaps[j]))
            if merged != div and merged > MAX_RUN_DIV:
                break
            div = merged
            j += 1
        divs.extend([Fraction(div)] * (j - i))
        i = j
    return divs


def _float(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(value)


# -- notes -----------------------------------------------------------------------


def _location(ev: Event) -> str:
    target = ev.target
    if isinstance(target, RingButton):
        return str(target.index + 1)
    if isinstance(target, TouchRegion):
        return target.name
    raise CodecError(f"maimai event without ring/touch target: {ev.role}")


def _decorations(ev: Event) -> str:
    try:
        return "".join(sx.MODIFIER_CHARS[m] for m in ev.modifiers)
    except KeyError as exc:
        raise CodecError(f"simai has no modifier {exc.args[0]!r}") from None


def _note(ev: Event, timing: TimingMap) -> str:
    text = _location(ev) + _decorations(ev)
    appearance = ev.native.get("appearance")
    if ev.role == "maimai:star" and appearance not in ("$", "$$"):
        appearance = "$"
    if appearance:
        text += appearance
    if ev.native.get("head"):
        text += ev.native["head"]
    if ev.primitive is Primitive.HOLD:
        assert ev.end is not None
        if "dur" in ev.native:
            return f"{text}h[{ev.native['dur']}]"
        if ev.end == ev.time:
            return f"{text}h"
        length = timing.beat_to_us_exact(ev.end.beat) - timing.beat_to_us_exact(ev.time.beat)
        bpm = timing.bpm_at(ev.time.beat)
        ratio = sx.ratio_for(length, bpm)
        return f"{text}h[{ratio or '#' + sx.seconds_for(length)}]"
    if ev.primitive is not Primitive.HIT:
        raise CodecError(f"unsupported maimai event {ev.role}/{ev.primitive}")
    return text


def _slide_note(events: list[Event], timing: TimingMap) -> str:
    heads = [e for e in events if e.role == "maimai:star"]
    slides = [e for e in events if e.role == "maimai:slide"]
    if len(heads) > 1 or not slides:
        raise CodecError("a slide group needs at most one star and at least one slide")
    if heads:
        head = heads[0]
        text = _location(head) + _decorations(head) + head.native.get("appearance", "")
    else:
        text = _location(slides[0]) + slides[0].native.get("head", "?")
    return text + "*".join(_slide_body(s, timing) for s in slides)


def _slide_body(ev: Event, timing: TimingMap) -> str:
    bpm = timing.bpm_at(ev.time.beat)
    start_us = timing.beat_to_us_exact((ev.path[0].start or ev.time).beat)
    wait = start_us - timing.beat_to_us_exact(ev.time.beat)
    first_bracket = True
    previous_us = start_us
    out: list[str] = []
    for seg in ev.path:
        out.append(seg.shape.removeprefix("maimai:") + "".join(_buttons(seg.points[1:])))
        if seg.end is None:
            continue
        end_us = timing.beat_to_us_exact(seg.end.beat)
        if "dur" in seg.native:
            body = seg.native["dur"]
        else:
            length = end_us - previous_us
            body = sx.ratio_for(length, bpm) or "#" + sx.seconds_for(length)
            if first_bracket and wait != sx.beats_us(Fraction(1), bpm):
                body = f"{sx.seconds_for(wait)}##{body.removeprefix('#')}"
        out.append(f"[{body}]")
        previous_us = end_us
        first_bracket = False
    if "break" in ev.modifiers:
        out.append("b")
    return "".join(out)


def _buttons(points: tuple) -> list[str]:
    out = []
    for point in points:
        if not isinstance(point, RingButton):
            raise CodecError("slide vertices must be ring buttons")
        out.append(str(point.index + 1))
    return out
