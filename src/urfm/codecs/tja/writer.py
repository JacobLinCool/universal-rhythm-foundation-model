"""IR -> TJA.

The writer is deterministic. Each measure gets the fewest characters that put
every note and command on an exact slot (the LCM of their position
denominators). Header lines are emitted only when they change the persistent
header state, mirroring how the reader interprets a file.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from fractions import Fraction
from typing import Any

from urfm.codecs.base import CodecError
from urfm.codecs.tja.common import (
    BALLOON_KEYS,
    FLAG_COMMANDS,
    NOTE_CHARS,
    RAW_PREFIX,
    ROLL_END,
    ROUTE_COMMANDS,
    SONG_FIELDS,
    US,
    format_decimal,
    format_float,
    parse_decimal,
)
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.controls import Control
from urfm.ir.events import Primitive
from urfm.ir.timing import TimingMap

Ctx = tuple[int, str] | None

COURSE_LABELS = {
    "easy": "Easy",
    "normal": "Normal",
    "hard": "Hard",
    "oni": "Oni",
    "edit": "Edit",
    "tower": "Tower",
    "dan": "Dan",
}


class _Item:
    """Something to place in the note stream: a slot character or a command line."""

    __slots__ = ("beat", "char", "command", "ctx", "on_emit", "order")

    def __init__(
        self,
        beat: Fraction,
        ctx: Ctx,
        order: int,
        *,
        char: str | None = None,
        command: str | None = None,
        on_emit: Callable[[], None] | None = None,
    ) -> None:
        self.beat = beat
        self.ctx = ctx
        self.order = order
        self.char = char
        self.command = command
        self.on_emit = on_emit


def write_tja(chartset: ChartSet) -> str:
    out: list[str] = []
    state: dict[str, str] = {}
    song = chartset.song
    timing = chartset.timing

    def header(key: str, value: str) -> None:
        out.append(f"{key}:{value}")
        state[key] = value

    header("TITLE", song.title)
    for key, attr in SONG_FIELDS.items():
        value = getattr(song, attr)
        if key != "TITLE" and value:
            header(key, value)
    header("BPM", format_decimal(timing.tempos[0].bpm))
    if chartset.audio is not None:
        header("WAVE", chartset.audio.path)
    header("OFFSET", format_decimal(-timing.offset_us / US))
    if song.preview is not None:
        header("DEMOSTART", format_decimal(Fraction(timing.us(song.preview), US)))
    for key, value in song.extra.items():
        header(key, value)
    song_state = dict(state)
    out.append("")

    for chart in chartset.charts:
        body, balloons = _ChartWriter(chart, chartset.timing_of(chart)).write()
        desired = _desired_headers(chart, chartset.timing_of(chart), balloons, song_state, state)
        for key, value in desired.items():
            if not _same(key, state.get(key), value):
                header(key, value)
        out.extend(chart.native.get("header_commands", []))
        player = chart.meta.get("player")
        out.append(f"#START P{player}" if player else "#START")
        out.extend(body)
        out.append("#END")
        out.append("")
    return "\n".join(out)


def _desired_headers(
    chart: Chart,
    timing: TimingMap,
    balloons: dict[str, list[int]],
    song_state: dict[str, str],
    state: dict[str, str],
) -> dict[str, str]:
    desired: dict[str, str] = {}
    label = chart.difficulty.label
    if label:
        desired["COURSE"] = COURSE_LABELS.get(label, label)
    if chart.difficulty.level is not None:
        desired["LEVEL"] = format_float(chart.difficulty.level)
    if timing.tempos and timing.tempos[0].beat == 0:
        desired["BPM"] = format_decimal(timing.tempos[0].bpm)
    desired["OFFSET"] = format_decimal(-timing.offset_us / US)
    for key, counts in balloons.items():
        desired[key] = ",".join(str(c) for c in counts)
    for key, value in chart.meta.items():
        if key != "player":
            desired[key] = value
    for key in state:
        if key not in desired and key in song_state:
            desired[key] = song_state[key]
    return desired


def _same(key: str, old: str | None, new: str) -> bool:
    if key == "BALLOON":
        return _ints(old) == _ints(new)
    if key in BALLOON_KEYS.values():  # a present-but-empty route list differs from a missing one
        return old is not None and _ints(old) == _ints(new)
    if old is None:
        return False
    if key in {"BPM", "OFFSET"}:
        try:
            return parse_decimal(old) == parse_decimal(new)
        except ValueError:
            return old == new
    return old == new


def _ints(text: str | None) -> list[int]:
    out = []
    for token in (text or "").split(","):
        try:
            out.append(int(token.strip()))
        except ValueError:
            continue
    return out


class _ChartWriter:
    def __init__(self, chart: Chart, timing: TimingMap) -> None:
        self.chart = chart
        self.timing = timing
        self.native = chart.native
        self.end_beat = Fraction(self.native.get("end_beat", "0"))
        self.segments = sorted(chart.branches, key=lambda b: b.start.beat)
        self.balloon_keys: list[str] = ["BALLOON", *self.native.get("balloon_keys", [])]
        self.balloons: dict[str, list[int]] = {k: [] for k in self.balloon_keys}
        self.items: list[_Item] = []
        self.order = 0
        self.lines: list[str] = []
        self.emitted: set[int] = set()
        self.divergent: dict[str, dict[str, list[list[str]]]] = self.native.get("route_timing", {})
        self.route_end: dict[str, str] = self.native.get("route_end", {})
        if self.segments:
            last = max(self._segment_end(i) for i in range(len(self.segments)))
            self.end_beat = max(self.end_beat, last)

    # -- spans -----------------------------------------------------------------

    def _segment_end(self, index: int) -> Fraction:
        seg = self.segments[index]
        if seg.end is not None:
            return seg.end.beat
        explicit = self.native.get("segment_end", {}).get(str(seg.id))
        if explicit is not None:
            return Fraction(explicit)
        if index + 1 < len(self.segments):
            return self.segments[index + 1].start.beat
        return Fraction(self.native.get("end_beat", "0"))

    def _route_end(self, index: int, route: str) -> Fraction:
        seg = self.segments[index]
        text = self.route_end.get(f"{seg.id}:{route}")
        return Fraction(text) if text is not None else self._segment_end(index)

    def _span_of(self, beat: Fraction) -> int | None:
        """Index of the branch segment covering ``beat`` (``None`` = shared region)."""
        for i, seg in enumerate(self.segments):
            if seg.start.beat <= beat < self._segment_end(i):
                return i
        return None

    # -- items -----------------------------------------------------------------

    def _add(self, beat: Fraction, ctx: Ctx, **kwargs: Any) -> None:
        self.order += 1
        self.items.append(_Item(beat, ctx, self.order, **kwargs))

    def _record_balloon(self, key: str, count: int | None) -> None:
        if count is not None:
            self.balloons.setdefault(key, []).append(count)

    def _collect(self) -> None:
        seg_ids = {seg.id for seg in self.segments}
        for ev in self.chart.events:
            ctx: Ctx = None if ev.branch is None else (ev.branch.segment, ev.branch.route)
            if ctx is not None and ctx[0] not in seg_ids:
                raise CodecError(f"event refers to unknown branch segment {ctx[0]}")
            char = NOTE_CHARS.get((ev.role, ev.modifiers))
            if char is None:
                raise CodecError(f"TJA has no note for {ev.role} {ev.modifiers}")
            if ev.time.residual_us:
                raise CodecError("TJA notes must lie exactly on beats (residual_us != 0)")
            on_emit: Callable[[], None] | None = None
            if ev.role == "taiko:balloon":
                key = BALLOON_KEYS[ctx[1]] if ctx is not None else "BALLOON"
                if key not in self.balloon_keys:
                    key = "BALLOON"
                on_emit = functools.partial(self._record_balloon, key, ev.count)

            self._add(ev.time.beat, ctx, char=char, on_emit=on_emit)
            if (
                ev.primitive is Primitive.REPEAT
                and ev.end is not None
                and not ev.native.get("implicit_end")
            ):
                self._add(ev.end.beat, self._ctx_for(ev.end.beat, ctx), char=ROLL_END)
        for control in self.chart.controls:
            ctx = None if control.branch is None else (control.branch.segment, control.branch.route)
            self._add(control.time.beat, ctx, command=_control_command(control))
        for beat_text, char, seg, route in self.native.get("unknown_notes", []):
            self._add(Fraction(beat_text), None if seg < 0 else (seg, route), char=char)

    def _ctx_for(self, beat: Fraction, preferred: Ctx) -> Ctx:
        span = self._span_of(beat)
        if span is None:
            return None
        seg = self.segments[span]
        if preferred is not None and preferred[1] in seg.routes:
            return (seg.id, preferred[1])
        return (seg.id, seg.routes[0])

    # -- timing ----------------------------------------------------------------

    def _shared_timing(self) -> list[tuple[Fraction, str]]:
        entries: list[tuple[Fraction, str]] = []
        for tempo in self.timing.tempos[1:]:
            entries.append((tempo.beat, f"#BPMCHANGE {format_decimal(tempo.bpm)}"))
        for stop in self.timing.stops:
            if not stop.before:
                raise CodecError("TJA can only express delays (Stop.before=True)")
            entries.append((stop.beat, f"#DELAY {format_decimal(stop.duration_us / US)}"))
        return entries

    def _route_timing(
        self, seg_id: int, route: str
    ) -> tuple[list[tuple[Fraction, str]], dict[Fraction, tuple[int, int]]] | None:
        lists = self.divergent.get(str(seg_id))
        if lists is None:
            return None
        entries: list[tuple[Fraction, str]] = []
        meters: dict[Fraction, tuple[int, int]] = {}
        for kind, beat_text, value in lists.get(route, []):
            beat = Fraction(beat_text)
            if kind == "bpm":
                entries.append((beat, f"#BPMCHANGE {format_decimal(Fraction(value))}"))
            elif kind == "delay":
                entries.append((beat, f"#DELAY {format_decimal(Fraction(value))}"))
            elif kind == "meter":
                num, den = value.split("/")
                meters[beat] = (int(num), int(den))
        return entries, meters

    # -- emission ----------------------------------------------------------------

    def write(self) -> tuple[list[str], dict[str, list[int]]]:
        self._collect()
        if self.items:  # charts not read from TJA carry no native end; cover every item
            last = max(i.beat for i in self.items)
            self.end_beat = max(self.end_beat, last + Fraction(1, 192))
        shared_timing = self._shared_timing()
        shared_meters = {m.beat: (m.numerator, m.denominator) for m in self.timing.meters}
        position = Fraction(0)
        for index, seg in enumerate(self.segments):
            self._span(None, position, seg.start.beat, shared_timing, shared_meters)
            if seg.condition:
                self.lines.append(f"#BRANCHSTART {seg.condition.get('raw', '')}".rstrip())
            for route in seg.routes:
                self.lines.append(f"#{ROUTE_COMMANDS[route]}")
                override = self._route_timing(seg.id, route)
                if override is None:
                    timing, meters = shared_timing, shared_meters
                else:
                    timing = override[0]
                    meters = dict(shared_meters)
                    for beat in [b for b in meters if b >= seg.start.beat]:
                        del meters[beat]
                    meters.update(override[1])
                self._span(
                    (seg.id, route),
                    seg.start.beat,
                    self._route_end(index, route),
                    timing,
                    meters,
                )
            if seg.end is not None:
                self.lines.append("#BRANCHEND")
            position = self._segment_end(index)
        self._span(None, position, self.end_beat, shared_timing, shared_meters)
        for key, extra in self.native.get("balloon_extra", {}).items():
            self.balloons.setdefault(key, []).extend(extra)
        return self.lines, self.balloons

    _after_cut = False
    """The previous measure was cut short, so the next one restates its meter."""

    def _span(
        self,
        ctx: Ctx,
        start: Fraction,
        end: Fraction,
        timing: list[tuple[Fraction, str]],
        meters: dict[Fraction, tuple[int, int]],
    ) -> None:
        own = [
            i
            for i in self.items
            if i.ctx == ctx and start <= i.beat <= end and id(i) not in self.emitted
        ]
        timed = [
            _Item(beat, ctx, -1 - n, command=text)
            for n, (beat, text) in enumerate(timing)
            if start <= beat < end
        ]
        meter_beats = sorted(meters)
        measure_start = start
        while measure_start < end:
            meter = _meter_at(meters, meter_beats, measure_start)
            measure_end = measure_start + Fraction(4 * meter[0], meter[1])
            # a meter change inside a measure starts a new measure there (as in
            # ``TimingMap.measure_starts``): the measure before it is cut short
            later = next((b for b in meter_beats if b > measure_start), None)
            cut = later is not None and later < measure_end
            if cut:
                assert later is not None
                measure_end = later
            inside = [
                i for i in own + timed if measure_start <= i.beat < measure_end and i.beat < end
            ]
            header = []
            if cut:
                short = (measure_end - measure_start) / 4
                header.append(f"#MEASURE {short.numerator}/{short.denominator}")
            elif (
                measure_start in meters and not (measure_start == 0 and meter == (4, 4))
            ) or self._after_cut:
                header.append(f"#MEASURE {meter[0]}/{meter[1]}")
            self._after_cut = cut
            self._measure(measure_start, measure_end, header, inside)
            measure_start = measure_end
        trailing = [i for i in own if i.beat == end and i.command is not None]
        for item in sorted(trailing, key=lambda i: i.order):
            self.lines.append(item.command or "")
            self.emitted.add(id(item))

    def _measure(
        self, start: Fraction, end: Fraction, header: list[str], items: list[_Item]
    ) -> None:
        self.emitted.update(id(i) for i in items)
        length = end - start
        positions = [(i.beat - start) / length for i in items]
        has_notes = any(i.char is not None for i in items)
        self.lines.extend(header)
        if not has_notes and all(p == 0 for p in positions):
            for item in sorted(items, key=_command_order):
                self.lines.append(item.command or "")
            self.lines.append(",")
            return
        count = 1
        for p in positions:
            count = math.lcm(count, p.denominator)
        slots: list[str] = ["0"] * count
        commands: dict[int, list[_Item]] = {}
        for item, p in zip(items, positions, strict=True):
            slot = int(p * count)
            if item.char is not None:
                if slots[slot] != "0":
                    raise CodecError(f"two notes on the same TJA slot at beat {item.beat}")
                slots[slot] = item.char
                if item.on_emit is not None:
                    item.on_emit()
            else:
                commands.setdefault(slot, []).append(item)
        line = ""
        for slot, char in enumerate(slots):
            if slot in commands:
                if line:
                    self.lines.append(line)
                    line = ""
                for item in sorted(commands[slot], key=_command_order):
                    self.lines.append(item.command or "")
            line += char
        self.lines.append(line + ",")


def _command_order(item: _Item) -> tuple[int, int]:
    # timing commands (negative order) first, then controls in IR order
    return (0, -item.order) if item.order < 0 else (1, item.order)


def _meter_at(
    meters: dict[Fraction, tuple[int, int]], beats: list[Fraction], beat: Fraction
) -> tuple[int, int]:
    current = (4, 4)
    for b in beats:
        if b <= beat:
            current = meters[b]
        else:
            break
    return current


def _control_command(control: Control) -> str:
    if isinstance(control.value, bool | None) and (control.kind, control.value) in FLAG_COMMANDS:
        return f"#{FLAG_COMMANDS[control.kind, control.value]}"
    if control.kind == "scroll":
        value = control.value
        return f"#SCROLL {format_float(value) if isinstance(value, int | float) else value}"
    if control.kind.startswith(RAW_PREFIX):
        name = control.kind[len(RAW_PREFIX) :].upper()
        return f"#{name} {control.value}" if control.value not in (None, "") else f"#{name}"
    raise CodecError(f"TJA cannot express control {control.kind!r}")
