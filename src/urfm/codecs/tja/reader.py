"""TJA -> IR.

Notes are placed on exact beats: the ``i``-th of ``n`` slots of a measure lies at
``measure_start + i / n * measure_beats``. Commands between slots take the beat
of the next slot. Header lines persist from one course to the next, matching how
simulators treat a TJA file as one stream of header assignments.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from urfm.codecs.base import CodecError
from urfm.codecs.tja.common import (
    BALLOON_KEYS,
    COURSE_KEYS,
    COURSE_NAMES,
    COURSE_TIER,
    FLAG_CONTROLS,
    NOTES,
    RAW_PREFIX,
    ROLL_END,
    ROUTES,
    SLOT_CHARS,
    SONG_FIELDS,
    STRUCTURED_SONG_KEYS,
    TIMING_KEYS,
    US,
    parse_decimal,
)
from urfm.ir.chart import (
    AudioRef,
    BranchSegment,
    Chart,
    ChartSet,
    Difficulty,
    Provenance,
    SongMeta,
)
from urfm.ir.controls import Control
from urfm.ir.events import BranchRef, Event, Primitive
from urfm.ir.time import Time
from urfm.ir.timing import MeterChange, Stop, TempoChange, TimingMap

Ctx = tuple[int, str] | None
"""Parsing context: ``None`` for shared notes, ``(segment, route)`` inside a branch."""
TimingEntry = tuple[str, Fraction, Any]


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if line:
            out.append(line)
    return out


def _split_command(line: str) -> tuple[str, str]:
    body = line[1:].strip()
    name, _, args = body.partition(" ")
    if not args:
        name, _, args = body.partition("\t")
    return name.strip().upper(), args.strip()


def _split_header(line: str) -> tuple[str, str] | None:
    key, sep, value = line.partition(":")
    key = key.strip().upper()
    if not sep or not key:
        return None
    return key, value.strip()


def _balloon_list(text: str | None) -> list[int]:
    if not text:
        return []
    out = []
    for token in text.split(","):
        token = token.strip()
        try:
            out.append(int(token))
        except ValueError:
            continue
    return out


def _parse_meter(args: str) -> tuple[int, int]:
    num, _, den = args.partition("/")
    ratio_num, ratio_den = parse_decimal(num or "4"), parse_decimal(den or "4")
    if ratio_num.denominator == 1 and ratio_den.denominator == 1 and ratio_den > 0:
        return int(ratio_num), int(ratio_den)
    ratio = ratio_num / ratio_den
    return ratio.numerator, ratio.denominator


def _branch_condition(args: str) -> dict[str, Any]:
    parts = [p.strip() for p in args.split(",")]
    condition: dict[str, Any] = {"raw": args}
    if parts and parts[0]:
        condition["type"] = parts[0].lower()
    if len(parts) >= 3:
        condition["expert"] = parts[1]
        condition["master"] = parts[2]
    return condition


@dataclass
class _RawChart:
    headers: dict[str, str]
    start_args: str
    header_commands: list[str]
    body: list[str]


@dataclass
class _Segment:
    id: int
    start: Fraction
    condition: dict[str, Any]
    routes: list[str] = field(default_factory=list)
    route_end: dict[str, Fraction] = field(default_factory=dict)
    end: Fraction | None = None
    saved_meter: tuple[int, int] = (4, 4)
    saved_last_meter: tuple[int, int] | None = None


class _BodyParser:
    def __init__(self, balloons: dict[str | None, list[int]]) -> None:
        self.beat = Fraction(0)
        self.meter: tuple[int, int] = (4, 4)
        self.last_meter: tuple[int, int] | None = None
        # ("note", char, "") or ("cmd", name, args), in input order
        self.buffer: list[tuple[str, str, str]] = []
        self.buffer_meter: tuple[int, int] | None = None
        self.seq = 0
        self.events: list[dict[str, Any]] = []
        self.controls: list[tuple[Fraction, Ctx, int, Control]] = []
        self.timing: dict[Ctx, list[TimingEntry]] = {None: []}
        self.unknown: list[list[Any]] = []
        self.segments: list[_Segment] = []
        self.segment: _Segment | None = None
        self.route: str | None = None
        self.open_rolls: dict[Ctx, int] = {}
        self.balloons = {k: list(v) for k, v in balloons.items()}
        self.route_timing: dict[str, dict[str, list[list[str]]]] = {}

    # -- context ----------------------------------------------------------

    @property
    def ctx(self) -> Ctx:
        if self.segment is None or self.route is None:
            return None
        return (self.segment.id, self.route)

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    # -- input -------------------------------------------------------------

    def feed(self, line: str) -> None:
        if not line.startswith("#") and _split_header(line) is not None:
            return  # header assignments inside a chart body are not supported
        if line.startswith("#"):
            name, args = _split_command(line)
            if name in {"BRANCHSTART", "N", "E", "M", "BRANCHEND"}:
                self.flush()
                self._branch_command(name, args)
                return
            if name == "MEASURE":
                self.meter = _parse_meter(args)
            self.buffer.append(("cmd", name, args))
            return
        for char in line.upper():
            if char == ",":
                self.flush(force=True)
            elif char in SLOT_CHARS:
                if self.buffer_meter is None:
                    self.buffer_meter = self.meter
                self.buffer.append(("note", char, ""))

    def finish(self) -> None:
        self.flush()
        if self.segment is not None:
            self._close_segment(explicit=False)
        for index in self.open_rolls.values():
            self._end_roll(index, Time(self.beat), implicit=True)
        self.open_rolls.clear()

    # -- measures ------------------------------------------------------------

    def flush(self, *, force: bool = False) -> None:
        """Close the pending measure. Without ``force`` only a measure holding notes is closed."""
        slots = sum(1 for kind, _, _ in self.buffer if kind == "note")
        if not force and slots == 0:
            # Commands without notes before a structural command or #END stay at the
            # current beat; they do not form a measure.
            for _, name, args in self.buffer:
                self._command(self.beat, name, args)
            self.buffer.clear()
            return
        meter = self.buffer_meter or self.meter
        measure_beats = Fraction(4 * meter[0], meter[1])
        if meter != self.last_meter:
            self.timing.setdefault(self.ctx, []).append(("meter", self.beat, meter))
        self.last_meter = meter
        step = measure_beats / slots if slots else Fraction(0)
        k = 0
        for kind, value, args in self.buffer:
            beat = self.beat + k * step
            if kind == "note":
                self._note(beat, value)
                k += 1
            else:
                self._command(beat, value, args)
        self.buffer.clear()
        self.buffer_meter = None
        self.beat += measure_beats

    # -- commands ------------------------------------------------------------

    def _command(self, beat: Fraction, name: str, args: str) -> None:
        ctx = self.ctx
        if name == "BPMCHANGE":
            self.timing.setdefault(ctx, []).append(("bpm", beat, parse_decimal(args)))
        elif name == "DELAY":
            self.timing.setdefault(ctx, []).append(("delay", beat, parse_decimal(args)))
        elif name == "MEASURE":
            pass  # applied to the measure grid when the command is read
        elif name == "SCROLL":
            try:
                value: Any = float(args)
            except ValueError:
                value = args
            self._control(beat, "scroll", value)
        elif name in FLAG_CONTROLS:
            kind, flag = FLAG_CONTROLS[name]
            self._control(beat, kind, flag)
        else:
            self._control(beat, RAW_PREFIX + name.lower(), args or None)

    def _control(self, beat: Fraction, kind: str, value: Any) -> None:
        ctx = self.ctx
        branch = None if ctx is None else BranchRef(*ctx)
        control = Control(time=Time(beat), kind=kind, value=value, branch=branch)
        self.controls.append((beat, ctx, self._next_seq(), control))

    def _branch_command(self, name: str, args: str) -> None:
        if name == "BRANCHSTART":
            if self.segment is not None:
                self._close_segment(explicit=False)
            self.segment = _Segment(
                id=len(self.segments),
                start=self.beat,
                condition=_branch_condition(args),
                saved_meter=self.meter,
                saved_last_meter=self.last_meter,
            )
            self.segments.append(self.segment)
            self.route = None
        elif name in ROUTES:
            if self.segment is None:
                self._branch_command("BRANCHSTART", "")
                assert self.segment is not None
                self.segment.condition = {}
            segment = self.segment
            if self.route is not None:
                segment.route_end[self.route] = self.beat
            self.route = ROUTES[name]
            if self.route not in segment.routes:
                segment.routes.append(self.route)
            self.beat = segment.start
            self.meter = segment.saved_meter
            self.last_meter = segment.saved_last_meter
        elif self.segment is not None:  # BRANCHEND
            self._close_segment(explicit=True)
        else:  # stray #BRANCHEND
            self._control(self.beat, RAW_PREFIX + "branchend", None)

    def _close_segment(self, *, explicit: bool) -> None:
        segment = self.segment
        assert segment is not None
        if self.route is not None:
            segment.route_end[self.route] = self.beat
        segment.end = self.beat if explicit else None
        self.segment = None
        self.route = None
        if not segment.routes:
            # A branch without routes carries no notes; keep its commands verbatim.
            self.segments.pop()
            self._control(segment.start, RAW_PREFIX + "branchstart", segment.condition.get("raw"))
            if explicit:
                self._control(segment.start, RAW_PREFIX + "branchend", None)
            return
        lists = [self.timing.pop((segment.id, r), []) for r in segment.routes]
        shared = self.timing[None]
        if lists:
            shared.extend(lists[-1])
            if any(entries != lists[0] for entries in lists):
                self.route_timing[str(segment.id)] = {
                    route: [[kind, str(beat), _entry_text(value)] for kind, beat, value in entries]
                    for route, entries in zip(segment.routes, lists, strict=True)
                }

    # -- notes ---------------------------------------------------------------

    def _note(self, beat: Fraction, char: str) -> None:
        ctx = self.ctx
        if char == "0":
            return
        if char == ROLL_END:
            index = self.open_rolls.pop(ctx, None)
            if index is None and ctx is not None:
                # a roll started in an earlier segment ends in the same-named route
                same_route = [c for c in self.open_rolls if c is not None and c[1] == ctx[1]]
                if same_route:
                    index = self.open_rolls.pop(same_route[-1])
            if index is None and self.open_rolls:
                last_ctx = max(self.open_rolls, key=lambda c: self.open_rolls[c])
                index = self.open_rolls.pop(last_ctx)
            if index is None:
                self._unknown(beat, char)
            else:
                self._end_roll(index, Time(beat), implicit=False)
            return
        spec = NOTES.get(char)
        if spec is None:
            self._unknown(beat, char)
            return
        primitive, role, modifiers = spec
        event: dict[str, Any] = {
            "beat": beat,
            "ctx": ctx,
            "seq": self._next_seq(),
            "primitive": primitive,
            "role": role,
            "modifiers": modifiers,
            "end": None,
            "count": None,
            "native": {},
        }
        if primitive is Primitive.REPEAT:
            previous = self.open_rolls.pop(ctx, None)
            if previous is not None:
                self._end_roll(previous, Time(beat), implicit=True)
            if role == "taiko:balloon":
                event["count"] = self._take_balloon(ctx)
            self.open_rolls[ctx] = len(self.events)
        self.events.append(event)

    def _end_roll(self, index: int, end: Time, *, implicit: bool) -> None:
        event = self.events[index]
        event["end"] = end
        if implicit:
            event["native"] = {"implicit_end": True}

    def _take_balloon(self, ctx: Ctx) -> int | None:
        key = BALLOON_KEYS[ctx[1]] if ctx is not None else "BALLOON"
        queue = self.balloons.get(key) if key in self.balloons else self.balloons.get("BALLOON")
        if not queue:
            return None
        return queue.pop(0)

    def _unknown(self, beat: Fraction, char: str) -> None:
        ctx = self.ctx
        self.unknown.append(
            [str(beat), char, -1 if ctx is None else ctx[0], "" if ctx is None else ctx[1]]
        )


def _entry_text(value: Any) -> str:
    if isinstance(value, tuple):
        return f"{value[0]}/{value[1]}"
    return str(value)


def read_tja(data: bytes, source: str | None = None) -> ChartSet:
    text = decode(data)
    state: dict[str, str] = {}
    pending_commands: list[str] = []
    raws: list[_RawChart] = []
    current: _RawChart | None = None
    for line in _lines(text):
        upper = line.upper()
        if current is not None:
            if upper.startswith("#START") and not current.body:
                current.start_args = line[6:].strip()  # duplicated #START line
            elif upper.startswith("#END"):
                raws.append(current)
                current = None
            else:
                current.body.append(line)
            continue
        if upper.startswith("#START"):
            current = _RawChart(dict(state), line[6:].strip(), pending_commands, [])
            pending_commands = []
        elif line.startswith("#"):
            pending_commands.append(line)
        elif (header := _split_header(line)) is not None:
            state[header[0]] = header[1]
    if current is not None:
        raws.append(current)
    if not raws and not state:
        raise CodecError("no TJA headers or charts found")

    song_state = raws[0].headers if raws else state
    song_timing = _timing(song_state, {None: []})
    song = SongMeta(
        **{attr: song_state.get(key, "") for key, attr in SONG_FIELDS.items()},
        preview=(
            song_timing.time_at_us(round(parse_decimal(song_state["DEMOSTART"]) * US))
            if song_state.get("DEMOSTART")
            else None
        ),
        extra={
            k: v
            for k, v in song_state.items()
            if k not in COURSE_KEYS and k not in STRUCTURED_SONG_KEYS
        },
    )
    wave = song_state.get("WAVE")
    charts = tuple(_chart(raw, song_state) for raw in raws)
    return ChartSet(
        song=song,
        audio=AudioRef(path=wave) if wave else None,
        timing=song_timing,
        charts=charts,
        provenance=Provenance(codec="tja", source=source, sha256=hashlib.sha256(data).hexdigest()),
    )


def _timing(headers: dict[str, str], entries: dict[Ctx, list[TimingEntry]]) -> TimingMap:
    bpm = parse_decimal(headers.get("BPM") or "120")
    offset = parse_decimal(headers.get("OFFSET") or "0")
    shared = sorted(entries[None], key=lambda e: e[1])
    return TimingMap(
        offset_us=-offset * US,
        tempos=(
            TempoChange(Fraction(0), bpm),
            *(TempoChange(b, v) for k, b, v in shared if k == "bpm"),
        ),
        stops=tuple(Stop(b, v * US, before=True) for k, b, v in shared if k == "delay"),
        meters=tuple(MeterChange(b, v[0], v[1]) for k, b, v in shared if k == "meter"),
    )


def _chart(raw: _RawChart, song_state: dict[str, str]) -> Chart:
    headers = raw.headers
    balloon_keys = [k for k in BALLOON_KEYS.values() if k in headers]
    parser = _BodyParser({k: _balloon_list(headers.get(k)) for k in balloon_keys})
    for line in raw.body:
        parser.feed(line)
    parser.finish()

    meta: dict[str, str] = {}
    level: float | None = None
    for key, value in headers.items():
        if key in TIMING_KEYS or key == "COURSE" or key in BALLOON_KEYS.values():
            continue
        if key == "LEVEL":
            try:
                level = float(value)
                continue
            except ValueError:
                pass
        if key in COURSE_KEYS or song_state.get(key) != value:
            meta[key] = value
    player = raw.start_args.upper().removeprefix("P")
    if player:
        meta["player"] = player

    course = COURSE_NAMES.get(headers.get("COURSE", "oni").strip().lower())
    if course is None:
        course = headers["COURSE"].strip().lower()

    route_rank = {
        (seg.id, route): i + 1 for seg in parser.segments for i, route in enumerate(seg.routes)
    }

    def rank(ctx: Ctx) -> int:
        return 0 if ctx is None else route_rank.get(ctx, 0)

    events = []
    for ev in sorted(parser.events, key=lambda e: (e["beat"], rank(e["ctx"]), e["seq"])):
        ctx = ev["ctx"]
        events.append(
            Event(
                time=Time(ev["beat"]),
                primitive=ev["primitive"],
                role=ev["role"],
                modifiers=ev["modifiers"],
                end=ev["end"],
                count=ev["count"],
                branch=None if ctx is None else BranchRef(*ctx),
                native=ev["native"],
            )
        )
    controls = tuple(
        c for _, _, _, c in sorted(parser.controls, key=lambda c: (c[0], rank(c[1]), c[2]))
    )
    branches = tuple(
        BranchSegment(
            id=seg.id,
            start=Time(seg.start),
            end=None if seg.end is None else Time(seg.end),
            routes=tuple(seg.routes),
            default="normal" if "normal" in seg.routes else (seg.routes or [""])[0],
            condition=seg.condition,
        )
        for seg in parser.segments
    )

    native: dict[str, Any] = {"end_beat": str(parser.beat)}
    if raw.header_commands:
        native["header_commands"] = raw.header_commands
    if parser.unknown:
        native["unknown_notes"] = parser.unknown
    route_keys = [k for k in balloon_keys if k != "BALLOON"]
    if route_keys:
        native["balloon_keys"] = route_keys
    extra = {k: v for k, v in parser.balloons.items() if v}
    if extra:
        native["balloon_extra"] = extra
    if parser.route_timing:
        native["route_timing"] = parser.route_timing
    route_end = {
        f"{seg.id}:{route}": str(end)
        for seg in parser.segments
        for route, end in seg.route_end.items()
        if seg.routes and end != seg.route_end.get(seg.routes[-1])
    }
    if route_end:
        native["route_end"] = route_end
    # Where an implicitly closed segment ends when that is not the next segment's start
    # (e.g. it was closed by a route-less #BRANCHSTART kept as a raw control).
    segment_end = {}
    for i, seg in enumerate(parser.segments):
        if seg.end is None and seg.routes:
            inferred = parser.segments[i + 1].start if i + 1 < len(parser.segments) else parser.beat
            actual = seg.route_end[seg.routes[-1]]
            if actual != inferred:
                segment_end[str(seg.id)] = str(actual)
    if segment_end:
        native["segment_end"] = segment_end

    return Chart(
        mechanics="taiko",
        difficulty=Difficulty(label=course, level=level, tier=COURSE_TIER.get(course)),
        events=tuple(events),
        controls=controls,
        branches=branches,
        timing=_timing(headers, parser.timing),
        meta=meta,
        native=native,
    )
