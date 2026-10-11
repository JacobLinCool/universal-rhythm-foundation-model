"""simai (``maidata.txt``) reader.

Parsing happens in two passes per difficulty: the first pass walks the note
stream to place every comma-delimited tick on an exact beat and to collect the
tempo map; the second pass parses each tick's note text and converts
second-based durations (holds, slide waits and lengths) through the finished
tempo map, so every resulting :class:`~urfm.ir.time.Time` is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from urfm.codecs.base import CodecError
from urfm.codecs.simai import syntax as sx
from urfm.ir.chart import Chart, ChartSet, Difficulty, Provenance, SongMeta
from urfm.ir.controls import Control
from urfm.ir.events import Event, PathSegment, Primitive
from urfm.ir.targets import RingButton, TouchRegion
from urfm.ir.time import Time
from urfm.ir.timing import TempoChange, TimingMap

MECHANICS = "maimai_dx"
SONG_FIELDS = ("title", "artist", "des", "genre")


# -- metadata ----------------------------------------------------------------


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp932"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise CodecError("maidata is neither UTF-8 nor Shift-JIS")


def split_fields(text: str) -> list[tuple[str, str]]:
    """``&key=value`` blocks in file order; a value runs until the next ``&`` line."""
    fields: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith("&") and "=" in line:
            key, _, value = line[1:].partition("=")
            fields.append((key.strip(), [value]))
        elif fields:
            fields[-1][1].append(line)
    return [(key, "\n".join(lines).strip()) for key, lines in fields]


def read(data: bytes, source: str | None = None) -> ChartSet:
    fields = split_fields(decode(data))
    values = dict(fields)
    offset_us = sx.parse_number(values.get("first") or "0") * sx.US_PER_SECOND
    whole_bpm = sx.parse_number(values["wholebpm"]) if values.get("wholebpm") else None

    slots = [
        int(key[6:])
        for key, value in fields
        if key.startswith("inote_") and key[6:].isdigit() and value.strip()
    ]
    used = {f"inote_{n}" for n in slots} | {f"lv_{n}" for n in slots} | {f"des_{n}" for n in slots}
    used |= set(SONG_FIELDS) | {"first"}

    charts: list[Chart] = []
    for slot in slots:
        timing, events, controls = read_fumen(values[f"inote_{slot}"], offset_us, whole_bpm)
        level_text = values.get(f"lv_{slot}", "")
        meta = {"level": level_text} if level_text else {}
        if values.get(f"des_{slot}"):
            meta["designer"] = values[f"des_{slot}"]
        charts.append(
            Chart(
                mechanics=MECHANICS,
                difficulty=Difficulty(
                    label=sx.DIFFICULTY_LABELS.get(slot, f"inote_{slot}"),
                    level=_level(level_text),
                    tier=sx.DIFFICULTY_TIERS.get(slot),
                ),
                events=events,
                controls=controls,
                timing=timing,
                meta=meta,
                native={"slot": slot},
            )
        )

    if charts:
        shared = charts[0].timing
        assert shared is not None
        charts = [c if c.timing != shared else _without_timing(c) for c in charts]
    else:
        shared = TimingMap.constant(whole_bpm or 120, offset_us)

    song = SongMeta(
        title=values.get("title", ""),
        artist=values.get("artist", ""),
        creator=values.get("des", ""),
        genre=values.get("genre", ""),
        extra={key: value for key, value in fields if key not in used},
    )
    return ChartSet(
        song=song,
        timing=shared,
        charts=tuple(charts),
        provenance=Provenance(codec="simai", source=source),
    )


def _without_timing(chart: Chart) -> Chart:
    return Chart(
        mechanics=chart.mechanics,
        difficulty=chart.difficulty,
        events=chart.events,
        controls=chart.controls,
        branches=chart.branches,
        meta=chart.meta,
        native=chart.native,
    )


def _level(text: str) -> float | None:
    """``13`` -> 13.0, ``13+`` -> 13.5, ``13.7`` -> 13.7; anything else -> None."""
    text = text.strip()
    plus = text.endswith("+")
    try:
        value = float(text.rstrip("+"))
    except ValueError:
        return None
    return value + 0.5 if plus else value


# -- note stream, pass 1: ticks and tempo -------------------------------------


@dataclass
class Tick:
    beat: Fraction
    bpm: Fraction
    text: str


def scan(
    fumen: str, whole_bpm: Fraction | None
) -> tuple[list[Tick], list[TempoChange], list[Control]]:
    ticks: list[Tick] = []
    tempos: dict[Fraction, Fraction] = {}
    controls: list[Control] = []
    beat = Fraction(0)
    bpm = whole_bpm
    div: Fraction = Fraction(4)
    step_seconds: Fraction | None = None
    buffer: list[str] = []
    i, n = 0, len(fumen)

    def section(open_: str, close: str) -> str:
        nonlocal i
        end = fumen.find(close, i + 1)
        if end < 0:
            raise CodecError(f"unterminated {open_!r} at offset {i}")
        body = fumen[i + 1 : end]
        i = end + 1
        return body

    while i < n:
        c = fumen[i]
        if c == "|" and fumen.startswith("||", i):
            end = fumen.find("\n", i)
            i = n if end < 0 else end
        elif c.isspace():
            i += 1
        elif c == "(":
            bpm = sx.parse_number(section("(", ")"))
            if bpm <= 0:
                raise CodecError(f"non-positive BPM {bpm}")
            tempos[beat] = bpm
        elif c == "{":
            body = section("{", "}").strip()
            if body.startswith("#"):
                step_seconds = sx.parse_number(body[1:])
            else:
                div, step_seconds = sx.parse_number(body), None
        elif c == "<" and fumen.startswith("<H", i):
            body = section("<", ">")
            value = body[3:] if body.startswith("HS*") else body[2:]
            controls.append(Control(time=Time(beat), kind="scroll", value=float(value)))
        elif c == ",":
            if buffer:
                if bpm is None:
                    raise CodecError("notes before the first (bpm)")
                ticks.append(Tick(beat, bpm, "".join(buffer)))
                buffer = []
            if bpm is None:
                raise CodecError("time step before the first (bpm)")
            beat += step_seconds * bpm / 60 if step_seconds is not None else 4 / div
            i += 1
        elif c == "E" and not (i + 1 < n and fumen[i + 1] in "12345678"):
            break
        else:
            buffer.append(c)
            i += 1
    if buffer and "".join(buffer).strip():
        if bpm is None:
            raise CodecError("notes before the first (bpm)")
        ticks.append(Tick(beat, bpm, "".join(buffer)))

    if Fraction(0) not in tempos:
        if whole_bpm is None:
            raise CodecError("chart does not start with (bpm) and has no &wholebpm")
        tempos[Fraction(0)] = whole_bpm
    tempo_changes = [TempoChange(b, v) for b, v in sorted(tempos.items())]
    return ticks, tempo_changes, controls


# -- note stream, pass 2: note text -------------------------------------------


@dataclass
class RawSegment:
    op: str
    vertices: list[int]
    duration: str | None = None


@dataclass
class RawSlide:
    segments: list[RawSegment] = field(default_factory=list)
    breaks: bool = False


@dataclass
class RawNote:
    button: int | None = None
    region: str | None = None
    modifiers: set[str] = field(default_factory=set)
    appearance: str | None = None
    headless: str | None = None
    hold: bool = False
    duration: str | None = None
    slides: list[RawSlide] = field(default_factory=list)


def parse_part(text: str) -> list[RawNote]:
    """Parse one each-part (no ``/`` or backtick); consecutive locations start new notes."""
    notes: list[RawNote] = []
    i, n = 0, len(text)

    def fail(message: str) -> CodecError:
        return CodecError(f"{message} in note text {text!r} at {i}")

    def bracket() -> str:
        nonlocal i
        end = text.find("]", i)
        if end < 0:
            raise fail("unterminated '['")
        body = text[i + 1 : end]
        i = end + 1
        return body

    def button() -> int:
        nonlocal i
        if i < n and text[i] in "12345678":
            i += 1
            return int(text[i - 1])
        raise fail("expected button 1-8")

    while i < n:
        note = RawNote()
        c = text[i]
        if c in "12345678":
            note.button = button()
        elif c in "ABDE":
            i += 1
            note.region = c + str(button())
        elif c == "C":
            i += 1
            note.region = "C"
            if i < n and text[i] in "12":
                note.region += text[i]
                i += 1
        else:
            raise fail(f"unexpected {c!r}")

        while i < n:
            c = text[i]
            if c in sx.NOTE_MODIFIERS:
                note.modifiers.add(sx.NOTE_MODIFIERS[c])
                i += 1
            elif c == "h":
                note.hold = True
                i += 1
            elif c == "$":
                note.appearance = "$$" if text.startswith("$$", i) else "$"
                i += len(note.appearance)
            elif c == "@":
                note.appearance = "@"
                i += 1
            elif c in sx.HEADLESS:
                note.headless = c
                i += 1
            elif c == "[":
                note.hold = True
                note.duration = bracket()
            elif c in sx.SLIDE_OP_CHARS:
                if note.button is None:
                    raise fail("slides must start on a button")
                while True:
                    slide = RawSlide()
                    while i < n and text[i] in sx.SLIDE_OP_CHARS:
                        op = next(o for o in sx.SLIDE_OPS if text.startswith(o, i))
                        i += len(op)
                        segment = RawSegment(
                            op, [button() for _ in range(sx.SLIDE_VERTICES.get(op, 1))]
                        )
                        while i < n and text[i] in "b[":
                            if text[i] == "b":
                                slide.breaks = True
                                i += 1
                            else:
                                segment.duration = bracket()
                        slide.segments.append(segment)
                    if not slide.segments:
                        raise fail("'*' must be followed by a slide")
                    note.slides.append(slide)
                    if i < n and text[i] == "*":
                        i += 1
                        continue
                    break
                break
            else:
                break
        if note.slides and (note.hold or note.duration):
            raise fail("hold notes cannot start slides")
        notes.append(note)
    return notes


def read_fumen(
    fumen: str, offset_us: Fraction, whole_bpm: Fraction | None
) -> tuple[TimingMap, tuple[Event, ...], tuple[Control, ...]]:
    ticks, tempos, controls = scan(fumen, whole_bpm)
    timing = TimingMap(offset_us=offset_us, tempos=tuple(tempos))
    events: list[Event] = []
    groups = 0
    for tick in ticks:
        for k, part in enumerate(tick.text.split("`")):
            beat = tick.beat + k * sx.PSEUDO_EACH_STEP
            for note_text in part.split("/"):
                for note in parse_part(note_text):
                    if note.slides:
                        events.extend(_slide_events(note, beat, k, tick.bpm, timing, groups))
                        groups += 1
                    else:
                        events.append(_note_event(note, beat, k, tick.bpm, timing))
    events.sort(key=lambda e: e.time)
    return timing, tuple(events), tuple(controls)


def _native(k: int, **items: Any) -> dict[str, Any]:
    out = {key: value for key, value in items.items() if value is not None}
    if k:
        out["pseudo"] = k
    return out


def _time_after(timing: TimingMap, start_us: Fraction, length_us: Fraction) -> Time:
    return Time(timing.us_to_beat(start_us + length_us))


def _note_event(note: RawNote, beat: Fraction, k: int, bpm: Fraction, timing: TimingMap) -> Event:
    touch = note.region is not None
    target = TouchRegion(note.region) if note.region is not None else RingButton(note.button - 1)  # type: ignore[operator]
    modifiers = tuple(sorted(note.modifiers))
    time = Time(beat)
    native = _native(k, appearance=note.appearance, head=note.headless)
    if note.hold:
        end = time
        if note.duration is not None:
            length = sx.parse_hold_duration(note.duration, bpm)
            end = _time_after(timing, timing.beat_to_us_exact(beat), length)
            native["dur"] = note.duration
        return Event(
            time=time,
            end=end,
            primitive=Primitive.HOLD,
            role="maimai:touch_hold" if touch else "maimai:hold",
            modifiers=modifiers,
            target=target,
            native=native,
        )
    if touch:
        role = "maimai:touch"
    elif note.appearance in ("$", "$$"):
        role = "maimai:star"
    else:
        role = "maimai:tap"
    return Event(
        time=time,
        primitive=Primitive.HIT,
        role=role,
        modifiers=modifiers,
        target=target,
        native=native,
    )


def _slide_events(
    note: RawNote, beat: Fraction, k: int, bpm: Fraction, timing: TimingMap, group: int
) -> list[Event]:
    assert note.button is not None
    head = RingButton(note.button - 1)
    time = Time(beat)
    origin = timing.beat_to_us_exact(beat)
    events: list[Event] = []
    if note.headless is None:
        events.append(
            Event(
                time=time,
                primitive=Primitive.HIT,
                role="maimai:star",
                modifiers=tuple(sorted(note.modifiers)),
                target=head,
                group=group,
                native=_native(k, appearance=note.appearance),
            )
        )
    for slide in note.slides:
        brackets = [s.duration for s in slide.segments if s.duration is not None]
        if not brackets:
            raise CodecError(f"slide from button {note.button} has no duration")
        wait, _ = sx.parse_slide_duration(brackets[0], bpm)
        start_us = origin + wait
        elapsed = Fraction(0)
        previous = head
        segments: list[PathSegment] = []
        for index, raw in enumerate(slide.segments):
            end = None
            native: dict[str, Any] = {}
            if raw.duration is not None:
                _, length = sx.parse_slide_duration(raw.duration, bpm)
                elapsed += length
                end = _time_after(timing, start_us, elapsed)
                native["dur"] = raw.duration
            points = (previous, *(RingButton(v - 1) for v in raw.vertices))
            previous = points[-1]
            segments.append(
                PathSegment(
                    shape=f"maimai:{raw.op}",
                    points=points,
                    start=Time(timing.us_to_beat(start_us)) if index == 0 else None,
                    end=end,
                    native=native,
                )
            )
        last_end = next(s.end for s in reversed(segments) if s.end is not None)
        events.append(
            Event(
                time=time,
                end=last_end,
                primitive=Primitive.PATH,
                role="maimai:slide",
                modifiers=("break",) if slide.breaks else (),
                target=head,
                path=tuple(segments),
                group=group,
                native=_native(k, head=note.headless),
            )
        )
    return events
