"""osu! ``.osu`` codec (file format v3-v14, all four rulesets).

Timing model
------------
osu! is time-native: objects carry millisecond timestamps and every uninherited
("red") timing point restarts the beat grid. Beat 0 is the first red line. Each
later red line is placed on the first whole beat at or after the position it
would have under the previous tempo, and a ``before=True`` :class:`Stop` on that
beat absorbs the (usually negative) difference. This keeps every red line on an
integer beat -- so measures restart there exactly as in osu! -- while the
absolute time of every beat stays exact.

Objects are snapped to the simplest beat subdivision of their governing red line
within :data:`SNAP_TOLERANCE_US`; ``Time.residual_us`` keeps the remaining
microseconds, so absolute times survive bit-exactly.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from urfm.codecs.base import Codec, CodecError, Fidelity
from urfm.codecs.osu import format as fmt
from urfm.ir.chart import AudioRef, Chart, ChartSet, Difficulty, Provenance, SongMeta
from urfm.ir.controls import Control
from urfm.ir.events import Event, PathSegment, Primitive
from urfm.ir.targets import Lane, Point2D
from urfm.ir.time import Time
from urfm.ir.timing import MeterChange, Stop, TempoChange, TimingMap

MODES = {0: "osu_std", 1: "osu_taiko", 2: "osu_catch", 3: "osu_mania"}
MODE_OF = {v: k for k, v in MODES.items()}

SNAP_DENOMINATORS = (1, 2, 4, 3, 8, 6, 16, 12, 24, 32, 48, 5, 7, 9, 10, 14, 18, 20, 28, 36, 64, 96)
SNAP_FALLBACK = 192
SNAP_TOLERANCE_US = 1500
EARLY_VERSION_OFFSET_US = 24_000
"""osu! shifts every object of files older than format v5 by 24 ms."""

CURVES = {"B": "osu:bezier", "L": "osu:linear", "P": "osu:perfect", "C": "osu:catmull"}
CURVE_LETTERS = {v: k for k, v in CURVES.items()}

TYPE_CIRCLE, TYPE_SLIDER, TYPE_NEW_COMBO, TYPE_SPINNER, TYPE_HOLD = 1, 2, 4, 8, 128
HS_WHISTLE, HS_FINISH, HS_CLAP = 2, 4, 8
TAIKO_KA_DEFAULT = HS_CLAP

TP_DEFAULTS = {"meter": 4, "sample_set": 0, "sample_index": 0, "volume": 100, "effects": 0}
TP_FIELDS = 8
KIAI = 1


@dataclass
class _Point:
    us: int
    beat_length: Fraction
    beat_length_text: str
    uninherited: bool
    values: dict[str, int]
    fields: int
    raw: str = ""


@dataclass
class _Red:
    us: int
    beat_length: Fraction  # ms per beat
    beat: Fraction  # beat assigned in the IR (integer)


class _Grid:
    """Red-line grid used to turn microseconds into :class:`Time`."""

    def __init__(self, reds: list[_Red], timing: TimingMap) -> None:
        self.reds = reds
        self.times = [r.us for r in reds]
        self.timing = timing

    def time(self, us: int) -> Time:
        red = self.reds[max(bisect_right(self.times, us) - 1, 0)]
        beat_us = red.beat_length * 1000
        rel = (us - red.us) / beat_us
        beat = red.beat + _snap(rel, beat_us)
        return Time(beat, us - round(self.timing.beat_to_us_exact(beat)))


def _snap(rel: Fraction, beat_us: Fraction) -> Fraction:
    for den in SNAP_DENOMINATORS:
        candidate = Fraction(round(rel * den), den)
        if abs(candidate - rel) * beat_us <= SNAP_TOLERANCE_US:
            return candidate
    return Fraction(round(rel * SNAP_FALLBACK), SNAP_FALLBACK)


class OsuCodec(Codec):
    id = "osu"
    extensions = (".osu",)
    mechanics = frozenset(MODES.values())
    fidelity = Fidelity.SEMANTIC

    # -- read ----------------------------------------------------------------

    def read(self, data: bytes, *, source: str | None = None) -> ChartSet:
        osu = fmt.parse(data)
        general = osu.pairs("General")
        metadata = osu.pairs("Metadata")
        difficulty = osu.pairs("Difficulty")
        general_map, meta_map, diff_map = dict(general), dict(metadata), dict(difficulty)

        try:
            mode = int(general_map.get("Mode", "0"))
            mechanics = MODES[mode]
        except (KeyError, ValueError) as exc:
            raise CodecError(f"unsupported osu! mode {general_map.get('Mode')!r}") from exc
        early = EARLY_VERSION_OFFSET_US if osu.version < 5 else 0

        points, rejected = _split_rejected(
            _parse_timing_points(osu.sections.get("TimingPoints", []), early)
        )
        implicit = not any(p.uninherited for p in points)
        if implicit:  # osu! falls back to 60 BPM from time 0
            points.insert(0, _Point(0, Fraction(1000), "1000", True, dict(TP_DEFAULTS), TP_FIELDS))
        timing, reds = _build_timing(points)
        if implicit:
            points.pop(0)
        grid = _Grid(reds, timing)
        controls = _build_controls(points, grid, mechanics)

        lanes = None
        meta: dict[str, str] = {}
        if mechanics == "osu_mania":
            lanes = max(1, int(float(diff_map.get("CircleSize", "4"))))
            meta["lanes"] = str(lanes)
        slider_multiplier = fmt.number(diff_map.get("SliderMultiplier", "1.4"))
        context = _ObjectContext(mechanics, points, reds, grid, slider_multiplier, lanes, early)
        events = [context.event(line) for line in osu.sections.get("HitObjects", [])]
        events = [e for e in events if e is not None]
        events.sort(key=lambda e: timing.us(e.time))

        preview_ms = general_map.get("PreviewTime")
        preview = None
        if preview_ms is not None and fmt.number(preview_ms) >= 0:
            preview = grid.time(round(fmt.number(preview_ms) * 1000))
        audio_path = general_map.get("AudioFilename")

        structured_general = {"AudioFilename", "PreviewTime", "Mode"}
        structured_meta = {"Title", "Artist", "Creator", "Version"}
        native: dict[str, Any] = {
            "format_version": osu.version,
            "implicit_timing": implicit,
            "rejected_timing_points": rejected,
            "general": [p for p in general if p[0] not in structured_general],
            "editor": osu.pairs("Editor"),
            "metadata": [p for p in metadata if p[0] not in structured_meta],
            "difficulty": [
                p for p in difficulty if not (lanes is not None and p[0] == "CircleSize")
            ],
            "events": list(osu.sections.get("Events", [])),
            "colours": osu.pairs("Colours"),
            "extra_sections": {
                name: lines for name, lines in osu.sections.items() if name not in fmt.SECTION_ORDER
            },
        }
        chart = Chart(
            mechanics=mechanics,
            difficulty=Difficulty(label=meta_map.get("Version")),
            events=tuple(events),
            controls=tuple(controls),
            meta=meta,
            native=native,
        )
        return ChartSet(
            song=SongMeta(
                title=meta_map.get("Title", ""),
                artist=meta_map.get("Artist", ""),
                creator=meta_map.get("Creator", ""),
                preview=preview,
            ),
            audio=AudioRef(path=audio_path) if audio_path else None,
            timing=timing,
            charts=(chart,),
            provenance=Provenance(
                codec=self.id, source=source, sha256=hashlib.sha256(data).hexdigest()
            ),
        )

    # -- write ---------------------------------------------------------------

    def write(self, chartset: ChartSet) -> bytes:
        self.check_writable(chartset)
        if len(chartset.charts) != 1:
            raise CodecError("an .osu file holds exactly one chart")
        chart = chartset.charts[0]
        timing = chartset.timing_of(chart)
        native = chart.native
        version = int(native.get("format_version", 14))
        early = EARLY_VERSION_OFFSET_US if version < 5 else 0
        mode = MODE_OF[chart.mechanics]

        general = [["AudioFilename", chartset.audio.path]] if chartset.audio else []
        general += native.get("general", [])
        preview = chartset.song.preview
        general.append(["PreviewTime", "-1" if preview is None else _ms(timing.us(preview), 0)])
        general.append(["Mode", str(mode)])

        metadata = [
            ["Title", chartset.song.title],
            ["Artist", chartset.song.artist],
            ["Creator", chartset.song.creator],
            *([["Version", chart.difficulty.label]] if chart.difficulty.label is not None else []),
            *native.get("metadata", []),
        ]
        difficulty = list(native.get("difficulty", []))
        lanes = None
        if chart.mechanics == "osu_mania":
            lanes = int(chart.meta.get("lanes", "4"))
            difficulty.insert(0, ["CircleSize", str(lanes)])
        diff_map = dict(difficulty)
        slider_multiplier = fmt.number(diff_map.get("SliderMultiplier", "1.4"))

        points, lines = _write_timing_points(chart, timing, early)
        context = _WriteContext(chart, timing, points, slider_multiplier, lanes, early)

        osu = fmt.OsuFile(version=version)
        osu.sections["General"] = [f"{k}: {v}" for k, v in general]
        osu.sections["Editor"] = [f"{k}: {v}" for k, v in native.get("editor", [])]
        osu.sections["Metadata"] = [f"{k}:{v}" for k, v in metadata]
        osu.sections["Difficulty"] = [f"{k}:{v}" for k, v in difficulty]
        osu.sections["Events"] = list(native.get("events", []))
        osu.sections["TimingPoints"] = lines
        if native.get("colours"):
            osu.sections["Colours"] = [f"{k} : {v}" for k, v in native["colours"]]
        osu.sections["HitObjects"] = [context.line(ev) for ev in chart.events]
        for name, extra in native.get("extra_sections", {}).items():
            osu.sections[name] = list(extra)
        return fmt.serialize(osu)


# -- timing points -----------------------------------------------------------------


def _parse_timing_points(lines: list[str], early: int) -> list[_Point]:
    points: list[_Point] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        beat_length = _beat_length(parts[1])
        values = dict(TP_DEFAULTS)
        for key, index in (
            ("meter", 2),
            ("sample_set", 3),
            ("sample_index", 4),
            ("volume", 5),
            ("effects", 7),
        ):
            if len(parts) > index and parts[index] != "":
                values[key] = int(fmt.number(parts[index]))
        uninherited = int(parts[6]) == 1 if len(parts) > 6 else beat_length > 0
        us = round(fmt.number(parts[0]) * 1000) + early
        points.append(_Point(us, beat_length, parts[1], uninherited, values, len(parts), line))
    points.sort(key=lambda p: p.us)
    return points


def _split_rejected(points: list[_Point]) -> tuple[list[_Point], list[str]]:
    """osu! ignores red lines whose beat length is not a positive finite number."""
    kept, rejected = [], []
    for p in points:
        if p.uninherited and (p.beat_length <= 0 or not _finite(p.beat_length_text)):
            rejected.append(p.raw)
        else:
            kept.append(p)
    return kept, rejected


def _finite(text: str) -> bool:
    try:
        fmt.number(text)
    except CodecError:
        return False
    return True


def _beat_length(text: str) -> Fraction:
    try:
        return fmt.number(text)
    except CodecError:
        if text.strip().lower() in ("nan", "-nan", "inf", "-inf", "infinity", "-infinity"):
            return Fraction(-100)  # osu! falls back to the default slider velocity
        raise


def _build_timing(points: list[_Point]) -> tuple[TimingMap, list[_Red]]:
    red_points = [p for p in points if p.uninherited]
    reds: list[_Red] = []
    tempos: list[TempoChange] = []
    meters: list[MeterChange] = []
    stops: list[Stop] = []
    for p in red_points:
        if not reds:
            beat = Fraction(0)
        else:
            prev = reds[-1]
            exact = prev.beat + (p.us - prev.us) / (prev.beat_length * 1000)
            beat = Fraction(math.ceil(exact))
            if beat != exact:
                natural = prev.us + (beat - prev.beat) * prev.beat_length * 1000
                stops.append(Stop(beat, p.us - natural, before=True))
        reds.append(_Red(p.us, p.beat_length, beat))
        tempos.append(TempoChange(beat, Fraction(60000) / p.beat_length))
        meters.append(MeterChange(beat, max(p.values["meter"], 1), 4))
    timing = TimingMap(
        offset_us=Fraction(reds[0].us),
        tempos=tuple(tempos),
        stops=tuple(stops),
        meters=tuple(meters),
    )
    return timing, reds


def _sv_kind(mechanics: str) -> str:
    return "scroll" if mechanics in ("osu_taiko", "osu_mania") else "osu:sv"


def _compact(values: dict[str, int]) -> dict[str, int]:
    return {k: v for k, v in values.items() if TP_DEFAULTS[k] != v}


def _build_controls(points: list[_Point], grid: _Grid, mechanics: str) -> list[Control]:
    controls: list[Control] = []
    kiai = False
    for p in points:
        time = grid.time(p.us)
        values = dict(p.values)
        effects = values["effects"]
        values["effects"] = effects & ~KIAI
        native: dict[str, Any] = _compact(values)
        if p.fields != TP_FIELDS:
            native["fields"] = p.fields
        if p.uninherited:
            native.pop("meter", None)  # lives in the timing map
            controls.append(Control(time=time, kind="osu:red_line", native=native))
        else:
            native["beat_length"] = p.beat_length_text
            sv = float(-100 / p.beat_length) if p.beat_length else 1.0
            controls.append(Control(time=time, kind=_sv_kind(mechanics), value=sv, native=native))
        if bool(effects & KIAI) != kiai:
            kiai = not kiai
            controls.append(Control(time=time, kind="highlight", value=kiai))
    return controls


def _write_timing_points(
    chart: Chart, timing: TimingMap, early: int
) -> tuple[list[_Point], list[str]]:
    if timing.warps:
        raise CodecError("osu! cannot represent warps")
    tempo_beats = {t.beat for t in timing.tempos}
    for stop in timing.stops:
        if not stop.before or stop.beat not in tempo_beats:
            raise CodecError("osu! can only represent stops that coincide with a red line")

    tempos = sorted(timing.tempos, key=lambda t: t.beat)
    meters = sorted(timing.meters, key=lambda m: m.beat)
    paired_meters = len(meters) == len(tempos)
    sv_kind = _sv_kind(chart.mechanics)
    controls = chart.controls
    red_indices = [i for i, c in enumerate(controls) if c.kind == "osu:red_line"]
    paired_reds = len(red_indices) == len(tempos)
    kiai_of = _kiai_per_control(controls)

    # (us, order, control index or -1, point)
    entries: list[tuple[int, int, int, _Point]] = []
    implicit = bool(chart.native.get("implicit_timing")) and len(tempos) == 1
    for i, tempo in enumerate([] if implicit else tempos):
        us = round(timing.beat_to_us_exact(tempo.beat))
        index = red_indices[i] if paired_reds else -1
        native = controls[index].native if index >= 0 else {}
        meter = meters[i] if paired_meters else timing.meter_at(tempo.beat)
        values = {**TP_DEFAULTS, **{k: v for k, v in native.items() if k in TP_DEFAULTS}}
        values["meter"] = meter.numerator
        beat_length = Fraction(60000) / tempo.bpm
        text = fmt.format_number(beat_length)
        fields = int(native.get("fields", TP_FIELDS))
        entries.append((us, 0, index, _Point(us, beat_length, text, True, values, fields)))

    point_times: set[int] = {e[0] for e in entries}
    for i, control in enumerate(controls):
        if control.kind != sv_kind:
            continue
        us = timing.us(control.time)
        native = control.native
        values = {**TP_DEFAULTS, **{k: v for k, v in native.items() if k in TP_DEFAULTS}}
        text = native.get("beat_length")
        sv = float(control.value) if isinstance(control.value, int | float) else 1.0
        if text is None or (_beat_length(text) and float(-100 / _beat_length(text)) != sv):
            text = repr(-100 / sv)
        fields = int(native.get("fields", TP_FIELDS))
        point = _Point(us, _beat_length(text), text, False, values, fields)
        entries.append((us, 1, i, point))
        point_times.add(us)

    highlights = [(timing.us(c.time), bool(c.value)) for c in controls if c.kind == "highlight"]
    highlights.sort(key=lambda h: h[0])
    for us, _ in highlights:
        if us not in point_times:  # a highlight switch needs a timing point to live on
            entries.append(
                (us, 2, -1, _Point(us, Fraction(-100), "-100", False, dict(TP_DEFAULTS), 8))
            )
            point_times.add(us)
    # Points that came from controls keep their IR order; synthesized ones go first.
    entries.sort(key=lambda e: (e[0], e[2] if e[2] >= 0 else -1, e[1]))

    highlight_times = [h[0] for h in highlights]
    points: list[_Point] = []
    lines: list[str] = []
    for us, _, index, p in entries:
        if index >= 0:
            kiai = kiai_of[index]
        else:
            k = bisect_right(highlight_times, us) - 1
            kiai = highlights[k][1] if k >= 0 else False
        effects = (p.values["effects"] & ~KIAI) | (KIAI if kiai else 0)
        fields = [
            _ms(p.us, early),
            p.beat_length_text,
            str(p.values["meter"]),
            str(p.values["sample_set"]),
            str(p.values["sample_index"]),
            str(p.values["volume"]),
            "1" if p.uninherited else "0",
            str(effects),
        ]
        points.append(p)
        lines.append(",".join(fields[: max(2, min(p.fields, TP_FIELDS))]))
    lines.extend(chart.native.get("rejected_timing_points", []))
    return points, lines


def _kiai_per_control(controls: tuple[Control, ...]) -> dict[int, bool]:
    """Kiai state of each timing-point control: the state after the highlight
    switches that directly follow it at the same time (as produced by the reader)."""
    state = False
    out: dict[int, bool] = {}
    for i, control in enumerate(controls):
        if control.kind == "highlight":
            state = bool(control.value)
            continue
        j = i + 1
        local = state
        while (
            j < len(controls)
            and controls[j].kind == "highlight"
            and controls[j].time == control.time
        ):
            local = bool(controls[j].value)
            j += 1
        out[i] = local
    return out


def _ms(us: int, early: int) -> str:
    return fmt.format_number(Fraction(us - early, 1000))


def _object_ms(us: int, early: int, native: dict) -> str:
    """Hit-object time. Events read from ``.osu`` keep their exact value; events without
    native fields (generated or converted) are written as integer milliseconds, which is
    what osu! stores."""
    if native:
        return _ms(us, early)
    return str(round((us - early) / 1000))


def _velocity_points(points: list[_Point]) -> tuple[list[int], list[Fraction]]:
    """Slider velocity multiplier in effect from each timing point onward."""
    times, svs = [], []
    for p in points:
        if p.uninherited:
            sv = Fraction(1)
        else:
            beat_length = min(max(p.beat_length, Fraction(-1000)), Fraction(-10))
            sv = -100 / beat_length
        times.append(p.us)
        svs.append(sv)
    return times, svs


def _slider_duration_us(
    length: Fraction, slides: int, sv: Fraction, multiplier: Fraction, beat_length: Fraction
) -> Fraction:
    return length * slides / (multiplier * 100 * sv) * beat_length * 1000


def _coord(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


# -- hit objects -------------------------------------------------------------------


class _ObjectContext:
    def __init__(
        self,
        mechanics: str,
        points: list[_Point],
        reds: list[_Red],
        grid: _Grid,
        multiplier: Fraction,
        lanes: int | None,
        early: int,
    ) -> None:
        self.mechanics = mechanics
        self.sv_times, self.svs = _velocity_points(points)
        self.red_times = [r.us for r in reds]
        self.reds = reds
        self.grid = grid
        self.multiplier = multiplier
        self.lanes = lanes
        self.early = early

    def _sv_at(self, us: int) -> Fraction:
        i = bisect_right(self.sv_times, us) - 1
        return self.svs[i] if i >= 0 else Fraction(1)

    def _beat_length_at(self, us: int) -> Fraction:
        return self.reds[max(bisect_right(self.red_times, us) - 1, 0)].beat_length

    def _us(self, text: str) -> int:
        return round(fmt.number(text) * 1000) + self.early

    def event(self, line: str) -> Event | None:
        parts = line.split(",")
        if len(parts) < 4:
            return None
        x, y = float(parts[0]), float(parts[1])
        us = self._us(parts[2])
        obj_type = int(parts[3])
        hitsound = int(parts[4]) if len(parts) > 4 and parts[4] else 0
        native: dict[str, Any] = {}
        if obj_type & TYPE_NEW_COMBO:
            native["new_combo"] = True
        if skip := (obj_type >> 4) & 7:
            native["combo_skip"] = skip
        time = self.grid.time(us)
        mode = self.mechanics

        if obj_type & TYPE_SLIDER:
            if len(parts) < 8:
                raise CodecError(f"malformed slider: {line!r}")
            curve, slides, length = parts[5], int(parts[6]), parts[7]
            native["length"] = length
            for key, index in (("edge_sounds", 8), ("edge_sets", 9), ("sample", 10)):
                if len(parts) > index:
                    native[key] = parts[index]
            duration = _slider_duration_us(
                fmt.number(length),
                slides,
                self._sv_at(us),
                self.multiplier,
                self._beat_length_at(us),
            )
            end = self.grid.time(us + round(duration))
            if mode == "osu_taiko":
                _keep_position(native, x, y)
                native.update(curve=curve, slides=slides)
                _taiko_hitsound_native(native, hitsound, ka=False)
                mods = ("big",) if hitsound & HS_FINISH else ()
                return Event(
                    time=time, end=end, primitive=Primitive.REPEAT, role="taiko:drumroll",
                    modifiers=mods, native=native,
                )  # fmt: skip
            if mode == "osu_mania":
                raise CodecError("sliders are not valid in osu!mania beatmaps")
            if hitsound:
                native["hitsound"] = hitsound
            role = "osu:slider" if mode == "osu_std" else "catch:juice_stream"
            return Event(
                time=time, end=end, primitive=Primitive.PATH, role=role,
                target=Point2D(x, y), path=_parse_curve(curve, x, y), count=slides,
                native=native,
            )  # fmt: skip

        if obj_type & (TYPE_SPINNER | TYPE_HOLD):
            if obj_type & TYPE_HOLD:
                end_text, _, sample = parts[5].partition(":") if len(parts) > 5 else ("", "", "")
                if sample:
                    native["sample"] = sample
            else:
                end_text = parts[5]
                if len(parts) > 6:
                    native["sample"] = parts[6]
            end = self.grid.time(self._us(end_text))
            if mode == "osu_mania":
                if hitsound:
                    native["hitsound"] = hitsound
                lane = self._lane(x, native, y)
                return Event(
                    time=time, end=end, primitive=Primitive.HOLD, role="mania:hold",
                    target=Lane(lane), native=native,
                )  # fmt: skip
            if obj_type & TYPE_HOLD:
                raise CodecError("hold notes are only valid in osu!mania beatmaps")
            _keep_position(native, x, y)
            if hitsound:
                native["hitsound"] = hitsound
            role = {
                "osu_std": "osu:spinner",
                "osu_taiko": "taiko:balloon",
                "osu_catch": "catch:banana_shower",
            }[mode]
            return Event(time=time, end=end, primitive=Primitive.REPEAT, role=role, native=native)

        if len(parts) > 5:
            native["sample"] = parts[5]
        if mode == "osu_taiko":
            ka = bool(hitsound & (HS_WHISTLE | HS_CLAP))
            _keep_position(native, x, y)
            _taiko_hitsound_native(native, hitsound, ka=ka)
            return Event(
                time=time, primitive=Primitive.HIT, role="taiko:ka" if ka else "taiko:don",
                modifiers=("big",) if hitsound & HS_FINISH else (), native=native,
            )  # fmt: skip
        if hitsound:
            native["hitsound"] = hitsound
        if mode == "osu_mania":
            lane = self._lane(x, native, y)
            return Event(
                time=time,
                primitive=Primitive.HIT,
                role="mania:note",
                target=Lane(lane),
                native=native,
            )
        role = "osu:circle" if mode == "osu_std" else "catch:fruit"
        return Event(
            time=time, primitive=Primitive.HIT, role=role, target=Point2D(x, y), native=native
        )

    def _lane(self, x: float, native: dict[str, Any], y: float) -> int:
        assert self.lanes is not None
        lane = min(max(math.floor(x * self.lanes / 512), 0), self.lanes - 1)
        if x != _lane_x(lane, self.lanes):
            native["x"] = x
        if y != 192:
            native["y"] = y
        return lane


def _keep_position(native: dict[str, Any], x: float, y: float) -> None:
    """Store a position that has no gameplay meaning only when it is not the default."""
    if x != 256:
        native["x"] = x
    if y != 192:
        native["y"] = y


def _lane_x(lane: int, lanes: int) -> int:
    return math.floor((lane + 0.5) * 512 / lanes)


def _taiko_hitsound(ka: bool, big: bool) -> int:
    return (TAIKO_KA_DEFAULT if ka else 0) | (HS_FINISH if big else 0)


def _taiko_hitsound_native(native: dict[str, Any], hitsound: int, *, ka: bool) -> None:
    if hitsound != _taiko_hitsound(ka, bool(hitsound & HS_FINISH)):
        native["hitsound"] = hitsound


def _parse_curve(curve: str, x: float, y: float) -> tuple[PathSegment, ...]:
    segments: list[PathSegment] = []
    shape: str | None = None
    native: dict[str, Any] = {}
    points: list[Point2D] = [Point2D(x, y)]
    for token in curve.split("|"):
        if not token:
            continue
        if ":" not in token:
            if shape is not None and len(points) > 1:
                segments.append(PathSegment(shape=shape, points=tuple(points), native=native))
                points = [points[-1]]
            shape, native = _curve_shape(token)
            continue
        px, _, py = token.partition(":")
        points.append(Point2D(float(px), float(py)))
    segments.append(PathSegment(shape=shape or "osu:bezier", points=tuple(points), native=native))
    return tuple(segments)


def _curve_shape(token: str) -> tuple[str, dict[str, Any]]:
    if token in CURVES:
        return CURVES[token], {}
    return "osu:other", {"token": token}


def _format_curve(path: tuple[PathSegment, ...]) -> str:
    tokens: list[str] = []
    for segment in path:
        letter = segment.native.get("token") or CURVE_LETTERS.get(segment.shape, "B")
        tokens.append(letter)
        for point in segment.points[1:]:
            if not isinstance(point, Point2D):
                raise CodecError("osu! slider paths need Point2D control points")
            tokens.append(f"{_coord(point.x)}:{_coord(point.y)}")
    return "|".join(tokens)


class _WriteContext:
    def __init__(
        self,
        chart: Chart,
        timing: TimingMap,
        points: list[_Point],
        multiplier: Fraction,
        lanes: int | None,
        early: int,
    ) -> None:
        self.chart = chart
        self.timing = timing
        self.sv_times, self.svs = _velocity_points(points)
        reds = [p for p in points if p.uninherited]
        self.red_times = [p.us for p in reds]
        self.red_lengths = [p.beat_length for p in reds]
        self.multiplier = multiplier
        self.lanes = lanes
        self.early = early

    def _sv_at(self, us: int) -> Fraction:
        i = bisect_right(self.sv_times, us) - 1
        return self.svs[i] if i >= 0 else Fraction(1)

    def _beat_length_at(self, us: int) -> Fraction:
        return self.red_lengths[max(bisect_right(self.red_times, us) - 1, 0)]

    def line(self, ev: Event) -> str:
        native = ev.native
        us = self.timing.us(ev.time)
        combo = (TYPE_NEW_COMBO if native.get("new_combo") else 0) | (
            int(native.get("combo_skip", 0)) << 4
        )
        hitsound = int(native.get("hitsound", 0))
        big = "big" in ev.modifiers
        if ev.role in ("taiko:don", "taiko:ka", "taiko:drumroll"):
            ka = ev.role == "taiko:ka"
            canonical = _taiko_hitsound(ka, big)
            raw = native.get("hitsound")
            hitsound = canonical
            if raw is not None:
                raw = int(raw)
                consistent = bool(raw & HS_FINISH) == big and (
                    ev.role == "taiko:drumroll" or bool(raw & (HS_WHISTLE | HS_CLAP)) == ka
                )
                hitsound = raw if consistent else canonical

        if isinstance(ev.target, Point2D):
            x, y = ev.target.x, ev.target.y
        elif isinstance(ev.target, Lane):
            assert self.lanes is not None
            x = native.get("x", _lane_x(ev.target.index, self.lanes))
            if math.floor(float(x) * self.lanes / 512) != ev.target.index:
                x = _lane_x(ev.target.index, self.lanes)
            y = native.get("y", 192)
        else:
            x, y = native.get("x", 256), native.get("y", 192)
        head = [_coord(x), _coord(y), _object_ms(us, self.early, native)]

        if ev.primitive is Primitive.PATH or ev.role == "taiko:drumroll":
            assert ev.end is not None
            if ev.role == "taiko:drumroll":
                curve = native.get("curve") or f"L|{_coord(float(x) + 100)}:{_coord(y)}"
                slides = int(native.get("slides", 1))
            else:
                curve = _format_curve(ev.path)
                slides = ev.count or 1
            length = native.get("length")
            if length is None:
                duration = self.timing.us(ev.end) - us
                value = (
                    Fraction(duration)
                    * self.multiplier
                    * 100
                    * self._sv_at(us)
                    / (self._beat_length_at(us) * 1000 * slides)
                )
                length = fmt.format_number(round(value * 10000) / Fraction(10000))
            fields = [*head, str(TYPE_SLIDER | combo), str(hitsound), curve, str(slides), length]
            for key in ("edge_sounds", "edge_sets", "sample"):
                if key in native:
                    fields.append(native[key])
                else:
                    break
            return ",".join(fields)

        if ev.primitive in (Primitive.REPEAT, Primitive.HOLD):
            assert ev.end is not None
            end = _object_ms(self.timing.us(ev.end), self.early, native)
            if ev.role == "mania:hold":
                sample = native.get("sample")
                last = f"{end}:{sample}" if sample is not None else end
                return ",".join([*head, str(TYPE_HOLD | combo), str(hitsound), last])
            fields = [*head, str(TYPE_SPINNER | combo), str(hitsound), end]
            if "sample" in native:
                fields.append(native["sample"])
            return ",".join(fields)

        fields = [*head, str(TYPE_CIRCLE | combo), str(hitsound)]
        if "sample" in native:
            fields.append(native["sample"])
        return ",".join(fields)
