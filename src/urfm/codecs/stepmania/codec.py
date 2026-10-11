"""StepMania ``.sm`` / ``.ssc`` codec.

The MSD tag container is handled by ``_msd`` (on top of ``msdparser``) and the
note data by ``_notedata``; this module maps StepMania's beat-based data to
and from the IR.

Mapping summary (see ``docs/formats/stepmania.md``):

* ``OFFSET`` (seconds, beat 0 at ``-OFFSET``) -> ``TimingMap.offset_us``;
  ``BPMS``/``STOPS``/``DELAYS``/``WARPS``/``TIMESIGNATURES`` -> tempo map.
* ``TICKCOUNTS``/``COMBOS``/``SPEEDS``/``SCROLLS``/``FAKES``/``LABELS`` ->
  chart controls (the effective ones: song-level unless the SSC chart has
  its own timing).
* Note data -> one event per note; hold/roll heads are joined to their tails.
* Every tag the IR does not model is kept verbatim in ``native["tags"]`` and
  the original tag order in ``native["tag_order"]``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from urfm.codecs.base import Codec, CodecError, Fidelity
from urfm.codecs.stepmania import _msd, _notedata
from urfm.ir.chart import AudioRef, Chart, ChartSet, Difficulty, Provenance, SongMeta
from urfm.ir.controls import Control
from urfm.ir.events import Event, Primitive
from urfm.ir.targets import Lane
from urfm.ir.time import Time
from urfm.ir.timing import MeterChange, Stop, TempoChange, TimingMap, Warp
from urfm.mechanics import registry

ENCODINGS = ("utf-8", "cp1252", "cp932", "cp949")
BEAT_TICKS = 48
"""StepMania stores timing-segment beats as rows: 48 per beat."""
SSC_SPLIT_TIMING_VERSION = 0.7

TIMING_TAGS = ("OFFSET", "BPMS", "STOPS", "DELAYS", "WARPS", "TIMESIGNATURES")
CONTROL_TAGS = ("TICKCOUNTS", "COMBOS", "SPEEDS", "SCROLLS", "FAKES", "LABELS")
CHART_TIMING_TAGS = (*TIMING_TAGS[1:], *CONTROL_TAGS)
"""SSC chart tags whose (non-empty) presence switches the chart to its own timing."""
SONG_META_TAGS = ("TITLE", "SUBTITLE", "ARTIST", "GENRE", "CREDIT", "MUSIC", "SAMPLESTART")

CONTROL_KIND = {
    "TICKCOUNTS": "sm:tickcount",
    "COMBOS": "sm:combo",
    "SPEEDS": "sm:speed",
    "SCROLLS": "scroll",
    "FAKES": "sm:fake",
    "LABELS": "section",
}

DIFFICULTY_TIER = {"beginner": 0, "easy": 1, "medium": 2, "hard": 3, "challenge": 4}

CHAR_ROLE = {
    "1": (Primitive.HIT, "sm:tap"),
    "M": (Primitive.HIT, "sm:mine"),
    "L": (Primitive.HIT, "sm:lift"),
    "F": (Primitive.HIT, "sm:fake"),
    "K": (Primitive.HIT, "sm:keysound"),
    "A": (Primitive.HIT, "sm:attack"),
    "2": (Primitive.HOLD, "sm:hold"),
    "4": (Primitive.HOLD, "sm:roll"),
}
NOTE_CHAR = {role: char for char, (_, role) in CHAR_ROLE.items()}

DEFAULT_RADAR = ",".join(["0.000000"] * 22)


# -- number formatting ---------------------------------------------------------


def _decimal(value: Fraction, min_places: int = 3, max_places: int = 6) -> str:
    """Exact decimal string when ``value`` has a finite expansion, else rounded."""
    places = next(
        (p for p in range(max(min_places, 0), 13) if (value * 10**p).denominator == 1),
        None,
    )
    if places is None:
        places = max_places
    scaled = round(value * 10**places)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    return f"{sign}{digits[:-places]}.{digits[-places:]}" if places else f"{sign}{digits}"


def _beat_str(beat: Fraction) -> str:
    return _decimal(Fraction(beat), 3, 6)


def _float_str(value: float) -> str:
    return str(int(value)) if value == int(value) and abs(value) < 1e15 else repr(value)


def _parse_beat(text: str) -> Fraction:
    return Fraction(round(Fraction(text.strip()) * BEAT_TICKS), BEAT_TICKS)


def _entries(raw: str | None) -> list[list[str]]:
    if not raw or not raw.strip():
        return []
    return [[part.strip() for part in row.split("=")] for row in raw.split(",") if row.strip()]


# -- timing --------------------------------------------------------------------


def _read_timing(src: Mapping[str, str], fallback_bpms: str | None = None) -> TimingMap:
    try:
        return _parse_timing(src, fallback_bpms)
    except (ValueError, ZeroDivisionError) as exc:
        raise CodecError(f"malformed timing data: {exc}") from exc


def _parse_timing(src: Mapping[str, str], fallback_bpms: str | None) -> TimingMap:
    offset_raw = (src.get("OFFSET") or "").strip()
    offset = Fraction(offset_raw) if offset_raw else Fraction(0)
    bpms_raw = src.get("BPMS") or fallback_bpms
    tempos = [TempoChange(_parse_beat(b), Fraction(v)) for b, v, *_ in _entries(bpms_raw)]
    if not tempos:
        tempos = [TempoChange(Fraction(0), Fraction(60))]  # StepMania's default BPM
    tempos, warps = _negative_bpms_to_warps(tempos)
    stops_raw = src.get("STOPS") or src.get("FREEZES")
    stops = [Stop(_parse_beat(b), Fraction(v) * 1_000_000) for b, v, *_ in _entries(stops_raw)]
    stops += [
        Stop(_parse_beat(b), Fraction(v) * 1_000_000, before=True)
        for b, v, *_ in _entries(src.get("DELAYS"))
    ]
    warps += [Warp(_parse_beat(b), Fraction(v)) for b, v, *_ in _entries(src.get("WARPS"))]
    meters = tuple(
        MeterChange(_parse_beat(b), int(n), int(d))
        for b, n, d, *_ in _entries(src.get("TIMESIGNATURES"))
    )
    return TimingMap(
        offset_us=-offset * 1_000_000,
        tempos=tuple(tempos),
        stops=tuple(stops),
        warps=tuple(sorted(warps, key=lambda w: w.beat)),
        meters=meters,
    )


def _negative_bpms_to_warps(
    tempos: list[TempoChange],
) -> tuple[list[TempoChange], list[Warp]]:
    """Convert old-style negative-BPM warps into explicit warps, as StepMania does.

    A segment ``[b, c)`` at ``-x`` BPM followed by ``y`` BPM rewinds time by
    ``(c - b) * 60 / x`` seconds; StepMania skips every beat whose time lies in
    the rewound span, i.e. a warp of ``(c - b) * (1 + y / x)`` beats from ``b``.
    """
    kept: list[TempoChange] = []
    warps: list[Warp] = []
    for i, tempo in enumerate(tempos):
        if tempo.bpm > 0:
            kept.append(tempo)
            continue
        after = tempos[i + 1] if i + 1 < len(tempos) else None
        if tempo.bpm == 0 or after is None or after.bpm <= 0 or not kept:
            raise CodecError(f"unsupported BPM {tempo.bpm} at beat {tempo.beat}")
        span = after.beat - tempo.beat
        warps.append(Warp(tempo.beat, span * (1 + after.bpm / -tempo.bpm)))
    return kept, warps


def _timing_tag(timing: TimingMap, tag: str) -> str:
    if tag == "OFFSET":
        return _decimal(-timing.offset_us / 1_000_000, 6, 9)
    if tag == "BPMS":
        return ",".join(f"{_beat_str(t.beat)}={_decimal(t.bpm)}" for t in timing.tempos)
    if tag in ("STOPS", "FREEZES", "DELAYS"):
        before = tag == "DELAYS"
        return ",".join(
            f"{_beat_str(s.beat)}={_decimal(s.duration_us / 1_000_000)}"
            for s in timing.stops
            if s.before == before
        )
    if tag == "WARPS":
        return ",".join(f"{_beat_str(w.beat)}={_decimal(w.length)}" for w in timing.warps)
    if tag == "TIMESIGNATURES":
        return ",".join(f"{_beat_str(m.beat)}={m.numerator}={m.denominator}" for m in timing.meters)
    raise KeyError(tag)


def _has_timing_tag(timing: TimingMap, tag: str) -> bool:
    if tag in ("STOPS", "FREEZES"):
        return any(not s.before for s in timing.stops)
    if tag == "DELAYS":
        return any(s.before for s in timing.stops)
    if tag == "WARPS":
        return bool(timing.warps)
    if tag == "TIMESIGNATURES":
        return bool(timing.meters)
    return True


def _append_missing_timing_tags(target: dict[str, str], timing: TimingMap) -> None:
    """Add timing tags the map needs but the original tag order lacked (e.g. synthesized warps)."""
    for tag in TIMING_TAGS[2:]:
        present = tag in target or (tag == "STOPS" and "FREEZES" in target)
        if not present and _has_timing_tag(timing, tag):
            target[tag] = _timing_tag(timing, tag)


# -- controls ------------------------------------------------------------------


def _read_control_tag(tag: str, raw: str) -> list[Control]:
    """Parse one control tag; raises ``ValueError`` on malformed input."""
    kind = CONTROL_KIND[tag]
    out: list[Control] = []
    if tag == "LABELS":
        rows = [row.partition("=") for row in raw.split(",") if row.strip()]
        for beat, sep, text in rows:
            if not sep:
                raise ValueError(f"malformed label {beat!r}")
            out.append(Control(time=Time(_parse_beat(beat)), kind=kind, value=text.strip()))
        return out
    for row in _entries(raw):
        beat = Time(_parse_beat(row[0]))
        if tag == "TICKCOUNTS":
            out.append(Control(time=beat, kind=kind, value=int(row[1])))
        elif tag == "COMBOS":
            native = {"miss": int(row[2])} if len(row) > 2 else {}
            out.append(Control(time=beat, kind=kind, value=int(row[1]), native=native))
        elif tag == "SPEEDS":
            native: dict[str, Any] = {"duration": float(row[2]), "unit": int(row[3])}
            out.append(Control(time=beat, kind=kind, value=float(row[1]), native=native))
        else:  # SCROLLS, FAKES
            out.append(Control(time=beat, kind=kind, value=float(row[1])))
    return out


def _write_control_tag(tag: str, controls: Sequence[Control]) -> str:
    kind = CONTROL_KIND[tag]
    rows: list[str] = []
    for c in controls:
        if c.kind != kind:
            continue
        beat = _beat_str(c.time.beat)
        if tag == "LABELS":
            rows.append(f"{beat}={c.value}")
        elif tag == "COMBOS":
            miss = c.native.get("miss")
            rows.append(f"{beat}={c.value}" + (f"={miss}" if miss is not None else ""))
        elif tag == "SPEEDS":
            assert isinstance(c.value, int | float)
            duration = _float_str(float(c.native.get("duration", 0.0)))
            unit = int(c.native.get("unit", 0))
            rows.append(f"{beat}={_float_str(float(c.value))}={duration}={unit}")
        elif tag == "TICKCOUNTS":
            rows.append(f"{beat}={c.value}")
        else:
            assert isinstance(c.value, int | float)
            rows.append(f"{beat}={_float_str(float(c.value))}")
    return ",".join(rows)


def _read_controls(src: Mapping[str, str], raw_out: dict[str, str]) -> tuple[Control, ...]:
    """Controls of all present control tags; malformed tags are kept raw in ``raw_out``."""
    controls: list[tuple[Fraction, int, int, Control]] = []
    for ti, tag in enumerate(CONTROL_TAGS):
        if tag not in src:
            continue
        try:
            parsed = _read_control_tag(tag, src[tag] or "")
        except (ValueError, IndexError, ZeroDivisionError):
            raw_out[tag] = src[tag]
            continue
        controls += [(c.time.beat, ti, i, c) for i, c in enumerate(parsed)]
    controls.sort(key=lambda item: item[:3])
    return tuple(c for *_, c in controls)


# -- notes ---------------------------------------------------------------------


def _mechanics_id(stepstype: str) -> str:
    return "sm_" + stepstype.strip().lower().replace("-", "_")


def _stepstype(mechanics: str) -> str:
    return mechanics.removeprefix("sm_").replace("_", "-")


def _read_events(notes: str) -> tuple[Event, ...]:
    rows: list[tuple[Fraction, int, int, dict[str, Any]]] = []
    pending: dict[tuple[int, int], dict[str, Any]] = {}

    def demote(fields: dict[str, Any]) -> None:
        """A hold head without a tail is played as a tap; keep its character."""
        fields["native"].setdefault("char", NOTE_CHAR[fields["role"]])
        fields["primitive"], fields["role"] = Primitive.HIT, "sm:tap"
        fields.pop("end", None)

    for cell in _notedata.parse(notes):
        native: dict[str, Any] = {}
        if cell.player:
            native["player"] = cell.player
        key = (cell.player, cell.column)
        if cell.base == "3" and key in pending:
            head = pending.pop(key)
            head["end"] = Time(cell.beat)
            if cell.raw is not None:
                head["native"]["tail_cell"] = cell.raw
            elif cell.keysound is not None:
                head["native"]["tail_keysound"] = cell.keysound
            continue
        if cell.raw is not None:
            native["cell"] = cell.raw
        elif cell.keysound is not None:
            native["keysound"] = cell.keysound
        primitive, role = CHAR_ROLE.get(cell.base, (Primitive.HIT, "sm:other"))
        if role == "sm:other" and cell.raw is None:
            native["char"] = cell.base
        fields: dict[str, Any] = {"primitive": primitive, "role": role, "native": native}
        if primitive is Primitive.HOLD:
            if key in pending:
                demote(pending.pop(key))
            pending[key] = fields
        rows.append((cell.beat, cell.player, cell.column, fields))
    for fields in pending.values():
        demote(fields)
    rows.sort(key=lambda r: r[:3])
    return tuple(
        Event(
            time=Time(beat),
            primitive=f["primitive"],
            role=f["role"],
            end=f.get("end"),
            target=Lane(column),
            native=f["native"],
        )
        for beat, _, column, f in rows
    )


def _write_notes(chart: Chart, timing: TimingMap, columns: int) -> str:
    def beat_of(time: Time) -> Fraction:
        if time.residual_us:
            return timing.time_at_us(timing.us(time), BEAT_TICKS * 4).beat
        return time.beat

    def text(native: dict[str, Any], cell_key: str, char: str, ks_key: str) -> str:
        if cell_key in native:
            return native[cell_key]
        keysound = native.get(ks_key)
        return char if keysound is None else f"{char}[{keysound}]"

    cells: list[tuple[int, Fraction, int, str]] = []
    for ev in chart.events:
        if not isinstance(ev.target, Lane):
            raise CodecError(f"StepMania events need a lane target, got {ev.target!r}")
        beat = beat_of(ev.time)
        if beat < 0:
            raise CodecError(f"StepMania cannot place notes before beat 0 ({beat})")
        if ev.role not in NOTE_CHAR and "char" not in ev.native and "cell" not in ev.native:
            raise CodecError(f"no StepMania note for role {ev.role!r}")
        player = int(ev.native.get("player", 0))
        char = ev.native.get("char") or NOTE_CHAR.get(ev.role, "1")
        cells.append((player, beat, ev.target.index, text(ev.native, "cell", char, "keysound")))
        if ev.primitive is Primitive.HOLD:
            if ev.end is None:
                raise CodecError("hold without end")
            tail = text(ev.native, "tail_cell", "3", "tail_keysound")
            cells.append((player, beat_of(ev.end), ev.target.index, tail))
    return _notedata.serialize(cells, columns)


# -- codec ---------------------------------------------------------------------


def _decode(data: bytes) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        return text.removeprefix("﻿"), encoding
    raise CodecError("undecodable simfile")


def _is_ssc(source: str | None, text: str) -> bool:
    suffix = Path(source).suffix.lower() if source else ""
    if suffix in (".ssc", ".sm"):
        return suffix == ".ssc"
    return "#VERSION" in text.upper() or "#NOTEDATA" in text.upper()


def _version(sim: Mapping[str, str]) -> float:
    try:
        return float(sim.get("VERSION") or 0)
    except ValueError:
        return 0.0


class StepManiaCodec(Codec):
    id = "stepmania"
    extensions = (".sm", ".ssc")
    fidelity = Fidelity.SEMANTIC

    mechanics = frozenset(spec for spec in registry.all_specs() if spec.startswith("sm_"))

    # -- read ------------------------------------------------------------------

    def read(self, data: bytes, *, source: str | None = None) -> ChartSet:
        text, encoding = _decode(data)
        ssc = _is_ssc(source, text)
        try:
            doc = _msd.parse(text, ssc=ssc)
        except ValueError as exc:
            raise CodecError(f"cannot parse simfile: {exc}") from exc

        sim, sim_charts = doc.tags, doc.charts
        song_timing = _read_timing(sim)
        split_timing = ssc and _version(sim) >= SSC_SPLIT_TIMING_VERSION
        own = [split_timing and any(c.get(t) for t in CHART_TIMING_TAGS) for c in sim_charts]
        song_controls_used = not all(own) or not sim_charts

        consumed = {*SONG_META_TAGS, *TIMING_TAGS, "FREEZES"}
        if song_controls_used:
            consumed |= set(CONTROL_TAGS)
        tags = {k: v for k, v in sim.items() if k not in consumed}
        song_controls = _read_controls(sim, tags) if song_controls_used else ()

        charts = tuple(
            self._read_chart(c, sim, song_controls, own=o, ssc=ssc)
            for c, o in zip(sim_charts, own, strict=True)
        )

        preview = None
        sample = (sim.get("SAMPLESTART") or "").strip()
        if sample:
            try:
                preview = song_timing.time_at_us(round(Fraction(sample) * 1_000_000), BEAT_TICKS)
            except ValueError:
                tags["SAMPLESTART"] = sim["SAMPLESTART"]
        music = (sim.get("MUSIC") or "").strip()
        native: dict[str, Any] = {
            "dialect": "ssc" if ssc else "sm",
            "encoding": encoding,
            "tag_order": list(sim.keys()),
            "tags": tags,
        }
        return ChartSet(
            song=SongMeta(
                title=sim.get("TITLE") or "",
                subtitle=sim.get("SUBTITLE") or "",
                artist=sim.get("ARTIST") or "",
                genre=sim.get("GENRE") or "",
                creator=sim.get("CREDIT") or "",
                preview=preview,
            ),
            audio=AudioRef(path=music) if music else None,
            timing=song_timing,
            charts=charts,
            provenance=Provenance(
                codec=self.id, source=source, sha256=hashlib.sha256(data).hexdigest()
            ),
            native=native,
        )

    def _read_chart(
        self,
        chart: Mapping[str, str],
        sim: Mapping[str, str],
        song_controls: tuple[Control, ...],
        *,
        own: bool,
        ssc: bool,
    ) -> Chart:
        stepstype = (chart.get("STEPSTYPE") or "").strip()
        mechanics = _mechanics_id(stepstype)
        try:
            spec = registry.get(mechanics)
        except KeyError:
            raise CodecError(f"unsupported StepMania stepstype {stepstype!r}") from None

        consumed = {"STEPSTYPE", "DESCRIPTION", "DIFFICULTY", "METER", "NOTES", "NOTES2"}
        native: dict[str, Any] = {}
        timing = None
        controls = song_controls
        if own:
            consumed |= {*TIMING_TAGS, *CONTROL_TAGS}
            timing = _read_timing(chart, fallback_bpms=sim.get("BPMS"))
            raw: dict[str, str] = {}
            controls = _read_controls(chart, raw)
            native.update({"tags": raw})
        tags = {k: v for k, v in chart.items() if k not in consumed}
        if ssc:
            native["tag_order"] = list(chart.keys())
            native["tags"] = {**tags, **native.get("tags", {})}
        else:
            native["radarvalues"] = chart.get("RADARVALUES") or ""
            native.pop("tags", None)
            extradata = chart.extradata if isinstance(chart, _msd.Tags) else None
            if extradata:
                native["extradata"] = list(extradata)

        label = (chart.get("DIFFICULTY") or "").strip()
        meter = (chart.get("METER") or "").strip()
        try:
            level = float(meter)
        except ValueError:
            level = None
            native["meter"] = meter
        meta = {}
        description = chart.get("DESCRIPTION") or ""
        if description:
            meta["description"] = description
        notes_key = "NOTES" if "NOTES" in chart or "NOTES2" not in chart else "NOTES2"
        events = _read_events(chart.get(notes_key) or "")
        if spec.space.lanes is not None:
            meta["lanes"] = str(spec.space.lanes)
        return Chart(
            mechanics=mechanics,
            difficulty=Difficulty(
                label=label or None, level=level, tier=DIFFICULTY_TIER.get(label.lower())
            ),
            events=events,
            controls=controls,
            timing=timing,
            meta=meta,
            native=native,
        )

    # -- write -----------------------------------------------------------------

    def write(self, chartset: ChartSet) -> bytes:
        self.check_writable(chartset)
        native = chartset.native
        dialect = native.get("dialect") or (
            "ssc" if any(c.timing for c in chartset.charts) else "sm"
        )
        ssc = dialect == "ssc"
        if not ssc and any(c.timing is not None for c in chartset.charts):
            raise CodecError("per-chart timing requires the .ssc dialect")

        doc = _msd.Document(ssc)
        sim = doc.tags
        song_chart = next((c for c in chartset.charts if c.timing is None), None)
        song_controls = song_chart.controls if song_chart else ()
        tags: dict[str, str] = dict(native.get("tags", {}))
        order: list[str] = (
            list(native["tag_order"])
            if "tag_order" in native
            else self._default_song_order(chartset, ssc)
        )
        timing = chartset.timing
        song = chartset.song

        values: dict[str, Callable[[], str]] = {
            "TITLE": lambda: song.title,
            "SUBTITLE": lambda: song.subtitle,
            "ARTIST": lambda: song.artist,
            "GENRE": lambda: song.genre,
            "CREDIT": lambda: song.creator,
            "MUSIC": lambda: chartset.audio.path if chartset.audio else "",
            "SAMPLESTART": lambda: (
                _decimal(Fraction(timing.us(song.preview), 1_000_000), 6)
                if song.preview is not None
                else ""
            ),
        }
        for key in order:
            if key in tags:
                sim[key] = tags[key]
            elif key == "VERSION":
                sim[key] = "0.83"
            elif key in values:
                sim[key] = values[key]()
            elif key in TIMING_TAGS or key == "FREEZES":
                sim[key] = _timing_tag(timing, key)
            elif key in CONTROL_TAGS:
                sim[key] = _write_control_tag(key, song_controls)
        _append_missing_timing_tags(sim, timing)
        doc.charts = [self._write_chart(chart, chartset, ssc) for chart in chartset.charts]

        encoding = native.get("encoding", "utf-8")
        text = _msd.serialize(doc)
        try:
            return text.encode(encoding)
        except UnicodeEncodeError:
            return text.encode("utf-8")

    @staticmethod
    def _default_song_order(chartset: ChartSet, ssc: bool) -> list[str]:
        timing = chartset.timing
        order = ["VERSION"] if ssc else []
        order += ["TITLE", "SUBTITLE", "ARTIST", "GENRE", "CREDIT", "MUSIC", "OFFSET"]
        if chartset.song.preview is not None:
            order.append("SAMPLESTART")
        order += [t for t in TIMING_TAGS[1:] if t == "BPMS" or _has_timing_tag(timing, t)]
        song_chart = next((c for c in chartset.charts if c.timing is None), None)
        if song_chart is not None:
            kinds = {c.kind for c in song_chart.controls}
            order += [t for t in CONTROL_TAGS if CONTROL_KIND[t] in kinds]
        return order

    def _write_chart(self, chart: Chart, chartset: ChartSet, ssc: bool) -> _msd.Tags:
        spec = registry.get(chart.mechanics)
        timing = chartset.timing_of(chart)
        lanes = spec.space.lanes or int(chart.meta.get("lanes", 0))
        if not lanes:
            lanes = 1 + max(
                (e.target.index for e in chart.events if isinstance(e.target, Lane)), default=0
            )
        notes = _write_notes(chart, timing, lanes)
        level = chart.difficulty.level
        meter = _float_str(level) if level is not None else chart.native.get("meter", "1")
        fields = {
            "STEPSTYPE": _stepstype(chart.mechanics),
            "DESCRIPTION": chart.meta.get("description", ""),
            "DIFFICULTY": chart.difficulty.label or "Edit",
            "METER": meter,
            "NOTES": notes,
        }
        if not ssc:
            sm_chart = _msd.Tags(fields)
            sm_chart["RADARVALUES"] = chart.native.get("radarvalues", DEFAULT_RADAR)
            if chart.native.get("extradata"):
                sm_chart.extradata = list(chart.native["extradata"])
            return sm_chart

        tags: dict[str, str] = dict(chart.native.get("tags", {}))
        order: list[str] = list(chart.native.get("tag_order", []))
        if "tag_order" not in chart.native:
            order = ["STEPSTYPE", "DESCRIPTION", "DIFFICULTY", "METER"]
            if chart.timing is not None:
                order += ["OFFSET", "BPMS"]
                order += [t for t in TIMING_TAGS[2:] if _has_timing_tag(chart.timing, t)]
                kinds = {c.kind for c in chart.controls}
                order += [t for t in CONTROL_TAGS if CONTROL_KIND[t] in kinds]
            order.append("NOTES")
        inherits_bpms = chart.timing is not None and chart.timing.tempos == chartset.timing.tempos
        if chart.timing is not None and "BPMS" not in order and not inherits_bpms:
            notes_at = next(
                (i for i, k in enumerate(order) if k in ("NOTES", "NOTES2")), len(order)
            )
            order.insert(notes_at, "BPMS")

        ssc_chart = _msd.Tags()
        for key in order:
            if key in fields:
                ssc_chart[key] = fields[key]
            elif key == "NOTES2":
                ssc_chart[key] = notes
            elif key in tags:
                ssc_chart[key] = tags[key]
            elif chart.timing is not None and key in TIMING_TAGS:
                ssc_chart[key] = _timing_tag(chart.timing, key)
            elif chart.timing is not None and key in CONTROL_TAGS:
                ssc_chart[key] = _write_control_tag(key, chart.controls)
        if chart.timing is not None:
            _append_missing_timing_tags(ssc_chart, chart.timing)
        if "NOTES" not in ssc_chart and "NOTES2" not in ssc_chart:
            ssc_chart["NOTES"] = notes
        return ssc_chart
