import os
from fractions import Fraction as F
from pathlib import Path

import pytest

from urfm.codecs.osu import OsuCodec
from urfm.codecs.osu.format import format_number, parse
from urfm.compare.roundtrip import roundtrip_bytes, roundtrip_path
from urfm.ir import Chart, ChartSet, Event, Lane, Primitive, Time, TimingMap
from urfm.ir.validate import validate

FIXTURES = Path(__file__).parent / "fixtures" / "osu"
FILES = sorted(FIXTURES.glob("*.osu"))
codec = OsuCodec()


def source_times_us(path: Path) -> list[int]:
    osu = parse(path.read_bytes())
    early = 24_000 if osu.version < 5 else 0
    return sorted(
        round(F(line.split(",")[2]) * 1000) + early for line in osu.sections.get("HitObjects", [])
    )


def test_has_enough_fixtures():
    assert len(FILES) >= 10


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_fixture_roundtrip_is_semantic_exact(path: Path):
    result = roundtrip_path(codec, path)
    assert result.error is None, result.error
    assert result.timing_diffs == []
    assert result.semantic_diffs == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_absolute_times_match_source(path: Path):
    cs = codec.read_path(path)
    got = sorted(cs.timing.us(ev.time) for ev in cs.charts[0].events)
    assert got == source_times_us(path)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_fixtures_are_valid_ir(path: Path):
    assert validate(codec.read_path(path)) == []


def test_mode_mapping():
    mechanics = {p.stem: codec.read_path(p).charts[0].mechanics for p in FILES}
    assert mechanics["std_basic"] == "osu_std"
    assert mechanics["taiko_basic"] == "osu_taiko"
    assert mechanics["catch_basic"] == "osu_catch"
    assert mechanics["mania_7k"] == "osu_mania"
    assert mechanics["std_v3_old"] == "osu_std"  # v3 files have no Mode key


def test_red_lines_restart_on_whole_beats():
    cs = codec.read_path(FIXTURES / "std_red_lines.osu")
    timing = cs.timing
    assert all(t.beat.denominator == 1 for t in timing.tempos)
    assert all(s.before for s in timing.stops)
    # Red line at 3100 ms lands on beat 6 even though 3100 is off the 333.33 ms grid.
    assert timing.tempos[1].beat == 6
    assert timing.us(Time.at(6)) == 3_100_000
    assert [m.numerator for m in timing.meters] == [4, 3, 7, 4]
    # Objects on the new grid snap without residual relative to their own red line.
    times = {timing.us(e.time): e.time for e in cs.charts[0].events}
    assert times[3_100_000] == Time.at(6)
    assert times[3_225_000] == Time.at(F(13, 2))


def test_taiko_roles_and_modifiers():
    cs = codec.read_path(FIXTURES / "taiko_basic.osu")
    summary = [(e.role, e.modifiers) for e in cs.charts[0].events[:9]]
    assert summary == [
        ("taiko:don", ()),
        ("taiko:ka", ()),
        ("taiko:ka", ()),
        ("taiko:don", ("big",)),
        ("taiko:ka", ("big",)),
        ("taiko:ka", ("big",)),
        ("taiko:ka", ()),
        ("taiko:drumroll", ()),
        ("taiko:drumroll", ("big",)),
    ]
    balloon = cs.charts[0].events[9]
    assert balloon.role == "taiko:balloon" and balloon.primitive is Primitive.REPEAT


def test_drumroll_duration_follows_slider_velocity():
    cs = codec.read_path(FIXTURES / "taiko_basic.osu")
    roll = cs.charts[0].events[7]
    # length 200 px, SV 1, multiplier 1.4 -> 200 / 140 beats of 375 ms
    assert cs.timing.us(roll.end) - cs.timing.us(roll.time) == round(F(200, 140) * 375_000)


def test_mania_lanes_and_holds():
    cs = codec.read_path(FIXTURES / "mania_7k.osu")
    chart = cs.charts[0]
    assert chart.meta == {"lanes": "7"}
    assert sorted({e.target.index for e in chart.events}) == list(range(7))
    holds = [e for e in chart.events if e.role == "mania:hold"]
    assert [(h.target, cs.timing.us(h.end)) for h in holds] == [
        (Lane(3), 500_000),
        (Lane(6), 812_000),
    ]


def test_slider_paths_and_repeats():
    cs = codec.read_path(FIXTURES / "std_basic.osu")
    sliders = [e for e in cs.charts[0].events if e.role == "osu:slider"]
    assert [s.path[0].shape for s in sliders] == [
        "osu:bezier",
        "osu:linear",
        "osu:perfect",
        "osu:catmull",
    ]
    assert sliders[0].count == 2
    assert sliders[0].path[0].points[0] == sliders[0].target  # path starts at the head


def test_kiai_becomes_highlight_controls():
    cs = codec.read_path(FIXTURES / "std_basic.osu")
    highlights = [
        (cs.timing.us(c.time), c.value) for c in cs.charts[0].controls if c.kind == "highlight"
    ]
    assert highlights == [(2_500_000, True), (4_500_000, False)]


def test_early_version_offset():
    cs = codec.read_path(FIXTURES / "std_v3_old.osu")
    assert cs.timing.offset_us == 1_024_000
    assert cs.charts[0].native["format_version"] == 3


def test_header_only_file_uses_implicit_timing():
    cs = codec.read(b"osu file format v6\r\n")
    assert cs.charts[0].events == ()
    assert roundtrip_bytes(codec, b"osu file format v6\r\n").semantic_exact


def test_rejected_red_line_is_preserved():
    data = (
        (FIXTURES / "std_basic.osu")
        .read_bytes()
        .replace(b"[TimingPoints]\r\n", b"[TimingPoints]\r\n0,NaN,4,2,0,100,1,0\r\n")
        .replace(b"[TimingPoints]\n", b"[TimingPoints]\n0,NaN,4,2,0,100,1,0\n")
    )
    result = roundtrip_bytes(codec, data)
    assert result.semantic_exact
    assert codec.read(data).charts[0].native["rejected_timing_points"] == ["0,NaN,4,2,0,100,1,0"]


def test_write_chart_without_native_fields():
    timing = TimingMap.constant(180, offset_us=250_000)
    notes = [
        Event(time=Time.at(F(i, 2)), primitive=Primitive.HIT, role="mania:note", target=Lane(i % 4))
        for i in range(8)
    ]
    hold = Event(
        time=Time.at(4), end=Time.at(6), primitive=Primitive.HOLD, role="mania:hold", target=Lane(2)
    )
    events = (*notes, hold)
    cs = ChartSet(
        timing=timing,
        charts=(Chart(mechanics="osu_mania", meta={"lanes": "4"}, events=events),),
    )
    again = codec.read(codec.write(cs))
    # osu! stores hit-object times as integer ms; charts without native fields are rounded
    got = again.charts[0]
    assert len(got.events) == len(events)
    for a, b in zip(events, got.events, strict=True):
        assert abs(timing.us(a.time) - again.timing_of(got).us(b.time)) <= 500
        assert (a.role, a.target, a.primitive) == (b.role, b.target, b.primitive)


def test_format_number():
    assert format_number(F(1000)) == "1000"
    assert format_number(F("333.333333333333")) == "333.333333333333"
    assert format_number(F("-0.5")) == "-0.5"
    assert format_number(F("1003.5")) == "1003.5"
    assert format_number(F(1, 3)) == repr(1 / 3)


CORPUS = os.environ.get("URFM_CORPUS_OSU")


@pytest.mark.skipif(not CORPUS, reason="set URFM_CORPUS_OSU to a directory of .osu files")
def test_corpus_roundtrip():
    assert CORPUS
    failures = []
    for path in sorted(Path(CORPUS).rglob("*.osu")):
        result = roundtrip_path(codec, path)
        if not result.semantic_exact:
            failures.append((path.name, result.error, result.semantic_diffs[:2]))
    assert failures == []
