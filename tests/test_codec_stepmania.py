import warnings
from fractions import Fraction as F
from pathlib import Path

import pytest

from urfm.codecs.base import CodecError
from urfm.codecs.stepmania import StepManiaCodec
from urfm.compare.roundtrip import roundtrip_path
from urfm.ir import Lane, Primitive, Time, Warp
from urfm.ir.validate import validate

try:  # dev-only timing oracle; simfile -> fs still needs setuptools' pkg_resources
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from simfile.sm import SMSimfile
        from simfile.ssc import SSCSimfile
        from simfile.timing import TimingData
        from simfile.timing.engine import TimingEngine
except ImportError as exc:
    ORACLE_MISSING: str | None = f"simfile timing oracle unavailable: {exc}"
else:
    ORACLE_MISSING = None

FIXTURES = Path(__file__).parent / "fixtures" / "stepmania"
FILES = sorted(p for p in FIXTURES.iterdir() if p.suffix in (".sm", ".ssc"))
codec = StepManiaCodec()
NORMALIZED = {"negative_bpm.sm"}
"""Written back with an explicit ``#WARPS`` tag, so the tag order (only) changes."""
SIMFILE_DIVERGES = {
    "negative_bpm.sm": "simfile does not convert negative BPMs into warps as StepMania does",
    "fractional_warp.ssc": "simfile rounds warp lengths to 1/48 beat; StepMania keeps them",
}


def read(name: str):
    return codec.read_path(FIXTURES / name)


def test_has_enough_fixtures():
    assert len(FILES) >= 10


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_roundtrip_semantic(path: Path):
    result = roundtrip_path(codec, path)
    assert result.error is None, result.error
    assert result.timing_exact, result.timing_diffs
    if path.name not in NORMALIZED:
        assert result.semantic_exact, result.semantic_diffs


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_fixtures_validate(path: Path):
    assert validate(codec.read_path(path)) == []


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_times_match_simfile_engine(path: Path):
    """Differential test: every note time equals simfile's TimingEngine within 1 µs."""
    if ORACLE_MISSING:
        pytest.skip(ORACLE_MISSING)
    if path.name in SIMFILE_DIVERGES:
        pytest.skip(SIMFILE_DIVERGES[path.name])
    text = path.read_text("utf-8")
    sim = SSCSimfile(string=text) if path.suffix == ".ssc" else SMSimfile(string=text)
    cs = codec.read_path(path)
    checked = 0
    for sim_chart, chart in zip(sim.charts, cs.charts, strict=True):
        if chart.timing is not None and not sim_chart.get("BPMS"):
            sim_chart["BPMS"] = sim["BPMS"]  # the codec inherits song BPMs; simfile does not
        engine = TimingEngine(TimingData(sim, sim_chart))
        timing = cs.timing_of(chart)
        for ev in chart.events:
            if not engine.hittable(ev.time.beat):
                continue  # warped / faked notes have no well-defined hit time
            expected = float(engine.time_at(ev.time.beat)) * 1_000_000
            assert abs(timing.us(ev.time) - expected) <= 1, (chart.difficulty, ev)
            checked += 1
    assert checked > 0


HAND_TIMES = {
    # fixture -> {(chart index, beat): expected µs}, derived by hand from the tags
    "bpm_stops.sm": {
        # OFFSET -0.050 -> beat 0 at 50 ms; 128 BPM = 468750 µs/beat
        (0, F(1)): 50_000 + 468_750,
        (0, F(2)): 50_000 + 2 * 468_750,  # stop at 2: the note precedes the pause
        (0, F(3)): 50_000 + 3 * 468_750 + 500_000,
        # beat 4: 4 beats @128 + 0.5 s stop; then 256 BPM = 234375 µs/beat
        (0, F(5)): 50_000 + 4 * 468_750 + 500_000 + 234_375,
        # stop 0.25 s at 6 (strictly after), 64.5 BPM from 8.5
        (0, F(17, 2)): 50_000 + 4 * 468_750 + 500_000 + F(9, 2) * 234_375 + 250_000,
    },
    "gimmicks.ssc": {
        # OFFSET -0.123456; 150 BPM = 400000 µs/beat; delay 0.1 s at beat 2
        (0, F(1)): 123_456 + 400_000,
        (0, F(2)): 123_456 + 2 * 400_000 + 100_000,  # delay: the note follows the pause
        # stop 0.2 s at beat 4 is applied strictly after it
        (0, F(4)): 123_456 + 4 * 400_000 + 100_000,
        (0, F(5)): 123_456 + 5 * 400_000 + 100_000 + 200_000,
        # warp [6, 7): beat 7 is hit at the time of beat 6
        (0, F(7)): 123_456 + 6 * 400_000 + 300_000,
    },
    "split_timing.ssc": {
        (0, F(3)): 3 * 500_000,  # song timing, 120 BPM
        # chart OFFSET -0.010, stop 0.25 s at 1, 240 BPM from beat 2
        (1, F(3)): 10_000 + 2 * 500_000 + 250_000 + 250_000,
        (2, F(3)): 3 * 500_000,  # song BPMs, own (empty) offset; stop at 3 is strict
    },
}


@pytest.mark.parametrize("name", sorted(HAND_TIMES))
def test_hand_computed_times(name: str):
    cs = read(name)
    for (ci, beat), expected in HAND_TIMES[name].items():
        chart = cs.charts[ci]
        assert cs.timing_of(chart).us(Time(beat)) == round(expected), (ci, beat)


def test_basic_metadata_and_events():
    cs = read("basic.sm")
    assert cs.song.title == "Basic Song" and cs.song.subtitle == "(Short Ver.)"
    assert cs.song.creator == "urfm" and cs.audio is not None and cs.audio.path == "song.ogg"
    assert cs.timing.offset_us == 9_000  # beat 0 at -OFFSET seconds
    assert cs.timing.us(cs.song.preview) == 12_345_000
    assert cs.native["dialect"] == "sm" and cs.native["tags"]["BANNER"] == "bn.png"
    (chart,) = cs.charts
    assert chart.mechanics == "sm_dance_single"
    assert chart.difficulty.label == "Medium" and chart.difficulty.level == 5
    assert chart.difficulty.tier == 2
    assert chart.meta["description"] == "urfm"
    first = chart.events[:4]
    assert [e.target for e in first] == [Lane(0), Lane(1), Lane(2), Lane(3)]
    assert [e.time.beat for e in first] == [0, 1, 2, 3]
    # second measure: 8 rows -> eighth notes, jump on beat 4
    assert {(e.time.beat, e.target.index) for e in chart.events if e.time.beat == 4} == {
        (F(4), 0),
        (F(4), 3),
    }


def test_holds_rolls_mines():
    hard = read("multi_difficulty.sm").charts[2]
    holds = [e for e in hard.events if e.primitive is Primitive.HOLD]
    assert {(e.role, e.target.index, e.time.beat, e.end.beat) for e in holds} == {
        ("sm:hold", 0, F(0), F(3)),
        ("sm:roll", 3, F(0), F(3)),
    }
    assert sum(e.role == "sm:mine" for e in hard.events) == 2
    edit = read("multi_difficulty.sm").charts[4]
    assert edit.difficulty.level is None and edit.native["meter"] == "?"
    assert edit.difficulty.tier is None


def test_subdivisions_are_exact():
    chart = read("subdivisions.sm").charts[0]
    beats = {e.time.beat for e in chart.events}
    assert F(2, 3) in beats  # 12th
    assert F(1, 6) + 4 in beats  # 24th
    assert F(8) + F(1, 48) in beats  # 192nd
    assert F(12) + F(3, 16) in beats  # 64th
    assert F(16) + F(1, 5) in beats  # 5-tuplet (20 rows per measure)


def test_lifts_fakes_keysounds():
    cs = read("lifts_fakes_keysounds.ssc")
    chart = cs.charts[0]
    roles = [e.role for e in chart.events]
    assert {"sm:lift", "sm:fake", "sm:mine", "sm:keysound", "sm:hold"} <= set(roles)
    tap = chart.events[0]
    assert tap.role == "sm:tap" and tap.native == {"keysound": 0}
    hold = next(e for e in chart.events if e.role == "sm:hold")
    assert hold.native == {"keysound": 0, "tail_keysound": 1}
    assert chart.native["tags"]["CHARTNAME"] == "Keys"
    assert cs.native["tags"]["KEYSOUNDS"] == "kick.wav,snare.wav"


def test_gimmick_timing_and_controls():
    cs = read("gimmicks.ssc")
    t = cs.timing
    assert [s.before for s in t.stops] == [False, True, True]
    assert t.warps[1].beat == F(21, 2) and t.warps[1].length == F(1, 4)
    assert [(m.beat, m.numerator, m.denominator) for m in t.meters][-1] == (F(11), 7, 8)
    chart = cs.charts[0]
    kinds = {c.kind for c in chart.controls}
    assert kinds == {"sm:tickcount", "sm:combo", "sm:speed", "scroll", "sm:fake", "section"}
    speed = [c for c in chart.controls if c.kind == "sm:speed"][1]
    assert speed.value == 0.5 and speed.native == {"duration": 2.0, "unit": 1}
    combo = [c for c in chart.controls if c.kind == "sm:combo"][1]
    assert combo.value == 2 and combo.native == {"miss": 3}
    labels = [c.value for c in chart.controls if c.kind == "section"]
    assert labels == ["Song Start", "Drop!", "End"]
    times = [c.time for c in chart.controls]
    assert times == sorted(times)


def test_split_timing():
    cs = read("split_timing.ssc")
    easy, hard, challenge = cs.charts
    assert easy.timing is None
    assert hard.timing is not None and hard.timing.offset_us == 10_000
    assert [c.value for c in hard.controls if c.kind == "scroll"] == [2.0]
    assert [c.value for c in easy.controls if c.kind == "section"] == ["Song Start"]
    # a chart with only STOPS keeps the song's BPMs but uses its own (zero) offset
    assert challenge.timing is not None
    assert challenge.timing.tempos == cs.timing.tempos
    assert challenge.controls == ()


def test_all_split_keeps_unused_song_controls_raw():
    cs = read("all_split.ssc")
    assert all(c.timing is not None for c in cs.charts)
    assert cs.native["tags"]["LABELS"] == "0.000=Never Used"
    assert b"Never Used" in codec.write(cs)


def test_routine_players():
    chart = read("routine.ssc").charts[0]
    assert chart.mechanics == "sm_dance_routine"
    players = {e.native.get("player", 0) for e in chart.events}
    assert players == {0, 1}
    p2_hold = next(e for e in chart.events if e.native.get("player") == 1 and e.end)
    assert p2_hold.target == Lane(6) and p2_hold.end == Time(F(3))


def test_negative_stop_is_preserved():
    t = read("negative_stop.sm").timing
    assert t.stops[0].duration_us == -250_000


def test_orphans_keep_their_characters():
    chart = read("orphans.sm").charts[0]
    chars = [(e.time.beat, e.native.get("char")) for e in chart.events]
    assert chars == [(F(0), "2"), (F(1), "3"), (F(2), "2"), (F(4), "4")]


def test_pump_charts():
    single, double = read("pump.ssc").charts
    assert single.mechanics == "sm_pump_single" and double.mechanics == "sm_pump_double"
    assert max(e.target.index for e in double.events) == 9


def test_dialect_detection_without_suffix():
    data = (FIXTURES / "gimmicks.ssc").read_bytes()
    assert codec.read(data).native["dialect"] == "ssc"
    data = (FIXTURES / "basic.sm").read_bytes()
    assert codec.read(data).native["dialect"] == "sm"


def test_unknown_stepstype_is_rejected():
    text = "#BPMS:0=120;\n#NOTES:foo-bar::Easy:1::0000;\n"
    with pytest.raises(CodecError):
        codec.read(text.encode(), source="x.sm")


def test_write_minimal_ir_chartset():
    """IR without StepMania native fields still serializes to a valid simfile."""
    from urfm.ir import Chart, ChartSet, Difficulty, Event, TimingMap

    t = Time.at
    cs = ChartSet(
        timing=TimingMap.constant(F("145.5"), offset_us=-20_000),
        charts=(
            Chart(
                mechanics="sm_dance_single",
                difficulty=Difficulty(label="Hard", level=9),
                events=(
                    Event(time=t(0), primitive=Primitive.HIT, role="sm:tap", target=Lane(0)),
                    Event(
                        time=t(F(1, 3)),
                        end=t(2),
                        primitive=Primitive.HOLD,
                        role="sm:roll",
                        target=Lane(2),
                    ),
                ),
            ),
        ),
    )
    data = codec.write(cs)
    back = codec.read(data, source="out.sm")
    assert back.timing == cs.timing
    assert [(e.time, e.end, e.role, e.target) for e in back.charts[0].events] == [
        (e.time, e.end, e.role, e.target) for e in cs.charts[0].events
    ]
    with pytest.raises(CodecError):
        codec.write(ChartSet(timing=cs.timing, charts=(Chart(mechanics="taiko"),)))


def test_negative_bpm_becomes_warp():
    """SM5 NotesLoaderSM: [b, c) at -x BPM then y BPM -> warp of (c-b)(1 + y/x) beats at b."""
    t = read("negative_bpm.sm").timing
    assert [(x.beat, x.bpm) for x in t.tempos] == [(F(0), F(120)), (F(6), F(120))]
    assert t.warps == (Warp(F(4), F(2) * (1 + F(120, 240))),)
    # beat 4 at 2.0 s, rewound 0.5 s over [4, 6), then 2 beats at 0.5 s -> beat 8 at 2.5 s
    assert t.us(Time(F(8))) == 2_500_000
    result = roundtrip_path(codec, FIXTURES / "negative_bpm.sm")
    assert result.timing_exact


def test_fractional_warp_length_is_exact():
    t = read("fractional_warp.ssc").timing
    assert t.warps[0].length == F(83, 1000)


def test_msd_escaping_roundtrip():
    """Escaped ':' ';' '//' '\\' in tag values survive read -> write -> read."""
    text = (
        "#TITLE:A\\:B\\;C\\//D\\\\E;\n#ARTIST:x;\n#BPMS:0=120;\n"
        "#NOTES:dance-single:desc\\:with colon:Easy:1:0,0,0,0,0:1000\n0000\n0000\n0000;\n"
    )
    cs = codec.read(text.encode(), source="esc.sm")
    assert cs.song.title == "A:B;C//D\\E"
    assert cs.charts[0].meta["description"] == "desc:with colon"
    back = codec.read(codec.write(cs), source="esc.sm")
    assert back.song.title == cs.song.title
    assert back.charts[0].meta == cs.charts[0].meta
