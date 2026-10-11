import os
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

import pytest

from urfm.codecs.tja import TjaCodec
from urfm.codecs.tja.common import format_decimal
from urfm.compare.roundtrip import roundtrip_bytes, roundtrip_path
from urfm.ir import Primitive
from urfm.ir.validate import validate

FIXTURES = Path(__file__).parent / "fixtures" / "tja"
CODEC = TjaCodec()


def fixtures() -> list[Path]:
    return sorted(FIXTURES.glob("*.tja"))


def read(name: str):
    return CODEC.read_path(FIXTURES / name)


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.stem)
def test_fixture_roundtrip_is_semantic_exact(path: Path):
    result = roundtrip_path(CODEC, path)
    assert result.error is None, result.error
    assert result.timing_exact, result.timing_diffs
    assert result.semantic_exact, result.semantic_diffs


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.stem)
def test_fixture_is_valid(path: Path):
    assert validate(CODEC.read_path(path)) == []


def test_writer_is_deterministic_and_idempotent():
    for path in fixtures():
        once = CODEC.write(CODEC.read_path(path))
        assert CODEC.write(CODEC.read(once)) == once


def test_exact_beats_and_times():
    cs = read("bpm_mid_measure.tja")
    chart = cs.charts[0]
    timing = cs.timing_of(chart)
    beats = [e.time.beat for e in chart.events]
    assert beats[:4] == [0, 1, 2, 3]
    # 120 BPM for two beats, then 180 BPM
    assert [timing.us(e.time) for e in chart.events[:4]] == [0, 500_000, 1_000_000, 1_333_333]
    assert [t.bpm for t in timing.tempos] == [120, 180, F(181, 2), 240, 200]


def test_measure_grid():
    chart = read("measure_changes.tja").charts[0]
    beats = [e.time.beat for e in chart.events]
    assert beats[:2] == [0, 2]  # 3/4: slots of one beat
    assert beats[2:6] == [3, 4, 5, 6]  # 7/8: slots of half a beat
    assert beats[6] == F(13, 2)
    measure_6_8 = [m for m in read("measure_changes.tja").charts[0].timing.meters]
    assert [(m.numerator, m.denominator) for m in measure_6_8] == [
        (3, 4),
        (7, 8),
        (5, 16),
        (4, 4),
        (6, 8),
    ]


def test_delay_is_a_stop_before_the_note():
    cs = read("delay.tja")
    timing = cs.timing_of(cs.charts[0])
    assert [(s.beat, s.duration_us, s.before) for s in timing.stops] == [
        (4, 250_000, True),
        (10, 500_000, True),
        (10, -125_000, True),
        (12, 1_000_000, True),
    ]
    times = [timing.us(e.time) for e in cs.charts[0].events]
    assert times[4] == 500_000 + 4 * 500_000 + 250_000  # OFFSET:-0.5 puts beat 0 at +0.5 s


def test_branches_and_balloons():
    chart = read("branches.tja").charts[0]
    assert [b.routes for b in chart.branches] == [("normal", "expert", "master")] * 2
    assert chart.branches[0].end is not None and chart.branches[1].end is None
    assert chart.branches[0].condition["type"] == "p"
    balloons = [
        (e.branch and e.branch.route, e.count) for e in chart.events if e.role == "taiko:balloon"
    ]
    assert balloons == [(None, 4), ("normal", 8), ("expert", 12), ("master", None)]
    kusudama = [e for e in chart.events if "kusudama" in e.modifiers]
    assert kusudama and kusudama[0].end is not None
    kinds = Counter(c.kind for c in chart.controls)
    assert kinds["highlight"] == 4 and kinds["taiko:section"] == 1 and kinds["taiko:levelhold"] == 1


def test_per_route_balloon_lists():
    chart = read("branch_balloons_per_route.tja").charts[0]
    counts = {
        e.branch and e.branch.route: e.count for e in chart.events if e.role == "taiko:balloon"
    }
    assert counts == {None: 2, "normal": 3, "expert": 6, "master": 9}


def test_courses_inherit_headers():
    cs = read("multi_course.tja")
    labels = [(c.difficulty.label, c.difficulty.level, c.difficulty.tier) for c in cs.charts]
    assert labels == [
        ("easy", 1, 0),
        ("normal", 3, 1),
        ("hard", 5, 2),
        ("oni", 8, 3),
        ("edit", 10, 4),
    ]
    oni = cs.charts[3]
    assert oni.timing is not None and oni.timing.tempos[0].bpm == 200
    assert cs.charts[4].timing.tempos[0].bpm == 200  # BPM header persists
    assert [e.count for e in cs.charts[1].events if e.role == "taiko:balloon"] == [15, 20]


def test_double_play():
    cs = read("double_play.tja")
    assert [c.meta.get("player") for c in cs.charts] == ["1", "2", None]
    assert [c.meta.get("STYLE") for c in cs.charts] == ["Double", "Double", "Single"]


def test_drumroll_spans_measures_and_tempo_changes():
    cs = read("drumroll_span.tja")
    rolls = [e for e in cs.charts[0].events if e.primitive is Primitive.REPEAT]
    assert [(r.time.beat, r.end.beat) for r in rolls if r.end] == [
        (0, 15),
        (16, 24),
        (28, F(61, 2)),
        (31, 34),
    ]
    assert rolls[-1].native == {"implicit_end": True}


def test_special_and_unknown_notes():
    chart = read("special_notes.tja").charts[0]
    roles = [(e.role, e.modifiers) for e in chart.events]
    assert ("taiko:don", ("big", "hand")) in roles and ("taiko:mine", ()) in roles
    assert [n[1] for n in chart.native["unknown_notes"]] == ["D", "E", "H"]


def test_song_metadata():
    cs = read("basic.tja")
    assert (cs.song.title, cs.song.subtitle, cs.song.genre) == ("Basic", "--fixture", "Test")
    assert cs.audio is not None and cs.audio.path == "basic.ogg"
    assert cs.timing.offset_us == 123_000
    assert cs.song.preview is not None and cs.timing.us(cs.song.preview) == 12_500_000
    assert cs.charts[0].meta == {"SCOREINIT": "400", "SCOREDIFF": "100"}


def test_shift_jis_title():
    assert read("shift_jis.tja").song.title == "太鼓"


def test_format_decimal():
    assert format_decimal(F("150.5")) == "150.5"
    assert format_decimal(F(-1, 8)) == "-0.125"
    assert format_decimal(F(3)) == "3"
    assert format_decimal(F(1, 3)) == repr(1 / 3)


def test_roundtrip_bytes_of_minimal_chart():
    result = roundtrip_bytes(CODEC, b"BPM:120\n#START\n1,\n#END\n")
    assert result.semantic_exact


# -- differential test against tja-rs -------------------------------------------

TJA_RS_TYPES = {
    "Don": "1",
    "Ka": "2",
    "DonBig": "3",
    "KaBig": "4",
    "Roll": "5",
    "RollBig": "6",
    "Balloon": "7",
    "BalloonAlt": "9",
}
OUR_TYPES = {
    ("taiko:don", ()): "1",
    ("taiko:ka", ()): "2",
    ("taiko:don", ("big",)): "3",
    ("taiko:ka", ("big",)): "4",
    ("taiko:drumroll", ()): "5",
    ("taiko:drumroll", ("big",)): "6",
    ("taiko:balloon", ()): "7",
    ("taiko:balloon", ("kusudama",)): "9",
}
ROUTE = {"N": "normal", "E": "expert", "M": "master", None: None}


def _differential(text: str) -> None:
    tja = pytest.importorskip("tja")
    try:
        parsed = tja.parse_tja(text)
    except BaseException as exc:  # tja-rs panics (pyo3 PanicException) on some inputs
        pytest.skip(f"tja-rs cannot parse this input: {exc!r}")
    ours = CODEC.read(text.encode())
    # tja-rs turns a duplicated "#START" line into an extra empty chart
    their_charts = [c for c in parsed.charts if c.segments]
    assert len(their_charts) == len([c for c in ours.charts if c.events])
    has_letters = any(
        ch in line
        for line in text.splitlines()
        if not line.startswith("#") and ":" not in line
        for ch in "ABCDEFGHI"
    )
    for their_chart, chart in zip(their_charts, [c for c in ours.charts if c.events], strict=True):
        timing = ours.timing_of(chart)
        mine = Counter()
        for ev in chart.events:
            code = OUR_TYPES.get((ev.role, ev.modifiers))
            if code is not None:
                route = ev.branch.route if ev.branch else None
                mine[(round(timing.us(ev.time)), code, route)] += 1
        delays_in_branches = bool(chart.branches) and bool(timing.stops)
        if (
            timing.tempos[0].bpm != ours.timing.tempos[0].bpm
            or timing.offset_us != ours.timing.offset_us
        ):
            continue  # tja-rs ignores BPM/OFFSET headers redefined after the first course
        for seg in their_chart.segments:
            for note in seg.notes:
                code = TJA_RS_TYPES.get(note.note_type)
                if code is None:
                    continue
                us = note.timestamp * 1_000_000
                key = next(
                    (
                        k
                        for k in mine
                        if k[1] == code and k[2] == ROUTE[seg.branch] and abs(k[0] - us) <= 1
                    ),
                    None,
                )
                if delays_in_branches or has_letters:
                    continue  # tja-rs accumulates delays across routes and ignores letter notes
                assert key is not None, (note.note_type, note.timestamp, seg.branch)
                mine[key] -= 1
                if not mine[key]:
                    del mine[key]
        if not chart.branches and not has_letters:
            assert not mine, f"notes missing in tja-rs: {list(mine)[:5]}"


@pytest.mark.parametrize("path", fixtures(), ids=lambda p: p.stem)
def test_differential_against_tja_rs(path: Path):
    from urfm.codecs.tja.reader import decode

    _differential(decode(path.read_bytes()))


@pytest.mark.skipif(not os.environ.get("URFM_CORPUS_TJA"), reason="set URFM_CORPUS_TJA")
def test_corpus():
    root = Path(os.environ["URFM_CORPUS_TJA"])
    failures = []
    for path in sorted(root.rglob("*.tja")):
        result = roundtrip_path(CODEC, path)
        if not result.semantic_exact:
            failures.append((path, result.error, result.semantic_diffs[:3]))
    assert not failures, failures[:10]


@pytest.mark.parametrize(
    "meters",
    [
        [(0, 2), (198, 5), (F(401, 2), 4)],
        [(0, 2), (199, 5)],
        [(0, 4), (F(3, 2), 3)],
        [(0, 4), (F(1, 3), 4)],
    ],
)
def test_meter_change_inside_a_measure_keeps_time(meters):
    """A meter change that falls inside a measure (generated charts do this) starts a new
    measure there; the writer cuts the measure before it short instead of shifting every later
    note (a generated Ura course drifted by seconds)."""
    import msgspec

    from urfm.ir.chart import Chart, ChartSet, Difficulty
    from urfm.ir.events import Event
    from urfm.ir.time import Time
    from urfm.ir.timing import MeterChange, TimingMap

    base = TimingMap.constant(150)
    timing = msgspec.structs.replace(base, meters=tuple(MeterChange(F(b), n, 4) for b, n in meters))
    events = tuple(
        Event(time=Time(F(i, 2)), primitive=Primitive.HIT, role="taiko:don") for i in range(0, 600)
    )
    chart = Chart(
        mechanics="taiko", events=events, timing=timing, difficulty=Difficulty(label="edit")
    )
    text = CODEC.write(ChartSet(timing=base, charts=(chart,)))
    back = CODEC.read(text)
    got = sorted(back.timing_of(back.charts[0]).us(e.time) for e in back.charts[0].events)
    assert got == [timing.us(e.time) for e in events]
    _differential(text.decode())
