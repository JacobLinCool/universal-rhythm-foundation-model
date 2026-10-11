from fractions import Fraction as F
from pathlib import Path

import pytest

from urfm.codecs import registry
from urfm.codecs.base import CodecError
from urfm.codecs.simai import SimaiCodec
from urfm.compare import roundtrip_bytes
from urfm.ir import Primitive, RingButton, Time, TouchRegion
from urfm.ir.validate import validate

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "simai").glob("*.txt"))
codec = SimaiCodec()


def chart_of(text: str, index: int = 0):
    cs = codec.read(text.encode())
    return cs, cs.charts[index], cs.timing_of(cs.charts[index])


def wrap(fumen: str, bpm: int = 120) -> str:
    return f"&title=t\n&inote_5=\n({bpm}){{4}}{fumen}\nE\n"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_roundtrip_is_semantic_exact(path: Path):
    result = roundtrip_bytes(codec, path.read_bytes(), str(path))
    assert result.error is None
    assert result.semantic_exact, result.semantic_diffs
    assert result.timing_exact


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_is_valid(path: Path):
    assert validate(codec.read_path(path)) == []


def test_fixture_count():
    assert len(FIXTURES) >= 10


def test_registry_and_sniff(tmp_path: Path):
    assert isinstance(registry.get("simai"), SimaiCodec)
    simai = tmp_path / "maidata.txt"
    simai.write_text(wrap("1,"))
    other = tmp_path / "notes.txt"
    other.write_text("just text")
    assert codec.sniff(simai, simai.read_bytes())
    assert not codec.sniff(other, other.read_bytes())
    assert isinstance(registry.for_path(simai), SimaiCodec)


def test_metadata_and_difficulties():
    cs = codec.read((FIXTURES[0].parent / "10_multi_difficulty.txt").read_bytes())
    assert cs.song.title == "Multi Difficulty"
    assert cs.song.creator == "designer"
    assert cs.song.genre == "maimai"
    assert cs.song.extra == {"wholebpm": "128", "version": "fixture"}
    labels = [c.difficulty.label for c in cs.charts]
    assert labels == ["Easy", "Basic", "Advanced", "Expert", "Master", "Re:Master"]
    assert [c.difficulty.level for c in cs.charts] == [1, 3, 7.5, 10, 13.7, 14.5]
    assert cs.charts[1].meta == {"level": "3", "designer": "basic designer"}
    assert [c.difficulty.tier for c in cs.charts] == [0, 1, 2, 3, 4, 4]
    # Re:Master has a different tempo, so it carries its own timing.
    assert all(c.timing is None for c in cs.charts[:5])
    assert cs.charts[5].timing is not None and cs.charts[5].timing.tempos[0].bpm == 130


def test_first_offset_and_tick_positions():
    _, chart, timing = chart_of("&first=0.5\n" + wrap("1,2,{8}3,4,{1}5,"))
    beats = [ev.time.beat for ev in chart.events]
    assert beats == [0, 1, 2, F(5, 2), 3]
    assert timing.offset_us == 500_000
    assert timing.us(chart.events[1].time) == 1_000_000


def test_taps_modifiers_and_appearance():
    _, chart, _ = chart_of(wrap("1b,2x,3bx,4m,5$,6$$,7@,56,"))
    ev = chart.events
    assert ev[0].modifiers == ("break",) and ev[0].target == RingButton(0)
    assert ev[1].modifiers == ("ex",)
    assert ev[2].modifiers == ("break", "ex")
    assert ev[3].modifiers == ("mine",)
    assert ev[4].role == "maimai:star" and ev[4].native == {"appearance": "$"}
    assert ev[5].role == "maimai:star" and ev[5].native == {"appearance": "$$"}
    assert ev[6].role == "maimai:tap" and ev[6].native == {"appearance": "@"}
    assert [e.target for e in ev[7:]] == [RingButton(4), RingButton(5)]
    assert ev[7].time == ev[8].time


def test_hold_durations_are_exact():
    _, chart, timing = chart_of(wrap("1h[4:1],2h[#1.5],3h[200#4:1],4h,", bpm=150))
    lengths = [timing.us(e.end) - timing.us(e.time) for e in chart.events]
    assert lengths == [400_000, 1_500_000, 300_000, 0]
    assert all(e.primitive is Primitive.HOLD for e in chart.events)
    assert chart.events[0].native == {"dur": "4:1"}


def test_hold_length_is_in_seconds_across_tempo_change():
    # 4:2 at 120 BPM lasts 1 s; the tempo doubles at beat 1 (0.5 s), so the
    # remaining 0.5 s spans two beats of 240 BPM and the hold ends on beat 3.
    _, chart, timing = chart_of(wrap("1h[4:2],(240)2,"))
    hold = chart.events[0]
    assert timing.us(hold.end) - timing.us(hold.time) == 1_000_000
    assert hold.end == Time(F(3))


def test_touch_and_touch_hold():
    _, chart, timing = chart_of(wrap("A1,Cf,C1,E8bx,Ch[4:1],Cfh[2:1],"))
    roles = [e.role for e in chart.events]
    assert roles[:4] == ["maimai:touch"] * 4
    assert roles[4:] == ["maimai:touch_hold"] * 2
    assert [e.target for e in chart.events[:3]] == [
        TouchRegion("A1"),
        TouchRegion("C"),
        TouchRegion("C1"),
    ]
    assert chart.events[1].modifiers == ("firework",)
    assert chart.events[3].modifiers == ("break", "ex")
    assert chart.events[5].modifiers == ("firework",)
    assert timing.us(chart.events[5].end) - timing.us(chart.events[5].time) == 1_000_000


def test_slide_star_head_and_group():
    _, chart, timing = chart_of(wrap("1b-5[8:1]*>7[4:1]b,"))
    star, first, second = chart.events
    assert star.role == "maimai:star" and star.modifiers == ("break",)
    assert first.role == second.role == "maimai:slide"
    assert star.group == first.group == second.group == 0
    assert first.modifiers == () and second.modifiers == ("break",)
    seg = first.path[0]
    assert seg.shape == "maimai:-" and seg.points == (RingButton(0), RingButton(4))
    # default wait = one beat; 8:1 = half a beat at 120 BPM.
    assert timing.us(seg.start) - timing.us(star.time) == 500_000
    assert timing.us(first.end) - timing.us(seg.start) == 250_000
    assert second.path[0].shape == "maimai:>"
    assert second.path[0].points == (RingButton(0), RingButton(6))


def test_headless_slides():
    _, chart, _ = chart_of(wrap("3?-7[4:1],4!-8[4:1]*-1[4:1],"))
    assert [e.role for e in chart.events] == ["maimai:slide"] * 3
    assert chart.events[0].native == {"head": "?"}
    assert chart.events[1].native == chart.events[2].native == {"head": "!"}
    assert chart.events[1].group == chart.events[2].group != chart.events[0].group


def test_all_slide_shapes():
    _, chart, _ = chart_of(
        wrap(
            "1-5[4:1],1^4[4:1],1<6[4:1],1>4[4:1],1v5[4:1],1V35[4:1],"
            "1p4[4:1],1q6[4:1],1pp5[4:1],1qq5[4:1],1s5[4:1],1z5[4:1],1w5[4:1],"
        )
    )
    shapes = [e.path[0].shape.removeprefix("maimai:") for e in chart.events if e.path]
    assert shapes == ["-", "^", "<", ">", "v", "V", "p", "q", "pp", "qq", "s", "z", "w"]
    v = next(e for e in chart.events if e.path and e.path[0].shape == "maimai:V")
    assert v.path[0].points == (RingButton(0), RingButton(2), RingButton(4))


def test_chained_slide_segment_times():
    _, chart, timing = chart_of(wrap("1-4q7-2[2:1],1-4[8:1]q7[8:1]-2[8:1],", bpm=120))
    shared, split = (e for e in chart.events if e.role == "maimai:slide")
    assert [s.end for s in shared.path[:2]] == [None, None]
    assert timing.us(shared.end) - timing.us(shared.path[0].start) == 1_000_000
    assert [p.points[-1] for p in shared.path] == [RingButton(3), RingButton(6), RingButton(1)]
    ends = [timing.us(s.end) - timing.us(split.path[0].start) for s in split.path]
    assert ends == [250_000, 500_000, 750_000]


def test_slide_duration_notations():
    _, chart, timing = chart_of(
        wrap("1-5[160#4:1],2-6[160#2.0],3-7[#1.25],4-8[0.75##1.5],5-1[0.5##8:3],")
    )
    slides = [e for e in chart.events if e.role == "maimai:slide"]

    def wait_len(e):
        start = timing.us(e.path[0].start)
        return start - timing.us(e.time), timing.us(e.end) - start

    assert [wait_len(e) for e in slides] == [
        (375_000, 375_000),
        (375_000, 2_000_000),
        (500_000, 1_250_000),
        (750_000, 1_500_000),
        (500_000, 750_000),
    ]


def test_bpm_changes_and_seconds_division():
    _, chart, timing = chart_of(wrap("1,(240)2,{#0.25}3,4,(60){4}5,"))
    assert [t.bpm for t in timing.tempos] == [120, 240, 60]
    # {#0.25} at 240 BPM is one beat per comma.
    assert [e.time.beat for e in chart.events] == [0, 1, 2, 3, 4]
    assert timing.us(chart.events[-1].time) == 500_000 + 250_000 * 3


def test_pseudo_each_offsets():
    _, chart, _ = chart_of(wrap("1`2`3,5/6`7,"))
    beats = [(e.time.beat, e.native.get("pseudo", 0)) for e in chart.events]
    assert beats == [(0, 0), (F(1, 32), 1), (F(2, 32), 2), (1, 0), (1, 0), (1 + F(1, 32), 1)]


def test_hispeed_controls_and_comments():
    _, chart, _ = chart_of(wrap("|| hello\n<HS*1.5>1, || bye\n<HS*0.5>2,"))
    assert [(c.kind, c.value, c.time.beat) for c in chart.controls] == [
        ("scroll", 1.5, 0),
        ("scroll", 0.5, 1),
    ]
    assert len(chart.events) == 2


def test_writer_derives_durations_without_native_notation():
    import msgspec

    cs, _, _ = chart_of(wrap("1h[4:1],2-6[0.75##8:3],3-4[8:1]q7[8:1],(240)4h[4:3],"))
    stripped = msgspec.structs.replace(
        cs,
        charts=tuple(
            msgspec.structs.replace(
                c,
                events=tuple(
                    msgspec.structs.replace(
                        e,
                        native={k: v for k, v in e.native.items() if k != "dur"},
                        path=tuple(msgspec.structs.replace(s, native={}) for s in e.path),
                    )
                    for e in c.events
                ),
            )
            for c in cs.charts
        ),
    )
    back = codec.read(codec.write(stripped))
    assert back.charts[0].events == tuple(
        msgspec.structs.replace(
            e,
            native=dict(e.native, **({"dur": d.native["dur"]} if "dur" in d.native else {})),
            path=tuple(
                msgspec.structs.replace(s, native=t.native)
                for s, t in zip(e.path, d.path, strict=True)
            ),
        )
        for e, d in zip(stripped.charts[0].events, back.charts[0].events, strict=True)
    )


@pytest.mark.parametrize(
    "fumen",
    ["1-5,", "C-5[4:1],", "1h[4:1]-5[4:1],", "9,", "1-[4:1],", "1[4:1"],
)
def test_malformed_input_raises(fumen: str):
    with pytest.raises(CodecError):
        codec.read(wrap(fumen).encode())


def test_missing_bpm_raises_unless_wholebpm():
    with pytest.raises(CodecError):
        codec.read(b"&inote_5=\n1,2,\nE\n")
    cs = codec.read(b"&wholebpm=150\n&inote_5=\n1,2,\nE\n")
    assert cs.timing.tempos[0].bpm == 150


def test_writer_rejects_other_mechanics():
    from urfm.ir import Chart, ChartSet, TimingMap

    with pytest.raises(CodecError):
        codec.write(ChartSet(timing=TimingMap.constant(120), charts=(Chart(mechanics="taiko"),)))


@pytest.mark.skipif("URFM_CORPUS_SIMAI" not in __import__("os").environ, reason="no simai corpus")
def test_corpus_roundtrip():
    import os

    files = sorted(Path(os.environ["URFM_CORPUS_SIMAI"]).rglob("maidata.txt"))
    assert files
    failures = [
        str(f) for f in files if not roundtrip_bytes(codec, f.read_bytes(), str(f)).semantic_exact
    ]
    assert not failures, failures[:10]
