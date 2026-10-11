from fractions import Fraction as F

from urfm.ir import (
    BranchRef,
    BranchSegment,
    Chart,
    ChartSet,
    Control,
    Difficulty,
    Event,
    Lane,
    PathSegment,
    Point2D,
    Primitive,
    RingButton,
    Time,
    TimingMap,
    TouchRegion,
)
from urfm.ir.serialize import from_json, from_msgpack, schema_digest, to_json, to_msgpack
from urfm.ir.validate import validate


def sample() -> ChartSet:
    t = Time.at
    taiko = Chart(
        mechanics="taiko",
        difficulty=Difficulty(label="oni", level=8),
        events=(
            Event(time=t(0), primitive=Primitive.HIT, role="taiko:don"),
            Event(time=t(F(1, 2)), primitive=Primitive.HIT, role="taiko:ka", modifiers=("big",)),
            Event(time=t(1), end=t(2), primitive=Primitive.REPEAT, role="taiko:balloon", count=5),
            Event(
                time=t(4),
                primitive=Primitive.HIT,
                role="taiko:don",
                branch=BranchRef(0, "master"),
            ),
        ),
        controls=(Control(time=t(0), kind="scroll", value=1.5),),
        branches=(BranchSegment(id=0, start=t(4), routes=("normal", "master"), default="normal"),),
    )
    mania = Chart(
        mechanics="osu_mania",
        meta={"lanes": "4"},
        events=(
            Event(time=t(0, 3), primitive=Primitive.HIT, role="mania:note", target=Lane(3)),
            Event(time=t(1), end=t(2), primitive=Primitive.HOLD, role="mania:hold", target=Lane(0)),
        ),
    )
    mai = Chart(
        mechanics="maimai_dx",
        events=(
            Event(
                time=t(0),
                primitive=Primitive.HIT,
                role="maimai:star",
                target=RingButton(0),
                group=1,
            ),
            Event(
                time=t(0),
                end=t(2),
                primitive=Primitive.PATH,
                role="maimai:slide",
                target=RingButton(0),
                group=1,
                path=(
                    PathSegment(
                        shape="maimai:-", points=(RingButton(0), RingButton(4)), start=t(1)
                    ),
                ),
            ),
            Event(time=t(1), primitive=Primitive.HIT, role="maimai:touch", target=TouchRegion("C")),
        ),
    )
    std = Chart(
        mechanics="osu_std",
        events=(
            Event(
                time=t(0),
                end=t(1),
                primitive=Primitive.PATH,
                role="osu:slider",
                target=Point2D(10, 20),
                count=2,
                path=(PathSegment(shape="osu:bezier", points=(Point2D(10, 20), Point2D(200, 20))),),
            ),
        ),
    )
    return ChartSet(timing=TimingMap.constant(F("150.5")), charts=(taiko, mania, mai, std))


def test_json_and_msgpack_roundtrip():
    cs = sample()
    assert from_json(to_json(cs)) == cs
    assert from_msgpack(to_msgpack(cs)) == cs
    assert b'"150.5"' not in to_json(cs) and b'"301/2"' in to_json(cs)


def test_schema_digest_is_stable():
    assert schema_digest() == schema_digest()
    assert len(schema_digest()) == 64


def test_sample_is_valid():
    assert validate(sample()) == []


def test_validator_catches_mechanics_errors():
    bad = Chart(
        mechanics="sm_dance_single",
        events=(
            Event(time=Time.at(1), primitive=Primitive.HIT, role="sm:tap", target=Lane(4)),
            Event(time=Time.at(0), primitive=Primitive.HOLD, role="sm:tap", target=Lane(0)),
            Event(time=Time.at(2), primitive=Primitive.HIT, role="taiko:don"),
        ),
    )
    codes = {i.code for i in validate(ChartSet(timing=TimingMap.constant(120), charts=(bad,)))}
    assert {"target.range", "order", "role.primitive", "end.missing", "role.unknown"} <= codes


def test_enforce_single_action_drops_overlaps():
    from urfm.ir.validate import enforce_single_action

    t = Time.at
    events = (
        Event(time=t(0), end=t(2), primitive=Primitive.REPEAT, role="taiko:drumroll"),
        Event(time=t(1), primitive=Primitive.HIT, role="taiko:don"),
        Event(time=t(2), primitive=Primitive.HIT, role="taiko:don"),
        Event(time=t(3), primitive=Primitive.HIT, role="taiko:ka"),
        Event(time=t(3), primitive=Primitive.HIT, role="taiko:don"),
    )
    timing = TimingMap.constant(120)
    chart, dropped = enforce_single_action(Chart(mechanics="taiko", events=events), timing.us)
    assert dropped == 3
    assert [e.role for e in chart.events] == ["taiko:drumroll", "taiko:don"]


def test_fraction_decoding_is_exact_and_normalized():
    from fractions import Fraction as Fr

    from urfm.ir.serialize import _dec_hook

    for text in ("3/4", "-7/12", "5", "0", "-2"):
        value = _dec_hook(Fr, text)
        assert value == Fr(text) and isinstance(value, Fr)
        assert (value.numerator, value.denominator) == (Fr(text).numerator, Fr(text).denominator)


def test_lazy_chartset_views_are_exact():
    from urfm.ir.serialize import LazyChartSet

    cs = sample()
    lazy = LazyChartSet(to_msgpack(cs))
    assert len(lazy) == len(cs.charts)
    for i, chart in enumerate(cs.charts):
        view = lazy.view(i)
        assert view.charts == (chart,) and view.timing == cs.timing and view.song == cs.song
