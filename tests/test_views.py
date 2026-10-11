from fractions import Fraction as F

from test_ir import sample

from urfm.interop.rhythm_rs import to_canonical_chart
from urfm.ir import Chart, ChartSet, Event, Lane, Primitive, Time, TimingMap
from urfm.ir.views import active_events, chart_stats, interaction


def test_branch_route_selection():
    cs = sample()
    taiko = cs.charts[0]
    assert len(active_events(taiko)) == 3  # default route "normal" has no event at beat 4
    assert len(active_events(taiko, route="master")) == 4


def test_grouped_events_count_as_one_action():
    cs = sample()
    star_and_slide = interaction(cs, cs.charts[2])[0]
    assert star_and_slide.simultaneity == 1
    assert star_and_slide.primitive is Primitive.PATH


def test_chords_movement_and_ioi_bins():
    t = Time.at
    events = (
        Event(time=t(0), primitive=Primitive.HIT, role="mania:note", target=Lane(0)),
        Event(time=t(0), primitive=Primitive.HIT, role="mania:note", target=Lane(3)),
        Event(time=t(F(1, 2)), primitive=Primitive.HIT, role="mania:note", target=Lane(3)),
        Event(time=t(1), primitive=Primitive.HIT, role="mania:note", target=Lane(3)),
    )
    chart = Chart(mechanics="osu_mania", meta={"lanes": "4"}, events=events)
    cs = ChartSet(timing=TimingMap.constant(120), charts=(chart,))
    seq = interaction(cs, chart)
    assert [x.simultaneity for x in seq] == [2, 1, 1]
    assert [x.moved for x in seq] == [None, True, False]
    stats = chart_stats(cs, chart)
    assert stats.chord_ratio == 1 / 3
    assert stats.ioi_beat_hist == {"1/2": 1.0}
    assert stats.ioi_median_ms == 250


def test_rhythm_rs_export_taiko():
    cs = sample()
    out = to_canonical_chart(cs, cs.charts[0])
    kinds = [(o["kind"], o["lane_or_region"], o["flags"]) for o in out["objects"]]
    assert kinds[:3] == [
        ("Tap", {"Lane": 0}, 0),
        ("Tap", {"Lane": 1}, 1),
        ("Roll", {"Lane": 2}, 2),
    ]
    assert [o["id"] for o in out["objects"]] == list(range(len(out["objects"])))
    assert out["objects"][2]["required_hits"] == 5
    assert out["objects"][3]["branch_segment_id"] == 0
    assert out["objects"][3]["branch_route_id"] == 1
    assert out["objects"][0]["scroll_scaled"] == 1_500_000
    assert out["tempo_map"] == [{"tick": 0, "micros_per_quarter": 398671}]


def test_rhythm_rs_export_maimai_slide():
    cs = sample()
    out = to_canonical_chart(cs, cs.charts[2])
    slide = next(o for o in out["objects"] if o["kind"] == "Slide")
    assert slide["slide_to"] == 4
    assert any(o["kind"] == "Touch" for o in out["objects"])
