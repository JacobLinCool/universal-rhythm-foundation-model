from fractions import Fraction
from pathlib import Path

import msgspec
import numpy as np
import pytest

from urfm.codecs import registry
from urfm.eval import aggregate, evaluate_pair, match_onsets, write_report
from urfm.ir import ChartSet

FIXTURES = Path(__file__).parent / "fixtures"


def load(codec: str, name: str) -> ChartSet:
    return registry.get(codec).read_path(FIXTURES / codec / name)


def shifted(cs: ChartSet, us: int) -> ChartSet:
    timing = msgspec.structs.replace(cs.timing, offset_us=cs.timing.offset_us + Fraction(us))
    charts = tuple(
        msgspec.structs.replace(
            c,
            timing=None
            if c.timing is None
            else msgspec.structs.replace(c.timing, offset_us=c.timing.offset_us + Fraction(us)),
        )
        for c in cs.charts
    )
    return msgspec.structs.replace(cs, timing=timing, charts=charts)


def test_match_onsets_one_to_one():
    ref = np.array([0, 100_000, 200_000])
    gen = np.array([5_000, 10_000, 195_000, 400_000])
    assert match_onsets(ref, gen, 25_000) == [(0, 0), (2, 2)]


@pytest.mark.parametrize(
    ("codec", "name"),
    [
        ("tja", "basic.tja"),
        ("osu", "std_basic.osu"),
        ("osu", "mania_7k.osu"),
        ("stepmania", "basic.sm"),
        ("simai", "04_slide_shapes.txt"),
    ],
)
def test_identity(codec, name):
    cs = load(codec, name)
    chart = cs.charts[0]
    row = evaluate_pair(cs, chart, cs, chart)
    assert row["f1@25"] == 1.0 and row["f1@50"] == 1.0
    assert row["issues"] == 0
    assert row["writable"] is True
    assert row["density_abs_err"] == 0
    assert row["primitive_ratio_l1"] == 0 and row["ioi_hist_l1"] == 0


def test_shift_30ms():
    cs = load("tja", "basic.tja")
    moved = shifted(cs, 30_000)
    row = evaluate_pair(moved, moved.charts[0], cs, cs.charts[0])
    assert row["f1@25"] == 0.0
    assert row["f1@50"] == 1.0
    assert row["taiko_role_acc@25"] is None


def test_drop_half_recall():
    cs = load("osu", "mania_4k.osu")
    chart = cs.charts[0]
    # keep every other onset group
    onsets = sorted({cs.timing_of(chart).us(ev.time) for ev in chart.events})
    keep = set(onsets[::2])
    events = tuple(ev for ev in chart.events if cs.timing_of(chart).us(ev.time) in keep)
    half = msgspec.structs.replace(chart, events=events)
    row = evaluate_pair(cs, half, cs, chart)
    assert row["precision@25"] == 1.0
    assert row["recall@25"] == pytest.approx(len(keep) / len(onsets))
    assert "lane_entropy_gen" in row and "jack_ratio_ref" in row


def test_density_and_target():
    cs = load("tja", "basic.tja")
    chart = cs.charts[0]
    row = evaluate_pair(cs, chart, cs, chart, target_density=row_density(cs, chart) * 2)
    assert row["target_rel_err"] == pytest.approx(0.5)
    assert row["taiko_role_acc@25"] == 1.0


def row_density(cs, chart):
    return evaluate_pair(cs, chart, cs, chart)["density_ref"]


def test_aggregate_and_report(tmp_path):
    rows = []
    for codec, name in [
        ("tja", "basic.tja"),
        ("osu", "taiko_basic.osu"),
        ("simai", "02_holds.txt"),
    ]:
        cs = load(codec, name)
        rows.append(evaluate_pair(cs, cs.charts[0], cs, cs.charts[0]))
    table = aggregate(rows)
    assert set(table) == {"taiko", "osu_taiko", "maimai_dx"}
    assert table["taiko"]["f1@25"] == 1.0
    write_report(rows, tmp_path / "r.jsonl", tmp_path / "r.md")
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 3
    assert "| taiko |" in (tmp_path / "r.md").read_text()


def test_structure_metrics_identity_and_monotony():
    from fractions import Fraction

    from urfm.eval.metrics import evaluate_pair, structure
    from urfm.ir import Chart, ChartSet, Event, Primitive, Time, TimingMap

    t = TimingMap.constant(120)
    varied = tuple(
        Event(time=Time(Fraction(k, 4) + 4 * (k // 16)), primitive=Primitive.HIT, role="taiko:don")
        for k in range(64)
        if k % 3 != 1
    )
    flat = tuple(
        Event(time=Time(Fraction(k, 2)), primitive=Primitive.HIT, role="taiko:don")
        for k in range(64)
    )
    cs_v = ChartSet(timing=t, charts=(Chart(mechanics="taiko", events=varied),))
    cs_f = ChartSet(timing=t, charts=(Chart(mechanics="taiko", events=flat),))
    row = evaluate_pair(cs_v, cs_v.charts[0], cs_v, cs_v.charts[0])
    assert row["rhythm_change_rate_diff"] == 0 and row["note_type_l1"] == 0
    assert row["first_onset_delay_s"] == 0
    flat_s = structure(cs_f, cs_f.charts[0])
    assert flat_s["rhythm_change_rate"] == 0 and flat_s["ioi_classes"] == 1


def test_monotony_flags_constant_streams_only():
    from fractions import Fraction

    from urfm.eval.monotony import monotony
    from urfm.ir.chart import Chart, ChartSet
    from urfm.ir.events import Event, Primitive
    from urfm.ir.time import Time
    from urfm.ir.timing import TimingMap

    def chart(beats, roles):
        events = tuple(
            Event(time=Time(Fraction(b)), primitive=Primitive.HIT, role=r)
            for b, r in zip(beats, roles, strict=True)
        )
        return ChartSet(
            timing=TimingMap.constant(150), charts=(Chart(mechanics="taiko", events=events),)
        )

    stream = [Fraction(k, 4) for k in range(200)]  # 20 s of 16th notes
    cs = chart(stream, ["taiko:don" if k % 3 else "taiko:ka" for k in range(200)])
    assert monotony(cs, cs.charts[0])["monotone_share"] == 1.0  # don don ka, over and over
    varied = sorted(
        {Fraction(k, 4) for k in range(0, 200, 1) if (k * 7) % 5 != 0} | {Fraction(201, 4)}
    )
    roles = ["taiko:don" if (k * 13) % 7 < 4 else "taiko:ka" for k in range(len(varied))]
    cs2 = chart(varied, roles)
    assert monotony(cs2, cs2.charts[0])["monotone_share"] < 1.0
    # a constant stream with random colours is still rhythmically monotone
    import random

    rng = random.Random(0)
    cs3 = chart(stream, [rng.choice(["taiko:don", "taiko:ka"]) for _ in range(200)])
    assert monotony(cs3, cs3.charts[0], rhythm_only=True)["monotone_share"] == 1.0
    assert monotony(cs3, cs3.charts[0])["monotone_share"] < 0.5


def test_excess_monotone_time_is_density_calibrated():
    import random
    from fractions import Fraction

    from urfm.eval.monotony import default_references, excess_monotone_time
    from urfm.ir.chart import Chart, ChartSet
    from urfm.ir.events import Event, Primitive
    from urfm.ir.time import Time
    from urfm.ir.timing import TimingMap

    rng = random.Random(0)

    def chart(beats):
        events = tuple(
            Event(time=Time(b), primitive=Primitive.HIT, role=rng.choice(["taiko:don", "taiko:ka"]))
            for b in beats
        )
        return ChartSet(
            timing=TimingMap.constant(150), charts=(Chart(mechanics="taiko", events=events),)
        )

    _, ref = default_references()
    stream = chart([Fraction(k, 4) for k in range(600)])  # 60 s of 16ths, shuffled colours
    m = excess_monotone_time(stream, stream.charts[0], ref)
    assert m["excess_s_per_min"] > 20 and m["max_excess_s"] > 30
    cells = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4), Fraction(1), Fraction(1, 3)]
    beats, b = [], Fraction(0)
    while b < 150:
        beats.append(b)
        b += rng.choice(cells)
    varied = chart(beats)
    assert excess_monotone_time(varied, varied.charts[0], ref)["excess_s_per_min"] < 1


def test_variation_penalty_orders_counterfactual_charts():
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "r4"))
    from variation_stress import synthetic  # pyright: ignore[reportMissingImports]

    from urfm.eval.variation import variation

    s = {
        k.split(" ", 1)[0]: variation(cs, cs.charts[0])
        for k, cs in synthetic(random.Random(0)).items()
    }
    assert (
        s["1"]["score"] > s["2"]["score"] > s["4"]["score"]
    )  # all don > alternating > varied rhythm
    assert s["1"]["first_penalty_s"] < s["2"]["first_penalty_s"]  # colour variation delays it
    assert s["6"]["score"] > 0 and s["7"]["score"] > 0  # small changes / ABAB do not clear the debt
    assert s["5"]["score"] == 0  # random variation is not caught here (needs coherence terms)
