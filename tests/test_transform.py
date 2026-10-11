from fractions import Fraction as F
from pathlib import Path

import numpy as np

from urfm.codecs import get
from urfm.data.canonical import CanonicalStats, canonicalize
from urfm.ir import Stop, TempoChange, TimingMap
from urfm.ir.timing import MeterChange
from urfm.ir.transform import rescale_beats, stretch_timing, warp_timing

FIXTURES = Path(__file__).parent / "fixtures" / "tja"


def timing() -> TimingMap:
    return TimingMap(
        offset_us=F(-300_000),
        tempos=(TempoChange(F(0), F(150)), TempoChange(F(10), F(180))),
        stops=(Stop(F(5), F(200_000)), Stop(F(13), F(50_000), before=True)),
        meters=(MeterChange(F(0), 4, 4), MeterChange(F(12), 3, 4)),
    )


def test_stretch_maps_every_beat_through_the_audio_warp():
    old = timing()
    new, warp = stretch_timing(old, F(5, 4))
    beats = [F(k, 4) for k in range(-8, 80)]
    want = warp.forward([float(old.beat_to_us_exact(b)) for b in beats])
    got = [float(new.beat_to_us_exact(b)) for b in beats]
    assert np.allclose(got, want, atol=1e-6)
    assert new.bpm_at(F(0)) == 150 * F(5, 4)


def test_local_warp_is_consistent_with_audio_map():
    old = timing()
    knots, speeds = [F(2), F(5, 2), F(7), F(8)], [F(1), F(6, 5), F(1), F(4, 5), F(1)]
    new, warp = warp_timing(old, knots, speeds)
    beats = [F(k, 8) for k in range(-16, 160)]
    want = warp.forward([float(old.beat_to_us_exact(b)) for b in beats])
    got = [float(new.beat_to_us_exact(b)) for b in beats]
    assert np.allclose(got, want, atol=1e-6)
    t = np.linspace(-1e6, 9e6, 50)
    assert np.allclose(warp.inverse(warp.forward(t)), t)


def test_rescale_keeps_absolute_times_and_measures():
    cs = get("tja").read_path(FIXTURES / "scroll_ramp_measure_trick.tja")
    for factor in (F(2), F(1, 2)):
        new = rescale_beats(cs, factor)
        for a, b in zip(cs.charts, new.charts, strict=True):
            ta, tb = cs.timing_of(a), new.timing_of(b)
            for x, y in zip(a.events, b.events, strict=True):
                assert ta.us(x.time) == tb.us(y.time)
            ma = [ta.beat_to_us_exact(m) for m in ta.measure_starts(F(60))]
            mb = [tb.beat_to_us_exact(m) for m in tb.measure_starts(F(60) * factor)]
            assert ma == mb


def test_canonicalize_folds_visual_tempo_into_scroll():
    cs = get("tja").read_path(FIXTURES / "visual_bpm.tja")
    chart = cs.charts[0]
    stats = CanonicalStats()
    _, new = canonicalize(cs, chart, stats)
    old_t, new_t = cs.timing_of(chart), new.timing
    assert new_t is not None
    for a, b in zip(chart.events, new.events, strict=True):
        assert old_t.beat_to_us_exact(a.time.beat) == new_t.beat_to_us_exact(b.time.beat)
    # the BPM 32 crawl folds into 128; the 256 bar is simpler as written (octave notation),
    # and 150 / 180 are real tempo changes
    assert [t.bpm for t in new_t.tempos] == [128, 256, 128, 150, 180]
    assert stats.reasons["simpler_grid"] == 1 and stats.kept == 3

    # every note keeps its on-screen speed (BPM x scroll)
    def speed(cs_t, ch, beat):
        scrolls = [c for c in ch.controls if c.kind == "scroll" and c.time.beat <= beat]
        value = scrolls[-1].value if scrolls else 1.0
        return float(cs_t.bpm_at(beat)) * float(value)

    for a, b in zip(chart.events, new.events, strict=True):
        assert abs(speed(old_t, chart, a.time.beat) - speed(new_t, new, b.time.beat)) < 1e-6
    # measures start at the same times
    ma = [old_t.beat_to_us_exact(m) for m in old_t.measure_starts(F(40))]
    mb = [new_t.beat_to_us_exact(m) for m in new_t.measure_starts(F(60))]
    assert ma == mb[: len(ma)]


def test_canonicalize_is_time_exact_on_all_fixtures():
    tja = get("tja")
    for path in sorted(FIXTURES.glob("*.tja")):
        cs = tja.read_path(path)
        for chart in cs.charts:
            _, new = canonicalize(cs, chart)
            old_t, new_t = cs.timing_of(chart), cs.timing_of(new)
            for a, b in zip(chart.events, new.events, strict=True):
                assert old_t.us(a.time) == new_t.us(b.time), path.name


def test_retime_keeps_absolute_times():
    from urfm.ir.transform import retime_chart

    cs = get("tja").read_path(FIXTURES / "visual_bpm.tja")
    chart = cs.charts[0]
    old = cs.timing_of(chart)
    new = TimingMap(offset_us=F(-480_000), tempos=(TempoChange(F(0), F(256)),))
    out = retime_chart(chart, old, new)
    for a, b in zip(chart.events, out.events, strict=True):
        assert old.us(a.time) == new.us(b.time)
