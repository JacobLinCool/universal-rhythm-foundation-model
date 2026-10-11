from fractions import Fraction as F

from hypothesis import given
from hypothesis import strategies as st

from urfm.ir import Stop, TempoChange, Time, TimingMap, Warp


def complex_map() -> TimingMap:
    return TimingMap(
        offset_us=F(-1_000_000),
        tempos=(TempoChange(F(0), F(120)), TempoChange(F(4), F(240))),
        stops=(Stop(F(2), F(500_000)), Stop(F(3), F(100_000), before=True)),
        warps=(Warp(F(6), F(1)),),
    )


def test_constant_tempo():
    t = TimingMap.constant(150, offset_us=2_000)
    assert t.beat_to_us_exact(F(0)) == 2_000
    assert t.beat_to_us_exact(F(1)) == 2_000 + 400_000
    assert t.beat_to_us_exact(F(-1)) == 2_000 - 400_000


def test_stop_delay_warp_semantics():
    t = complex_map()
    assert t.beat_to_us_exact(F(2)) == 0  # stop: object on the beat precedes the pause
    assert t.beat_to_us_exact(F(5, 2)) == 750_000
    assert t.beat_to_us_exact(F(3)) == 1_100_000  # delay: object on the beat follows it
    assert t.beat_to_us_exact(F(5)) == 1_850_000
    assert t.beat_to_us_exact(F(13, 2)) == t.beat_to_us_exact(F(7)) == 2_100_000


def test_inverse_outside_stops():
    t = complex_map()
    for b in [F(-3), F(0), F(1, 3), F(5, 2), F(4), F(9, 2), F(8)]:
        assert t.us_to_beat(t.beat_to_us_exact(b)) == b


@given(st.integers(min_value=-2_000_000, max_value=20_000_000))
def test_time_at_us_is_exact(us: int):
    t = complex_map()
    time = t.time_at_us(us)
    assert t.us(time) == us


def test_measure_starts():
    from urfm.ir import MeterChange

    t = TimingMap(
        tempos=(TempoChange(F(0), F(120)),),
        meters=(MeterChange(F(0), 4, 4), MeterChange(F(8), 3, 4), MeterChange(F(14), 7, 8)),
    )
    assert t.measure_starts(F(18)) == [F(0), F(4), F(8), F(11), F(14), F(35, 2)]


def test_time_ordering():
    assert Time.at(1) < Time.at(F(3, 2)) < Time.at(2, -5) < Time.at(2)


def test_us_array_matches_exact():
    t = complex_map()
    beats = [F(-3), F(0), F(1, 3), F(2), F(5, 2), F(3), F(4), F(9, 2), F(13, 2), F(7), F(8)]
    approx = t.us_array(beats, [5] * len(beats))
    for b, a in zip(beats, approx, strict=True):
        assert abs(a - (t.us(Time(b, 5)))) < 1.0
