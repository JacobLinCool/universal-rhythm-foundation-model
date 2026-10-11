"""Codecs must write charts that were never read from their format (generated or converted).

Round-trip tests only cover ``read -> write -> read``; generated charts carry no native
fields, which once made the TJA writer emit zero measures. Every writer is checked here
with plain IR charts.
"""

from fractions import Fraction as F

import pytest

from urfm.codecs import get
from urfm.ir import (
    Chart,
    ChartSet,
    Difficulty,
    Event,
    Lane,
    Point2D,
    Primitive,
    RingButton,
    SongMeta,
    Time,
    TimingMap,
)

TIMING = TimingMap(offset_us=F(1_234_000), tempos=TimingMap.constant(F("146")).tempos)


def hits(role: str, target=None, n: int = 24) -> tuple[Event, ...]:
    return tuple(
        Event(
            time=Time(F(i, 2)),
            primitive=Primitive.HIT,
            role=role,
            target=target(i) if target else None,
        )
        for i in range(n)
    )


CASES = {
    "taiko": ("tja", hits("taiko:don"), {}, 0),
    "osu_taiko": ("osu", hits("taiko:ka"), {}, 1_000),
    "osu_mania": ("osu", hits("mania:note", lambda i: Lane(i % 4)), {"lanes": "4"}, 1_000),
    "osu_std": ("osu", hits("osu:circle", lambda i: Point2D(64 + 16 * i, 192)), {}, 1_000),
    "osu_catch": ("osu", hits("catch:fruit", lambda i: Point2D(64 + 16 * i, 192)), {}, 1_000),
    "sm_dance_single": ("stepmania", hits("sm:tap", lambda i: Lane(i % 4)), {}, 0),
    "maimai_dx": ("simai", hits("maimai:tap", lambda i: RingButton(i % 8)), {}, 0),
}


@pytest.mark.parametrize("mechanics", sorted(CASES))
def test_generated_chart_survives_write_and_read(mechanics):
    codec_id, events, meta, tolerance_us = CASES[mechanics]
    chart = Chart(
        mechanics=mechanics,
        difficulty=Difficulty(label="Hard", level=5),
        events=events,
        meta=meta,
    )
    cs = ChartSet(timing=TIMING, charts=(chart,), song=SongMeta(title="generated"))
    codec = get(codec_id)
    back = codec.read(codec.write(cs), source=f"generated{codec.extensions[0]}")
    assert len(back.charts) == 1
    got = back.charts[0]
    assert len(got.events) == len(events), "events lost on write"
    timing = back.timing_of(got)
    for a, b in zip(events, sorted(got.events, key=lambda e: e.time), strict=True):
        assert abs(TIMING.us(a.time) - timing.us(b.time)) <= tolerance_us
        assert (a.role, a.target) == (b.role, b.target) or a.target is None
