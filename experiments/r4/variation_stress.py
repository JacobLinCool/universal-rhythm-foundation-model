"""Stress tests for the variation penalties (urfm.eval.variation): synthetic counterfactual
charts whose intended ranking is known, then real charts (human vs model versions).

    uv run python experiments/r4/variation_stress.py
"""

from __future__ import annotations

import random
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from urfm.eval.monotony import default_references, excess_monotone_time
from urfm.eval.variation import (
    calibrated_variation,
    default_calibration,
    default_stagnation_limits,
    stagnation_penalty,
    variation,
)
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event, Primitive
from urfm.ir.time import Time
from urfm.ir.timing import TimingMap

sys.path.insert(0, str(Path(__file__).parent))
from monotony_eval import SET_A, SET_B, charts

CALIBRATED = "--calibrated" in sys.argv
STAGNATION = "--stagnation" in sys.argv

BEATS = 300  # 120 s at 150 BPM
D, K = "taiko:don", "taiko:ka"


def make(notes: list[tuple[F, str]]) -> ChartSet:
    events = tuple(Event(time=Time(b), primitive=Primitive.HIT, role=r) for b, r in sorted(notes))
    return ChartSet(
        timing=TimingMap.constant(150), charts=(Chart(mechanics="taiko", events=events),)
    )


def sixteenths():
    return [F(k, 4) for k in range(BEATS * 4)]


RHYTHMS = [  # one-beat cells (offsets in beats)
    [F(0)],
    [F(0), F(1, 2)],
    [F(0), F(1, 4), F(1, 2)],
    [F(0), F(1, 2), F(3, 4)],
    [F(0), F(1, 4), F(1, 2), F(3, 4)],
    [F(0), F(1, 3), F(2, 3)],
    [],
    [F(0), F(3, 4)],
]


def synthetic(rng: random.Random) -> dict[str, ChartSet]:
    out = {}
    out["1 all don, 16ths"] = make([(b, D) for b in sixteenths()])
    out["2 don-ka alternating, 16ths"] = make(
        [(b, D if i % 2 == 0 else K) for i, b in enumerate(sixteenths())]
    )
    motifs = ["ddkd", "dkkd", "ddkk", "kddk", "dkdd", "kdkd", "ddkd", "kkdd", "dkdk", "kddd"]
    notes = []
    for unit in range(BEATS // 4):  # each unit: 4 one-beat colour motifs, slowly shifting
        for beat in range(4):
            m = motifs[(unit + beat * 3) % len(motifs)]
            for j, c in enumerate(m):
                notes.append((F(unit * 4 + beat) + F(j, 4), D if c == "d" else K))
    out["3 evolving colours, 16ths"] = make(notes)
    notes, colour = [], 0
    for beat in range(BEATS):  # rhythm phrases change every two beats, colours alternate
        cell = RHYTHMS[(beat // 2 * 5 + beat % 2) % len(RHYTHMS)]
        for off in cell:
            notes.append((F(beat) + off, D if colour % 2 == 0 else K))
            colour += 1
    out["4 alternating colours, rhythm phrases vary"] = make(notes)
    notes = []
    for beat in range(BEATS):
        for off in rng.choice(RHYTHMS):
            notes.append((F(beat) + off, rng.choice([D, K])))
    out["5 random colours and rhythm"] = make(notes)
    base = [
        (F(0), D),
        (F(1, 2), K),
        (F(3, 4), D),
        (F(1), D),
        (F(3, 2), K),
        (F(2), D),
        (F(9, 4), D),
        (F(5, 2), K),
        (F(3), D),
        (F(13, 4), K),
        (F(7, 2), D),
    ]
    notes = []
    for unit in range(BEATS // 4):
        for k, (off, r) in enumerate(base):
            if unit % 3 == 2 and k == unit % len(base):
                r = K if r == D else D  # one note changed every third unit
            notes.append((F(unit * 4) + off, r))
    out["6 repeated unit, one note changed every 3 units"] = make(notes)
    a = [(F(0), D), (F(1, 2), D), (F(1), K), (F(2), D), (F(5, 2), K), (F(3), K)]
    b = [(F(0), K), (F(1, 4), K), (F(1, 2), D), (F(3, 2), D), (F(2), K), (F(3), D), (F(7, 2), D)]
    notes = [
        (F(unit * 4) + off, r)
        for unit in range(BEATS // 4)
        for off, r in (a if unit % 2 == 0 else b)
    ]
    out["7 ABAB units"] = make(notes)
    notes = []  # user's example: 2 s of fast don, 2 s of fast ka, inside varied music
    for beat in range(BEATS):
        phrase = (beat // 10) % 3
        if phrase == 2:  # 10 beats (4 s) of 16ths: 5 beats don, 5 beats ka
            for j in range(4):
                notes.append((F(beat) + F(j, 4), D if beat % 10 < 5 else K))
        else:
            for off in RHYTHMS[(beat * 7 + beat // 3) % 6]:
                notes.append((F(beat) + off, rng.choice([D, K])))
    out["9 varied music with 2 s don + 2 s ka bursts"] = make(notes)
    notes = [(F(k, 4), D if (k // 20) % 2 == 0 else K) for k in range(BEATS * 4)]
    out["10 2 s don / 2 s ka bursts, 16ths, for 120 s"] = make(notes)
    out["8 case 1 with 40% of notes deleted"] = make(
        [(b, D) for b in sixteenths() if rng.random() > 0.4]
    )
    return out


def real() -> dict[str, list[ChartSet]]:
    sources = {
        "human": [],
        "R3 best (criticised)": [],
        "R4 final, audio": [],
        "R5 RL v2, audio": [],
        "R5 RL v2, audio, doubled grid (1 song)": [],
    }
    files = {
        "human": ("human.tja",),
        "R3 best (criticised)": ("urfm-best.tja",),
        "R4 final, audio": ("urfm-r4-final-audio.tja",),
        "R5 RL v2, audio": ("urfm-r5-rl-audio.tja",),
        "R5 RL v2, audio, doubled grid (1 song)": ("urfm-r5-rl-audio-x2grid.tja",),
    }
    for root in (SET_A, SET_B):
        for song in sorted(root.iterdir()):
            for name, fs in files.items():
                for f in fs:
                    if (song / f).exists():
                        for cs, c, label in charts(song / f):
                            if label in ("hard", "oni", "edit"):
                                sources[name].append(ChartSet(timing=cs.timing_of(c), charts=(c,)))
    return sources


def row(cs: ChartSet) -> dict:
    if STAGNATION:
        v = stagnation_penalty(cs, cs.charts[0], default_stagnation_limits())
        v["E_time"] = excess_monotone_time(cs, cs.charts[0], default_references()[1])[
            "excess_s_per_min"
        ]
        return v
    v = (
        calibrated_variation(cs, cs.charts[0], default_calibration())
        if CALIBRATED
        else variation(cs, cs.charts[0])
    )
    v["E_time"] = excess_monotone_time(cs, cs.charts[0], default_references()[1])[
        "excess_s_per_min"
    ]
    return v


def main() -> int:
    rng = random.Random(0)
    if STAGNATION:
        cols_s = (
            "score",
            "p_time",
            "p_space",
            "p_any",
            "peak_time_s",
            "peak_any_s",
            "first_penalty_s",
            "E_time",
        )
        cols = ("score", "p_time", "p_space", "p_any", "peak_time_s", "E_time")
    else:
        cols_s = ("score", "p_short", "p_long_time", "p_long_space", "first_penalty_s", "E_time")
        cols = ("score", "p_short", "p_long_time", "p_long_space", "E_time")
    print("| synthetic, 120 s | " + " | ".join(cols_s) + " |")
    print("|---" * (len(cols_s) + 1) + "|")
    for name, cs in synthetic(rng).items():
        r = row(cs)
        cells = ["never" if r[c] is None else f"{r[c]:.2f}" for c in cols_s]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    print("| real (hard/oni/edit) | charts | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 2) + "|")
    for name, items in real().items():
        rs = [row(cs) for cs in items]
        rs = [r for r in rs if r["score"] is not None]
        cells = [f"{np.mean([r[c] for r in rs]):.2f}" for c in cols]
        print(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
