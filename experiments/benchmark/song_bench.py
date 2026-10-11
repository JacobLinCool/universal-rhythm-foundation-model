"""Single-song benchmark against a human chart (default: Crystal Hail, a held-out test song).

Measures what a player notices, not only onset agreement: how long the chart stays silent at the
start, the rhythmic vocabulary (inter-onset intervals), how much the per-measure rhythm changes,
note-type usage (big, drumroll, balloon), density, presentation controls, and F1 against the
human chart. Runs on CPU in about a minute per decoding variant.

    uv run python experiments/benchmark/song_bench.py \\
        --ckpt runs/checkpoints/ft-r3-song.pt \\
        --song runs/v1b-taiko/playable/taiko1000_test-00004-Crystal_Hail \\
        --variant t0.6 --variant t0.9 --variant t0.6:plan-mean --variant t0.6:plan-oracle
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from urfm.codecs import get
from urfm.data.features import MEL, mel_transform, quantize
from urfm.eval.metrics import match_onsets
from urfm.ir.chart import AudioRef, Chart, ChartSet, Difficulty, SongMeta
from urfm.ir.views import active_events
from urfm.tokenize import audio_windows, build_tokenizer
from urfm.tokenize.tokenizer import onset_rate
from urfm.train.generate import Job, generate
from urfm.train.tiers import chart_prefix, load_tiers
from urfm.train.train import load_checkpoint

IOI_BINS = [
    Fraction(1, 8),
    Fraction(1, 6),
    Fraction(1, 4),
    Fraction(1, 3),
    Fraction(1, 2),
    Fraction(2, 3),
    Fraction(1),
    Fraction(3, 2),
    Fraction(2),
]


TIER_TEMPERATURE = {"easy": 0.6, "normal": 0.65, "hard": 0.75, "oni": 0.85, "edit": 0.85}
"""Easy courses repeat a few patterns; hard ones vary constantly (human charts, Taiko-1000)."""


def song_mel(audio: Path) -> np.ndarray:
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio),
            "-ac",
            "1",
            "-ar",
            str(MEL.sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    wav = torch.from_numpy(np.frombuffer(pcm, np.float32).copy())
    with torch.no_grad():
        return quantize(mel_transform(device="cpu")(wav)).numpy()


def ioi_class(x: Fraction) -> str:
    best = min(IOI_BINS, key=lambda b: abs(b - x))
    return str(best) if abs(best - x) <= Fraction(1, 96) else "other"


def chart_metrics(cs: ChartSet, chart: Chart) -> dict:
    timing = cs.timing_of(chart)
    events = sorted(active_events(chart), key=lambda e: e.time)
    if not events:
        return {"events": 0}
    onsets = sorted({e.time.beat for e in events})
    iois = Counter(ioi_class(b - a) for a, b in itertools.pairwise(onsets))
    total = max(sum(iois.values()), 1)
    # per-measure rhythm signature: onset offsets inside the measure (1/48 beat grid)
    starts = timing.measure_starts(onsets[-1] + 4)
    sigs = []
    j = 0
    for k, m0 in enumerate(starts[:-1]):
        m1 = starts[k + 1]
        sig = []
        while j < len(onsets) and onsets[j] < m1:
            if onsets[j] >= m0:
                sig.append(round((onsets[j] - m0) * 48))
            j += 1
        if sig:
            sigs.append(tuple(sig))
    roles = Counter(e.role.split(":")[1] + ("+big" if "big" in e.modifiers else "") for e in events)
    return {
        "events": len(events),
        "first_note_s": round(timing.seconds(events[0].time), 2),
        "ioi": {k: round(v / total, 3) for k, v in iois.most_common()},
        "ioi_classes_2pct": sum(1 for v in iois.values() if v / total >= 0.02),
        "measures": len(sigs),
        "distinct_measure_rhythms": round(len(set(sigs)) / max(len(sigs), 1), 3),
        "rhythm_change_rate": round(
            sum(a != b for a, b in itertools.pairwise(sigs)) / max(len(sigs) - 1, 1), 3
        ),
        "roles": dict(roles.most_common()),
        "controls": dict(Counter(c.kind for c in chart.controls)),
        "meter_changes": len(timing.meters),
    }


def ioi_l1(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return round(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys), 3)


def window_rates(cs: ChartSet, chart: Chart, duration_us: int) -> list[float]:
    timing = cs.timing_of(chart)
    us = sorted({timing.us(e.time) for e in active_events(chart)})
    out = []
    for s, e in audio_windows(timing, duration_us):
        n = sum(1 for x in us if s <= x < e)
        out.append(n / max((e - s) / 1e6, 1e-3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--song", required=True, help="folder with human.tja and song.ogg")
    ap.add_argument(
        "--variant",
        action="append",
        default=[],
        help="t<temp>[:plan-mean|plan-oracle|plan-cal|plan-t<temp>]",
    )
    ap.add_argument("--courses", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    song = Path(args.song)
    out = Path(args.out or song / "bench")
    out.mkdir(parents=True, exist_ok=True)
    tja = get("tja")
    human = tja.read_path(song / "human.tja")
    mel = song_mel(next(song.glob("song.*")))
    duration_us = round(mel.shape[0] / MEL.fps * 1e6)
    model, meta = load_checkpoint(args.ckpt, "cpu")
    tiers = load_tiers(meta)
    tok = build_tokenizer()
    charts = [c for c in human.charts if c.events and c.mechanics == "taiko"]
    if args.courses:
        charts = [c for c in charts if (c.difficulty.label or "").lower() in args.courses]

    report = {"human": {c.difficulty.label: chart_metrics(human, c) for c in charts}}
    for spec in args.variant or ["t0.6"]:
        parts = spec.split(":")
        temperature = float(parts[0][1:])
        plan, plan_t, oracle, per_tier, top_p = "sample", temperature, False, False, 0.95
        for p in parts[1:]:
            if p == "plan-mean":
                plan = "mean"
            elif p == "plan-oracle":
                oracle = True
            elif p == "plan-cal":
                plan = "calibrated"
            elif p == "tier-t":
                per_tier = True
            elif p.startswith("topp"):
                top_p = float(p[4:])
            elif p.startswith("plan-t"):
                plan_t = float(p[6:])
        jobs = [
            Job(
                mel=mel,
                timing=human.timing_of(c),
                mechanics="taiko",
                prefix=chart_prefix(tok, tiers, human, c),
                difficulty=Difficulty(label=c.difficulty.label, level=c.difficulty.level),
                density_plan=window_rates(human, c, duration_us) if oracle else None,
                target_rate=onset_rate(human, c),
            )
            for c in charts
        ]
        if per_tier:  # one call per course: easier courses get a lower temperature
            generated = []
            for c, job in zip(charts, jobs, strict=True):
                t = TIER_TEMPERATURE.get((c.difficulty.label or "").lower(), temperature)
                generated += generate(
                    model,
                    tok,
                    [job],
                    temperature=t,
                    top_p=top_p,
                    device="cpu",
                    seed=args.seed,
                    plan=plan,
                    plan_temperature=plan_t,
                )
        else:
            generated = generate(
                model,
                tok,
                jobs,
                temperature=temperature,
                top_p=top_p,
                device="cpu",
                seed=args.seed,
                plan=plan,
                plan_temperature=plan_t,
            )
        gen_cs = ChartSet(
            timing=human.timing,
            charts=tuple(generated),
            song=SongMeta(title=f"{human.song.title} ({spec})"),
            audio=AudioRef(path=next(song.glob("song.*")).name),
        )
        (out / f"{spec.replace(':', '_')}.tja").write_bytes(tja.write(gen_cs))
        rows = {}
        for ref, gen in zip(charts, generated, strict=True):
            m = chart_metrics(gen_cs, gen)
            h = report["human"][ref.difficulty.label]
            ref_us = [human.timing_of(ref).us(e.time) for e in active_events(ref)]
            gen_us = [gen_cs.timing_of(gen).us(e.time) for e in active_events(gen)]
            tp = len(match_onsets(np.array(sorted(ref_us)), np.array(sorted(gen_us)), 25_000))
            m["f1@25"] = round(2 * tp / max(len(ref_us) + len(gen_us), 1), 3)
            m["ioi_l1_vs_human"] = ioi_l1(m.get("ioi", {}), h.get("ioi", {}))
            rows[ref.difficulty.label] = m
        report[spec] = rows
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    keys = [
        "events",
        "first_note_s",
        "f1@25",
        "ioi_l1_vs_human",
        "ioi_classes_2pct",
        "distinct_measure_rhythms",
        "rhythm_change_rate",
    ]
    for course in [c.difficulty.label for c in charts]:
        print(f"\n## {course}")
        print("| variant | " + " | ".join(keys) + " | roles |")
        print("|---" * (len(keys) + 2) + "|")
        for name, rows in report.items():
            m = rows.get(course, {})
            print(
                f"| {name} | "
                + " | ".join(str(m.get(k, "-")) for k in keys)
                + f" | {m.get('roles')} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
