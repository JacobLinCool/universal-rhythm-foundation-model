"""R4 benchmark songs (P3): Crystal Hail (Taiko-1000 test) and three newer charts that sit
at or beyond the training distribution's tails. Runs on CPU.

    uv run python experiments/r4/bench.py --ckpt runs/checkpoints/ft-r3-song.pt \\
        --timing audio --beat-ckpt runs/checkpoints/beat-v1.pt --out runs/r4/bench/r3

Each song folder holds ``human.tja`` and ``song.*``. For every course the model is asked for
a chart like the human one (``urfm.eval.protocol``); generated charts are written as
playable TJA next to a copy of the audio name, and the metrics table compares them with the
human chart on the human beat grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from urfm.codecs import get
from urfm.data.features import audio_mel
from urfm.eval import evaluate_pair
from urfm.eval.protocol import choose_octave, plan_job, predicted_timing, request, speed_up
from urfm.eval.structure import alignment
from urfm.ir.chart import AudioRef, ChartSet, SongMeta
from urfm.tokenize import build_tokenizer
from urfm.tokenize.tokenizer import onset_rate
from urfm.train.generate import generate
from urfm.train.train import load_checkpoint

SONGS = (
    "runs/v1b-taiko/playable/taiko1000_test-00004-Crystal_Hail",
    "runs/newcharts/Nivalis＊Anima",
    "runs/newcharts/銀の黎明か、黒の晶華か。",
    "runs/newcharts/魔宵月",
)
KEYS = (
    "n_gen",
    "n_ref",
    "f1@25",
    "ioi_ref_l1",
    "section_r_density",
    "section_r_fine",
    "section_r_rest",
    "section_r_ka",
    "fine_share_gen",
    "fine_share_ref",
    "rest_share_gen",
    "rest_share_ref",
    "measure_rhythm_variety_gen",
    "measure_rhythm_variety_ref",
    "first_onset_delay_s",
    "taiko_role_acc@25",
    "controls_gen",
    "controls_ref",
    "lt60ms_gen",
    "lt60ms_ref",
    "placement_median_ms",
    "placement_within_10ms",
    "recall_sub4",
    "recall_sub6",
    "recall_fine8",
    "tuplet_share_gen",
    "structure_score_gen",
    "structure_score_ref",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--songs", nargs="*", default=list(SONGS))
    ap.add_argument("--timing", default="oracle", choices=["oracle", "audio"])
    ap.add_argument("--beat-ckpt", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--courses", nargs="*", default=["oni", "edit"])
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--plan", default="calibrated")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--octave-search", action="store_true")
    ap.add_argument("--planner", default=None, help="R6 song planner checkpoint")
    ap.add_argument("--plan-temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tja = get("tja")
    tok = build_tokenizer()
    model, meta = load_checkpoint(args.ckpt, args.device)
    planner = None
    if args.planner:
        from urfm.train.planner import load_planner

        planner = load_planner(args.planner, args.device)
    beat = None
    if args.timing == "audio":
        from urfm.beat.train import load as load_beat

        beat = load_beat(args.beat_ckpt, args.device)
    rows = []
    for folder in map(Path, args.songs):
        human = tja.read_path(folder / "human.tja")
        audio = next(folder.glob("song.*"))
        mel = audio_mel(audio)
        human, mel = speed_up(human, mel, args.speed)
        timing_audio = (
            predicted_timing(beat, mel, args.device, "taiko") if beat is not None else None
        )
        charts = [
            c
            for c in human.charts
            if c.events
            and c.mechanics == "taiko"
            and (not args.courses or (c.difficulty.label or "").lower() in args.courses)
        ]
        if timing_audio is not None and args.octave_search and charts:
            hardest = max(charts, key=lambda c: onset_rate(human, c))
            factor, timing_audio = choose_octave(
                model,
                meta,
                tok,
                human,
                hardest,
                mel,
                timing_audio,
                temperature=args.temperature,
                device=args.device,
                seed=0,
                plan=args.plan,
            )
            print(f"{folder.name}: octave x{factor}", flush=True)
        for seed in args.seeds:
            jobs = [
                request(model, meta, tok, human, c, mel, timing_audio or human.timing_of(c))
                for c in charts
            ]
            for i, job in enumerate(jobs):
                plan_job(job, meta, planner, args.plan_temperature, seed * 7919 + i)
            gen = generate(
                model,
                tok,
                jobs,
                temperature=args.temperature,
                device=args.device,
                seed=seed,
                plan="sample" if planner is not None else args.plan,
                plan_temperature=args.plan_temperature if planner is not None else 1.0,
            )
            timing = timing_audio or human.timing
            gen_cs = ChartSet(
                timing=timing,
                charts=tuple(gen),
                song=SongMeta(title=f"{human.song.title} (URFM {out.name})"),
                audio=AudioRef(path=audio.name),
            )
            tag = f"{folder.name}-s{seed}"
            (out / f"{tag}.tja").write_bytes(tja.write(gen_cs))
            for ref, g in zip(charts, gen, strict=True):
                row = evaluate_pair(gen_cs, g, human, ref)
                row["structure_score_gen"] = alignment(gen_cs, g, mel)["score"]
                row["structure_score_ref"] = alignment(human, ref, mel)["score"]
                row = {k: row.get(k) for k in KEYS}
                row |= {"song": folder.name, "course": ref.difficulty.label, "seed": seed}
                rows.append(row)
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"| song | course | {' | '.join(KEYS)} |")
    print("|---" * (len(KEYS) + 2) + "|")
    for r in rows:
        cells = [
            "–" if r[k] is None else (f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]))
            for k in KEYS
        ]
        print(f"| {r['song'][:12]} | {r['course']} | {' | '.join(cells)} |")
    means = {
        k: float(np.mean([r[k] for r in rows if r[k] is not None]))
        for k in KEYS
        if any(r[k] is not None for r in rows)
    }
    (out / "summary.json").write_text(json.dumps(means, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
