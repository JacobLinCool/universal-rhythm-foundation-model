"""Why do condition models chart sparser on audio-predicted grids? (validation split only)

    python diag_audio.py --ckpt /content/runs/r4a-full-15m/best.pt \\
        --beat-ckpt /content/runs/beat-v1/best.pt --songs 40

Variants differ in one factor each: timing (oracle / audio), plan temperature for the
GRID/REST tokens, and whether notation-dependent conditions are given.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.parquet as pq
import torch

from urfm.beat.decode import for_game, predict_timing
from urfm.beat.train import load as load_beat
from urfm.data.features import MEL
from urfm.eval import evaluate_pair
from urfm.eval.protocol import request
from urfm.ir.chart import SongMeta
from urfm.ir.serialize import from_msgpack
from urfm.tokenize import build_tokenizer
from urfm.train.conditions import without_notation
from urfm.train.data import shard_files
from urfm.train.generate import generate, reference_inputs, to_chartset
from urfm.train.train import load_checkpoint

VARIANTS = {
    "oracle": dict(timing="oracle", plan_t=1.0, notation=True),
    "oracle t0.6": dict(timing="oracle", plan_t=1.0, notation=True, t=0.6),
    "oracle t0.7": dict(timing="oracle", plan_t=1.0, notation=True, t=0.7),
    "oracle t1.0": dict(timing="oracle", plan_t=1.0, notation=True, t=1.0),
    "audio": dict(timing="audio", plan_t=1.0, notation=False),
    "audio+notation": dict(timing="audio", plan_t=1.0, notation=True),
    "audio planT0.6": dict(timing="audio", plan_t=0.6, notation=False),
    "oracle planT0.6": dict(timing="oracle", plan_t=0.6, notation=True),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beat-ckpt", required=True)
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--songs", type=int, default=40)
    ap.add_argument("--only", nargs="*", default=None, help="variant names to run")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.ckpt, device)
    beat = load_beat(args.beat_ckpt, device)
    tok = build_tokenizer()
    items = []
    for f in shard_files(args.data, ["taiko1000"], "validation"):
        t = pq.read_table(f, columns=["mel", "chartset"])
        for mel_b, blob in zip(t["mel"].to_pylist(), t["chartset"].to_pylist(), strict=True):
            cs = from_msgpack(blob)
            charts = [
                c
                for c in cs.charts
                if c.events and (c.difficulty.label or "").lower() in ("oni", "edit")
            ]
            if not charts:
                continue
            mel = np.frombuffer(mel_b, np.uint8).reshape(-1, MEL.n_mels)
            audio = for_game(predict_timing(beat, mel, device), "taiko")
            items += [(cs, c, mel, audio) for c in charts]
            if len(items) >= args.songs:
                break
    for name, v in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        jobs = []
        for cs, c, mel, audio in items:
            timing = audio if v["timing"] == "audio" else cs.timing_of(c)
            job = request(model, meta, tok, cs, c, mel, timing)
            if v["notation"] and job.cond is not None:
                job.cond = reference_inputs(model, meta, tok, cs, c)[1]
            elif not v["notation"] and job.cond is not None:
                job.cond = without_notation(job.cond)
            jobs.append(job)
        gen = generate(
            model,
            tok,
            jobs,
            temperature=float(v.get("t", 0.85)),
            device=device,
            seed=0,
            plan="calibrated",
            plan_temperature=float(v["plan_t"]),
        )
        rows = []
        for (cs, c, _mel, _a), job, g in zip(items, jobs, gen, strict=True):
            rows.append(evaluate_pair(to_chartset(g, job.timing, SongMeta()), g, cs, c))
        keys = (
            "f1@25",
            "density_rel_err",
            "rest_share_gen",
            "rest_share_ref",
            "ioi_ref_l1",
            "section_r_density",
            "section_r_fine",
            "section_r_rest",
            "section_r_ka",
            "measure_rhythm_variety_gen",
            "measure_rhythm_variety_ref",
        )
        summary = {
            k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 3) for k in keys
        }
        ratio = np.mean([r["n_gen"] / max(r["n_ref"], 1) for r in rows])
        print(
            json.dumps({"variant": name, "n_ratio": round(float(ratio), 3), **summary}), flush=True
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
