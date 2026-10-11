"""Generate charts for held-out songs with a checkpoint and score them against the references.

Runs on the Colab VM next to the shards:

    python evaluate.py --ckpt /content/runs/v1-taiko/best.pt --sources taiko1000 \
        --mechanics taiko --out /content/runs/v1-taiko/eval

For every test song and every reference chart of the requested mechanics, the model
generates a chart under the reference's conditioning (``urfm.eval.protocol``). The timing
map is the reference's (``--timing oracle``) or predicted from audio by a beat model
(``--timing audio --beat-ckpt ...``); ``--speed 1.25`` plays every song faster first (stress
test). Writes ``rows.jsonl`` and ``report.md`` (urfm.eval) and the first ``--samples``
generated charts as native files.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from urfm.beat.decode import for_game
from urfm.data.features import MEL
from urfm.eval import evaluate_pair, write_report
from urfm.eval.protocol import choose_octave, plan_job, predicted_timing, request, speed_up
from urfm.eval.structure import alignment
from urfm.ir.chart import SongMeta
from urfm.ir.serialize import from_msgpack
from urfm.tokenize import build_tokenizer
from urfm.tokenize.tokenizer import onset_rate
from urfm.train.data import shard_files
from urfm.train.generate import generate, to_chartset, write_native
from urfm.train.train import load_checkpoint

EXT = {"taiko": ".tja", "maimai_dx": ".maidata.txt"}


def reference_traits(cs, chart) -> dict:
    """What makes a reference hard (panel P2): tempo changes, segments shorter than a
    measure, and density."""
    from urfm.tokenize.tokenizer import onset_rate

    timing = cs.timing_of(chart)
    tempos = sorted(timing.tempos, key=lambda t: t.beat)
    short = 0
    for a, b in itertools.pairwise(tempos):
        if b.beat - a.beat < timing.meter_at(a.beat).measure_beats:
            short += 1
    return {
        "ref_tempo_changes": len(tempos) - 1,
        "ref_short_segments": short,
        "ref_rate": onset_rate(cs, chart),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--sources", nargs="+", default=["taiko1000"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--mechanics", nargs="+", default=["taiko"])
    ap.add_argument("--max-songs", type=int, default=0)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--plan", default="calibrated", choices=["calibrated", "sample", "mean"])
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--timing", default="oracle", choices=["oracle", "audio"])
    ap.add_argument("--beat-ckpt", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument(
        "--octave-search",
        action="store_true",
        help="audio timing: pick the notation octave per song (urfm.eval.protocol.choose_octave)",
    )
    ap.add_argument(
        "--notation",
        type=float,
        default=1.0,
        help="P7: re-notate the timing map at this BPM factor (0.5 = half BPM, same sound)",
    )
    ap.add_argument("--planner", default=None, help="R6 song planner checkpoint")
    ap.add_argument("--plan-temperature", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.ckpt, device)
    tok = build_tokenizer()
    feats = meta.get("features", {})
    planner = None
    if feats.get("sections"):
        from urfm.train.planner import load_planner

        if not args.planner:
            raise SystemExit("a section-plan model needs --planner")
        planner = load_planner(args.planner, device)
    beat_model = None
    if args.timing == "audio":
        from urfm.beat.train import load as load_beat

        if not args.beat_ckpt:
            raise SystemExit("--timing audio needs --beat-ckpt")
        beat_model = load_beat(args.beat_ckpt, device)

    items = []  # (song_id, ref chartset, ref chart, job)
    octaves: dict[str, int] = {}
    for file in shard_files(args.data, args.sources, args.split):
        table = pq.read_table(file, columns=["song_id", "mel", "chartset", "title"])
        for song_id, mel, blob, title in zip(
            *(table[c].to_pylist() for c in ("song_id", "mel", "chartset", "title")), strict=True
        ):
            cs = from_msgpack(blob)
            if not any(c.mechanics in args.mechanics and c.events for c in cs.charts):
                continue
            mel_arr = np.frombuffer(mel, np.uint8).reshape(-1, MEL.n_mels)
            cs, mel_arr = speed_up(cs, mel_arr, args.speed)
            audio_timing = (
                predicted_timing(beat_model, mel_arr, device, "")
                if beat_model is not None
                else None
            )
            if audio_timing is not None and args.octave_search:
                charts = [c for c in cs.charts if c.mechanics in args.mechanics and c.events]
                hardest = max(charts, key=lambda c: onset_rate(cs, c))
                factor, audio_timing = choose_octave(
                    model,
                    meta,
                    tok,
                    cs,
                    hardest,
                    mel_arr,
                    for_game(audio_timing, hardest.mechanics),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                    seed=0,
                    plan=args.plan,
                )
                octaves[str(factor)] = octaves.get(str(factor), 0) + 1
            for chart in cs.charts:
                if chart.mechanics not in args.mechanics or not chart.events:
                    continue
                timing = (
                    audio_timing
                    if audio_timing is not None and args.octave_search
                    else for_game(audio_timing, chart.mechanics)
                    if audio_timing is not None
                    else cs.timing_of(chart)
                )
                if args.notation != 1:
                    from fractions import Fraction

                    from urfm.ir.transform import rescale_timing

                    timing = rescale_timing(timing, Fraction(args.notation).limit_denominator(8))
                job = request(model, meta, tok, cs, chart, mel_arr, timing)
                items.append((song_id, title, cs, chart, job, mel_arr))
            if args.max_songs and len({i[0] for i in items}) >= args.max_songs:
                break
    print(f"{len(items)} charts from {len({i[0] for i in items})} songs", flush=True)

    rows, t0 = [], time.time()
    for seed in args.seeds:
        for b in range(0, len(items), args.batch):
            chunk = items[b : b + args.batch]
            for it in chunk:  # jobs accumulate output: start each seed afresh
                job = it[4]
                job.events, job.controls, job.meters = [], [], []
                job.density_plan = None
                plan_job(job, meta, planner, args.plan_temperature, seed * 7919 + b)
            charts = generate(
                model,
                tok,
                [it[4] for it in chunk],
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                seed=seed * 100_003 + b,
                plan="sample" if planner is not None else args.plan,
                plan_temperature=args.plan_temperature,
            )
            for (song_id, title, ref_cs, ref_chart, job, mel_arr), gen_chart in zip(
                chunk, charts, strict=True
            ):
                gen_cs = to_chartset(gen_chart, job.timing, SongMeta(title=title or song_id))
                row = evaluate_pair(gen_cs, gen_chart, ref_cs, ref_chart)
                g_al = alignment(gen_cs, gen_chart, mel_arr)
                r_al = alignment(ref_cs, ref_chart, mel_arr)
                row |= {
                    "structure_score_gen": g_al["score"],
                    "structure_null_gen": g_al.get("null"),
                    "structure_score_ref": r_al["score"],
                }
                row |= {"song_id": song_id, "label": ref_chart.difficulty.label, "seed": seed}
                row |= reference_traits(ref_cs, ref_chart)
                rows.append(row)
                if len(rows) <= args.samples:
                    safe = song_id.replace(":", "_")
                    ext = EXT.get(job.mechanics) or (
                        ".osu" if job.mechanics.startswith("osu") else ".ssc"
                    )
                    try:
                        write_native(
                            gen_cs, out / "samples" / f"{safe}-{ref_chart.difficulty.label}{ext}"
                        )
                    except Exception as exc:
                        print(f"cannot write sample {song_id}: {exc}", flush=True)
            print(f"seed {seed}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

    write_report(rows, out / "rows.jsonl", out / "report.md")
    (out / "summary.json").write_text(
        json.dumps({"charts": len(rows), "seconds": time.time() - t0, "octaves": octaves})
    )
    print((out / "report.md").read_text(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
