"""Whole-song evaluation of the beat model (P1 timing, P5 synthetic tempo warps).

For each held-out song: activations over the whole song, the decoded TimingMap, and
* beat F@70 / F@25 / any-level F@70 of the map's beats against the chart's (canonical) beats,
* downbeat F@70 of the map's measure starts,
* grid fit: share of the human chart's note onsets on 1/4 or 1/3 beats of the predicted map
  (and of the human map, the ceiling).
``--warp`` first applies random short tempo changes (half a beat to a measure, up to six,
±25%) to audio and labels: the model has to follow tempo changes it was never given.

    python eval_beat.py --ckpt /content/runs/beat-v1/best.pt --out /content/runs/beat-v1/eval
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from urfm.beat.data import random_warp, resample
from urfm.beat.decode import activations, decode_beats, to_timing
from urfm.beat.labels import representative, song_labels
from urfm.beat.metrics import beat_scores, f_measure, grid_fit
from urfm.beat.train import load
from urfm.data.canonical import canonicalize
from urfm.data.features import MEL
from urfm.ir.serialize import from_msgpack
from urfm.tokenize.tokenizer import playable_events
from urfm.train.data import shard_files


def grid_times(timing, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    lo = int(np.floor(float(timing.us_to_beat(0))))
    hi = int(np.ceil(float(timing.us_to_beat(int(duration_s * 1e6)))))
    beats = timing.us_array(list(range(lo, hi + 1))) / 1e6
    starts = [m for m in timing.measure_starts(Fraction(hi)) if m >= lo]
    downs = timing.us_array(starts) / 1e6 if starts else np.zeros(0)
    keep = (beats >= 0) & (beats <= duration_s)
    return beats[keep], downs[(downs >= 0) & (downs <= duration_s)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--sources", nargs="+", default=["taiko1000", "osu"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-songs", type=int, default=150, help="per source")
    ap.add_argument("--warp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.ckpt, device)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for source in args.sources:
        n = 0
        for file in shard_files(args.data, [source], args.split):
            table = pq.read_table(file, columns=["song_id", "mel", "chartset", "duration_s"])
            for sid, mel_b, blob, dur in zip(
                *(table[c].to_pylist() for c in ("song_id", "mel", "chartset", "duration_s")),
                strict=True,
            ):
                if n >= args.max_songs:
                    break
                cs = from_msgpack(blob)
                labels = song_labels(cs, dur)
                chart = representative(cs)
                if labels is None or chart is None:
                    continue
                n += 1
                ref, ref_d, span = labels
                cs_c, chart_c = canonicalize(cs, chart)
                human = cs_c.timing_of(chart_c)
                onsets = np.unique(
                    np.round([human.us(e.time) / 1e6 for e in playable_events(chart_c)], 4)
                )
                mel = np.frombuffer(mel_b, np.uint8).reshape(-1, MEL.n_mels)
                if args.warp:
                    warp = random_warp(rng, ref, p_speed=0.0, p_local=1.0, local=(0.75, 1.33))
                    new_dur = float(warp.forward(np.asarray([dur]))[0])
                    mel = resample(mel, warp, 0.0, int(new_dur * MEL.fps))
                    ref, ref_d, onsets = (warp.forward(x) for x in (ref, ref_d, onsets))
                    lo_w, hi_w = (float(x) for x in warp.forward(np.asarray(span)))
                    span = (lo_w, hi_w)
                    dur = new_dur
                act = activations(model, mel, device)
                est = decode_beats(act[:, 0])
                timing = to_timing(est, act[:, 1])
                beats, downs = grid_times(timing, dur)
                row = {"source": source, "song_id": sid, "tempos": len(timing.tempos)}
                row |= beat_scores(ref, beats, span)
                row["peaks_f70"] = beat_scores(ref, est, span)["beat_f70"]
                downs = downs[(downs >= span[0] - 0.035) & (downs <= span[1] + 0.035)]
                row["downbeat_f70"] = f_measure(ref_d, downs, 0.07)
                row["grid_fit"] = grid_fit(timing, onsets)
                if not args.warp:
                    row["grid_fit_human"] = grid_fit(human, onsets)
                rows.append(row)
    (out / "songs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    summary: dict[str, dict[str, float]] = defaultdict(dict)
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)] if rows else []
    for source in args.sources:
        sub = [r for r in rows if r["source"] == source]
        if sub:
            summary[source] = {k: round(float(np.mean([r[k] for r in sub])), 4) for k in keys}
            summary[source]["songs"] = len(sub)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
