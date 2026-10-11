"""How do charters choose the notation octave? (training songs only)

For Taiko-1000 training charts (canonical tempo): fit log2(onsets per beat) as a linear function
of log2(onsets per second). Then, for each training song, compare the human chart's tempo with
the audio-predicted map (``urfm-timing-v1``) and check how often the rule "pick the octave
(x1/2, x1, x2) whose onsets per beat is closest to the fit for the requested density"
reproduces the human choice, against always keeping the predicted octave.

    python octave_rule.py --out /content/runs/r4/octave.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from urfm.data.canonical import canonicalize
from urfm.ir.serialize import from_msgpack, timing_from_msgpack
from urfm.tokenize.tokenizer import onset_rate, playable_events
from urfm.train.data import extreme_tempo, shard_files


def main_bpm(timing, chart) -> float:
    """Tempo (BPM) covering most of the chart's notes."""
    beats = np.asarray([float(e.time.beat) for e in playable_events(chart)])
    tempos = sorted(timing.tempos, key=lambda t: t.beat)
    starts = np.asarray([float(t.beat) for t in tempos])
    idx = np.clip(np.searchsorted(starts, beats, side="right") - 1, 0, None)
    counts = np.bincount(idx, minlength=len(tempos))
    return float(tempos[int(counts.argmax())].bpm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--timing", default="/content/data/urfm-timing-v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []  # (song key, log2 rate, log2 per beat, human main bpm, chart)
    for f in shard_files(args.data, ["taiko1000"], "train"):
        pred_file = Path(args.timing) / "data" / "taiko1000" / Path(f).name
        pred = {}
        if pred_file.exists():
            t = pq.read_table(pred_file, columns=["row_group", "row", "timing"])
            for g, r, b in zip(
                *(t[c].to_pylist() for c in ("row_group", "row", "timing")), strict=True
            ):
                pred[(g, r)] = timing_from_msgpack(b)
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            for r, blob in enumerate(
                pf.read_row_group(g, columns=["chartset"])["chartset"].to_pylist()
            ):
                cs = from_msgpack(blob)
                for c in cs.charts:
                    if c.mechanics != "taiko" or len(c.events) < 16:
                        continue
                    if extreme_tempo(cs.timing_of(c)):
                        continue
                    cs2, c2 = canonicalize(cs, c)
                    timing = cs2.timing_of(c2)
                    rate = onset_rate(cs2, c2)
                    bpm = main_bpm(timing, c2)
                    if rate <= 0:
                        continue
                    per_beat = rate * 60 / bpm
                    p = pred.get((g, r))
                    pbpm = float(np.median([float(t.bpm) for t in p.tempos])) if p else None
                    rows.append(
                        (
                            f"{f}:{g}:{r}",
                            math.log2(rate),
                            math.log2(per_beat),
                            bpm,
                            pbpm,
                            (c.difficulty.label or "").lower(),
                        )
                    )
    x = np.asarray([r[1] for r in rows])
    y = np.asarray([r[2] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    resid = float(np.std(y - (slope * x + intercept)))
    # octave choice on songs with a prediction: rule vs keep, against the human octave
    hits_rule = hits_keep = n = 0
    for _key, lr, _lpb, bpm, pbpm, _label in rows:
        if pbpm is None:
            continue
        human = round(math.log2(bpm / pbpm))
        if abs(math.log2(bpm / pbpm) - human) > 0.15 or abs(human) > 1:
            continue  # not an octave relation (tempo error or unusual notation)
        want = slope * lr + intercept
        best = min((-1, 0, 1), key=lambda o: abs(lr - math.log2(pbpm * 2**o / 60) - want))
        hits_rule += best == human
        hits_keep += human == 0
        n += 1
    out = {
        "charts": len(rows),
        "fit": {"slope": float(slope), "intercept": float(intercept), "resid_std": resid},
        "octave_songs": n,
        "rule_accuracy": hits_rule / max(n, 1),
        "keep_accuracy": hits_keep / max(n, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
