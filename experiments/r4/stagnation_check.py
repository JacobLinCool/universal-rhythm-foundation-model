"""Human check for the time-based stagnation penalty: peak debts of training charts and the
penalty of held-out test charts per course (design limits: grace 10 s / limit 20 s, both
dimensions plain: 5 s / 10 s).

    python stagnation_check.py --out /content/runs/r4/stagnation_check.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from urfm.eval.variation import (
    STAG_KINDS,
    StagnationLimits,
    stagnation_episodes,
    stagnation_penalty,
    stagnation_trace,
)
from urfm.ir.serialize import from_msgpack
from urfm.train.data import extreme_tempo, shard_files


def charts(file: str):
    for blob in pq.read_table(file, columns=["chartset"])["chartset"].to_pylist():
        cs = from_msgpack(blob)
        for c in cs.charts:
            if c.mechanics == "taiko" and c.events and not extreme_tempo(cs.timing_of(c)):
                yield cs, c


def peaks(file: str) -> tuple[list[tuple[str, str, float]], list[tuple[str, float, float]]]:
    out, episodes = [], []
    for cs, c in charts(file):
        rows = stagnation_trace(cs, c)
        label = (c.difficulty.label or "").lower()
        for k in STAG_KINDS:
            out.append((k, label, max((r[k] for r in rows), default=0.0)))
        episodes += stagnation_episodes(rows)
    return out, episodes


def tests(args: tuple[str, StagnationLimits | None]) -> list[tuple[str, dict]]:
    file, limits = args
    return [
        ((c.difficulty.label or "").lower(), stagnation_penalty(cs, c, limits))
        for cs, c in charts(file)
    ]


def summary(held: list[tuple[str, dict]]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for label, m in held:
        if m["score"] is not None:
            by[label].append(m)
    return {
        label: {
            "charts": len(ms),
            "mean": float(np.mean([m["score"] for m in ms])),
            "share_penalised": float(np.mean([m["score"] > 0 for m in ms])),
        }
        for label, ms in by.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with ProcessPoolExecutor(8) as pool:
        parts = list(pool.map(peaks, shard_files(args.data, ["taiko1000"], "train")))
        train = [p for part, _e in parts for p in part]
        limits = StagnationLimits.build([e for _p, eps in parts for e in eps])
        test_files = shard_files(args.data, ["taiko1000"], "test")
        held = {
            name: [t for part in pool.map(tests, [(f, lim) for f in test_files]) for t in part]
            for name, lim in (("fixed", None), ("calibrated", limits))
        }
    out: dict = {"train_peak_debt_s": {}, "stagnation": limits.table}
    for k in STAG_KINDS:
        v = np.asarray([p for kk, _l, p in train if kk == k])
        out["train_peak_debt_s"][k] = {
            f"q{q}": float(np.quantile(v, q / 100)) for q in (50, 90, 95, 99)
        }
        out["train_peak_debt_s"][k]["max"] = float(v.max())
        out["train_peak_debt_s"][k]["share_over_10s"] = float((v > 10).mean())
    out["test_penalty"] = {name: summary(h) for name, h in held.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
