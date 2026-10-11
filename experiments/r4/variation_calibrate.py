"""Calibrate the variation penalties on Taiko-1000 training charts; check held-out humans.

python variation_calibrate.py --out /content/runs/r4/variation_calibration.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from urfm.eval.variation import Calibration, calibrated_variation, trace
from urfm.ir.serialize import from_msgpack
from urfm.train.data import extreme_tempo, shard_files


def taiko_charts(file: str):
    for blob in pq.read_table(file, columns=["chartset"])["chartset"].to_pylist():
        cs = from_msgpack(blob)
        for c in cs.charts:
            if c.mechanics == "taiko" and c.events and not extreme_tempo(cs.timing_of(c)):
                yield cs, c


def records(file: str) -> list[dict]:
    return [r for cs, c in taiko_charts(file) for r in trace(cs, c)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = shard_files(args.data, ["taiko1000"], "train")
    recs: list[dict] = []
    with ProcessPoolExecutor(8) as pool:
        for part in pool.map(records, files):
            recs += part
    cal = Calibration.build(recs)
    # evidence for cross-dimension compensation: q95 of the time debt by space variation,
    # within density bins and own (time) variation bins
    evidence: dict[str, dict[str, float]] = defaultdict(dict)
    for key, q in cal.table["q"]["time"].items():
        parts = key.split(",")
        if len(parts) == 3:
            evidence[f"density bin {parts[0]}, V_time bin {parts[1]}"][
                f"V_space bin {parts[2]}"
            ] = q
    held: dict[str, list[float]] = defaultdict(list)
    for f in shard_files(args.data, ["taiko1000"], "test"):
        for cs, c in taiko_charts(f):
            m = calibrated_variation(cs, c, cal)
            if m["score"] is not None:
                held[(c.difficulty.label or "").lower()].append(m["score"])
    out = {
        "source": "Taiko-1000 training charts, all courses; experiments/r4/variation_calibrate.py",
        "units": len(recs),
        **cal.table,
        "evidence_time_debt_q95_by_space_variation": evidence,
        "test_human": {
            k: {"charts": len(v), "mean": float(np.mean(v)), "p90": float(np.quantile(v, 0.9))}
            for k, v in held.items()
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("units", "tau", "test_human")}, indent=1))
    print(json.dumps(evidence, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
