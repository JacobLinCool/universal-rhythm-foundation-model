"""Typical condition values per osu! mode at five density levels (train charts), the presets
the generation app offers for osu! (Easy, Normal, Hard, Insane, Expert = the 10/30/50/70/90th
percentile of mean density, with the median peak, scroll and meter rates of charts near it).

    python osu_defaults.py --out /content/runs/r6/osu_defaults.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from urfm.ir.serialize import from_msgpack
from urfm.tokenize.tokenizer import chart_lanes
from urfm.train.conditions import NAMES, measure
from urfm.train.data import extreme_tempo, shard_files

MODES = ("osu_taiko", "osu_std", "osu_catch", "osu_mania")
LEVELS = (("easy", 0.1), ("normal", 0.3), ("hard", 0.5), ("insane", 0.7), ("expert", 0.9))
KEEP = ("log2_rate", "log2_peak_rate", "log2p_scroll_per_min", "log2p_meter_per_min")


def work(file: str) -> list[tuple[str, list[float], int | None]]:
    out = []
    for blob in pq.read_table(file, columns=["chartset"])["chartset"].to_pylist():
        cs = from_msgpack(blob)
        for c in cs.charts:
            if c.mechanics in MODES and c.events and not extreme_tempo(cs.timing_of(c)):
                v, _ = measure(cs, c)
                out.append((c.mechanics, [float(v[NAMES.index(k)]) for k in KEEP], chart_lanes(c)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with ProcessPoolExecutor(6) as pool:
        rows = [
            r for part in pool.map(work, shard_files(args.data, ["osu"], "train")) for r in part
        ]
    by: dict[str, list[list[float]]] = defaultdict(list)
    lanes: dict[int, int] = defaultdict(int)
    for mode, v, n in rows:
        by[mode].append(v)
        if mode == "osu_mania" and n:
            lanes[n] += 1
    out: dict = {"_mania_lanes": dict(sorted(lanes.items()))}
    for mode, vals in by.items():
        a = np.asarray(vals)
        out[mode] = {}
        for name, q in LEVELS:
            target = np.quantile(a[:, 0], q)
            near = a[np.abs(a[:, 0] - target) < 0.25]
            med = np.median(near if len(near) else a, 0)
            out[mode][name] = {
                "charts": len(near),
                **{k: float(med[i]) for i, k in enumerate(KEEP)},
            }
            out[mode][name]["log2_rate"] = float(target)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
