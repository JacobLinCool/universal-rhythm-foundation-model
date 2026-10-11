"""Typical condition values per Taiko course and level (Taiko-1000 train charts), the defaults
the generation app asks for when the user gives only a course and a level.

    python course_defaults.py --out /content/runs/r6/course_defaults.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import msgspec
import numpy as np
import pyarrow.parquet as pq

from urfm.data.canonical import canonicalize
from urfm.ir.serialize import from_msgpack
from urfm.train.conditions import NAMES, measure
from urfm.train.data import extreme_tempo, shard_files

KEEP = ("log2_rate", "log2_peak_rate", "log2p_scroll_per_min", "log2p_meter_per_min")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    by: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for f in shard_files(args.data, ["taiko1000"], "train"):
        for blob in pq.read_table(f, columns=["chartset"])["chartset"].to_pylist():
            cs = from_msgpack(blob)
            for chart in cs.charts:
                label = (chart.difficulty.label or "").lower()
                level = chart.difficulty.level
                if chart.mechanics != "taiko" or not chart.events or not level:
                    continue
                if extreme_tempo(cs.timing_of(chart)):
                    continue
                c1, ch = canonicalize(msgspec.structs.replace(cs, charts=(chart,)), chart)
                values, _mask = measure(msgspec.structs.replace(c1, charts=(ch,)), ch)
                by[label, int(level)].append(values)
    out: dict = {}
    for (label, level), rows in sorted(by.items()):
        v = np.median(np.stack(rows), 0)
        out.setdefault(label, {})[str(level)] = {
            "charts": len(rows),
            **{k: float(v[NAMES.index(k)]) for k in KEEP},
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
