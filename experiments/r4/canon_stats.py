"""How often visual tempo tricks are folded into scroll (urfm.data.canonical), per source.

python canon_stats.py --data /content/data/urfm-ingest-v0 --out /content/runs/r4/canon.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from urfm.data.canonical import SCROLL_FOLLOWS_BPM, CanonicalStats, canonicalize
from urfm.ir.serialize import from_msgpack


def one(file: str) -> dict:
    stats = CanonicalStats()
    out: Counter[str] = Counter()
    pf = pq.ParquetFile(file)
    blobs = (
        blob
        for g in range(pf.num_row_groups)
        for blob in pf.read_row_group(g, columns=["chartset"])["chartset"].to_pylist()
    )
    for blob in blobs:
        cs = from_msgpack(blob)
        for chart in cs.charts:
            if chart.mechanics not in SCROLL_FOLLOWS_BPM or not chart.events:
                continue
            timing = cs.timing_of(chart)
            out[f"{chart.mechanics}/charts"] += 1
            if len(timing.tempos) < 2:
                continue
            out[f"{chart.mechanics}/charts_with_changes"] += 1
            before = stats.merged
            try:
                _, new = canonicalize(cs, chart, stats)
            except Exception as exc:
                out[f"{chart.mechanics}/error:{type(exc).__name__}"] += 1
                continue
            out[f"{chart.mechanics}/tempos_before"] += len(timing.tempos)
            out[f"{chart.mechanics}/tempos_after"] += len(cs.timing_of(new).tempos)
            if stats.merged > before:
                out[f"{chart.mechanics}/charts_changed"] += 1
    out.update({f"reason/{k}": v for k, v in stats.reasons.items()})
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = sorted(str(p) for p in Path(args.data, "data").glob("*/*.parquet"))
    total: Counter[str] = Counter()
    with ProcessPoolExecutor(6) as pool:
        for part in pool.map(one, files):
            total.update(part)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(sorted(total.items())), indent=1))
    print(json.dumps(dict(sorted(total.items())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
