"""Human reference for density-calibrated monotony: R of every 10 s window of Taiko-1000
training charts (all courses), as quantiles by local density.

    python monotony_reference.py --out /content/runs/r4/monotony_reference.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from urfm.eval.monotony import (
    Reference,
    StretchReference,
    excess_monotone_time,
    excess_monotony,
    stretches,
    window_scores,
)
from urfm.ir.serialize import from_msgpack
from urfm.train.data import extreme_tempo, shard_files


def stretch_items(file: str) -> list[tuple[float, float]]:
    out = []
    for blob in pq.read_table(file, columns=["chartset"])["chartset"].to_pylist():
        cs = from_msgpack(blob)
        for c in cs.charts:
            if c.mechanics != "taiko" or not c.events or extreme_tempo(cs.timing_of(c)):
                continue
            out += stretches(cs, c)[0]
    return out


def one(file: str) -> list[tuple[float, float, float]]:
    out = []
    for blob in pq.read_table(file, columns=["chartset"])["chartset"].to_pylist():
        cs = from_msgpack(blob)
        for c in cs.charts:
            if c.mechanics != "taiko" or not c.events or extreme_tempo(cs.timing_of(c)):
                continue
            out += [(d, rf, rr) for _s, d, rf, rr in window_scores(cs, c)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = shard_files(args.data, ["taiko1000"], "train")
    windows = []
    with ProcessPoolExecutor(8) as pool:
        for part in pool.map(one, files):
            windows += part
    refs = {f"q{int(q * 100)}": Reference.build(windows, q).table for q in (0.5, 0.9, 0.95)}
    items = []
    with ProcessPoolExecutor(8) as pool:
        for part in pool.map(stretch_items, files):
            items += part
    sref = StretchReference.build(items)
    refs["stretch_q95"] = sref.table  # type: ignore[assignment]
    held_stretch: dict[str, list] = {}
    for f in shard_files(args.data, ["taiko1000"], "test"):
        for blob in pq.read_table(f, columns=["chartset"])["chartset"].to_pylist():
            cs = from_msgpack(blob)
            for c in cs.charts:
                if c.mechanics != "taiko" or not c.events or extreme_tempo(cs.timing_of(c)):
                    continue
                m = excess_monotone_time(cs, c, sref)
                if m["excess_s_per_min"] is not None:
                    held_stretch.setdefault((c.difficulty.label or "").lower(), []).append(m)
    refs["test_human_stretch"] = {  # type: ignore[assignment]
        k: {
            "charts": len(v),
            "excess_s_per_min": sum(m["excess_s_per_min"] for m in v) / len(v),
            "max_excess_s": sum(m["max_excess_s"] for m in v) / len(v),
        }
        for k, v in held_stretch.items()
    }
    # held-out check: human test charts should be flagged ~5% at every course
    ref = Reference(refs["q95"])
    by_course: dict[str, list] = {}
    for f in shard_files(args.data, ["taiko1000"], "test"):
        for blob in pq.read_table(f, columns=["chartset"])["chartset"].to_pylist():
            cs = from_msgpack(blob)
            for c in cs.charts:
                if c.mechanics != "taiko" or not c.events or extreme_tempo(cs.timing_of(c)):
                    continue
                label = (c.difficulty.label or "").lower()
                for kind in ("rhythm", "full"):
                    m = excess_monotony(cs, c, ref, kind)
                    if m["excess_share"] is not None:
                        by_course.setdefault(f"{kind}/{label}", []).append(m["excess_share"])
    held_out = {k: {"charts": len(v), "mean_excess": sum(v) / len(v)} for k, v in by_course.items()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"windows": len(windows), "test_human": held_out, **refs}, indent=1)
    )
    print(json.dumps(held_out, indent=1))
    print(len(windows), json.dumps(refs["q95"], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
