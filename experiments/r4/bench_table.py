"""P3 benchmark table over seeds (deterministic from ``bench.py`` outputs).

python experiments/r4/bench_table.py runs/r4/bench/final-*
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    cols = (
        "f1@25",
        "density_err",
        "ioi_ref_l1",
        "fine_gap",
        "rest_gap",
        "section_r_density",
        "section_r_fine",
        "section_r_rest",
        "section_r_ka",
    )
    print("| run | charts | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 2) + "|")
    for d in map(Path, sys.argv[1:]):
        rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x]
        for r in rows:
            r["density_err"] = abs(r["n_gen"] / max(r["n_ref"], 1) - 1)
            r["fine_gap"] = abs(r["fine_share_gen"] - r["fine_share_ref"])
            r["rest_gap"] = abs(r["rest_share_gen"] - r["rest_share_ref"])
        cells = []
        for c in cols:
            vals = [r[c] for r in rows if isinstance(r.get(c), int | float)]
            cells.append(f"{np.mean(vals):.3f}" if vals else "–")
        print(f"| {d.name} | {len(rows)} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
