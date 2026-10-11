"""Canonical R4 tables from evaluation outputs (deterministic; rerun any time).

    python experiments/r4/tables.py runs/r4/results   # folders holding <run>/<panel>/rows.jsonl

Panels: P1 = all rows; P2 = rows whose reference has tempo changes, a segment shorter than a
measure, or a density in the top 20% of the panel. Means over seeds and charts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KEYS = (
    "f1@25",
    "f1@50",
    "density_rel_err",
    "ioi_ref_l1",
    "section_r_density",
    "section_r_fine",
    "section_r_rest",
    "section_r_ka",
    "fine_share_gen",
    "fine_share_ref",
    "rest_share_gen",
    "rest_share_ref",
    "measure_rhythm_variety_gen",
    "first_onset_delay_s",
    "taiko_role_acc@25",
    "issues_per_100",
    "lt60ms_gen",
    "lt60ms_ref",
)


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def mean(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), int | float)]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    root = Path(sys.argv[1])
    lines = ["| run | panel | rows | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 3) + "|"]
    for rows_file in sorted(root.glob("*/*/rows.jsonl")):
        rows = load(rows_file)
        run, panel = rows_file.parent.parent.name, rows_file.parent.name
        rates = [r.get("ref_rate", 0) for r in rows]
        cut = float(np.quantile(rates, 0.8)) if rates else 0
        hard = [
            r
            for r in rows
            if r.get("ref_tempo_changes", 0) > 0
            or r.get("ref_short_segments", 0) > 0
            or r.get("ref_rate", 0) >= cut
        ]
        for name, sub in (("P1", rows), ("P2", hard)):
            cells = []
            for k in KEYS:
                v = mean(sub, k)
                cells.append("–" if v is None else f"{v:.3f}")
            lines.append(f"| {run} | {panel}:{name} | {len(sub)} | " + " | ".join(cells) + " |")
    out = "\n".join(lines) + "\n"
    (root / "tables.md").write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
