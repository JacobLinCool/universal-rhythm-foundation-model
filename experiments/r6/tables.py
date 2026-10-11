"""R6 canonical tables from evaluation rows (deterministic).

    python experiments/r6/tables.py runs/r6/results

Each ``<results>/<model>/<eval>/rows.jsonl`` is one panel run; Taiko rows are averaged per
seed, then mean and standard deviation over seeds are reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KEYS = (
    ("f1@25", "F1@25"),
    ("placement_median_ms", "place med ms"),
    ("placement_within_10ms", "place ≤10ms"),
    ("density_rel_err", "dens err"),
    ("ioi_ref_l1", "IOI L1"),
    ("section_r_density", "sec r dens"),
    ("section_r_fine", "sec r fine"),
    ("section_r_rest", "sec r rest"),
    ("section_r_ka", "sec r ka"),
    ("structure_score_gen", "struct"),
    ("structure_null_gen", "struct null"),
    ("recall_sub4", "rec 16th"),
    ("recall_sub6", "rec 24th"),
    ("recall_fine8", "rec 32nd+"),
    ("lt60ms_gen", "<60ms"),
    ("tuplet_share_gen", "tuplets"),
    ("monotony_s_per_min_gen", "E_time"),
    ("stagnation_gen", "stag"),
)
HUMAN = (
    ("structure_score_ref", "struct"),
    ("lt60ms_ref", "<60ms"),
    ("monotony_s_per_min_ref", "E_time"),
    ("stagnation_ref", "stag"),
)


def summarise(path: Path) -> dict[str, tuple[float, float, int]]:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("mechanics") == "taiko"]
    seeds = sorted({r.get("seed", 0) for r in rows})
    out = {}
    for key, _ in (*KEYS, *HUMAN):
        per_seed = []
        for s in seeds:
            v = [
                r[key]
                for r in rows
                if r.get("seed", 0) == s and isinstance(r.get(key), int | float)
            ]
            if v:
                per_seed.append(float(np.mean(v)))
        if per_seed:
            out[key] = (float(np.mean(per_seed)), float(np.std(per_seed)), len(per_seed))
    return out


def cell(v: tuple[float, float, int] | None) -> str:
    if v is None:
        return "–"
    m, s, n = v
    return f"{m:.3f} ± {s:.3f}" if n > 1 else f"{m:.3f}"


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/r6/results")
    panels: dict[str, dict[str, dict]] = {}
    for f in sorted(root.glob("*/*/rows.jsonl")):
        panels.setdefault(f.parent.name, {})[f.parent.parent.name] = summarise(f)
    for panel, models in panels.items():
        print(f"\n### {panel}\n")
        print("| model | " + " | ".join(n for _, n in KEYS) + " |")
        print("|---" * (len(KEYS) + 1) + "|")
        for model, m in models.items():
            print(f"| {model} | " + " | ".join(cell(m.get(k)) for k, _ in KEYS) + " |")
        first = next(iter(models.values()))
        print(
            "| human | "
            + " | ".join(
                cell(first.get(dict((n, k) for k, n in HUMAN).get(n, "")))
                if n in dict(HUMAN).values()
                else "–"
                for _, n in KEYS
            )
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
