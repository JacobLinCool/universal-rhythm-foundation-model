"""Density-calibrated monotony on every local chart (reference: Taiko-1000 training charts).

uv run python experiments/r4/monotony_eval2.py --ref runs/r4/monotony/reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from urfm.eval.monotony import (
    Reference,
    StretchReference,
    excess_monotone_time,
    excess_monotony,
)

sys.path.insert(0, str(Path(__file__).parent))
from monotony_eval import (
    COURSES,
    SET_A,
    SET_A_FILES,
    SET_B,
    SET_B_FILES,
    SET_C,
    SET_C_RUNS,
    charts,
)


def rows_for(score) -> list[dict]:
    rows = []
    for set_name, root, files in (("A", SET_A, SET_A_FILES), ("B", SET_B, SET_B_FILES)):
        for song in sorted(root.iterdir()):
            for name, f in files.items():
                if (song / f).exists():
                    for cs, c, label in charts(song / f):
                        rows.append(
                            {"set": set_name, "source": name, "course": label, **score(cs, c)}
                        )
    for name, run in SET_C_RUNS.items():
        for f in sorted((SET_C / run).glob("*-s[0-9].tja")):
            for cs, c, label in charts(f):
                rows.append({"set": "C", "source": name, "course": label, **score(cs, c)})
    return rows


def table(rows, set_name, key_share, key_long) -> str:
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["set"] == set_name and r.get(key_share) is not None:
            groups[r["source"]][r["course"]].append(r)
    present = [c for c in COURSES if any(c in g for g in groups.values())]
    lines = ["| source | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|"]
    for source, by in groups.items():
        cells = []
        for c in present:
            v = by.get(c, [])
            if not v:
                cells.append("–")
                continue
            share = np.mean([r[key_share] for r in v])
            longest = np.mean([r[key_long] for r in v])
            cells.append(
                f"{share:.0%} / {longest:.0f}s"
                if key_share.endswith("share")
                else f"{share:.1f} / {longest:.0f}s"
            )
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="runs/r4/monotony/reference.json")
    ap.add_argument("--quantile", default="q95")
    ap.add_argument("--out", default="runs/r4/monotony/report2.md")
    args = ap.parse_args()
    data = json.loads(Path(args.ref).read_text())
    ref = Reference(data[args.quantile])
    sref = StretchReference(data["stretch_q95"])
    rows = rows_for(
        lambda cs, c: excess_monotony(cs, c, ref, "rhythm") | excess_monotone_time(cs, c, sref)
    )
    report = []
    for s in ("A", "B", "C"):
        report += [
            f"## Set {s}",
            "",
            "Window level, rhythm only (excess share / longest excess stretch):",
            "",
            table(rows, s, "excess_share", "excess_longest_s"),
            "",
            "Stretch level (excess monotone seconds per minute / largest single excess):",
            "",
            table(rows, s, "excess_s_per_min", "max_excess_s"),
            "",
        ]
    text = "\n".join(report)
    Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
