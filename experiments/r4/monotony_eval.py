"""Validate the monotony metric (urfm.eval.monotony) on every local chart: human charts and
the charts each model version generated for the same songs.

    uv run python experiments/r4/monotony_eval.py --out runs/r4/monotony
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from urfm.codecs import get
from urfm.eval.monotony import monotony

COURSES = ("easy", "normal", "hard", "oni", "edit")
SET_A = Path("runs/v1b-taiko/playable")  # R1-R3 era, human timing
SET_A_FILES = {
    "human": "human.tja",
    "R1 (v1)": "urfm-v1.tja",
    "R2 (v2)": "urfm-v2.tja",
    "R2 all games (v3)": "urfm-v3.tja",
    "R3 best": "urfm-best.tja",
}
SET_B = Path("runs/r4/playable")  # benchmark songs, all courses
SET_B_FILES = {
    "human": "human.tja",
    "R4a oracle": "urfm-r4a-oracle.tja",
    "R4a audio": "urfm-r4a-audio.tja",
    "R4 final audio": "urfm-r4-final-audio.tja",
    "R5 RL audio": "urfm-r5-rl-audio.tja",
    "R5 RL audio x2 grid": "urfm-r5-rl-audio-x2grid.tja",
}
SET_C = Path("runs/r4/bench")  # benchmark oni/edit, 3 seeds each, audio timing
SET_C_RUNS = {
    "R3": "final-ft-r3-song",
    "R4a": "final-r4a-full-15m",
    "R4 final": "final-r4a-ft-pred",
    "R5 SFT": "final-r5-sft",
    "R5 RL v2": "final-r5-rl2",
}
SET_C_SONGS = {"taiko1000_test-00004-Crystal_Hail": "Crystal_Hail"}


def charts(path: Path):
    cs = get("tja").read_path(path)
    for c in cs.charts:
        label = (c.difficulty.label or "").lower()
        if c.events and c.mechanics == "taiko" and label in COURSES:
            yield cs, c, label


def collect(params: dict) -> list[dict]:
    rows = []
    for song in sorted(SET_A.iterdir()):
        for name, f in SET_A_FILES.items():
            if (song / f).exists():
                for cs, c, label in charts(song / f):
                    rows.append(
                        {
                            "set": "A",
                            "song": song.name,
                            "source": name,
                            "course": label,
                            **monotony(cs, c, **params),
                        }
                    )
    for song in sorted(SET_B.iterdir()):
        for name, f in SET_B_FILES.items():
            if (song / f).exists():
                for cs, c, label in charts(song / f):
                    rows.append(
                        {
                            "set": "B",
                            "song": song.name,
                            "source": name,
                            "course": label,
                            **monotony(cs, c, **params),
                        }
                    )
    for name, run in SET_C_RUNS.items():
        for f in sorted((SET_C / run).glob("*-s[0-9].tja")):
            for cs, c, label in charts(f):
                rows.append(
                    {
                        "set": "C",
                        "song": f.name,
                        "source": name,
                        "course": label,
                        **monotony(cs, c, **params),
                    }
                )
    return rows


def table(rows: list[dict], set_name: str, courses=COURSES) -> str:
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["set"] == set_name and r["monotone_share"] is not None:
            groups[r["source"]][r["course"]].append(r)
    present = [c for c in courses if any(c in g for g in groups.values())]
    lines = ["| source | " + " | ".join(present) + " |", "|---" * (len(present) + 1) + "|"]
    for source, by in groups.items():
        cells = []
        for c in present:
            v = by.get(c, [])
            if not v:
                cells.append("–")
                continue
            share = np.mean([r["monotone_share"] for r in v])
            longest = np.mean([r["longest_monotone_s"] for r in v])
            cells.append(f"{share:.0%} / {longest:.0f}s")
        lines.append(f"| {source} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/r4/monotony")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect({})
    rhythm = collect({"rhythm_only": True})
    (out / "rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (out / "rows_rhythm.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rhythm)
    )
    report = []
    for s, title in (
        ("A", "Set A: 6 Taiko-1000 test songs, human timing (R1-R3)"),
        ("B", "Set B: 4 benchmark songs, all courses (R4-R5)"),
        ("C", "Set C: benchmark oni/edit, 3 seeds, audio timing"),
    ):
        report += [
            f"## {title}",
            "cells: monotone share / mean longest monotone stretch",
            "",
            "Rhythm and colour:",
            "",
            table(rows, s),
            "",
            "Rhythm only (colours ignored):",
            "",
            table(rhythm, s),
            "",
        ]
    # sensitivity: window length and match threshold, human vs generated, sets A+B oni/edit
    report += [
        "## Sensitivity (sets A+B, hard/oni/edit; human vs all generated)",
        "",
        "| window | match | human share | generated share | human longest | generated longest |",
        "|---|---|---|---|---|---|",
    ]
    for window in (5.0, 10.0, 15.0):
        for match in (0.85, 0.9, 0.95):
            rs = [
                r
                for r in collect({"window_s": window, "match": match})
                if r["set"] in ("A", "B")
                and r["course"] in ("hard", "oni", "edit")
                and r["monotone_share"] is not None
            ]
            h = [r for r in rs if r["source"] == "human"]
            g = [r for r in rs if r["source"] != "human"]
            report.append(
                f"| {window:.0f}s | {match} | {np.mean([r['monotone_share'] for r in h]):.0%} | "
                f"{np.mean([r['monotone_share'] for r in g]):.0%} | "
                f"{np.mean([r['longest_monotone_s'] for r in h]):.0f}s | "
                f"{np.mean([r['longest_monotone_s'] for r in g]):.0f}s |"
            )
    text = "\n".join(report) + "\n"
    (out / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
