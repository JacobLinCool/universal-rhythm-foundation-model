"""Human-data analysis that fixes the R6 design choices (no model involved).

    python analysis.py --out /content/runs/r6/analysis.json

1. **Subdivisions**: share of onsets needing each subdivision (per beat: 1, 2, 3, 4, 6, 8, 12,
   16, 24, 48, tuplets 5/7, other) on the chart's canonical grid and on the audio-predicted
   grid (`urfm-timing-v1`), per game; tempo ratio predicted / canonical.
2. **Section contrast**: agreement of two human courses of one song across 8-beat sections
   (the ceiling for section r), on test songs.
3. **Audio predictability of sections**: ridge regression from per-section audio descriptors
   (within-song z-scores) to section density / fine / rest share (within-chart z-scores),
   trained on train songs, scored as per-chart correlation on test songs.
4. **Structure alignment** (`urfm.eval.structure`) of human charts vs the circular-shift null.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import msgspec
import numpy as np
import pyarrow.parquet as pq

from urfm.beat.decode import for_game
from urfm.data.canonical import canonicalize
from urfm.data.features import MEL
from urfm.eval.metrics import SECTION_BEATS, rhythm_in_reference_beats
from urfm.eval.structure import alignment, audio_descriptors
from urfm.ir.serialize import from_msgpack
from urfm.tokenize.tokenizer import beats_at_us, playable_events
from urfm.train.data import PredictedTimings, extreme_tempo, shard_files

GAMES = ("taiko", "osu_taiko")
DENOMS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 5, 7, 10, 14)
COURSES = ("easy", "normal", "hard", "oni", "edit")
PAIRS = (("normal", "hard"), ("hard", "oni"), ("oni", "edit"))
PRED_TOL_MS = 8.0
PREDICTED: PredictedTimings | None = None


def subdivision(beats: np.ndarray, beat_ms: np.ndarray, tol_ms: float | None) -> list[str]:
    """Smallest subdivision per beat that holds each onset (exact, or within ``tol_ms``)."""
    out = []
    for b, ms in zip(beats, beat_ms, strict=True):
        name = "other"
        for d in DENOMS:
            err = abs(b * d - round(b * d)) / d
            # on a predicted grid, a quarter of the candidate spacing at most, so a fine grid
            # cannot absorb every slightly-off onset
            tol = None if tol_ms is None else min(tol_ms, ms / d / 4)
            if (err * ms <= tol) if tol is not None else err < 1e-6:
                name = str(d)
                break
        out.append(name)
    return out


def onset_beats(cs, chart, timing) -> tuple[np.ndarray, np.ndarray]:
    events = playable_events(chart)
    us = np.unique(
        np.asarray(
            cs.timing_of(chart).us_array(
                [e.time.beat for e in events], [e.time.residual_us for e in events]
            ),
            np.float64,
        )
    )
    beats = beats_at_us(timing, us)
    eps = 1e-3
    beat_ms = (timing.us_array(beats + eps) - timing.us_array(beats)) / eps / 1000
    return beats, np.asarray(beat_ms)


def median_bpm(timing) -> float:
    return float(np.median([float(t.bpm) for t in timing.tempos]))


def section_features(cs, chart, mel):
    """Per 8-beat section of the canonical grid: (audio descriptor, density, fine, rest)."""
    timing = cs.timing_of(chart)
    beats, _ = onset_beats(cs, chart, timing)
    if len(beats) < 16:
        return None
    first = float(np.floor(beats[0] / SECTION_BEATS) * SECTION_BEATS)
    n = int(np.ceil((beats[-1] + 1e-6 - first) / SECTION_BEATS))
    if n < 8:
        return None
    a = audio_descriptors(mel, timing, SECTION_BEATS, n, first)
    flux = a[:, :-32].reshape(n, -1, 4)  # positions x bands (positions are 12 per beat)
    on = flux[:, ::3].mean(1)  # 16th positions
    off = flux[:, 1::3].mean(1) + flux[:, 2::3].mean(1)  # between 16ths (triplets, 32nds)
    beat_pos = flux[:, ::12].mean(1)
    x = np.concatenate([flux.mean(1), on, off, beat_pos, flux.max(1), a[:, -32:]], axis=1)
    edges = first + SECTION_BEATS * np.arange(n + 1)
    secs = np.diff(timing.us_array(edges)) / 1e6
    y = []
    for i in range(n):
        sel = beats[(beats >= edges[i]) & (beats < edges[i + 1])]
        gaps = np.diff(np.concatenate([[edges[i]], sel, [edges[i + 1]]])).astype(np.float64)
        inner = np.diff(sel)
        y.append(
            (
                np.log1p(len(sel) / max(secs[i], 1e-3)),
                float((inner <= 0.25 + 1 / 48).mean()) if len(inner) else 0.0,
                float(gaps[gaps >= 1].sum() / SECTION_BEATS),
            )
        )
    return x, np.array(y)


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean(0)) / (v.std(0) + 1e-6)


def work(args: tuple[str, str]) -> dict:
    file, split = args
    sub: dict[str, Counter] = defaultdict(Counter)
    ratio: Counter = Counter()
    pairs: dict[str, list[float]] = defaultdict(list)
    sections: list[tuple[np.ndarray, np.ndarray, str]] = []
    struct: dict[str, list[tuple[float, float]]] = defaultdict(list)
    pf = pq.ParquetFile(file)
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=["mel", "chartset"])
        for row, (mel_b, blob) in enumerate(
            zip(t["mel"].to_pylist(), t["chartset"].to_pylist(), strict=True)
        ):
            cs = from_msgpack(blob)
            mel = np.frombuffer(mel_b, np.uint8).reshape(-1, MEL.n_mels)
            pred = PREDICTED.get(file, g, row) if PREDICTED else None
            by_course = {}
            for chart in cs.charts:
                if chart.mechanics not in GAMES or not chart.events:
                    continue
                if extreme_tempo(cs.timing_of(chart)):
                    continue
                one = msgspec.structs.replace(cs, charts=(chart,))
                c1, ch = canonicalize(one, chart)
                c1 = msgspec.structs.replace(c1, charts=(ch,))
                timing = c1.timing_of(ch)
                game = ch.mechanics
                beats, beat_ms = onset_beats(c1, ch, timing)
                sub[f"{split}/{game}/canonical"].update(subdivision(beats, beat_ms, None))
                if pred is not None:
                    p = for_game(pred, game)
                    pb, pms = onset_beats(c1, ch, p)
                    sub[f"{split}/{game}/predicted"].update(subdivision(pb, pms, PRED_TOL_MS))
                    r = median_bpm(p) / median_bpm(timing)
                    ratio[
                        f"{split}/{game}/"
                        + next(
                            (
                                k
                                for k, v in (
                                    ("x0.5", 0.5),
                                    ("x1", 1.0),
                                    ("x2", 2.0),
                                    ("x0.67", 2 / 3),
                                    ("x1.5", 1.5),
                                    ("x0.25", 0.25),
                                    ("x4", 4.0),
                                )
                                if abs(np.log2(r / v)) < 0.06
                            ),
                            "other",
                        )
                    ] += 1
                label = (ch.difficulty.label or "").lower()
                if game == "taiko" and label in COURSES:
                    by_course[label] = (c1, ch)
                    if label in ("hard", "oni", "edit"):
                        s = section_features(c1, ch, mel)
                        if s is not None:
                            sections.append((s[0], s[1], label))
                        if split == "test":
                            a = alignment(c1, ch, mel)
                            if a["score"] is not None and a["null"] is not None:
                                struct[label].append((a["score"], a["null"]))
            if split == "test":
                for lo, hi in PAIRS:
                    if lo in by_course and hi in by_course:
                        (ca, a), (cb, b) = by_course[lo], by_course[hi]
                        m = rhythm_in_reference_beats(ca, a, cb, b)
                        for k in ("density", "fine", "rest", "ka"):
                            v = m.get(f"section_r_{k}")
                            if v is not None:
                                pairs[f"{lo}-{hi}/{k}"].append(v)
    return {
        "sub": {k: dict(v) for k, v in sub.items()},
        "ratio": dict(ratio),
        "pairs": dict(pairs),
        "sections": [(x.astype(np.float32), y.astype(np.float32), lab) for x, y, lab in sections],
        "struct": dict(struct),
        "split": split,
    }


def init(pred_root: str | None) -> None:
    global PREDICTED
    PREDICTED = PredictedTimings(pred_root) if pred_root else None


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 10.0) -> np.ndarray:
    x1 = np.concatenate([x, np.ones((len(x), 1))], 1)
    return np.linalg.solve(x1.T @ x1 + lam * np.eye(x1.shape[1]), x1.T @ y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--predicted", default="/content/data/urfm-timing-v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [
        (f, split)
        for split in ("train", "test")
        for f in shard_files(args.data, ["taiko1000", "osu"], split)
    ]
    pred = args.predicted if Path(args.predicted).is_dir() else None
    with ProcessPoolExecutor(8, initializer=init, initargs=(pred,)) as pool:
        parts = list(pool.map(work, jobs))

    out: dict = {"subdivision_share": {}, "tempo_ratio_predicted_over_canonical": {}}
    sub: dict[str, Counter] = defaultdict(Counter)
    ratio: Counter = Counter()
    for p in parts:
        for k, v in p["sub"].items():
            sub[k].update(v)
        ratio.update(p["ratio"])
    for k, c in sorted(sub.items()):
        n = sum(c.values())
        out["subdivision_share"][k] = {"onsets": n} | {
            d: round(c[d] / n, 5) for d in [*map(str, DENOMS), "other"] if c[d]
        }
    groups = sorted({k.rsplit("/", 1)[0] for k in ratio})
    for g in groups:
        n = sum(v for k, v in ratio.items() if k.startswith(g + "/"))
        out["tempo_ratio_predicted_over_canonical"][g] = {"charts": n} | {
            k.rsplit("/", 1)[1]: round(v / n, 4) for k, v in ratio.items() if k.startswith(g + "/")
        }

    pairs: dict[str, list[float]] = defaultdict(list)
    for p in parts:
        for k, v in p["pairs"].items():
            pairs[k] += v
    out["course_agreement_section_r"] = {
        k: {"pairs": len(v), "mean": float(np.mean(v)), "median": float(np.median(v))}
        for k, v in sorted(pairs.items())
    }

    train = [s for p in parts if p["split"] == "train" for s in p["sections"]]
    test = [s for p in parts if p["split"] == "test" for s in p["sections"]]
    xs = np.concatenate([_z(x) for x, _y, _l in train])
    ys = np.concatenate([_z(y) for _x, y, _l in train])
    w = ridge(xs, ys)
    pred_r: dict[str, list[float]] = defaultdict(list)
    for x, y, label in test:
        yh = np.concatenate([_z(x), np.ones((len(x), 1))], 1) @ w
        for i, k in enumerate(("density", "fine", "rest")):
            if y[:, i].std() > 0:
                pred_r[f"{label}/{k}"].append(float(np.corrcoef(yh[:, i], y[:, i])[0, 1]))
                pred_r[f"all/{k}"].append(pred_r[f"{label}/{k}"][-1])
    out["audio_linear_section_r"] = {
        "train_sections": len(xs),
        "test_charts": len(test),
    } | {k: float(np.mean(v)) for k, v in sorted(pred_r.items())}

    struct: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for p in parts:
        for k, v in p["struct"].items():
            struct[k] += v
    out["structure_alignment_human_test"] = {
        k: {
            "charts": len(v),
            "score_mean": float(np.mean([a for a, _ in v])),
            "score_q10": float(np.quantile([a for a, _ in v], 0.1)),
            "null_mean": float(np.mean([b for _, b in v])),
            "share_above_null": float(np.mean([a > b for a, b in v])),
        }
        for k, v in sorted(struct.items())
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
