"""Which extra section-plan values are worth planning? (R6.1, human data only)

    python analysis_plan2.py --out /content/runs/r6/analysis_plan2.json

Candidates per R6 section (``urfm.tokenize.sections``), Taiko-1000 hard/oni/edit:

- ``peak``: most onsets in any 2 s span of the section, per second;
- ``repeat``: highest cosine similarity of the section's onset grid (don/ka rows) to an earlier
  section at least 2 back (1 = an exact repeat);
- ``variety``: distinct inter-onset intervals (to 1/48 beat) in the section;
- ``switch``: share of consecutive onsets that change colour (don <-> ka);
- ``long``: share of the section covered by drumrolls and balloons.

Kept only if (1) two human courses of the same song agree across sections (test songs, mean
per-pair r > 0.3) and (2) audio predicts it (ridge from the planner's section descriptors,
within-song z-scores, trained on train songs, mean per-chart r on test songs > 0.2; for
``repeat`` the predictor is the audio's own max earlier-section similarity).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import msgspec
import numpy as np
import pyarrow.parquet as pq

from urfm.data.canonical import canonicalize
from urfm.data.features import MEL
from urfm.data.sections import section_audio, similarity
from urfm.ir.events import Primitive
from urfm.ir.serialize import from_msgpack
from urfm.tokenize.sections import section_beats
from urfm.tokenize.tokenizer import beats_at_us, playable_events
from urfm.train.data import extreme_tempo, shard_files

NAMES = ("peak", "repeat", "variety", "switch", "long")
COURSES = ("hard", "oni", "edit")
PAIRS = (("hard", "oni"), ("oni", "edit"))
SUBS = 12


def chart_sections(cs, chart, mel):
    """``(first, values [S, 5], audio feats [S, F], audio max earlier similarity [S])``."""
    timing = cs.timing_of(chart)
    k = section_beats(timing)
    first, feats = section_audio(mel, timing, k)
    n = len(feats)
    events = playable_events(chart)
    if len(events) < 16 or n < 8:
        return None
    us = np.asarray(
        timing.us_array([e.time.beat for e in events], [e.time.residual_us for e in events]),
        np.float64,
    )
    beats = beats_at_us(timing, us)
    ka = np.array([e.role == "taiko:ka" for e in events])
    long_spans = [
        (float(e.time.beat), float(e.end.beat))
        for e in chart.events
        if e.primitive in (Primitive.HOLD, Primitive.REPEAT) and e.end is not None
    ]
    grid = np.zeros((n, 2, k * SUBS), np.float32)
    vals = np.zeros((n, len(NAMES)))
    sec_us = np.asarray(timing.us_array(first * k + k * np.arange(n + 1, dtype=np.float64)))
    for i in range(n):
        a, b = (first + i) * k, (first + i + 1) * k
        sel = (beats >= a - 1e-6) & (beats < b - 1e-6)
        sb, su, sk = beats[sel], us[sel], ka[sel]
        dur = (sec_us[i + 1] - sec_us[i]) / 1e6
        if len(su):
            j = np.searchsorted(su, su + 2e6)
            vals[i, 0] = float((j - np.arange(len(su))).max()) / min(2.0, max(dur, 1e-3))
        gaps = np.round(np.diff(sb) * 48)
        vals[i, 2] = len(np.unique(gaps)) if len(gaps) else 0
        vals[i, 3] = float((sk[1:] != sk[:-1]).mean()) if len(sk) > 1 else 0.0
        covered = sum(max(0.0, min(e, b) - max(s, a)) for s, e in long_spans)
        vals[i, 4] = covered / k
        pos = np.rint((sb - a) * SUBS).astype(np.int64)
        ok = (pos >= 0) & (pos < k * SUBS)
        grid[i, sk[ok].astype(np.int64), pos[ok]] = 1.0
    g = grid.reshape(n, -1)
    norm = np.linalg.norm(g, axis=1)
    csim = (g @ g.T) / np.maximum(norm[:, None] * norm[None], 1e-12)
    asim = similarity(feats)
    amax = np.zeros(n)
    for i in range(n):
        if i >= 2:
            vals[i, 1] = csim[i, : i - 1].max() if norm[i] > 0 else 0.0
            amax[i] = asim[i, : i - 1].max()
    return first, vals, feats, amax


def work(args):
    file, split = args
    sections, pairs = [], defaultdict(list)
    t = pq.read_table(file, columns=["mel", "chartset"])
    for mel_b, blob in zip(t["mel"].to_pylist(), t["chartset"].to_pylist(), strict=True):
        cs = from_msgpack(blob)
        mel = np.frombuffer(mel_b, np.uint8).reshape(-1, MEL.n_mels)
        by = {}
        for chart in cs.charts:
            label = (chart.difficulty.label or "").lower()
            if chart.mechanics != "taiko" or label not in COURSES or not chart.events:
                continue
            if extreme_tempo(cs.timing_of(chart)):
                continue
            c1, ch = canonicalize(msgspec.structs.replace(cs, charts=(chart,)), chart)
            c1 = msgspec.structs.replace(c1, charts=(ch,))
            r = chart_sections(c1, ch, mel)
            if r is None:
                continue
            by[label] = r
            sections.append((label, r[1].astype(np.float32), r[2], r[3].astype(np.float32)))
        if split == "test":
            for lo, hi in PAIRS:
                if lo in by and hi in by and by[lo][0] == by[hi][0]:
                    va, vb = by[lo][1], by[hi][1]
                    n = min(len(va), len(vb))
                    for j, name in enumerate(NAMES):
                        x, y = va[:n, j], vb[:n, j]
                        if x.std() > 0 and y.std() > 0:
                            pairs[f"{lo}-{hi}/{name}"].append(float(np.corrcoef(x, y)[0, 1]))
    return {"split": split, "sections": sections, "pairs": dict(pairs)}


def _z(v):
    return (v - v.mean(0)) / (v.std(0) + 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = [(f, s) for s in ("train", "test") for f in shard_files(args.data, ["taiko1000"], s)]
    with ProcessPoolExecutor(8) as pool:
        parts = list(pool.map(work, jobs))
    out: dict = {"course_agreement": {}, "audio_r": {}, "human_quantiles": {}}
    pairs = defaultdict(list)
    for p in parts:
        for k, v in p["pairs"].items():
            pairs[k] += v
    for k, v in sorted(pairs.items()):
        out["course_agreement"][k] = {"pairs": len(v), "mean": float(np.mean(v))}
    train = [s for p in parts if p["split"] == "train" for s in p["sections"]]
    test = [s for p in parts if p["split"] == "test" for s in p["sections"]]
    xs = np.concatenate([_z(f[:, :-2]) for _l, _v, f, _a in train])
    ys = np.concatenate([_z(v) for _l, v, _f, _a in train])
    x1 = np.concatenate([xs, np.ones((len(xs), 1))], 1)
    w = np.linalg.solve(x1.T @ x1 + 10 * np.eye(x1.shape[1]), x1.T @ ys)
    rs = defaultdict(list)
    for _l, v, f, a in test:
        pred = np.concatenate([_z(f[:, :-2]), np.ones((len(f), 1))], 1) @ w
        for j, name in enumerate(NAMES):
            y = v[:, j]
            p_ = a if name == "repeat" else pred[:, j]
            if y.std() > 0 and p_.std() > 0:
                rs[name].append(float(np.corrcoef(p_, y)[0, 1]))
    out["audio_r"] = {k: float(np.mean(v)) for k, v in rs.items()}
    allv = np.concatenate([v for _l, v, _f, _a in train])
    for j, name in enumerate(NAMES):
        out["human_quantiles"][name] = [
            float(np.quantile(allv[:, j], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.97)
        ]
    out["train_sections"], out["test_charts"] = len(allv), len(test)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
