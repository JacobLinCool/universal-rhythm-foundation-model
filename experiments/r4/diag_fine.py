"""Why do generated charts avoid very fast notes (intervals under ~60 ms)? Validation only.

    python diag_fine.py --ckpt /content/runs/r4a-ft-pred/best.pt --out /content/runs/r4/diag-fine

1. **Data**: share of inter-onset intervals under 60 / 45 ms in training charts (Taiko-1000 and
   osu!taiko), and how many charts have at least 5% of them.
2. **Model belief** (teacher forcing on validation charts that use fast notes): for each event
   that follows its predecessor within 60 ms, the probability of its position tokens, and
   whether its ``POS_SUB`` would survive nucleus sampling (temperature 0.85, top-p 0.95).
3. **Decoding**: generate those charts with top-p 0.95 and 1.0 (human timing) and compare the
   share of fast intervals with the human charts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgspec
import numpy as np
import pyarrow.parquet as pq
import torch

from urfm.data.canonical import canonicalize
from urfm.data.features import MEL
from urfm.eval.protocol import request
from urfm.ir.serialize import from_msgpack
from urfm.tokenize import build_tokenizer
from urfm.tokenize.tokenizer import playable_events, window_start_beat
from urfm.tokenize.vocab import SUB_GRID
from urfm.train.data import Features, collate, extreme_tempo, make_sample, prepare, shard_files
from urfm.train.generate import generate
from urfm.train.train import load_checkpoint

FAST_MS = 60.0
GAMES = ("taiko", "osu_taiko")


def intervals_ms(cs, chart) -> np.ndarray:
    t = cs.timing_of(chart)
    events = playable_events(chart)
    if len(events) < 2 or extreme_tempo(t):  # same rule as training
        return np.zeros(0)
    us = np.unique(
        np.round(t.us_array([e.time.beat for e in events], [e.time.residual_us for e in events]))
    )
    return np.diff(us) / 1000


def data_stats(root: str, max_files: int) -> dict:
    out: dict[str, dict] = {}
    for source, games in (("taiko1000", ("taiko",)), ("osu", ("osu_taiko",))):
        fast, fast45, total, charts, heavy = 0, 0, 0, 0, 0
        for f in shard_files(root, [source], "train")[:max_files]:
            for blob in pq.read_table(f, columns=["chartset"])["chartset"].to_pylist():
                cs = from_msgpack(blob)
                for c in cs.charts:
                    if c.mechanics not in games or not c.events:
                        continue
                    ms = intervals_ms(cs, c)
                    if not len(ms):
                        continue
                    charts += 1
                    total += len(ms)
                    fast += int((ms < FAST_MS).sum())
                    fast45 += int((ms < 45).sum())
                    heavy += float((ms < FAST_MS).mean()) >= 0.05
        out[source] = {
            "charts": charts,
            "share_lt60": fast / max(total, 1),
            "share_lt45": fast45 / max(total, 1),
            "charts_ge5pct_lt60": heavy / max(charts, 1),
        }
    return out


def fast_charts(root: str, limit: int):
    """Validation charts (both games) with at least 5% of intervals under 60 ms."""
    items = []
    for f in shard_files(root, ["taiko1000", "osu"], "validation"):
        t = pq.read_table(f, columns=["mel", "chartset"])
        for mel_b, blob in zip(t["mel"].to_pylist(), t["chartset"].to_pylist(), strict=True):
            cs = from_msgpack(blob)
            for c in cs.charts:
                if c.mechanics in GAMES and c.events:
                    ms = intervals_ms(cs, c)
                    if len(ms) and (ms < FAST_MS).mean() >= 0.05:
                        mel = np.frombuffer(mel_b, np.uint8).reshape(-1, MEL.n_mels)
                        items.append((cs, c, mel))
            if len(items) >= limit:
                return items
    return items


@torch.no_grad()
def belief(model, meta, tok, items, device) -> dict:
    v = tok.vocab
    feats = Features(cond=bool(model.cfg.cond_dim), texture=bool(model.cfg.song_feat_dim))
    probs_beat, probs_sub, in_nucleus, n_slow, slow_in = [], [], [], 0, 0
    for cs, chart, mel in items:
        cs1, ch = canonicalize(msgspec.structs.replace(cs, charts=(chart,)), chart)
        cs1 = msgspec.structs.replace(cs1, charts=(ch,))
        prep = prepare(tok, cs1, ch, mel.shape[0], None, feats)
        timing = cs1.timing_of(ch)
        for w in prep.windows:
            s = make_sample(tok, cs1, prep, mel, w, model.cfg.max_dec_len, feats)
            if (s.tokens >= model.cfg.vocab_size).any():
                continue
            b = {k: x.to(device) for k, x in collate([s], tok.pad_id).items() if k != "stats"}
            logits = model(
                b["mel"],
                b["feats"],
                b["tokens"][:, :-1],
                b.get("song"),
                b.get("song_marker"),
                b.get("song_len"),
                b.get("cond"),
                b.get("cond_mask"),
            )[0].float()
            toks = s.tokens.tolist()
            start_beat = window_start_beat(timing, w[0])
            prev_us = None
            for i in range(s.sep + 1, len(toks) - 1):
                if v.kinds[toks[i]] != "POS_BEAT" or v.kinds[toks[i + 1]] != "POS_SUB":
                    continue
                beat = start_beat + v.values[toks[i]] + SUB_GRID[v.values[toks[i + 1]]]
                us = float(timing.beat_to_us_exact(beat))
                fast = prev_us is not None and (us - prev_us) / 1000 < FAST_MS
                prev_us = us
                p_beat = torch.softmax(logits[i - 1], -1)[toks[i]].item()
                dist = torch.softmax(logits[i] / 0.85, -1)
                p_sub = dist[toks[i + 1]].item()
                sorted_p, _ = dist.sort(descending=True)
                cum_before = sorted_p[sorted_p > p_sub].sum().item()
                kept = cum_before < 0.95
                if fast:
                    probs_beat.append(p_beat)
                    probs_sub.append(torch.softmax(logits[i], -1)[toks[i + 1]].item())
                    in_nucleus.append(kept)
                else:
                    n_slow += 1
                    slow_in += kept
    return {
        "fast_events": len(probs_sub),
        "fast_p_beat_mean": float(np.mean(probs_beat)) if probs_beat else None,
        "fast_p_sub_mean": float(np.mean(probs_sub)) if probs_sub else None,
        "fast_p_sub_median": float(np.median(probs_sub)) if probs_sub else None,
        "fast_sub_in_nucleus": float(np.mean(in_nucleus)) if in_nucleus else None,
        "other_sub_in_nucleus": slow_in / max(n_slow, 1),
    }


def decoding(model, meta, tok, items, device) -> dict:
    out = {}
    ref = np.concatenate([intervals_ms(cs, c) for cs, c, _ in items])
    out["human_share_lt60"] = float((ref < FAST_MS).mean())
    for top_p in (0.95, 1.0):
        jobs = [request(model, meta, tok, cs, c, mel, cs.timing_of(c)) for cs, c, mel in items]
        gen = generate(
            model,
            tok,
            jobs,
            temperature=0.85,
            top_p=top_p,
            device=device,
            seed=0,
            plan="calibrated",
        )
        ms = np.concatenate(
            [
                intervals_ms(msgspec.structs.replace(cs, charts=(g,)), g)
                for (cs, _c, _m), g in zip(items, gen, strict=True)
            ]
        )
        n_gen = sum(len(g.events) for g in gen)
        n_ref = sum(len(c.events) for _cs, c, _m in items)
        out[f"top_p={top_p}"] = {
            "share_lt60": float((ms < FAST_MS).mean()) if len(ms) else 0.0,
            "share_lt45": float((ms < 45).mean()) if len(ms) else 0.0,
            "notes_ratio": n_gen / max(n_ref, 1),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--charts", type=int, default=40)
    ap.add_argument("--data-files", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.ckpt, device)
    tok = build_tokenizer()
    report = {"data": data_stats(args.data, args.data_files)}
    print(json.dumps(report, indent=1), flush=True)
    items = fast_charts(args.data, args.charts)
    report["validation_fast_charts"] = {
        g: sum(1 for _cs, c, _m in items if c.mechanics == g) for g in GAMES
    }
    report["belief"] = belief(model, meta, tok, items, device)
    print(json.dumps(report["belief"], indent=1), flush=True)
    report["decoding"] = decoding(model, meta, tok, items, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
