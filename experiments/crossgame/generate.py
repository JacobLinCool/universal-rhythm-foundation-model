"""Cross-game generation (quick Exp C): one song, many games, one model.

For each held-out song the model generates a chart for every target game from the same
audio and timing. Native files are written for listening/playing, and a consistency table
compares the games' onset-density curves (2 s bins) with each other and with the human
reference charts of the song:

    python generate.py --ckpt /content/runs/v3-all/best.pt --out /content/runs/v3-all/crossgame

High correlation between generated games means the shared musical-interaction structure
(where the music asks for more or less action) survives across mechanics.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from urfm.data.features import MEL
from urfm.ir.chart import Chart, Difficulty, SongMeta
from urfm.ir.serialize import from_msgpack
from urfm.ir.timing import TimingMap
from urfm.ir.views import active_events
from urfm.tokenize import build_tokenizer
from urfm.tokenize.tokenizer import onset_rate
from urfm.train.data import shard_files
from urfm.train.generate import Job, generate, to_chartset, write_native
from urfm.train.tiers import load_tiers
from urfm.train.train import load_checkpoint

TARGETS = {  # same per-game percentile tier for every game (see urfm.train.tiers)
    "taiko": {"tier": 3},
    "osu_taiko": {"tier": 3},
    "osu_mania": {"tier": 3, "lanes": 4},
    "osu_catch": {"tier": 3},
    "osu_std": {"tier": 3},
}
EXT = {"taiko": ".tja"}
BIN_S = 2.0


def density_curve(chart: Chart, timing: TimingMap, duration_s: float) -> np.ndarray:
    bins = np.zeros(max(int(duration_s / BIN_S), 1))
    for ev in active_events(chart):
        i = int(timing.seconds(ev.time) / BIN_S)
        if 0 <= i < len(bins):
            bins[i] += 1
    return bins


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 4 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--sources", nargs="+", default=["taiko1000"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--songs", type=int, default=24)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_checkpoint(args.ckpt, device)
    tiers = load_tiers(meta)
    tok = build_tokenizer()

    songs = []  # (song_id, title, mel, timing, duration_s, reference charts)
    for file in shard_files(args.data, args.sources, args.split):
        table = pq.read_table(file, columns=["song_id", "title", "mel", "chartset", "duration_s"])
        for sid, title, mel, blob, dur in zip(
            *(table[c].to_pylist() for c in ("song_id", "title", "mel", "chartset", "duration_s")),
            strict=True,
        ):
            cs = from_msgpack(blob)
            refs = [c for c in cs.charts if c.events]
            if not refs:
                continue
            mel_arr = np.frombuffer(mel, np.uint8).reshape(-1, MEL.n_mels)
            songs.append((sid, title, mel_arr, cs.timing_of(refs[0]), dur, cs, refs))
            if len(songs) >= args.songs:
                break
        if len(songs) >= args.songs:
            break

    jobs, keys = [], []
    for sid, _title, mel, timing, _dur, _cs, _refs in songs:
        for game in args.targets:
            cfg = TARGETS[game]
            jobs.append(
                Job(
                    mel=mel,
                    timing=timing,
                    mechanics=game,
                    prefix=tok.conditioning(
                        game,
                        tier=cfg["tier"],
                        onsets_per_s=tiers.density(game, cfg["tier"]) if tiers else None,
                        lanes=cfg.get("lanes"),
                    ),
                    difficulty=Difficulty(label=f"urfm-{game}", tier=cfg["tier"]),
                    meta={"lanes": str(cfg["lanes"])} if "lanes" in cfg else {},
                    lanes=cfg.get("lanes"),
                )
            )
            keys.append((sid, game))
    charts = generate(model, tok, jobs, device=device, seed=0)

    by_song: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    info = {s[0]: s for s in songs}
    for (sid, game), chart in zip(keys, charts, strict=True):
        _, title, _mel, timing, dur, _cs, _refs = info[sid]
        by_song[sid][game] = density_curve(chart, timing, dur)
        ext = EXT.get(game, ".osu")
        try:
            write_native(
                to_chartset(chart, timing, SongMeta(title=title or sid)),
                out / f"{sid.replace(':', '_')}-{game}{ext}",
            )
        except Exception as exc:
            print(f"cannot write {sid} {game}: {exc}", flush=True)

    rows = []
    for sid, curves in by_song.items():
        _, _title, _mel, _timing, dur, cs, refs = info[sid]
        ref = max(refs, key=lambda c: onset_rate(cs, c))  # densest human chart of the song
        ref_curve = density_curve(ref, cs.timing_of(ref), dur)
        row: dict[str, object] = {"song_id": sid, "reference": ref.mechanics}
        for a, b in itertools.combinations(sorted(curves), 2):
            row[f"{a}~{b}"] = corr(curves[a], curves[b])
        for g, c in curves.items():
            row[f"{g}~human"] = corr(c, ref_curve)
        rows.append(row)
    (out / "consistency.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cols = sorted({k for r in rows for k in r if "~" in k})
    lines = ["| pair | mean r | songs |", "|---|---|---|"]
    for col in cols:
        vals = [r[col] for r in rows if isinstance(r.get(col), float)]
        if vals:
            lines.append(f"| {col} | {np.mean(vals):.3f} | {len(vals)} |")
    (out / "consistency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
