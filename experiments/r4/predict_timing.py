"""Audio-predicted timing maps for every song (training with predicted grids, evaluation).

    python predict_timing.py --ckpt /content/runs/beat-v1/best.pt \\
        --out /content/data/urfm-timing-v1 --hub-repo JacobLinCool/urfm-timing-v1

Writes one Parquet file per shard with ``song_id``, the TimingMap (msgpack) and ``grid_fit``:
the share of the representative human chart's notes on 1/4 or 1/3 beats of the predicted map
(training only uses songs whose fit is high).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from urfm.beat.decode import activations, decode_beats, to_timing
from urfm.beat.labels import representative
from urfm.beat.metrics import grid_fit
from urfm.beat.train import load
from urfm.data.canonical import canonicalize
from urfm.data.features import MEL
from urfm.ir.serialize import from_msgpack, timing_to_msgpack
from urfm.tokenize.tokenizer import playable_events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/content/data/urfm-ingest-v0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hub-repo", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.ckpt, device)
    root = Path(args.data) / "data"
    out = Path(args.out)
    t0, n = time.time(), 0
    for file in sorted(root.glob("*/*.parquet")):
        dest = out / "data" / file.parent.name / file.name
        if dest.exists():
            continue
        pf = pq.ParquetFile(file)
        rows: list[dict] = []
        for g in range(pf.num_row_groups):
            table = pf.read_row_group(g, columns=["song_id", "mel", "chartset"])
            for r, (sid, mel_b, blob) in enumerate(
                zip(*(table[c].to_pylist() for c in ("song_id", "mel", "chartset")), strict=True)
            ):
                mel = np.frombuffer(mel_b, np.uint8).reshape(-1, MEL.n_mels)
                act = activations(model, mel, device)
                timing = to_timing(decode_beats(act[:, 0]), act[:, 1])
                fit = None
                cs = from_msgpack(blob)
                chart = representative(cs)
                if chart is not None:
                    cs, chart = canonicalize(cs, chart)
                    human = cs.timing_of(chart)
                    onsets = np.unique([human.us(e.time) / 1e6 for e in playable_events(chart)])
                    fit = grid_fit(timing, onsets) if len(onsets) else None
                rows.append(
                    {
                        "song_id": sid,
                        "row_group": g,
                        "row": r,
                        "timing": timing_to_msgpack(timing),
                        "grid_fit": fit,
                        "tempos": len(timing.tempos),
                    }
                )
                n += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), dest)
        print(
            f"{file.parent.name}/{file.name}: {len(rows)} songs, total {n}, "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    if args.hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.hub_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=args.hub_repo, repo_type="dataset", folder_path=str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
