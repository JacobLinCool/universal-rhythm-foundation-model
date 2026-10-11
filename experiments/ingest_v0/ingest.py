"""URFM ingest v0: build per-song training records on a Colab VM and push them to the Hub.

Sources (audio + charts):
  taiko1000  JacobLinCool/taiko-1000-parsed-clean @ pinned revision, its own song-level split
  osu        project-riz/osu-beatmaps (compressed: 64 kbps mono Opus), streamed tar by tar;
             every song with a taiko/catch/mania map is kept, standard-only songs are sampled

Audio is decoded with FFmpeg and turned into uint8 log-mel (urfm.data.features) in worker
processes; charts are parsed into the IR by the urfm codecs. Records (urfm.data.records) are
written as zstd Parquet shards and uploaded to a private dataset repo.

    python ingest.py taiko1000
    python ingest.py osu --std-percent 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import traceback
import unicodedata
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from urfm.data.features import MEL, mel_transform, quantize
from urfm.data.records import SongRecord, arrow_schema
from urfm.ir.chart import ChartSet
from urfm.ir.serialize import to_msgpack

REPO = "JacobLinCool/urfm-ingest-v0"
OUT = Path("/content/data/urfm-ingest-v0")
TAIKO_REPO = "JacobLinCool/taiko-1000-parsed-clean"
TAIKO_REV = "b72da4616d643018e81f372cea06ce51349285e0"
OSU_REPO = "project-riz/osu-beatmaps"
SHARD_BYTES = 512 * 2**20
WORKERS = max((os.cpu_count() or 2) - 1, 1)

# -- worker side ---------------------------------------------------------------

_mel = None


def _decode(data: bytes) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            str(MEL.sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, np.float32)


def _mel_bytes(wav: np.ndarray) -> tuple[bytes, int]:
    global _mel
    import torch

    torch.set_num_threads(1)
    if _mel is None:
        _mel = mel_transform(device="cpu")
    with torch.no_grad():
        q = quantize(_mel(torch.from_numpy(wav.copy())))
    return q.numpy().tobytes(), int(q.shape[0])


def _record(meta: dict, audio: bytes, chartset: ChartSet) -> dict:
    wav = _decode(audio)
    mel, frames = _mel_bytes(wav)
    rec = SongRecord(
        **meta,
        duration_s=len(wav) / MEL.sample_rate,
        mel_version=MEL.version,
        mel_frames=frames,
        mel=mel,
        chartset=to_msgpack(chartset),
        mechanics=sorted({c.mechanics for c in chartset.charts}),
        n_charts=len(chartset.charts),
        n_events=sum(len(c.events) for c in chartset.charts),
    )
    return msgspec.structs.asdict(rec)


def taiko_job(meta: dict, audio: bytes, tja: str) -> dict:
    from urfm.codecs.tja import TjaCodec

    cs = TjaCodec().read(tja.encode("utf-8"), source=meta["song_id"])
    meta = meta | {"title": cs.song.title, "artist": cs.song.artist}
    return _record(meta, audio, cs)


def osu_job(meta: dict, audio: bytes, contents: list[str]) -> dict:
    from urfm.codecs.osu import OsuCodec

    codec = OsuCodec()
    sets = []
    for i, text in enumerate(contents):
        try:
            sets.append(codec.read(text.encode("utf-8"), source=f"{meta['song_id']}#{i}"))
        except Exception:
            continue
    if not sets:
        raise ValueError("no parsable beatmap")
    charts = tuple(msgspec.structs.replace(c, timing=cs.timing) for cs in sets for c in cs.charts)
    merged = msgspec.structs.replace(sets[0], charts=charts, provenance=None)
    return _record(meta, audio, merged)


# -- main side -------------------------------------------------------------------


def norm_title(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


class ShardWriter:
    def __init__(self, source: str) -> None:
        self.source = source
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.sizes: dict[str, int] = defaultdict(int)
        self.counts: dict[str, int] = defaultdict(int)
        self.dir = OUT / "data" / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest = (OUT / "manifest").joinpath(f"{source}.jsonl")
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_fh = self.manifest.open("a")

    def add(self, row: dict) -> None:
        split = row["split"]
        self.buffers[split].append(row)
        self.sizes[split] += len(row["mel"]) + len(row["chartset"])
        light = {k: v for k, v in row.items() if k not in ("mel", "chartset")}
        self.manifest_fh.write(json.dumps(light, ensure_ascii=False) + "\n")
        if self.sizes[split] >= SHARD_BYTES:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers.pop(split, [])
        self.sizes[split] = 0
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=arrow_schema())
        path = self.dir / f"{split}-{self.counts[split]:05d}.parquet"
        pq.write_table(table, path, compression="zstd", row_group_size=64)
        self.counts[split] += 1
        print(f"[{self.source}] wrote {path.name} ({len(rows)} songs)", flush=True)

    def close(self) -> None:
        for split in list(self.buffers):
            self.flush(split)
        self.manifest_fh.close()


def run_pool(jobs, writer: ShardWriter, errors_path: Path, max_inflight: int = 4 * WORKERS):
    """Submit ``(fn, meta, *args)`` jobs with bounded memory; write results as they finish."""
    t0, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(WORKERS) as pool, errors_path.open("a") as err:
        pending: dict[Future, dict] = {}

        def drain(block: bool) -> None:
            nonlocal done, failed
            finished, _ = (
                wait(pending, return_when=FIRST_COMPLETED)
                if block
                else ([f for f in pending if f.done()], None)
            )
            for fut in finished:
                meta = pending.pop(fut)
                try:
                    writer.add(fut.result())
                    done += 1
                except Exception as exc:
                    failed += 1
                    err.write(
                        json.dumps({"song_id": meta["song_id"], "error": repr(exc)[:500]}) + "\n"
                    )
                if (done + failed) % 200 == 0:
                    rate = (done + failed) / (time.time() - t0)
                    print(
                        f"[{writer.source}] {done} ok, {failed} failed, {rate:.1f} songs/s",
                        flush=True,
                    )

        for fn, meta, *args in jobs:
            while len(pending) >= max_inflight:
                drain(block=True)
            pending[pool.submit(fn, meta, *args)] = meta
        while pending:
            drain(block=True)
    writer.close()
    print(
        f"[{writer.source}] finished: {done} ok, {failed} failed in {time.time() - t0:.0f}s",
        flush=True,
    )
    return done, failed


def taiko_jobs():
    for split in ("train", "validation", "test"):
        files = [
            f
            for f in HfApi().list_repo_files(TAIKO_REPO, repo_type="dataset", revision=TAIKO_REV)
            if f.startswith(f"data/{split}-")
        ]
        index = 0
        for name in sorted(files):
            path = hf_hub_download(TAIKO_REPO, name, repo_type="dataset", revision=TAIKO_REV)
            table = pq.read_table(path, columns=["audio", "tja", "group_id", "audio_sha256"])
            for row in table.to_pylist():
                meta = {
                    "song_id": f"taiko1000:{split}-{index:05d}",
                    "source": "taiko1000",
                    "split": split,
                    "group": f"taiko1000:{row['group_id']}",
                    "audio_sha256": row["audio_sha256"],
                }
                index += 1
                yield taiko_job, meta, row["audio"]["bytes"], row["tja"]


def osu_jobs(std_percent: int, limit_tars: int | None):
    taiko_split = {}
    manifest = OUT / "manifest" / "taiko1000.jsonl"
    if manifest.exists():
        for line in manifest.open():
            row = json.loads(line)
            if row.get("title"):
                taiko_split[norm_title(row["title"])] = row["split"]
    print(f"[osu] {len(taiko_split)} taiko titles for cross-game split alignment", flush=True)

    files = sorted(
        f
        for f in HfApi().list_repo_files(OSU_REPO, repo_type="dataset")
        if f.startswith("compressed/") and f.endswith(".tar")
    )
    files = files[:limit_tars] if limit_tars else files
    print(f"[osu] {len(files)} tars", flush=True)
    downloads = ThreadPoolExecutor(3)
    futures = [
        downloads.submit(hf_hub_download, OSU_REPO, f, repo_type="dataset") for f in files[:3]
    ]
    next_file = 3
    kept = skipped = 0
    for i in range(len(files)):
        path = futures[i].result()
        if next_file < len(files):
            futures.append(
                downloads.submit(hf_hub_download, OSU_REPO, files[next_file], repo_type="dataset")
            )
            next_file += 1
        members: dict[str, dict[str, bytes]] = defaultdict(dict)
        with tarfile.open(path) as tar:
            for m in tar:
                if m.isfile():
                    key, _, ext = m.name.rpartition(".")
                    members[key][ext] = tar.extractfile(m).read()  # type: ignore[union-attr]
        for parts in members.values():
            if "json" not in parts:
                continue
            info = json.loads(parts["json"])
            audio = next((v for k, v in parts.items() if k != "json"), None)
            maps = info.get("beatmaps", [])
            modes = {int(b.get("mode", 0)) for b in maps}
            ahash = info.get("audio_hash") or hashlib.sha256(audio or b"").hexdigest()
            bucket = int(ahash[:8], 16) % 100
            if audio is None or not maps or (modes <= {0} and bucket >= std_percent):
                skipped += 1
                continue
            titles = {
                norm_title(str(b.get(f, ""))) for b in maps for f in ("title", "title_unicode")
            }
            split = next((taiko_split[t] for t in titles if t in taiko_split), None)
            split = split or (
                "test" if bucket % 50 == 0 else "validation" if bucket % 50 == 1 else "train"
            )
            meta = {
                "song_id": f"osu:{ahash[:16]}",
                "source": "osu",
                "split": split,
                "group": f"osu:{ahash}",
                "audio_sha256": ahash,
                "title": str(maps[0].get("title_unicode") or maps[0].get("title", "")),
                "artist": str(maps[0].get("artist_unicode") or maps[0].get("artist", "")),
            }
            kept += 1
            yield osu_job, meta, audio, [b["content"] for b in maps if b.get("content")]
        os.remove(Path(path).resolve())
        print(f"[osu] tar {i + 1}/{len(files)} scanned; kept {kept}, skipped {skipped}", flush=True)


def upload(source: str) -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=REPO,
        repo_type="dataset",
        folder_path=str(OUT),
        allow_patterns=[f"data/{source}/*", f"manifest/{source}*"],
        commit_message=f"ingest v0: {source}",
    )
    print(f"[{source}] uploaded to {REPO}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["taiko1000", "osu"])
    parser.add_argument("--std-percent", type=int, default=5)
    parser.add_argument("--limit-tars", type=int)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    errors = OUT / "manifest" / f"{args.source}.errors.jsonl"
    errors.parent.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(args.source)
    try:
        jobs = (
            taiko_jobs()
            if args.source == "taiko1000"
            else osu_jobs(args.std_percent, args.limit_tars)
        )
        run_pool(jobs, writer, errors)
    except Exception:
        traceback.print_exc()
        return 1
    if not args.no_upload:
        upload(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
