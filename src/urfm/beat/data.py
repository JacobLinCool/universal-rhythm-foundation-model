"""Beat-model data: song mel + beat labels -> warped, augmented fixed-length chunks.

The index pass (parallel, cached per shard) stores each song's beat and downbeat times
(:mod:`urfm.beat.labels`). Workers stream row groups, read only the ``mel`` column, and
cut ``chunks_per_song`` chunks per visit. Augmentation happens on the source song before
cropping:

* tempo: the whole song ``k`` times faster (log-uniform), and up to six short spans
  (half a beat to a measure) faster or slower -- the short tempo changes newer charts use;
* level: random gain and spectral tilt in dB, and one frequency mask.

Mel is resampled by linear interpolation along time with the inverse warp, and labels go
through the forward warp, so they stay aligned to the frame.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from urfm.beat.labels import song_labels
from urfm.data.features import MEL
from urfm.ir.serialize import from_msgpack
from urfm.ir.transform import TimeWarp

CHUNK = 2560
"""Frames per training chunk (~29.7 s)."""
LABEL_VERSION = 1


@dataclass(frozen=True)
class SongRef:
    file: str
    row_group: int
    row: int
    beats: np.ndarray  # seconds
    downbeats: np.ndarray
    span: tuple[float, float]


def _index_file(file: str) -> list[tuple[int, int, list[float], list[float], float, float]]:
    pf = pq.ParquetFile(file)
    out = []
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g, columns=["chartset", "duration_s"])
        for r, (blob, dur) in enumerate(
            zip(table["chartset"].to_pylist(), table["duration_s"].to_pylist(), strict=True)
        ):
            try:
                labels = song_labels(from_msgpack(blob), dur)
            except Exception:
                labels = None
            if labels is None:
                continue
            beats, downbeats, (lo, hi) = labels
            out.append((g, r, np.round(beats, 5).tolist(), np.round(downbeats, 5).tolist(), lo, hi))
    return out


def _cached(file: str) -> list:
    path = Path(file)
    st = path.stat()
    cache = path.with_suffix(f".beats{LABEL_VERSION}-{st.st_size}-{int(st.st_mtime)}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    rows = _index_file(file)
    with contextlib.suppress(OSError):
        cache.write_text(json.dumps(rows))
    return rows


def build_index(files: list[str], workers: int = 0) -> list[SongRef]:
    workers = workers or min(len(files), os.cpu_count() or 1, 8) or 1
    if workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_cached, files))
    else:
        results = [_cached(f) for f in files]
    return [
        SongRef(f, g, r, np.asarray(b, np.float64), np.asarray(d, np.float64), (lo, hi))
        for f, rows in zip(files, results, strict=True)
        for g, r, b, d, lo, hi in rows
    ]


# -- augmentation ---------------------------------------------------------------------


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


SPAN_BEATS = (0.5, 1.0, 2.0, 4.0)


def random_warp(
    rng: random.Random,
    beats: np.ndarray,
    speed: tuple[float, float] = (0.75, 1.4),
    p_speed: float = 0.6,
    p_local: float = 0.5,
    local: tuple[float, float] = (0.75, 1.33),
    max_spans: int = 6,
) -> TimeWarp:
    """Global speed ``k`` plus short beat-aligned spans played at their own speed."""
    k = log_uniform(rng, *speed) if rng.random() < p_speed else 1.0
    spans: list[tuple[float, float, float]] = []
    if len(beats) > 8 and rng.random() < p_local:
        for _ in range(rng.randint(1, max_spans)):
            length = rng.choice(SPAN_BEATS)
            i = rng.randrange(0, len(beats) - 5)
            start = _beat_time(beats, i + rng.choice((0.0, 0.5)))
            end = _beat_time(beats, i + rng.choice((0.0, 0.5)) + length)
            if end - start > 0.05:
                spans.append((start, end, log_uniform(rng, *local)))
    spans.sort()
    kept: list[tuple[float, float, float]] = []
    for s in spans:  # non-overlapping
        if not kept or s[0] >= kept[-1][1]:
            kept.append(s)
    src, dst = [], []
    t_prev, d_prev = 0.0, 0.0
    for a, b, s in kept:
        d_a = d_prev + (a - t_prev) / k
        d_b = d_a + (b - a) / (k * s)
        src += [a, b]
        dst += [d_a, d_b]
        t_prev, d_prev = b, d_b
    return TimeWarp(tuple(src), tuple(dst), 1 / k, 1 / k)


def _beat_time(beats: np.ndarray, x: float) -> float:
    i = min(int(x), len(beats) - 2)
    return float(beats[i] + (x - i) * (beats[i + 1] - beats[i]))


def resample(mel: np.ndarray, warp: TimeWarp, start_s: float, frames: int) -> np.ndarray:
    """``frames`` output frames from output time ``start_s``, read through the inverse warp."""
    t_out = start_s + np.arange(frames) / MEL.fps
    src = warp.inverse(t_out) * MEL.fps
    i0 = np.floor(src).astype(np.int64)
    w = (src - i0).astype(np.float32)[:, None]
    n = mel.shape[0]
    valid = (i0 >= 0) & (i0 + 1 < n)
    a = mel[np.clip(i0, 0, n - 1)].astype(np.float32)
    b = mel[np.clip(i0 + 1, 0, n - 1)].astype(np.float32)
    out = a * (1 - w) + b * w
    out[~valid] = 0.0
    return np.rint(out).astype(np.uint8)


def level_augment(mel: np.ndarray, rng: random.Random) -> np.ndarray:
    """Gain (±6 dB), tilt (±6 dB across the spectrum) and one frequency mask, in uint8 units."""
    step = 255.0 / MEL.db_range  # uint8 steps per dB
    x = mel.astype(np.float32)
    nonzero = x > 0
    gain = rng.uniform(-6, 6) * step
    tilt = rng.uniform(-6, 6) * step * np.linspace(-0.5, 0.5, x.shape[1], dtype=np.float32)
    x = np.where(nonzero, x + gain + tilt[None], 0.0)
    if rng.random() < 0.3:
        w = rng.randint(4, 16)
        f0 = rng.randrange(0, x.shape[1] - w)
        x[:, f0 : f0 + w] = 0.0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# -- chunks ---------------------------------------------------------------------------


@dataclass
class Chunk:
    mel: np.ndarray  # uint8 [T, n_mels]
    beats: np.ndarray  # uint8 [4, T]: 2 = beat frame, 1 = neighbour (target 0.5)
    downbeats: np.ndarray  # uint8 [T]
    mask: np.ndarray  # bool [T]


def _track(times: np.ndarray, start_s: float, frames: int) -> np.ndarray:
    out = np.zeros(frames, np.uint8)
    f = np.rint((times - start_s) * MEL.fps).astype(np.int64)
    for d, v in ((-1, 1), (1, 1), (0, 2)):
        g = f + d
        g = g[(g >= 0) & (g < frames)]
        out[g] = np.maximum(out[g], v)
    return out


def levels(beats: np.ndarray) -> list[np.ndarray]:
    """As labelled, double (midpoints added), and the two halves."""
    if len(beats) < 2:
        return [beats, beats, beats, beats]
    gaps = np.diff(beats)
    mids = (beats[:-1] + gaps / 2)[gaps < 2.0]
    return [beats, np.sort(np.concatenate([beats, mids])), beats[0::2], beats[1::2]]


def make_chunk(
    mel: np.ndarray, ref: SongRef, rng: random.Random, augment: bool, frames: int = CHUNK
) -> Chunk:
    warp = random_warp(rng, ref.beats) if augment else TimeWarp((), (), 1.0, 1.0)
    beats = warp.forward(ref.beats)
    downs = warp.forward(ref.downbeats)
    lo, hi = (float(x) for x in warp.forward(np.asarray(ref.span)))
    duration = float(warp.forward(np.asarray([mel.shape[0] / MEL.fps]))[0])
    length = frames / MEL.fps
    first = max(0.0, lo - length / 2)
    last = max(first, min(hi - length / 2, duration - length))
    start = rng.uniform(first, last)
    out = resample(mel, warp, start, frames)
    if augment and rng.random() < 0.5:
        out = level_augment(out, rng)
    t = start + np.arange(frames) / MEL.fps
    mask = (t >= lo) & (t <= hi) & (t <= duration)
    return Chunk(
        mel=out,
        beats=np.stack([_track(x, start, frames) for x in levels(beats)]),
        downbeats=_track(downs, start, frames),
        mask=mask,
    )


def collate(chunks: list[Chunk]) -> dict[str, torch.Tensor]:
    return {
        "mel": torch.from_numpy(np.stack([c.mel for c in chunks])),
        "beats": torch.from_numpy(np.stack([c.beats for c in chunks])),
        "downbeats": torch.from_numpy(np.stack([c.downbeats for c in chunks])),
        "mask": torch.from_numpy(np.stack([c.mask for c in chunks])),
    }


class ChunkStream(IterableDataset):
    def __init__(
        self,
        songs: list[SongRef],
        batch_size: int,
        seed: int = 0,
        chunks_per_song: int = 2,
        augment: bool = True,
    ) -> None:
        self.songs = songs
        self.batch_size = batch_size
        self.seed = seed
        self.chunks_per_song = chunks_per_song
        self.augment = augment

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 7919 + wid)
        by_group: dict[tuple[str, int], list[SongRef]] = {}
        for s in self.songs:
            by_group.setdefault((s.file, s.row_group), []).append(s)
        groups = sorted(by_group)[wid::nw] or sorted(by_group)
        buffer: list[Chunk] = []
        while True:
            rng.shuffle(groups)
            for key in groups:
                mels = pq.ParquetFile(key[0]).read_row_group(key[1], columns=["mel"])["mel"]
                for ref in by_group[key]:
                    mel = np.frombuffer(mels[ref.row].as_py(), np.uint8).reshape(-1, MEL.n_mels)
                    for _ in range(self.chunks_per_song):
                        buffer.append(make_chunk(mel, ref, rng, self.augment))
                while len(buffer) >= self.batch_size * 4:
                    rng.shuffle(buffer)
                    batch, buffer = buffer[: self.batch_size], buffer[self.batch_size :]
                    yield collate(batch)


def loader(stream: ChunkStream, workers: int) -> DataLoader:
    return DataLoader(
        stream,
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def fixed_chunks(songs: list[SongRef], limit: int, seed: int = 0) -> list[Chunk]:
    """Deterministic, unaugmented validation chunks (one per song)."""
    rng = random.Random(seed)
    picked = rng.sample(songs, min(limit, len(songs)))
    out = []
    cache: dict[tuple[str, int], object] = {}
    for ref in picked:
        key = (ref.file, ref.row_group)
        if key not in cache:
            cache[key] = pq.ParquetFile(ref.file).read_row_group(ref.row_group, columns=["mel"])
        mels = cache[key]["mel"]  # type: ignore[index]
        mel = np.frombuffer(mels[ref.row].as_py(), np.uint8).reshape(-1, MEL.n_mels)
        out.append(make_chunk(mel, ref, rng, augment=False))
    return out
