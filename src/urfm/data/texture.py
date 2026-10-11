"""Beat-synchronous rhythm texture of a whole song (song-level context for the note model).

The song is cut into units of ``UNIT_BEATS`` beats on its beat grid. Each unit describes:

* *where onsets are*: spectral flux (rise in dB) in ``BANDS`` frequency bands at
  ``SUBS`` positions per beat (24 covers 16ths, 32nds and triplets), max over ±1 frame;
* *what it sounds like*: the mean log-mel in ``COARSE`` bands.

Positions are in beats, so a passage of 16th-note hi-hats looks the same at any tempo, and a
song played faster (tempo augmentation) has the same texture: units only move in time.
"""

from __future__ import annotations

import numpy as np

from urfm.data.features import MEL
from urfm.ir.timing import TimingMap
from urfm.tokenize.tokenizer import beats_at_us

UNIT_BEATS = 2
SUBS = 24
BANDS = 4
COARSE = 32
FEATURES = UNIT_BEATS * SUBS * BANDS + COARSE
MAX_UNITS = 512
"""~1024 beats (7 min at 150 BPM); longer songs keep uniformly spaced units."""
FLUX_UNIT = 10 * 255 / MEL.db_range
"""Flux is expressed in 10 dB steps (uint8 mel steps are 80/255 dB)."""


def onset_envelope(mel: np.ndarray) -> np.ndarray:
    """``[T, BANDS]`` mean positive dB rise per frame in equal mel-bin bands."""
    x = mel.astype(np.float32)
    rise = np.maximum(np.diff(x, axis=0, prepend=x[:1]), 0.0)
    return rise.reshape(len(x), BANDS, -1).mean(-1) / FLUX_UNIT


_SONG_CACHE: list[tuple[object, tuple[int, ...], dict]] = []


def song_cache(mel: np.ndarray) -> dict:
    """Per-song scratch space for derived arrays, shared by every chart of the song while its
    mel buffer is alive (a song's difficulties are visited one after another in training).
    Keyed by the identity of the buffer the mel views, which the cache keeps alive."""
    base = mel.base if mel.base is not None else mel
    for owner, shape, store in _SONG_CACHE:
        if owner is base and shape == mel.shape:
            return store
    store: dict = {}
    _SONG_CACHE.insert(0, (base, mel.shape, store))
    del _SONG_CACHE[2:]
    return store


def cached_envelope(mel: np.ndarray) -> np.ndarray:
    """:func:`onset_envelope` max-pooled over +-1 frame, cached per song."""
    store = song_cache(mel)
    if "env" not in store:
        env = onset_envelope(mel)
        store["env"] = np.maximum(env, np.maximum(np.roll(env, 1, 0), np.roll(env, -1, 0)))
    return store["env"]


def cached_band_cumsum(mel: np.ndarray) -> np.ndarray:
    """``[T + 1, COARSE]`` cumulative sums of the mel summed in ``COARSE`` bands, per song."""
    store = song_cache(mel)
    if "cum" not in store:
        t = mel.shape[0]
        bands = mel.reshape(t, COARSE, -1).sum(-1, dtype=np.int32)
        store["cum"] = np.concatenate(
            [np.zeros((1, COARSE), np.int64), np.cumsum(bands, 0, dtype=np.int64)]
        )
    return store["cum"]


def song_texture(
    mel: np.ndarray, timing: TimingMap, duration_us: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(features float16 [S, FEATURES], unit start µs int64 [S])`` with ``S <= MAX_UNITS``."""
    t = mel.shape[0]
    duration_us = duration_us or round(t / MEL.fps * 1e6)
    lo, hi = beats_at_us(timing, np.array([0.0, float(duration_us)]))
    first = np.floor(lo / UNIT_BEATS) * UNIT_BEATS
    n = max(int(np.ceil((hi - first) / UNIT_BEATS)), 1)
    starts = first + UNIT_BEATS * np.arange(n)
    if n > MAX_UNITS:
        keep = np.unique(np.linspace(0, n - 1, MAX_UNITS).round().astype(np.int64))
        starts = starts[keep]
    env = cached_envelope(mel)
    offsets = np.arange(UNIT_BEATS * SUBS) / SUBS
    pos = (starts[:, None] + offsets[None]).ravel()
    frames = np.rint(timing.us_array(pos) / 1e6 * MEL.fps).astype(np.int64)
    inside = (frames >= 0) & (frames < t)
    flux = np.where(inside[:, None], env[np.clip(frames, 0, t - 1)], 0.0)
    flux = flux.reshape(len(starts), -1)
    unit_us = timing.us_array(starts)
    end_us = timing.us_array(starts + UNIT_BEATS)
    f0 = np.clip(np.rint(unit_us / 1e6 * MEL.fps).astype(np.int64), 0, t)
    f1 = np.clip(np.rint(end_us / 1e6 * MEL.fps).astype(np.int64), 0, t)
    cum = cached_band_cumsum(mel)
    width = mel.shape[1] // COARSE
    coarse = (cum[f1] - cum[f0]) / (np.maximum(f1 - f0, 1)[:, None] * width * 255.0)
    feats = np.concatenate([flux, coarse], axis=1).astype(np.float16)
    return feats, np.round(unit_us).astype(np.int64)


def unit_marker(unit_us: np.ndarray, start_us: int, end_us: int) -> np.ndarray:
    """``int8 [S]``: 1 for units overlapping ``[start_us, end_us)``."""
    nxt = np.append(
        unit_us[1:], unit_us[-1] + (unit_us[-1] - unit_us[-2] if len(unit_us) > 1 else 1)
    )
    marker = (unit_us < end_us) & (nxt > start_us)
    if not marker.any():
        marker[int(np.argmin(np.abs(unit_us - start_us)))] = True
    return marker.astype(np.int8)
