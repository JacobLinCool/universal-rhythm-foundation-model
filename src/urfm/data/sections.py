"""Per-section audio descriptors (R6): the song planner's input and the similarity used to pick
the structure-memory section.

Sections are ``k`` beats of a timing map (``urfm.tokenize.sections``). A section's descriptor
is its onset envelope in 4 bands folded onto one beat at ``PHASES`` positions (so 16ths,
triplets, 32nds and 64ths show as peaks at their phases), the mean and maximum envelope per
band, the mean log-mel in 32 bands, and the tempo (log2 BPM and log2 ``k``).
"""

from __future__ import annotations

import math

import numpy as np

from urfm.data.features import MEL
from urfm.data.texture import COARSE, cached_band_cumsum, cached_envelope
from urfm.ir.timing import TimingMap
from urfm.tokenize.tokenizer import beats_at_us

PHASES = 48
BANDS = 4
FEATURES = PHASES * BANDS + 2 * BANDS + COARSE + 2
MAX_SECTIONS = 384
"""About 20 minutes at 3.2 s per section."""
TEMPO_DIMS = 2


def section_range(timing: TimingMap, duration_us: int, k: int) -> tuple[int, int]:
    """First and last section index covering ``[0, duration_us)``."""
    lo, hi = beats_at_us(timing, np.array([0.0, float(max(duration_us - 1, 0))]))
    return math.floor(lo / k), math.floor(hi / k)


def section_audio(mel: np.ndarray, timing: TimingMap, k: int) -> tuple[int, np.ndarray]:
    """``(first section index, float32 [S, FEATURES])`` for the whole song, ``S`` at most
    ``MAX_SECTIONS`` (later sections are cut)."""
    t = mel.shape[0]
    first, last = section_range(timing, round(t / MEL.fps * 1e6), k)
    n = min(last - first + 1, MAX_SECTIONS)
    env = cached_envelope(mel)
    pos = first * k + np.arange(n * k * PHASES) / PHASES
    us = timing.us_array(pos)
    frames = np.rint(np.asarray(us) / 1e6 * MEL.fps).astype(np.int64)
    inside = (frames >= 0) & (frames < t)
    flux = np.where(inside[:, None], env[np.clip(frames, 0, t - 1)], 0.0)
    flux = flux.reshape(n, k, PHASES, BANDS)
    folded = flux.mean(1).reshape(n, -1)
    edges_us = np.asarray(timing.us_array(first * k + k * np.arange(n + 1, dtype=np.float64)))
    edges = np.clip(np.rint(edges_us / 1e6 * MEL.fps).astype(np.int64), 0, t)
    cum = cached_band_cumsum(mel)
    width = np.maximum(edges[1:] - edges[:-1], 1)[:, None] * (mel.shape[1] // COARSE)
    coarse = (cum[edges[1:]] - cum[edges[:-1]]) / width / 255.0
    beat_s = np.diff(edges_us) / k / 1e6
    tempo = np.stack(
        [np.log2(60.0 / np.maximum(beat_s, 1e-3)) - 7.0, np.full(n, math.log2(k) - 3.0)], axis=1
    )
    feats = np.concatenate(
        [folded, flux.mean((1, 2)), flux.max((1, 2)), coarse, tempo], axis=1
    ).astype(np.float32)
    return first, feats


def similarity(feats: np.ndarray) -> np.ndarray:
    """Cosine similarity of sections after per-song standardisation (tempo dims left out)."""
    x = feats[:, :-TEMPO_DIMS].astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    norm = np.linalg.norm(x, axis=1)
    return (x @ x.T) / np.maximum(norm[:, None] * norm[None], 1e-12)


REF_QUANTILE = 0.8
REF_MIN_LAG = 2


def ref_threshold(sim: np.ndarray) -> float:
    """The song's ``REF_QUANTILE`` similarity over section pairs ``REF_MIN_LAG`` apart."""
    iu = np.triu_indices(len(sim), REF_MIN_LAG)
    return float(np.quantile(sim[iu], REF_QUANTILE)) if len(iu[0]) else 1.0


def choose_ref(
    sim: np.ndarray, threshold: float, first: int, window_first: int, window_last: int
) -> tuple[int, int] | None:
    """``(memory section, window section)`` with the most similar audio, the memory section
    ending before the window and at least ``REF_MIN_LAG`` sections before its partner; ``None``
    when no pair beats the song's ``REF_QUANTILE`` similarity. Indices are absolute."""
    n = len(sim)
    lo, hi = window_first - first, min(window_last - first, n - 1)
    if hi < 0 or lo < 1 or n < 4:
        return None
    best, pair = threshold, None
    for i in range(max(lo, 0), hi + 1):
        top = min(i - REF_MIN_LAG, lo - 1)
        if top < 0:
            continue
        j = int(np.argmax(sim[i, : top + 1]))
        if sim[i, j] > best:
            best, pair = float(sim[i, j]), (j + first, i + first)
    return pair
