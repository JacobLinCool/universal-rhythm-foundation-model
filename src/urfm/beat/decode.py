"""From frame activations to a :class:`TimingMap`.

1. Peaks of the beat activation (sub-frame refined) are beats.
2. Clean-up against the local beat period: drop a beat much closer to its neighbour than
   the period, fill a gap that is a whole number of periods.
3. Tempo segments: the longest runs of beats that a straight line through their end beats
   fits within ``tol`` -- no smoothness prior, so a tempo may change on any beat (half a
   measure, a single beat) and change again right after.
4. Measures from downbeat peaks; beat 0 is the first beat (a pickup becomes a short first
   measure).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch

from urfm.beat.data import CHUNK
from urfm.beat.model import BeatModel
from urfm.data.features import MEL
from urfm.ir.timing import MeterChange, TempoChange, TimingMap

US = 1_000_000
CONVENTION_OFFSET_US = {"taiko": 15_000}
"""Per-game timing convention relative to the beat model, which follows the osu! majority
of its training data. Official Taiko charts sit 15 ms later on the same audio (median over
300 Taiko-1000 training songs: beat model −15.0 ms, osu! test songs +0.7 ms), so maps meant
for Taiko are shifted by this much."""


def for_game(timing: TimingMap, mechanics: str) -> TimingMap:
    """``timing`` moved to ``mechanics``' timing convention (see ``CONVENTION_OFFSET_US``)."""
    shift = CONVENTION_OFFSET_US.get(mechanics, 0)
    if not shift:
        return timing
    return TimingMap(
        offset_us=timing.offset_us + shift,
        tempos=timing.tempos,
        stops=timing.stops,
        warps=timing.warps,
        meters=timing.meters,
    )


@torch.no_grad()
def activations(
    model: BeatModel,
    mel: np.ndarray,
    device: str | torch.device = "cpu",
    hop: int = 1920,
    batch: int = 16,
) -> np.ndarray:
    """Sigmoid ``[T, 2]`` (beat, downbeat) for a whole song: overlapping chunks (run in
    batches), blended with triangular weights so chunk edges (little context) count less."""
    device = torch.device(device)
    model.eval()
    t = mel.shape[0]
    out = np.zeros((t, 2), np.float64)
    weight = np.zeros(t, np.float64)
    ramp = np.minimum(np.arange(1, CHUNK + 1), np.arange(CHUNK, 0, -1)).astype(np.float64)
    starts = list(range(0, max(t - CHUNK, 0) + 1, hop))
    if starts[-1] + CHUNK < t:
        starts.append(t - CHUNK)
    cuda = device.type == "cuda"
    chunks = np.zeros((len(starts), CHUNK, mel.shape[1]), np.uint8)
    for i, s in enumerate(starts):
        part = mel[s : s + CHUNK]
        chunks[i, : len(part)] = part
    probs = []
    for i in range(0, len(starts), batch):
        x = torch.from_numpy(chunks[i : i + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            probs.append(torch.sigmoid(model(x).float()).cpu().numpy())
    p_all = np.concatenate(probs)
    for s, p in zip(starts, p_all, strict=True):
        n = min(CHUNK, t - s)
        w = ramp[:n]
        out[s : s + n] += p[:n] * w[:, None]
        weight[s : s + n] += w
    return (out / np.maximum(weight, 1e-9)[:, None]).astype(np.float32)


def peaks(act: np.ndarray, threshold: float = 0.4, radius: int = 4) -> np.ndarray:
    """Times (s) of local maxima above ``threshold``, at least ``radius`` frames apart,
    refined by a parabola through the peak and its neighbours."""
    a = np.asarray(act, np.float64)
    n = len(a)
    if n < 3:
        return np.zeros(0)
    pad = np.pad(a, radius, constant_values=-1)
    window = np.lib.stride_tricks.sliding_window_view(pad, 2 * radius + 1)
    is_max = (a >= window.max(1)) & (a >= threshold)
    idx = np.nonzero(is_max)[0]
    # plateaus: keep the first frame of equal maxima
    idx = idx[np.concatenate([[True], np.diff(idx) > radius])] if len(idx) else idx
    left = a[np.clip(idx - 1, 0, n - 1)]
    right = a[np.clip(idx + 1, 0, n - 1)]
    denom = left - 2 * a[idx] + right
    shift = np.where(np.abs(denom) > 1e-9, 0.5 * (left - right) / np.where(denom == 0, 1, denom), 0)
    return (idx + np.clip(shift, -0.5, 0.5)) / MEL.fps


def clean(beats: np.ndarray, context: int = 8) -> np.ndarray:
    """Remove beats much closer than the local period; fill gaps of whole periods."""
    b = list(np.asarray(beats, np.float64))
    if len(b) < 4:
        return np.asarray(b)

    def local_period(i: int, xs: list[float]) -> float:
        lo, hi = max(0, i - context), min(len(xs) - 1, i + context)
        return float(np.median(np.diff(xs[lo : hi + 1])))

    out = [b[0]]
    for i in range(1, len(b)):
        if b[i] - out[-1] < 0.55 * local_period(i, b):
            continue
        out.append(b[i])
    filled = [out[0]]
    for i in range(1, len(out)):
        gap = out[i] - out[i - 1]
        p = local_period(i, out)
        n = round(gap / p)
        if n >= 2 and abs(gap / p - n) < 0.25:
            filled += [out[i - 1] + gap * k / n for k in range(1, n)]
        filled.append(out[i])
    return np.asarray(filled)


def track(
    times: np.ndarray,
    strength: np.ndarray,
    min_ioi: float = 0.15,
    max_ioi: float = 1.5,
    change_cost: float = 4.0,
    beat_cost: float = 0.35,
) -> np.ndarray:
    """Choose beats among candidate peaks by dynamic programming.

    A path scores ``sum(strength - beat_cost) - change_cost * sum(log(ioi_k / ioi_{k-1})^2)``.
    Tempo may change anywhere (the penalty is finite: a real change of a few beats pays it
    once), but a lone spurious or missing beat distorts two intervals and is not worth it,
    and ``beat_cost`` keeps weak peaks from doubling the level. Stretches without candidates
    longer than ``max_ioi`` split the song into independently tracked parts.
    """
    t = np.asarray(times, np.float64)
    a = np.asarray(strength, np.float64) - beat_cost
    if len(t) < 3:
        return t
    parts = np.split(np.arange(len(t)), np.nonzero(np.diff(t) > max_ioi)[0] + 1)
    out: list[float] = []
    for idx in parts:
        out += _track_part(t[idx], a[idx], min_ioi, max_ioi, change_cost)
    return np.asarray(out)


def _track_part(t, a, min_ioi, max_ioi, change_cost) -> list[float]:
    n = len(t)
    if n < 3:
        return t.tolist()
    # predecessors of each candidate: those between min_ioi and max_ioi earlier
    lo = np.searchsorted(t, t - max_ioi, side="left")
    hi = np.searchsorted(t, t - min_ioi, side="right")
    score: dict[tuple[int, int], float] = {}
    back: dict[tuple[int, int], tuple[int, int] | None] = {}
    for k in range(n):
        for j in range(lo[k], hi[k]):
            ioi = t[k] - t[j]
            best, arg = a[j] + a[k], None  # a path may start with the pair (j, k)
            for i in range(lo[j], hi[j]):
                prev = score.get((i, j))
                if prev is None:
                    continue
                r = np.log(ioi / (t[j] - t[i]))
                cand = prev + a[k] - change_cost * r * r
                if cand > best:
                    best, arg = cand, (i, j)
            score[(j, k)] = best
            back[(j, k)] = arg
    if not score:
        return t.tolist()
    state: tuple[int, int] | None = max(score, key=score.__getitem__)
    path: list[int] = []
    while state is not None:
        j, k = state
        path.append(k)
        prev = back[state]
        if prev is None:
            path.append(j)
        state = prev
    return sorted(t[i] for i in path)


LEVEL_RATIO = {0.5: (0.4, 0.62), 2.0: (1.65, 2.45)}
MIN_RUN = 4


def one_level(beats: np.ndarray, strength: np.ndarray | None = None) -> np.ndarray:
    """Keep one metrical level across the song.

    The beat model may choose a different level (half or double tempo) in different chunks,
    which turns into spurious x2 tempo jumps. Intervals are compared with the song's
    dominant interval (the median): runs of at least ``MIN_RUN`` half-length intervals lose
    every other beat (the weaker phase by ``strength``), runs of double-length intervals get
    their midpoints. Other ratios (real tempo changes) are left alone.
    """
    b = np.asarray(beats, np.float64)
    if len(b) < 8:
        return b
    s = np.ones(len(b)) if strength is None else np.asarray(strength, np.float64)
    ref = float(np.median(np.diff(b)))
    out_b: list[float] = []
    out_s: list[float] = []
    i = 0
    n = len(b)
    while i < n:
        if i + 1 < n:
            r = (b[i + 1] - b[i]) / ref
            for level, (lo, hi) in LEVEL_RATIO.items():
                if not lo <= r <= hi:
                    continue
                j = i + 1  # run of intervals at this level: beats i..j
                while j + 1 < n and lo <= (b[j + 1] - b[j]) / ref <= hi:
                    j += 1
                if j - i >= MIN_RUN:
                    run, strength_run = b[i : j + 1], s[i : j + 1]
                    if level == 0.5:
                        even, odd = strength_run[0::2].sum(), strength_run[1::2].sum()
                        keep = slice(0, None, 2) if even >= odd else slice(1, None, 2)
                        out_b += run[keep].tolist()
                        out_s += strength_run[keep].tolist()
                    else:
                        mids = (run[:-1] + run[1:]) / 2
                        both = np.empty(len(run) + len(mids))
                        both[0::2], both[1::2] = run, mids
                        out_b += both.tolist()
                        out_s += [1.0] * len(both)
                    i = j + 1
                    break
            else:
                out_b.append(b[i])
                out_s.append(s[i])
                i += 1
            continue
        out_b.append(b[i])
        out_s.append(s[i])
        i += 1
    return np.asarray(sorted(set(out_b)))


def segments(beats: np.ndarray, tol: float = 0.012) -> list[tuple[int, int]]:
    """Greedy longest runs ``(i, j)`` of beats whose times a line through beats ``i`` and
    ``j`` fits within ``tol`` seconds; consecutive runs share their boundary beat."""
    n = len(beats)
    out: list[tuple[int, int]] = []
    i = 0
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            k = np.arange(i, j + 2)
            line = beats[i] + (k - i) * (beats[j + 1] - beats[i]) / (j + 1 - i)
            if np.abs(beats[k] - line).max() > tol:
                break
            j += 1
        out.append((i, j))
        i = j
    return out


def to_timing(
    beats: np.ndarray, downbeat_act: np.ndarray | None = None, tol: float = 0.015
) -> TimingMap:
    """A tempo map through ``beats`` (s); measures from ``downbeat_act`` (frame-level)."""
    if len(beats) < 2:
        return TimingMap.constant(120)
    tempos: list[TempoChange] = []
    for i, j in segments(beats, tol):
        us = max(round((beats[j] - beats[i]) / (j - i) * US), 1)
        bpm = Fraction(60 * US, us)
        if not tempos or tempos[-1].bpm != bpm:
            tempos.append(TempoChange(Fraction(i), bpm))
    meters = _meters(beats, downbeat_act)
    return TimingMap(offset_us=Fraction(round(beats[0] * US)), tempos=tuple(tempos), meters=meters)


def _meters(beats: np.ndarray, act: np.ndarray | None) -> tuple[MeterChange, ...]:
    """One meter for the song: 4 or 3 beats per measure, with the phase whose beats carry
    the most downbeat activation (a pickup becomes a short first measure). Downbeat
    predictions are too noisy to place meter changes; visual meter tricks are generated by
    the chart model instead."""
    if act is None or len(beats) < 8:
        return ()
    frames = np.clip(np.rint(beats * MEL.fps).astype(np.int64), 0, len(act) - 1)
    near = np.stack([act[np.clip(frames + d, 0, len(act) - 1)] for d in (-1, 0, 1)]).max(0)
    best = (-1.0, 4, 0)
    for length, prior in ((4, 0.03), (3, 0.0)):
        for phase in range(length):
            score = float(near[phase::length].mean()) + prior
            if score > best[0]:
                best = (score, length, phase)
    _, length, phase = best
    meters = []
    if phase:
        meters.append(MeterChange(Fraction(0), phase, 4))
    meters.append(MeterChange(Fraction(phase), length, 4))
    return tuple(meters)


def _consistent(meters: list[MeterChange]) -> list[MeterChange]:
    """Meter changes must fall on measure starts of the previous meter; a change between
    them is moved to the next measure start."""
    out: list[MeterChange] = []
    for m in meters:
        if out:
            prev = out[-1]
            span = m.beat - prev.beat
            k = -(-span // prev.measure_beats)  # ceil
            beat = prev.beat + k * prev.measure_beats
            if beat != m.beat:
                m = MeterChange(beat, m.numerator, m.denominator)
            if beat <= prev.beat:
                continue
        out.append(m)
    return out


def predict_timing(
    model: BeatModel, mel: np.ndarray, device: str | torch.device = "cpu", threshold: float = 0.4
) -> TimingMap:
    act = activations(model, mel, device)
    beats = decode_beats(act[:, 0], threshold)
    return to_timing(beats, act[:, 1])


def decode_beats(act: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """Beat times from a beat activation: candidate peaks (a lower threshold, so missed beats
    can be recovered), dynamic-programming tracking, one metrical level, gap filling."""
    cand = peaks(act, min(threshold, 0.2))
    frames = np.clip(np.rint(cand * MEL.fps).astype(np.int64), 0, len(act) - 1)
    beats = track(cand, act[frames])
    frames = np.clip(np.rint(beats * MEL.fps).astype(np.int64), 0, len(act) - 1)
    return clean(one_level(beats, act[frames]))
