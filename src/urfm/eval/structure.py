"""Music-structure alignment: does the chart repeat where the music repeats and change where
the music changes?

Both the chart and the audio are cut into the same units on the chart's beat grid (a whole
number of beats close to ``UNIT_S`` seconds, so the choice does not depend on the notation
octave). Each unit gets a descriptor:

- chart: onsets at ``SUBS`` positions per beat, one row per action colour (Taiko don/ka;
  other games one row);
- audio: the onset envelope in 4 bands at the same positions plus the mean log-mel in 32 bands.

Self-similarity matrices (cosine) of the two are compared over unit pairs at least
``MIN_LAG`` units apart; the score is their Pearson correlation. A circular shift of the chart
against the audio gives the same statistic under no alignment (``null``).
"""

from __future__ import annotations

import numpy as np

from urfm.data.features import MEL
from urfm.data.texture import COARSE, onset_envelope
from urfm.ir.chart import Chart, ChartSet
from urfm.tokenize.tokenizer import beats_at_us, playable_events

UNIT_S = 2.5
SUBS = 12
MIN_LAG = 2
SHIFTS = 8


def unit_beats(chartset: ChartSet, chart: Chart) -> int:
    """Beats per unit: the power of two nearest ``UNIT_S`` at the chart's median tempo."""
    bpms = [float(t.bpm) for t in chartset.timing_of(chart).tempos] or [120.0]
    beat_s = 60.0 / float(np.median(bpms))
    return int(2 ** np.clip(np.round(np.log2(UNIT_S / beat_s)), 0, 4))


def _colour(event) -> int:
    role = event.role or ""
    return 1 if role.endswith(":ka") or role.endswith(":big_ka") else 0


def chart_events(chartset: ChartSet, chart: Chart) -> tuple[np.ndarray, np.ndarray]:
    """Onset beats (float, on the chart's timing) and colours of the playable events."""
    timing = chartset.timing_of(chart)
    events = playable_events(chart)
    if not events:
        return np.zeros(0), np.zeros(0, np.int64)
    us = timing.us_array([e.time.beat for e in events], [e.time.residual_us for e in events])
    beats = beats_at_us(timing, np.asarray(us, np.float64))
    return beats, np.array([_colour(e) for e in events], np.int64)


def grid_descriptors(beats: np.ndarray, colour: np.ndarray, k: int, n_units: int, first: float):
    """``[n_units, 2 * k * SUBS]`` onset grids of onsets at ``beats`` with ``colour``."""
    out = np.zeros((n_units, 2, k * SUBS), np.float32)
    if len(beats):
        pos = np.rint((np.asarray(beats) - first) * SUBS).astype(np.int64)
        unit, slot = pos // (k * SUBS), pos % (k * SUBS)
        ok = (unit >= 0) & (unit < n_units)
        out[unit[ok], colour[ok], slot[ok]] = 1.0
    return out.reshape(n_units, -1)


def chart_descriptors(chartset: ChartSet, chart: Chart, k: int, n_units: int, first: float):
    """``[n_units, 2 * k * SUBS]`` onset grids of the chart's playable events."""
    beats, colour = chart_events(chartset, chart)
    return grid_descriptors(beats, colour, k, n_units, first)


def audio_descriptors(mel: np.ndarray, timing, k: int, n_units: int, first: float):
    """``[n_units, 4 * k * SUBS + COARSE]``: band onset envelopes on the grid + mean mel."""
    t = mel.shape[0]
    env = onset_envelope(mel)
    env = np.maximum(env, np.maximum(np.roll(env, 1, 0), np.roll(env, -1, 0)))
    pos = first + np.arange(n_units * k * SUBS) / SUBS
    frames = np.clip(np.rint(timing.us_array(pos) / 1e6 * MEL.fps).astype(np.int64), 0, t - 1)
    flux = env[frames].reshape(n_units, -1)
    edges = np.clip(
        np.rint(timing.us_array(first + k * np.arange(n_units + 1)) / 1e6 * MEL.fps).astype(
            np.int64
        ),
        0,
        t,
    )
    bands = mel.reshape(t, COARSE, -1).mean(-1, dtype=np.float64)
    cum = np.concatenate([np.zeros((1, COARSE)), np.cumsum(bands, 0)])
    coarse = (cum[edges[1:]] - cum[edges[:-1]]) / np.maximum(edges[1:] - edges[:-1], 1)[:, None]
    return np.concatenate([flux, coarse / 255.0], axis=1).astype(np.float32)


def _ssm(x: np.ndarray, centre: bool) -> np.ndarray:
    if centre:
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    norm = np.linalg.norm(x, axis=1)
    empty = norm == 0
    s = (x @ x.T) / np.maximum(norm[:, None] * norm[None], 1e-12)
    s[empty[:, None] & empty[None]] = 1.0  # two silent units are the same
    return s


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 10 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def alignment(chartset: ChartSet, chart: Chart, mel: np.ndarray, seed: int = 0) -> dict:
    """``score`` (chart SSM vs audio SSM correlation), ``null`` (mean over circular shifts),
    ``units`` and ``unit_beats``; ``score`` is None for charts too short or uniform."""
    timing = chartset.timing_of(chart)
    events = playable_events(chart)
    if len(events) < 8:
        return {"score": None}
    k = unit_beats(chartset, chart)
    us = timing.us_array([events[0].time.beat, events[-1].time.beat])
    lo, hi = beats_at_us(timing, np.asarray(us, np.float64))
    first = float(np.floor(lo / k) * k)
    n = int(np.ceil((hi + 1e-6 - first) / k))
    if n < 2 * MIN_LAG + 8:
        return {"score": None}
    c = _ssm(chart_descriptors(chartset, chart, k, n, first), centre=False)
    a = _ssm(audio_descriptors(mel, timing, k, n, first), centre=True)
    iu = np.triu_indices(n, MIN_LAG)
    score = _corr(c[iu], a[iu])
    rng = np.random.default_rng(seed)
    nulls = []
    offsets = np.arange(MIN_LAG + 2, n - MIN_LAG - 1)
    for s in rng.choice(offsets, size=min(SHIFTS, len(offsets)), replace=False):
        cs = np.roll(np.roll(c, s, 0), s, 1)
        v = _corr(cs[iu], a[iu])
        if v is not None:
            nulls.append(v)
    return {
        "score": score,
        "null": float(np.mean(nulls)) if nulls else None,
        "units": n,
        "unit_beats": k,
    }


class SongStructure:
    """A song's audio SSM and a reference chart's unit descriptors, for scoring a generated
    window in place (RL): the window's units are replaced by the generated onsets and their
    SSM rows are correlated with the audio's rows (pairs at least ``MIN_LAG`` units apart)."""

    def __init__(self, chartset: ChartSet, chart: Chart, mel: np.ndarray) -> None:
        timing = chartset.timing_of(chart)
        self.timing = timing
        self.k = unit_beats(chartset, chart)
        lo, hi = beats_at_us(timing, np.array([0.0, mel.shape[0] / MEL.fps * 1e6 - 1]))
        self.first = float(np.floor(lo / self.k) * self.k)
        self.n = max(int(np.ceil((hi - self.first) / self.k)), 1)
        self.desc = chart_descriptors(chartset, chart, self.k, self.n, self.first)
        self.audio = _ssm(audio_descriptors(mel, timing, self.k, self.n, self.first), True)

    def window_score(
        self, beats: np.ndarray, colour: np.ndarray, start_beat: float, end_beat: float
    ) -> float | None:
        """Row alignment of the units in ``[start_beat, end_beat)`` written as ``beats``."""
        a = int(np.floor((start_beat - self.first) / self.k))
        b = int(np.ceil((end_beat - self.first) / self.k))
        a, b = max(a, 0), min(b, self.n)
        if b - a < 1 or self.n < 2 * MIN_LAG + 4:
            return None
        desc = self.desc.copy()
        sel = (beats >= self.first + a * self.k) & (beats < self.first + b * self.k)
        desc[a:b] = grid_descriptors(beats[sel], colour[sel], self.k, self.n, self.first)[a:b]
        norm = np.linalg.norm(desc, axis=1)
        rows = desc[a:b] @ desc.T / np.maximum(norm[a:b, None] * norm[None], 1e-12)
        rows[(norm[a:b] == 0)[:, None] & (norm == 0)[None]] = 1.0
        idx = np.arange(self.n)
        keep = np.abs(idx[None] - np.arange(a, b)[:, None]) >= MIN_LAG
        return _corr(rows[keep], self.audio[a:b][keep])
