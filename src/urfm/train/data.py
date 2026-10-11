"""Training data: song records -> (mel window, timing features, decoder tokens) batches.

Records come from Parquet shards (``urfm.data.records``). An index pass reads only
the light columns and the IR chart sets; workers then stream row groups (with mel)
in shuffled order, draw game-balanced windows, and yield length-bucketed batches.
Mel stays ``uint8`` until it is on the GPU.
"""

from __future__ import annotations

import bisect
import contextlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import msgspec
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from urfm.data.canonical import canonicalize, rescroll
from urfm.data.features import MEL
from urfm.data.sections import choose_ref, ref_threshold, section_audio, similarity
from urfm.data.texture import MAX_UNITS, song_texture, unit_marker
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event, Primitive
from urfm.ir.serialize import LazyChartSet, from_msgpack
from urfm.ir.time import Time
from urfm.ir.timing import TimingMap
from urfm.ir.transform import (
    TimeWarp,
    map_chart_times,
    rescale_beats,
    retime_chart,
    warp_timing,
)
from urfm.tokenize import CONTEXT_BEATS, audio_windows, take_drops
from urfm.tokenize.api import CONTEXT_EVENTS, WINDOW_S, Tokenizer
from urfm.tokenize.sections import (
    REP_BINS,
    VAR_BINS,
    SectionPlan,
    plan_extras,
    plan_sections,
    section_beats,
    window_sections,
)
from urfm.tokenize.tokenizer import beats_at_us, onset_rate, playable_events, window_start_beat
from urfm.tokenize.vocab import GRID_VALUES, SHARE_BINS
from urfm.train.conditions import chart_conditions, cond_prefix
from urfm.train.tiers import TierTable, chart_prefix

N_FRAMES = 2 * math.ceil(WINDOW_S * MEL.fps / 2)
"""Mel frames per window (even, for the 2x encoder stem)."""
WINDOW_US = round(WINDOW_S * 1_000_000)
DEC_BUCKETS = (128, 256, 384, 512, 768, 1024, 1536, 2048)


def download_shards(
    local_dir: str | Path = "/content/data/urfm-ingest-v0",
    repo: str = "JacobLinCool/urfm-ingest-v0",
    sources: tuple[str, ...] = ("taiko1000", "osu"),
    revision: str | None = None,
) -> Path:
    """Fetch shard files (VM side) with ``snapshot_download``; returns the local directory."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=[f"data/{s}/*" for s in sources],
    )
    return Path(path)


# -- index ----------------------------------------------------------------------


@dataclass(frozen=True)
class ChartRef:
    file: str
    row_group: int
    row: int
    chart: int
    mechanics: str
    weight: float = 1.0
    rate: float = 0.0
    """Interactive onsets per second of the chart (for per-game tiers)."""


@dataclass
class Index:
    charts: list[ChartRef]
    files: dict[str, int] = field(default_factory=dict)
    """file -> number of row groups"""

    def tier_table(self) -> TierTable:
        rates: dict[str, list[float]] = defaultdict(list)
        for c in self.charts:
            rates[c.mechanics].append(c.rate)
        return TierTable.build(rates)

    def counts(self) -> Counter[str]:
        return Counter(c.mechanics for c in self.charts)


def shard_files(root: str | Path, sources: list[str] | None, split: str) -> list[str]:
    root = Path(root) / "data" if (Path(root) / "data").is_dir() else Path(root)
    files = sorted(root.glob(f"*/{split}-*.parquet")) + sorted(root.glob(f"{split}-*.parquet"))
    return [str(f) for f in files if sources is None or f.parent.name in sources]


INDEX_VERSION = 2


def _index_file(file: str) -> tuple[int, list[tuple[str, int, int, int, str, float, float]]]:
    pf = pq.ParquetFile(file)
    refs = []
    for g in range(pf.num_row_groups):
        table = pf.read_row_group(g, columns=["chartset", "duration_s"])
        for r, (blob, dur) in enumerate(
            zip(table["chartset"].to_pylist(), table["duration_s"].to_pylist(), strict=True)
        ):
            cs = from_msgpack(blob)
            n_win = max(1.0, dur / WINDOW_S)
            for ci, chart in enumerate(cs.charts):
                if chart.events:
                    rate = 0.0 if extreme_tempo(cs.timing_of(chart)) else onset_rate(cs, chart)
                    refs.append((file, g, r, ci, chart.mechanics, n_win, rate))
    return pf.num_row_groups, refs


def _cached_index_file(
    file: str,
) -> tuple[int, list[tuple[str, int, int, int, str, float, float]]]:
    """Index one shard, caching the result next to it (keyed by size and mtime)."""
    path = Path(file)
    stat = path.stat()
    cache = path.with_suffix(f".index{INDEX_VERSION}-{stat.st_size}-{int(stat.st_mtime)}.json")
    if cache.exists():
        groups, refs = json.loads(cache.read_text())
        return groups, [tuple(r) for r in refs]  # type: ignore[misc]
    groups, refs = _index_file(file)
    with contextlib.suppress(OSError):
        cache.write_text(json.dumps([groups, refs]))
    return groups, refs


def build_index(
    files: list[str],
    mechanics: set[str] | None = None,
    alpha: float = 0.5,
    workers: int = 0,
    boost: dict[str, float] | None = None,
) -> Index:
    """Light pass over shards; weights give each mechanics a share ∝ n_windows^alpha, times
    ``boost[mechanics]`` (e.g. more official Taiko in a mixed run).

    Shards are indexed in parallel processes and cached, so repeated runs start instantly.
    """
    workers = workers or min(len(files), os.cpu_count() or 1) or 1
    if workers > 1 and len(files) > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_cached_index_file, files))
    else:
        results = [_cached_index_file(f) for f in files]
    groups = {f: g for f, (g, _) in zip(files, results, strict=True)}
    refs = [r for _, rs in results for r in rs if mechanics is None or r[4] in mechanics]
    windows: dict[str, float] = defaultdict(float)
    for *_, mech, n_win, _rate in refs:
        windows[mech] += n_win
    total = sum(windows.values()) or 1.0
    share = {m: n**alpha * (boost or {}).get(m, 1.0) for m, n in windows.items()}
    norm = sum(share.values()) or 1.0
    # expected windows drawn per chart per epoch: (share_m * total) spread over mechanics m
    rate = {m: share[m] / norm * total / windows[m] for m in windows}
    charts = [
        ChartRef(f, g, r, c, m, rate[m] * n_win, chart_rate)
        for f, g, r, c, m, n_win, chart_rate in refs
    ]
    return Index(charts, groups)


TEMPO_RANGE = (10, 2000)
"""BPM outside this range only appears in gimmick sections (osu! red-line tricks); such
charts are skipped for training and counted as ``skip_extreme_tempo``."""


def extreme_tempo(timing) -> bool:
    lo, hi = TEMPO_RANGE
    return any(not lo <= t.bpm <= hi for t in timing.tempos)


PIPELINE: Counter[str] = Counter()
"""Per-process failure counts (``mechanics/reason``), shipped to the trainer in batches."""


# -- samples ----------------------------------------------------------------------


SONG_POOL = round(2 * MEL.fps)
"""Mel frames per song-summary block (~2 s)."""
SONG_MAX = 256
"""Maximum song blocks (~8.5 min); longer songs keep uniformly spaced blocks."""
SONG_BLOCK_US = round(SONG_POOL / MEL.fps * 1_000_000)


def song_summary(mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-pool a whole-song uint8 mel ``[T, n_mels]`` into ~2 s blocks.

    Returns ``(blocks uint8 [S, n_mels], block start µs int64 [S])`` with ``S <= SONG_MAX``.
    """
    t = mel.shape[0]
    starts = np.arange(0, max(t, 1), SONG_POOL)
    if t == 0:
        return np.zeros((1, mel.shape[1]), np.uint8), np.zeros(1, np.int64)
    sums = np.add.reduceat(mel.astype(np.uint32), starts, axis=0)
    counts = np.minimum(SONG_POOL, t - starts)[:, None]
    blocks = np.rint(sums / counts).astype(np.uint8)
    keep = np.arange(len(starts))
    if len(keep) > SONG_MAX:
        keep = np.unique(np.linspace(0, len(starts) - 1, SONG_MAX).round().astype(np.int64))
    block_us = (starts[keep] / MEL.fps * 1_000_000).round().astype(np.int64)
    return blocks[keep], block_us


def song_marker(block_us: np.ndarray, start_us: int, end_us: int) -> np.ndarray:
    """``int8 [S]``: 1 for summary blocks overlapping the window ``[start_us, end_us)``."""
    marker = (block_us < end_us) & (block_us + SONG_BLOCK_US > start_us)
    if not marker.any():  # window inside a gap of a subsampled long song
        marker[int(np.argmin(np.abs(block_us - start_us)))] = True
    return marker.astype(np.int8)


@dataclass
class Sample:
    mel: np.ndarray  # uint8 [N_FRAMES, n_mels]
    feats: np.ndarray  # float32 [N_FRAMES, F]
    tokens: np.ndarray  # int64 [L]
    sep: int  # index of SEP in tokens
    truncated: bool
    mechanics: str
    song: np.ndarray | None = None
    """Song units: uint8 ``[S, n_mels]`` pooled mel, or float16 ``[S, F]`` texture."""
    song_marker: np.ndarray | None = None  # int8 [S], 1 = units of this window
    cond: np.ndarray | None = None  # float32 [K] normalised conditions
    cond_mask: np.ndarray | None = None  # bool [K]


@dataclass(frozen=True)
class Features:
    """What a model consumes besides mel and tokens (fixed per run, stored in checkpoints)."""

    cond: bool = False
    """Continuous conditions (``GAME [LANES] COND`` prefix) instead of ``TIER``/``DENS``."""
    texture: bool = False
    """Beat-synchronous texture units as song context instead of pooled mel blocks."""
    cond_drop: float = 0.15
    """Training: chance to hide each condition (the model must also work without it)."""
    cond_drop_all: float = 0.1
    """Training: chance to hide all conditions."""
    sections: bool = False
    """R6: section plans (``SBEAT SDENS SFINE SREST SKA SSUB`` per section) replace the window
    plan; teacher-forced from the chart in training, given by the song planner at inference."""
    ref: bool = False
    """R6: structure memory -- the events of the earlier section whose audio is most like a
    section of the window (``REF ... REFEND`` before the context)."""
    plan_extra: bool = False
    """R6.1: section plans carry peak density, repetition, rhythm variety, colour switches and
    long-note share (``urfm.tokenize.sections.plan_extras``)."""
    plan_noise: float = 0.1
    """Training: chance to move each section-plan value by one bin (planner errors)."""
    ctx_noise: float = 0.0
    """Training: chance a sample's context and memory are corrupted (drop 10% of events, swap
    10% of don/ka, shift 5% by a 16th) -- at inference they are the model's own output."""


@dataclass(frozen=True)
class Augment:
    """Per-visit chart and audio transformations (training only); see ``augment_chart``."""

    canonical: bool = True
    """Fold visual tempo tricks into scroll (``urfm.data.canonical``)."""
    p_speed: float = 0.0
    speed: tuple[float, float] = (0.8, 1.6)
    p_local: float = 0.0
    local: tuple[float, float] = (0.8, 1.25)
    max_spans: int = 6
    p_octave: float = 0.0
    octave_factors: tuple[tuple[float, float], ...] = ()
    """``(factor, weight)`` choices for octave re-notation, e.g. ``((0.5, .6), (2, .25), (0.25,
    .15))``; factors that would leave ``[min_bpm, max_bpm / 1.25]`` are skipped. Empty: the R4
    rule (double slow charts, else half)."""
    min_bpm: float = 40.0
    max_bpm: float = 420.0
    """Upper bound on any tempo after augmentation (16 s windows must fit in MAX_BEATS)."""
    p_predicted: float = 0.0
    """Chance a visit re-notates the chart on the audio-predicted timing map (when the song
    has one), so the model learns to chart on the grid it will get at inference."""
    min_grid_fit: float = 0.9
    """...but only when that many of the chart's notes sit on the predicted grid."""
    windows_per_visit: int = 1
    """Draw this many windows per chart visit and visit charts that much less often (same
    expected windows per chart): the per-visit work (augmentation, song features, section
    descriptors) is shared by more samples."""
    fast_boost: float = 0.0
    """Windows are drawn with weight ``1 + fast_boost * share of intervals < FAST_MS``, and a
    visit to a chart with fast passages draws ``1 + fast_boost/4`` times more windows: very
    fast passages are rare (1.5% of Taiko-1000 intervals) and the model rarely starts them."""


FAST_MS = 60.0


def window_weights(prep: PreparedChart, boost: float) -> list[float] | None:
    """Per-window sampling weights favouring fast passages (``None`` = uniform)."""
    if boost <= 0 or len(prep.onsets) < 2:
        return None
    us = np.unique(np.asarray(prep.onsets, dtype=np.int64))
    fast_at = us[1:][np.diff(us) < FAST_MS * 1000]
    if not len(fast_at):
        return None
    out = []
    for a, b in prep.windows:
        n = int(np.searchsorted(us, b) - np.searchsorted(us, a))
        f = int(np.searchsorted(fast_at, b) - np.searchsorted(fast_at, a))
        out.append(1.0 + boost * f / max(n, 1))
    return out


def chart_windows(tok: Tokenizer, cs: ChartSet, chart: Chart) -> list[tuple[int, int]]:
    """Measure-aligned ``(start_us, end_us)`` windows tiling the chart (shared with inference)."""
    method = getattr(tok, "windows", None)
    if method is not None:
        return list(method(cs, chart))
    from urfm.tokenize import windows  # tokenizers predating the ``windows`` method

    return list(windows(cs, chart))


def slice_mel(mel: np.ndarray, start_us: int) -> np.ndarray:
    start = round(start_us / 1_000_000 * MEL.fps)
    lo, hi = max(start, 0), min(start + N_FRAMES, mel.shape[0])
    if lo == start and hi - lo == N_FRAMES:
        return mel[lo:hi]
    out = np.zeros((N_FRAMES, mel.shape[1]), dtype=np.uint8)
    if hi > lo:
        out[lo - start : hi - start] = mel[lo:hi]
    return out


@dataclass
class SongAudio:
    """A song's mel as heard after a tempo warp (``None`` = as recorded)."""

    mel: np.ndarray
    warp: TimeWarp | None = None

    @property
    def frames(self) -> int:
        if self.warp is None:
            return self.mel.shape[0]
        return int(self.warp.forward(np.asarray([self.mel.shape[0] / MEL.fps]))[0] * MEL.fps)

    def window(self, start_us: int) -> np.ndarray:
        if self.warp is None:
            return slice_mel(self.mel, start_us)
        from urfm.beat.data import resample

        return resample(self.mel, _seconds(self.warp), start_us / 1e6, N_FRAMES)


def _seconds(warp: TimeWarp) -> TimeWarp:
    """A µs warp expressed in seconds."""
    return TimeWarp(
        tuple(x / 1e6 for x in warp.src),
        tuple(x / 1e6 for x in warp.dst),
        warp.first_slope,
        warp.last_slope,
    )


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def augment_chart(
    cs: ChartSet,
    chart: Chart,
    mel: np.ndarray,
    rng: random.Random,
    aug: Augment,
    predicted: TimingMap | None = None,
) -> tuple[ChartSet, Chart, SongAudio, ChartSet]:
    """Apply ``aug`` to a one-chart set. Returns the chart set and chart to train on, the
    audio as heard, and the pre-warp chart set (whose timing places texture units).

    Order: canonical tempo -> (sometimes) the audio-predicted grid -> BPM octave
    re-notation (x2 or x1/2, same sound) -> tempo warp
    (whole song ``k`` times faster, and short beat spans at their own speed; beats stay,
    the timing map and the audio change).
    """
    if aug.canonical:
        cs, chart = canonicalize(cs, chart)
        cs = msgspec.structs.replace(cs, charts=(chart,))
    if predicted is not None and rng.random() < aug.p_predicted:
        from urfm.beat.decode import for_game
        from urfm.beat.metrics import grid_fit

        old = cs.timing_of(chart)
        grid = for_game(predicted, chart.mechanics)
        onsets = old.us_array(
            [e.time.beat for e in chart.events], [e.time.residual_us for e in chart.events]
        )
        if len(onsets) and grid_fit(grid, onsets / 1e6) >= aug.min_grid_fit:
            moved = retime_chart(chart, old, grid)
            chart = rescroll(chart, old, moved, grid)
            cs = msgspec.structs.replace(cs, charts=(chart,))
            PIPELINE[f"{chart.mechanics}/predicted_grid"] += 1
    timing = cs.timing_of(chart)
    bpms = [float(t.bpm) for t in timing.tempos]
    hi, lo = max(bpms), min(bpms)
    octave = rng.random() < aug.p_octave
    if octave and aug.octave_factors:
        ok = [
            (f, w)
            for f, w in aug.octave_factors
            if lo * f >= aug.min_bpm and hi * f <= aug.max_bpm / 1.25
        ]
        if ok:
            factor = Fraction(rng.choices([f for f, _ in ok], [w for _, w in ok])[0])
            cs = rescale_beats(cs, factor)
            chart = cs.charts[0]
            timing = cs.timing_of(chart)
            hi, lo = hi * float(factor), lo * float(factor)
    elif octave:
        if hi * 2 <= aug.max_bpm / 1.25 and (lo < 150 or rng.random() < 0.5):
            factor = Fraction(2)
        elif lo / 2 >= 50:
            factor = Fraction(1, 2)
        else:
            factor = Fraction(1)
        if factor != 1:
            cs = rescale_beats(cs, factor)
            chart = cs.charts[0]
            timing = cs.timing_of(chart)
            hi, lo = hi * float(factor), lo * float(factor)
    plain = cs
    k = _log_uniform(rng, *aug.speed) if rng.random() < aug.p_speed else 1.0
    k = min(k, aug.max_bpm / (hi * aug.local[1]))
    spans: list[tuple[Fraction, Fraction, float]] = []
    events = chart.events
    if events and rng.random() < aug.p_local:
        first = min(e.time.beat for e in events)
        last = max(e.time.beat for e in events)
        stops = {s.beat for s in timing.stops}
        for _ in range(rng.randint(1, aug.max_spans)):
            length = Fraction(rng.choice((1, 2, 4, 8)), 2)
            if last - first <= length + 1:
                break
            start = Fraction(rng.randrange(int(first * 2), int((last - length) * 2)), 2)
            end = start + length
            if start in stops or end in stops:
                continue
            spans.append((start, end, _log_uniform(rng, *aug.local)))
    spans.sort()
    knots: list[Fraction] = []
    speeds = [_frac(k)]
    for a, b, s in spans:
        if knots and a < knots[-1]:
            continue
        if knots and a == knots[-1]:
            speeds[-1] = _frac(k * s)
            knots.append(b)
        else:
            knots += [a, b]
            speeds += [_frac(k * s)]
        speeds.append(_frac(k))
    if k == 1.0 and not knots:
        return cs, chart, SongAudio(mel), plain
    new_timing, warp = warp_timing(timing, knots, speeds)
    if any(e.time.residual_us for e in chart.events):  # osu! off-grid timestamps
        chart = map_chart_times(chart, _residual_fn(knots, speeds))
    chart = msgspec.structs.replace(chart, timing=new_timing)
    cs = msgspec.structs.replace(cs, charts=(chart,))
    return cs, chart, SongAudio(mel, warp), plain


def _frac(x: float) -> Fraction:
    return Fraction(round(x * 4096), 4096)


def _residual_fn(knots: list[Fraction], speeds: list[Fraction]):
    def fn(t: Time) -> Time:
        if not t.residual_us:
            return t
        return Time(t.beat, round(t.residual_us / speeds[bisect.bisect_right(knots, t.beat)]))

    return fn


SLICE_MARGIN_US = 4


@dataclass
class PreparedChart:
    """Per-chart work done once per visit, so each window costs O(events in window)."""

    chart: Chart
    prefix: list[int]
    windows: list[tuple[int, int]]
    onsets: list[int]
    events: tuple[Event, ...]
    """Events sorted by absolute onset; ``onsets[i]`` is the onset of ``events[i]``."""
    cache: dict = field(default_factory=dict)
    """Per-event token bodies (everything but the position), shared by all windows."""
    song: tuple[np.ndarray, np.ndarray] | None = None
    """Song units and their start µs, computed on first use and shared by windows."""
    compiled: object | None = None
    """``CompiledChart`` from the tokenizer when it supports compilation (integer fast path)."""
    cond: tuple[np.ndarray, np.ndarray] | None = None
    """Normalised conditions and mask (condition models)."""
    audio: SongAudio | None = None
    texture_timing: TimingMap | None = None
    """Timing that places texture units before any tempo warp (``None``: the chart's)."""
    sections: SectionInfo | None = None
    """Section length, onsets and audio similarity (R6), computed on first use."""

    def between(self, lo: int, hi: int, tail: int | None = None) -> Chart:
        """Events with onset near ``[lo, hi)``; onsets are float-derived, so the slice keeps
        a small margin and the tokenizer's filter makes the final decision. ``tail`` keeps
        only the last ``tail`` events (context only needs the most recent ones)."""
        a = bisect.bisect_left(self.onsets, lo - SLICE_MARGIN_US)
        b = bisect.bisect_left(self.onsets, hi + SLICE_MARGIN_US)
        if tail is not None:
            a = max(a, b - tail)
        return msgspec.structs.replace(self.chart, events=self.events[a:b])


def prepare(
    tok: Tokenizer,
    cs: ChartSet,
    chart: Chart,
    mel_frames: int,
    tiers: TierTable | None = None,
    features: Features | None = None,
    audio: SongAudio | None = None,
    texture_timing: TimingMap | None = None,
) -> PreparedChart:
    """Windows tile the whole audio (shared with generation), not just the charted span."""
    features = features or Features()
    timing = cs.timing_of(chart)
    events = chart.events
    approx = timing.us_array([e.time.beat for e in events], [e.time.residual_us for e in events])
    order = np.argsort(approx, kind="stable")
    if audio is not None:
        mel_frames = audio.frames
    if features.cond:
        from urfm.tokenize.tokenizer import chart_lanes

        prefix = cond_prefix(tok, chart.mechanics, chart_lanes(chart))
        cond = chart_conditions(cs, chart)
    else:
        prefix, cond = chart_prefix(tok, tiers, cs, chart), None
    return PreparedChart(
        chart=chart,
        prefix=prefix,
        windows=audio_windows(timing, round(mel_frames / MEL.fps * 1_000_000)),
        onsets=np.round(approx[order]).astype(np.int64).tolist(),
        events=tuple(events[i] for i in order),
        compiled=compile_fn(cs, chart) if (compile_fn := getattr(tok, "compile", None)) else None,
        cond=cond,
        audio=audio,
        texture_timing=texture_timing,
    )


@dataclass
class SectionInfo:
    k: int
    beats: np.ndarray
    """Sorted unique onset beats (float)."""
    ka: np.ndarray
    """Whether each onset is a ``ka`` hit."""
    first: int
    """Index of the first section with audio descriptors."""
    sim: np.ndarray
    threshold: float
    extras: dict | None = None
    """R6.1 plan values by section (whole song), when requested."""


def onset_beats(cs: ChartSet, chart: Chart, timing: TimingMap) -> tuple[np.ndarray, np.ndarray]:
    """Unique onset beats of ``chart`` on ``timing`` and whether each is a ``ka`` hit."""
    events = playable_events(chart)
    if not events:
        return np.zeros(0), np.zeros(0, bool)
    us = cs.timing_of(chart).us_array(
        [e.time.beat for e in events], [e.time.residual_us for e in events]
    )
    beats = beats_at_us(timing, np.asarray(us, np.float64))
    ka = np.array([e.role == "taiko:ka" for e in events])
    order = np.argsort(beats, kind="stable")
    beats, ka = beats[order], ka[order]
    keep = np.concatenate([[True], np.diff(beats) > 1e-6])
    starts = np.flatnonzero(keep)
    return beats[keep], np.logical_or.reduceat(ka, starts) if len(starts) else ka


def long_spans(chart: Chart) -> list[tuple[float, float]]:
    """Beat spans of holds, rolls and balloons."""
    return [
        (float(e.time.beat), float(e.end.beat))
        for e in chart.events
        if e.primitive in (Primitive.HOLD, Primitive.REPEAT) and e.end is not None
    ]


def section_info(
    prep: PreparedChart, cs: ChartSet, mel: np.ndarray, extras: bool = False
) -> SectionInfo:
    if prep.sections is not None and extras and prep.sections.extras is None:
        info = prep.sections
        n = len(info.sim)
        timing = cs.timing_of(prep.chart)
        info.extras = plan_extras(
            timing,
            info.beats,
            info.ka,
            long_spans(prep.chart),
            info.first,
            info.first + n - 1,
            info.k,
        )
    if prep.sections is None:
        timing = cs.timing_of(prep.chart)
        k = section_beats(timing)
        beats, ka = onset_beats(cs, prep.chart, timing)
        first, feats = section_audio(mel, prep.texture_timing or timing, k)
        sim = similarity(feats)
        prep.sections = SectionInfo(k, beats, ka, first, sim, ref_threshold(sim))
        if extras:
            prep.sections.extras = plan_extras(
                timing, beats, ka, long_spans(prep.chart), first, first + len(sim) - 1, k
            )
    return prep.sections


def perturb_plan(p: SectionPlan, rng: random.Random, prob: float) -> SectionPlan:
    """Move each plan value by one bin with probability ``prob`` (within range)."""
    sizes = {"dens": 16, "fine": SHARE_BINS, "rest": SHARE_BINS, "ka": SHARE_BINS}
    sizes["sub"] = len(GRID_VALUES)
    sizes |= {"peak": 16, "rep": REP_BINS, "var": VAR_BINS, "sw": SHARE_BINS, "long": SHARE_BINS}
    out = p._asdict()
    for name, n in sizes.items():
        if out[name] is not None and rng.random() < prob:
            out[name] = min(max(out[name] + rng.choice((-1, 1)), 0), n - 1)
    return SectionPlan(**out)


def noisy(chart: Chart, rng: random.Random) -> Chart:
    """Corrupt events like a model's own output: drop 10%, swap don/ka on 10%, move 5% by a
    16th."""
    swap = {"taiko:don": "taiko:ka", "taiko:ka": "taiko:don"}
    taken = {e.time for e in chart.events}
    out = []
    for e in chart.events:
        if rng.random() < 0.1:
            continue
        if e.role in swap and rng.random() < 0.1:
            e = msgspec.structs.replace(e, role=swap[e.role])
        if rng.random() < 0.05:
            step = Fraction(rng.choice((-1, 1)), 4)
            moved = Time(e.time.beat + step, e.time.residual_us)
            if moved.beat >= 0 and moved not in taken:  # never onto another note
                taken.add(moved)
                e = msgspec.structs.replace(e, time=moved)
        out.append(e)
    return msgspec.structs.replace(chart, events=tuple(out))


def song_units(
    prep: PreparedChart, mel: np.ndarray, timing: TimingMap, texture: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Song context units for ``prep`` (cached on it): texture (positions in beats, so a
    tempo warp only moves units in time) or pooled mel blocks."""
    if prep.song is None:
        if texture:
            base = prep.texture_timing or timing
            feats, unit_us = song_texture(mel, base)
            warp = prep.audio.warp if prep.audio is not None else None
            if warp is not None:
                unit_us = np.round(warp.forward(unit_us.astype(np.float64))).astype(np.int64)
            prep.song = (feats, unit_us)
        else:
            prep.song = song_summary(mel)
    return prep.song


def make_sample(
    tok: Tokenizer,
    cs: ChartSet,
    chart: Chart | PreparedChart,
    mel: np.ndarray,
    window: tuple[int, int],
    max_dec_len: int,
    features: Features | None = None,
    rng: random.Random | None = None,
) -> Sample:
    """``prefix BOS context SEP WDENS GRID REST events [EOS]``; context and body come from
    the tokenizer's shared definitions so generation sees exactly the same layout.
    ``rng`` (training) enables condition dropout."""
    features = features or Features()
    prep = chart if isinstance(chart, PreparedChart) else prepare(tok, cs, chart, mel.shape[0])
    timing = cs.timing_of(prep.chart)
    start_us, end_us = window
    lo = round(timing.beat_to_us_exact(timing.us_to_beat(start_us) - CONTEXT_BEATS - 1))
    corrupt = rng is not None and features.ctx_noise > 0 and rng.random() < features.ctx_noise
    # 2x headroom covers non-interactive events and same-onset groups at the cut
    ctx_chart = prep.between(lo, start_us, tail=2 * CONTEXT_EVENTS + 8)
    if corrupt:
        assert rng is not None
        ctx = tok.context(cs, noisy(ctx_chart, rng), start_us)
    else:
        ctx = tok.context(cs, ctx_chart, start_us, cache=prep.cache, compiled=prep.compiled)
    plans, k, ref = None, None, []
    if features.sections or features.ref:
        info = section_info(prep, cs, mel, features.plan_extra)
        k = info.k
        a, b = window_sections(timing, start_us, end_us, k)
        if features.sections:
            plans = plan_sections(timing, info.beats, info.ka, a, b, k, info.extras)
            if rng is not None:
                plans = [perturb_plan(p, rng, features.plan_noise) for p in plans]
        if features.ref:
            pair = choose_ref(info.sim, info.threshold, info.first, a, b)
            if pair is not None:
                j, i = pair
                r0 = round(timing.beat_to_us_exact(Fraction(j * k)))
                r1 = round(timing.beat_to_us_exact(Fraction((j + 1) * k)))
                wb = window_start_beat(timing, start_us)
                if corrupt:
                    assert rng is not None
                    ref = tok.ref_block(cs, noisy(prep.between(r0, r1), rng), j, i, wb, k)
                else:
                    ref = tok.ref_block(
                        cs,
                        prep.between(r0, r1),
                        j,
                        i,
                        wb,
                        k,
                        cache=prep.cache,
                        compiled=prep.compiled,
                    )
    head = [*prep.prefix, tok.bos_id]
    room = max_dec_len - len(head) - 2  # SEP + EOS
    body, truncated = tok.encode_window(
        cs,
        prep.between(start_us, end_us),
        start_us,
        end_us,
        room,
        cache=prep.cache,
        compiled=prep.compiled,
        **({"sections": plans, "k": k} if plans is not None else {}),
    )
    if len(ref) + len(ctx) + len(body) > room:  # memory, then context, are dropped whole
        ref = []
    if len(ctx) + len(body) > room:
        ctx = []
    tokens = [*head, *ref, *ctx, tok.sep_id, *body]
    if not truncated:
        tokens.append(tok.eos_id)
    feats = tok.frame_features(timing, start_us, N_FRAMES).astype(np.float32, copy=False)
    units, unit_us = song_units(prep, mel, timing, features.texture)
    marker = (
        unit_marker(unit_us, start_us, end_us)
        if features.texture
        else song_marker(unit_us, start_us, end_us)
    )
    cond = cond_mask = None
    if prep.cond is not None:
        cond, cond_mask = prep.cond[0].copy(), prep.cond[1].copy()
        if rng is not None:
            if rng.random() < features.cond_drop_all:
                cond_mask[:] = False
            else:
                for i in range(len(cond_mask)):
                    if rng.random() < features.cond_drop:
                        cond_mask[i] = False
            cond = np.where(cond_mask, cond, 0.0).astype(np.float32)
    return Sample(
        mel=prep.audio.window(start_us) if prep.audio is not None else slice_mel(mel, start_us),
        feats=feats,
        tokens=np.asarray(tokens, dtype=np.int64),
        sep=len(head) + len(ref) + len(ctx),
        truncated=truncated,
        mechanics=prep.chart.mechanics,
        song=units,
        song_marker=marker,
        cond=cond,
        cond_mask=cond_mask,
    )


def collate(samples: list[Sample], pad_id: int) -> dict[str, torch.Tensor]:
    longest = max(len(s.tokens) for s in samples)
    length = next((b for b in DEC_BUCKETS if b >= longest), longest)
    tokens = np.full((len(samples), length), pad_id, dtype=np.int64)
    mask = np.zeros((len(samples), length), dtype=bool)
    for i, s in enumerate(samples):
        tokens[i, : len(s.tokens)] = s.tokens
        mask[i, s.sep + 1 : len(s.tokens)] = True
    stats = Counter(take_drops())
    stats.update(PIPELINE)
    PIPELINE.clear()
    for smp in samples:
        stats[f"{smp.mechanics}/windows"] += 1
        stats[f"{smp.mechanics}/truncated_windows"] += smp.truncated
    batch = {
        "stats": dict(stats),  # type: ignore[dict-item]
        "mel": torch.from_numpy(np.stack([s.mel for s in samples])),
        "feats": torch.from_numpy(np.stack([s.feats for s in samples])),
        "tokens": torch.from_numpy(tokens),
        "loss_mask": torch.from_numpy(mask),
        "truncated": torch.tensor([s.truncated for s in samples]),
    }
    if all(s.song is not None and s.song_marker is not None for s in samples):
        batch.update(collate_song([(s.song, s.song_marker) for s in samples]))  # type: ignore[misc]
    if all(s.cond is not None and s.cond_mask is not None for s in samples):
        batch["cond"] = torch.from_numpy(np.stack([s.cond for s in samples]))  # type: ignore[misc]
        batch["cond_mask"] = torch.from_numpy(np.stack([s.cond_mask for s in samples]))  # type: ignore[misc]
    return batch


def collate_song(items: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Pad song units to a fixed size (static shapes for ``torch.compile``): ``SONG_MAX``
    pooled mel blocks, or ``MAX_UNITS`` texture units (float features)."""
    first = items[0][0]
    size = SONG_MAX if first.dtype == np.uint8 else MAX_UNITS
    song = np.zeros((len(items), size, first.shape[1]), dtype=first.dtype)
    marker = np.zeros((len(items), size), dtype=np.int64)
    length = np.zeros(len(items), dtype=np.int64)
    for i, (blocks, mark) in enumerate(items):
        n = min(len(blocks), size)
        song[i, :n], marker[i, :n], length[i] = blocks[:n], mark[:n], n
    return {
        "song": torch.from_numpy(song),
        "song_marker": torch.from_numpy(marker),
        "song_len": torch.from_numpy(length),
    }


class PredictedTimings:
    """Audio-predicted timing maps (``experiments/r4/predict_timing.py``), keyed like
    :class:`ChartRef` rows. Whether a chart fits its map is checked per chart at use
    (``Augment.min_grid_fit``), after the game's timing convention is applied."""

    def __init__(self, root: str | Path, min_fit: float = 0.0) -> None:
        self.root, self.min_fit = Path(root), min_fit
        self._maps: dict[tuple[str, int, int], bytes] | None = None

    def _load(self) -> dict[tuple[str, int, int], bytes]:
        out: dict[tuple[str, int, int], bytes] = {}
        for f in sorted(self.root.glob("data/*/*.parquet")):
            t = pq.read_table(f, columns=["row_group", "row", "timing", "grid_fit"])
            key = f"{f.parent.name}/{f.name}"
            for g, r, blob, fit in zip(
                *(t[c].to_pylist() for c in ("row_group", "row", "timing", "grid_fit")),
                strict=True,
            ):
                if fit is not None and fit >= self.min_fit:
                    out[(key, g, r)] = blob
        return out

    def get(self, file: str, row_group: int, row: int) -> TimingMap | None:
        if self._maps is None:
            self._maps = self._load()
        p = Path(file)
        blob = self._maps.get((f"{p.parent.name}/{p.name}", row_group, row))
        if blob is None:
            return None
        from urfm.ir.serialize import timing_from_msgpack

        return timing_from_msgpack(blob)


# -- streaming dataset -------------------------------------------------------------


class WindowStream(IterableDataset):
    """Infinite stream of length-bucketed batches drawn with game-balanced weights."""

    def __init__(
        self,
        index: Index,
        tokenizer: Tokenizer,
        batch_size: int,
        max_dec_len: int,
        seed: int = 0,
        buffer_batches: int = 8,
        tiers: TierTable | None = None,
        features: Features | None = None,
        augment: Augment | None = None,
        predicted: PredictedTimings | None = None,
    ) -> None:
        self.index = index
        self.tok = tokenizer
        self.batch_size = batch_size
        self.max_dec_len = max_dec_len
        self.seed = seed
        self.buffer_batches = buffer_batches
        self.tiers = tiers
        self.features = features or Features()
        self.augment = augment or Augment(canonical=False)
        self.predicted = predicted

    def _groups(self) -> list[tuple[str, int]]:
        return sorted({(c.file, c.row_group) for c in self.index.charts})

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        if info is not None:  # forked workers inherit the parent's counters; start clean
            take_drops()
            PIPELINE.clear()
        rng = random.Random(self.seed * 1009 + wid)
        groups = self._groups()[wid::nw] or self._groups()
        by_group: dict[tuple[str, int], list[ChartRef]] = {}
        for c in self.index.charts:
            by_group.setdefault((c.file, c.row_group), []).append(c)
        buffer: list[Sample] = []
        while True:
            made = 0
            rng.shuffle(groups)
            for key in groups:
                table = pq.ParquetFile(key[0]).read_row_group(key[1], columns=["mel", "chartset"])
                mels = table["mel"].to_pylist()
                sets = table["chartset"].to_pylist()
                lazies: dict[int, LazyChartSet] = {}
                m = max(self.augment.windows_per_visit, 1)
                for ref in by_group.get(key, []):
                    k = m * math.floor(ref.weight / m + rng.random())
                    if not k:
                        continue
                    lazy = lazies.get(ref.row)
                    if lazy is None:
                        lazy = lazies[ref.row] = LazyChartSet(sets[ref.row])
                    cs = lazy.view(ref.chart)  # decode only the sampled difficulty
                    mel = np.frombuffer(mels[ref.row], dtype=np.uint8).reshape(-1, MEL.n_mels)
                    if extreme_tempo(cs.timing_of(cs.charts[0])):
                        PIPELINE[f"{ref.mechanics}/skip_extreme_tempo"] += 1
                        continue
                    try:
                        pred = None
                        if self.predicted is not None and self.augment.p_predicted > 0:
                            pred = self.predicted.get(ref.file, ref.row_group, ref.row)
                        cs, chart, audio, plain = augment_chart(
                            cs, cs.charts[0], mel, rng, self.augment, pred
                        )
                        prep = prepare(
                            self.tok,
                            cs,
                            chart,
                            mel.shape[0],
                            self.tiers,
                            self.features,
                            audio,
                            plain.timing_of(plain.charts[0]),
                        )
                    except Exception as exc:
                        PIPELINE[f"{ref.mechanics}/prepare_error:{type(exc).__name__}"] += 1
                        continue
                    if not prep.windows:
                        continue
                    weights = window_weights(prep, self.augment.fast_boost)
                    if weights is not None:
                        k = math.floor(k * (1 + self.augment.fast_boost / 4) + rng.random())
                    for _ in range(k):
                        try:
                            window = (
                                rng.choices(prep.windows, weights)[0]
                                if weights is not None
                                else rng.choice(prep.windows)
                            )
                            buffer.append(
                                make_sample(
                                    self.tok,
                                    cs,
                                    prep,
                                    mel,
                                    window,
                                    self.max_dec_len,
                                    self.features,
                                    rng,
                                )
                            )
                            made += 1
                        except Exception as exc:
                            PIPELINE[f"{ref.mechanics}/sample_error:{type(exc).__name__}"] += 1
                            continue
                if len(buffer) >= self.batch_size * self.buffer_batches:
                    yield from self._batches(buffer, rng)
                    buffer = []
            if not self.index.charts:
                return
            if not made:
                raise RuntimeError(f"a full pass produced no samples; failures: {dict(PIPELINE)}")

    def _batches(
        self, buffer: list[Sample], rng: random.Random
    ) -> Iterator[dict[str, torch.Tensor]]:
        rng.shuffle(buffer)
        buffer.sort(key=lambda s: len(s.tokens))
        batches = [
            buffer[i : i + self.batch_size]
            for i in range(0, len(buffer) - self.batch_size + 1, self.batch_size)
        ]
        rng.shuffle(batches)
        for b in batches:
            yield collate(b, self.tok.pad_id)


def loader(stream: IterableDataset, workers: int, prefetch: int = 4) -> DataLoader:
    """Batches as soon as any worker has one (``in_order=False``): workers yield in bursts (a
    buffer of sorted batches at a time), and in order the slowest worker would stall all."""
    return DataLoader(
        stream,
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=prefetch if workers > 0 else None,
        in_order=False,
    )


def fixed_windows(
    index: Index,
    tokenizer: Tokenizer,
    max_dec_len: int,
    limit: int = 256,
    seed: int = 0,
    tiers: TierTable | None = None,
    features: Features | None = None,
    canonical: bool = False,
) -> list[Sample]:
    """Deterministic validation windows: up to two per chart, at most ``limit`` overall
    (no augmentation beyond the canonical tempo, no condition dropout)."""
    rng = random.Random(seed)
    refs = list(index.charts)
    rng.shuffle(refs)
    out: list[Sample] = []
    cache: dict[tuple[str, int], tuple[list, list]] = {}
    for ref in refs:
        if len(out) >= limit:
            break
        key = (ref.file, ref.row_group)
        if key not in cache:
            t = pq.ParquetFile(ref.file).read_row_group(ref.row_group, columns=["mel", "chartset"])
            cache[key] = (t["mel"].to_pylist(), t["chartset"].to_pylist())
        mels, sets = cache[key]
        cs = from_msgpack(sets[ref.row])
        chart = cs.charts[ref.chart]
        mel = np.frombuffer(mels[ref.row], dtype=np.uint8).reshape(-1, MEL.n_mels)
        if extreme_tempo(cs.timing_of(chart)):
            PIPELINE[f"{ref.mechanics}/val_skip_extreme_tempo"] += 1
            continue
        if canonical:
            cs, chart = canonicalize(cs, chart)
            cs = msgspec.structs.replace(cs, charts=(chart,))
        prep = prepare(tokenizer, cs, chart, mel.shape[0], tiers, features)
        for s in rng.sample(prep.windows, min(2, len(prep.windows))):
            try:
                out.append(make_sample(tokenizer, cs, prep, mel, s, max_dec_len, features))
            except Exception as exc:
                PIPELINE[f"{ref.mechanics}/val_sample_error:{type(exc).__name__}"] += 1
    return out[:limit]
