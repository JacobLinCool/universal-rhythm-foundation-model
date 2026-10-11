"""Evaluation protocol shared by every generation benchmark.

A benchmark item is a song (mel + human chart set). The protocol fixes three choices so
results are comparable across scripts and checkpoints:

* **speed** -- the song may be played ``k`` times faster first (stress test beyond the
  training distribution); the human charts are re-timed the same way and stay the reference;
* **timing** -- the model gets either the human timing map (``oracle``) or one predicted from
  audio by a beat model (``audio``, the real use case);
* **request** -- the model is asked for a chart like each human chart: same game and lanes,
  conditions measured on the human chart (old checkpoints: tier/density tokens), and the
  human chart's mean density as the calibrated plan's target.
"""

from __future__ import annotations

from fractions import Fraction

import msgspec
import numpy as np

from urfm.data.features import MEL
from urfm.ir.chart import Chart, ChartSet, Difficulty
from urfm.ir.timing import TimingMap
from urfm.ir.transform import TimeWarp, warp_chartset
from urfm.tokenize.tokenizer import chart_lanes, onset_rate


def speed_up(cs: ChartSet, mel: np.ndarray, speed: float) -> tuple[ChartSet, np.ndarray]:
    """Play the song ``speed`` times faster: mel resampled, every timing map re-timed."""
    if speed == 1:
        return cs, mel
    from urfm.beat.data import resample

    k = Fraction(speed).limit_denominator(1000)
    cs, warp = warp_chartset(cs, (), (k,))
    seconds = TimeWarp((), (), warp.first_slope, warp.last_slope)
    frames = int(mel.shape[0] / float(k))
    return cs, resample(mel, seconds, 0.0, frames)


def predicted_timing(beat_model, mel: np.ndarray, device, mechanics: str = "taiko") -> TimingMap:
    """Timing map from audio, in ``mechanics``' timing convention."""
    from urfm.beat.decode import for_game, predict_timing

    return for_game(predict_timing(beat_model, mel, device), mechanics)


def request(model, meta: dict, tok, cs: ChartSet, chart: Chart, mel: np.ndarray, timing):
    """A :class:`~urfm.train.generate.Job` asking ``model`` for a chart like ``chart``.

    On a timing map other than the chart's own (audio-predicted), notation-dependent
    conditions (onsets per beat, share off the 16th grid) are left out, as in real use."""
    from urfm.train.conditions import without_notation
    from urfm.train.generate import Job, reference_inputs

    prefix, cond = reference_inputs(model, meta, tok, cs, chart)
    if cond is not None and timing is not cs.timing_of(chart):
        cond = without_notation(cond)
    lanes = chart_lanes(chart)
    return Job(
        mel=mel,
        timing=timing,
        mechanics=chart.mechanics,
        prefix=prefix,
        cond=cond,
        difficulty=Difficulty(label=chart.difficulty.label, level=chart.difficulty.level),
        meta={"lanes": str(lanes)} if lanes else {},
        lanes=lanes,
        target_rate=onset_rate(cs, chart),
    )


def plan_job(job, meta: dict, planner, temperature: float = 0.7, seed: int = 0) -> None:
    """R6 models: plan the job's sections with the song planner and switch on the structure
    memory, as the model was trained (no-op for older models)."""
    feats = meta.get("features", {})
    job.use_ref = bool(feats.get("ref"))
    if not feats.get("sections"):
        return
    if planner is None:
        raise ValueError("a section-plan model needs the song planner")
    from urfm.train.planner import plan_song

    job.section_k, job.section_plans = plan_song(
        planner, job.mel, job.timing, job.mechanics, job.cond, temperature, seed
    )


def one_chart(cs: ChartSet, chart: Chart) -> ChartSet:
    return msgspec.structs.replace(cs, charts=(chart,))


def duration_s(mel: np.ndarray) -> float:
    return mel.shape[0] / MEL.fps


OCTAVES = (Fraction(1, 2), Fraction(1), Fraction(2))


def choose_octave(model, meta, tok, cs: ChartSet, chart: Chart, mel: np.ndarray, timing, **gen):
    """Notation octave for an audio-predicted map: generate ``chart``'s request on the map at
    half, same and double BPM and keep the one whose chart density is closest (in log) to the
    requested density. The beat model often picks half the tempo charters use for very fast
    songs, and 32nd notes at 150 BPM are almost absent from training data, so the model writes
    those passages only on the doubled grid. Returns ``(factor, rescaled timing)``."""
    import math

    from urfm.ir.transform import rescale_timing
    from urfm.train.generate import generate

    target = onset_rate(cs, chart)
    best = (math.inf, Fraction(1), timing)
    for f in OCTAVES:
        t = timing if f == 1 else rescale_timing(timing, f)
        job = request(model, meta, tok, cs, chart, mel, t)
        (out,) = generate(model, tok, [job], **gen)
        got = onset_rate(one_chart(ChartSet(timing=t, charts=(out,)), out), out)
        err = abs(math.log((got + 1e-3) / (target + 1e-3)))
        if err < best[0]:
            best = (err, f, t)
    return best[1], best[2]
