import random
from pathlib import Path

import numpy as np
import torch

from urfm.beat.data import SongRef, collate, make_chunk
from urfm.beat.decode import clean, peaks, to_timing
from urfm.beat.labels import song_labels
from urfm.beat.metrics import beat_scores, grid_fit
from urfm.beat.model import BeatConfig, BeatModel, beat_loss
from urfm.codecs import get
from urfm.data.features import MEL

FIXTURES = Path(__file__).parent / "fixtures" / "tja"


def tempo_changing_beats() -> np.ndarray:
    """120 BPM, then two beats at 180, half a measure at 90, then 150."""
    periods = [0.5] * 16 + [1 / 3] * 2 + [2 / 3] * 2 + [0.4] * 30
    return 1.0 + np.concatenate([[0], np.cumsum(periods)])


def click_mel(beats: np.ndarray, seconds: float) -> np.ndarray:
    mel = np.zeros((int(seconds * MEL.fps), MEL.n_mels), np.uint8)
    for t in beats:
        f = round(t * MEL.fps)
        if f < len(mel):
            mel[f] = 250
    return mel


def test_chunk_labels_follow_the_warped_audio():
    beats = tempo_changing_beats()
    mel = click_mel(beats, beats[-1] + 2)
    ref = SongRef("x", 0, 0, beats, beats[::4], (beats[0], beats[-1]))
    rng = random.Random(3)
    for _ in range(20):
        c = make_chunk(mel, ref, rng, augment=True, frames=1024)
        loud = set(np.nonzero(c.mel.max(1) > 5)[0].tolist())
        labelled = np.nonzero(c.beats[0] == 2)[0]
        inside = [f for f in labelled if c.mask[f]]
        assert inside, "chunk overlaps the labelled span"
        # every labelled beat frame sits on (or next to) a click in the resampled mel
        assert all(any(f + d in loud for d in (-1, 0, 1)) for f in inside)


def test_decode_recovers_short_tempo_changes():
    beats = tempo_changing_beats()
    act = np.zeros(int((beats[-1] + 1) * MEL.fps))
    for t in beats:
        f = t * MEL.fps
        i = int(np.floor(f))
        act[i], act[i + 1] = 1 - (f - i), f - i  # sub-frame position in two frames
    est = clean(peaks(act, 0.3))
    assert beat_scores(beats, est)["beat_f25"] == 1.0
    timing = to_timing(est)
    got = timing.us_array(list(range(len(beats)))) / 1e6
    assert np.abs(got - beats).max() < 0.012
    bpms = [float(t.bpm) for t in timing.tempos]
    assert len(bpms) == 4
    assert all(abs(b / w - 1) < 0.02 for b, w in zip(bpms, [120, 180, 90, 150], strict=True))
    assert grid_fit(timing, beats[:-1] + np.diff(beats) / 2) == 1.0


def test_labels_from_a_chart():
    cs = get("tja").read_path(FIXTURES / "visual_bpm.tja")
    out = song_labels(cs, 60.0)
    assert out is not None
    beats, _downbeats, (lo, hi) = out
    assert np.all(np.diff(beats) > 0)
    # the BPM-32 crawl is visual: beats stay at 128 BPM through it
    period = np.diff(beats[(beats > 2.0) & (beats < 5.0)])
    assert np.allclose(period, 60 / 128, atol=1e-6)
    assert lo <= beats[0] and hi >= beats[-1]


def test_model_and_loss_run():
    model = BeatModel(BeatConfig(d_model=32, layers=1, heads=2, channels=(4, 4, 4)))
    beats = tempo_changing_beats()
    mel = click_mel(beats, beats[-1] + 2)
    ref = SongRef("x", 0, 0, beats, beats[::4], (beats[0], beats[-1]))
    rng = random.Random(0)
    batch = collate([make_chunk(mel, ref, rng, True, frames=256) for _ in range(2)])
    logits = model(batch["mel"])
    assert logits.shape == (2, 256, 2)
    loss, level = beat_loss(logits, batch["beats"] / 2, batch["downbeats"] / 2, batch["mask"], 0.5)
    loss.backward()
    assert torch.isfinite(loss) and level.shape == (2,)
