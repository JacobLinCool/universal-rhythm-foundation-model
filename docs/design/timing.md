# Timing from audio (`urfm.beat`, `urfm.data.canonical`)

At inference a model gets audio only: no BPM, no measures. R1–R3 were evaluated with the human
timing map, which hides the hardest part of charting newer songs: tempo changes that last half
a measure, several in a row, and visual tempo tricks.

## Musical vs visual tempo (`canonicalize`)

Taiko note speed is proportional to `BPM x scroll`, so charters change the BPM to change how
notes move (a few beats at BPM 32 inside a 257 BPM song, then back). That tempo is not
audible. `canonicalize(chartset, chart)` rewrites such segments as the surrounding musical
tempo plus a scroll multiplier `old BPM / musical BPM`:

- A segment folds into the previous musical tempo, or the song's dominant one, when its notes
  and its end sit on a **simpler** grid of that tempo than of its own (max denominator, at
  most 1/48). A real 150 → 180 change keeps its tempo (5/24-beat positions at 150); a crawl at
  1/8 of the base tempo folds (1/16 → 1/2 beats).
- Octave ratios with equally simple grids keep whichever tempo is closer to the dominant one.
- Measures keep their start times (lengths re-expressed in new beats); scroll controls are
  multiplied by the segment's factor, and one is added where the factor changes.
- Every event keeps its exact absolute time and on-screen speed (tested on all fixtures).
- Only games with BPM-dependent scrolling (`taiko`, `osu_taiko`); maps with warps, BPM outside
  10–2000 or zero-length meters are left alone.

## Beat model

- **Labels** (`labels.song_labels`): one representative chart per song (official Taiko, then
  osu! modes), canonical tempo, whole beats and measure starts within one measure of the
  charted notes.
- **Model** (`model.BeatModel`, ~5M): 2D conv frontend over (time, mel) that keeps the frame
  rate (11.6 ms) and shrinks frequency 128 → 4, then 6 pre-LN Transformer blocks with rotary
  positions, two logits per frame.
- **Loss**: class-balanced BCE (positive and negative frames averaged separately; neighbours of
  a beat frame have target 0.5). The beat term takes the best of four metrical levels (as
  labelled, doubled, and the two halves), because charts write the same music at 120 or 240
  BPM; balancing keeps the minimum from favouring sparser levels.
- **Augmentation** (`data.random_warp`): the song 0.75–1.4x faster, and up to six spans of half a
  beat to a measure at 0.75–1.33x, so the model sees more short tempo changes than any chart
  has. Gain, spectral tilt and a frequency mask on top.
- **Decoding** (`decode`): peaks (sub-frame parabola) → clean-up against the local period
  (drop beats much closer than it, fill whole-period gaps) → tempo segments, the longest runs a
  straight line through their end beats fits within 12 ms. No smoothness prior: a tempo can
  change on any beat and again right after. Measures come from downbeat peaks.

## Evaluation (`experiments/r4/eval_beat.py`, `urfm.beat.metrics`)

Beat F@70 ms and F@25 ms, the best F@70 over metrical levels, downbeat F@70, and **grid fit**: the
share of the human chart's notes within 15 ms of a 1/4 or 1/3 beat of the predicted map, which
is what a chart needs. `--warp` adds random short tempo changes with known ground truth.
