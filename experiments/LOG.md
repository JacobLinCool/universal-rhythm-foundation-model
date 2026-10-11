# Experiment log (12-hour quick-validation sprint, 2026-10-09)

Exploratory runs. Every run uses one seed, so there are no significance claims. The goals are a
working model and early signals. Compute: Colab L4 at 1.54 CU/h per VM, cap 50 CU. The ledger is
`runs/compute-ledger.jsonl`.

## Data: `JacobLinCool/urfm-ingest-v0` (private)

The ingest script is [`ingest_v0/ingest.py`](ingest_v0/ingest.py). It runs on the VM, and no data
passes through the local machine.

| Source | Songs | Failures | Notes |
| --- | --- | --- | --- |
| taiko1000 | 1,155 | 0 | Clean split pinned at `b72da46` (813/80/120 groups) |
| osu (`project-riz/osu-beatmaps`, 64 kbps Opus) | 21,314 | 1 | Every song with taiko, catch or mania maps; 5% of std-only songs |

- **Size on disk**: 35 GB for osu; 1.2 GB for Taiko.
- **Split**: osu songs whose normalised titles match a Taiko-1000 song inherit that song's split. All other songs are split by audio hash.
- **Charts without audio**: StepMania and maimai have no audio in the current corpora, so they are not in v0.

## Runs

| Run | Data | Model | Status | Key numbers |
| --- | --- | --- | --- | --- |
| V1 | Taiko | 10M | Done; superseded | Context origin inconsistent with generation; its vocabulary predates `WDENS` |
| **V1′** (`v1b-taiko`) | Taiko | 10M, 6k steps | Done | val loss 0.460; Taiko test (599 charts) F1@25 **0.553**, density error 12%, role accuracy 0.49 |
| **V2** (`v2-taiko-osutaiko`) | Taiko + osu!taiko | 15M, 6k steps | Done | val loss 0.439; Taiko test F1@25 **0.533**, role accuracy 0.45, 0 violations |
| V3-45M (`v3-all-45m`) | Taiko + all four osu! modes | 45M | Cancelled (time) | Earlier attempt: 110 windows/s, GPU 100% |
| **V3-15M** (`v3-all-15m`) | Taiko + all four osu! modes | 15M, stopped at 6.4k steps (best 6k) | Done | val loss 1.038 (still falling); Taiko F1@25 **0.561**; osu! mania 0.705, taiko 0.600, catch 0.458, std 0.406 |

V1′ F1@25 by difficulty:

| easy | normal | hard | oni | edit |
| --- | --- | --- | --- | --- |
| 0.50 | 0.53 | 0.56 | 0.61 | 0.66 |

V2 against V1′ on the same Taiko test set (F1@25):

| | easy | normal | hard | oni | edit | all |
| --- | --- | --- | --- | --- | --- | --- |
| V1′ (Taiko only, 10M) | 0.502 | 0.526 | 0.555 | 0.608 | 0.655 | 0.553 |
| V2 (+ osu!taiko, 15M) | 0.459 | 0.480 | 0.546 | **0.622** | **0.680** | 0.533 |

| V3 (+ all osu! modes, 15M) | **0.518** | **0.536** | **0.565** | 0.610 | 0.654 | **0.561** |

Adding osu!taiko helps the hard courses and hurts the easy ones. One hypothesis is that the
cross-game `TIER` derived from osu! difficulty names does not mean the same as Taiko's courses,
which is a concrete case of "shared token ≠ shared semantics". The next step is per-game
percentile tiers. This is not conclusive: one seed, and the model size and step count differ
between the two runs. V3 (all four osu! modes) recovers the easy and normal courses and gives the
best overall Taiko score.

Other results:

- **V2 on osu!taiko test songs**: F1@25 0.584 (403 charts).
- **V3 on osu! test songs**:

  | mania | osu!taiko | catch | std |
  | --- | --- | --- | --- |
  | 0.705 | 0.600 | 0.458 | 0.406 |

**Cross-game consistency (quick Exp C).** V3 generated Taiko, osu!taiko, mania, catch and std
charts for 24 Taiko test songs. Correlations of the 2 s onset-density curves:

| Comparison | r |
| --- | --- |
| Generated games, pairwise | 0.20–0.29 |
| Generated vs the human Taiko chart | 0.13–0.20 |

These are weak. Song-level density arcs are not yet shared or controlled. Caveats:

- Generation used the median density bin and let the model plan `WDENS` itself.
- The human reference is the last-listed course, which is usually easy.
- This matches review point 5: a 16 s window model cannot plan a whole song.

Playable V1′ samples (six test songs, every course, original audio, human `.tja` alongside):
`runs/v1b-taiko/playable/` (local, git-ignored); each song has `human.tja`, `urfm-v1.tja`, `urfm-v2.tja` and `urfm-v3.tja`. Results (no checkpoints): `runs/results/`. Checkpoints: HF `JacobLinCool/urfm-runs` (private).

## Findings and fixes, in discovery order

1. **Data-loader cost scaled with chart length.** Every window re-scanned the whole chart with
   `Fraction` math. Charts are now prepared once and sliced per window, and sub-beat snapping is
   cached. A window went from about 7 ms to about 1 ms.
2. **Training and generation used different context and windows** (raised in review). There is now
   one shared definition:
   - `audio_windows` covers the whole audio, including silence.
   - `context` uses an integral-beat origin.
   - Truncation happens only at event boundaries.
   - Every drop is counted.
   - Added the `WDENS` per-window density plan token.
3. **GPU generation crashed.** The KV cache was fp32 while autocast produced bf16. Fixed, with a
   CPU bf16 check.
4. **Writers assumed their own native fields.** The TJA writer emitted zero measures for
   generated charts, and the osu! writer emitted fractional milliseconds. Both are fixed, and a
   new test writes and re-reads plain IR charts for every codec.
5. **Taiko generation produced simultaneous and overlapping notes.**
   - The grammar now forbids `POS_SAME` when the mechanics spec has `max_simultaneous == 1`.
   - A post-pass (`enforce_single_action`) drops notes that fall inside a drumroll.
6. **osu!mania was truncated in 18–23% of windows at 1,024 tokens** (also 3–6% for catch, about
   0% for Taiko). This systematically dropped dense or complex charts. With `max_dec_len` 2048
   (buckets up to 2048), truncation is 0%.
7. **`TimingMap.us_to_beat` returned the earliest beat** when negative stops (osu! red lines) make
   time non-monotonic. Windows that start on a red line got an off-grid origin, which shifted every
   position in the window by a fraction of a beat. It now returns the latest beat, consistent with
   the vectorised tokenizer path. A regression test covers this.
8. **Pipeline accounting double-counted at start-up.** Forked workers inherited the parent's drop
   counters. Workers now start with clean counters, and validation drops are recorded in
   `config.json`.
9. **Known and visible, but not fixed in this sprint.** Some osu!taiko gimmick sections (extreme
   BPM) put events more than 128 beats into a 16 s window. These events show up as
   `outside_position_range` drops. The fix is to fold extreme tempi into the musical range for
   tokenization.
10. **V2 attempt 1 crashed** (DataLoader worker died). The cause was memory pressure from a
    concurrent export job on the same VM. Rule: never run side jobs next to training.
11. **All three runs (V2, V3-15M, V3-45M) were OOM-killed.** In each case a single data worker
    grew to about 39 GB.
    - **Cause**: osu! gimmick red lines at absurd BPM turn a song into astronomically many
      beats. Window tiling enumerated every measure (`measure_starts`).
    - **Fix**: tiling now locates measures arithmetically (O(windows)). Charts with BPM outside
      [10, 2000] are skipped for training and counted as `skip_extreme_tempo`. Regression test:
      10^7 BPM tiles in under 1 s.
    - **Cost**: roughly 1.5 h of both VMs was lost (about 4.6 CU) because the crash went
      unnoticed. Monitoring now alerts on `abort` or OOM as well as on completion.

# Round 2 (2026-10-09 afternoon)

## Loader throughput on real osu! data

The synthetic benchmark hid the real bottleneck. Each fix below was measured on real shards on
the VM, with one worker:

| Change | Samples/s per worker |
| --- | --- |
| Baseline (round 1) | ~30 |
| Compiled timing maps cached by identity (`_compile` hashed every tempo/stop `Fraction`) | 35 |
| Per-event token body cache | 45 |
| Integer-tick compiled charts (`CompiledChart`, 8,648,640 ticks per beat), lazy bodies; a test proves the output matches the `Fraction` path token for token | **76** |

During training this gave R2-all a median GPU utilisation of 93% at 245 windows/s, and R2-taiko
100% at 339 windows/s. In round 1, GPU utilisation was about 30%.

## Fair transfer comparison

Both runs: 15M model, 8k steps, per-game tiers, the same loader. Only the training data differs.

| Run | Data | easy | normal | hard | oni | edit | **Taiko F1@25** | role acc | density err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2-taiko | Taiko | 0.509 | 0.522 | 0.556 | 0.613 | 0.674 | **0.555** | 0.494 | 0.112 |
| R2-all | Taiko + osu! (4 modes) | 0.499 | 0.525 | 0.554 | 0.591 | 0.622 | 0.546 | 0.429 | 0.150 |

**The round-1 impression that V3 was best is retracted.** That comparison mixed model sizes and
step counts. Under equal conditions, mixed training is slightly worse on Taiko, especially on the
hard courses and on don/ka choice.

A likely reason: with a fixed number of steps, balanced sampling gives Taiko only about 15–20% of
the updates, and the model's capacity is shared across games. The next test pre-trains on all
games, then fine-tunes on Taiko, and compares against an equal-recipe continuation of R2-taiko
(`chain_ft.sh`).

R2-all on osu! test songs (F1@25):

| mania | osu!taiko | catch | std |
| --- | --- | --- | --- |
| 0.693 | 0.560 | 0.405 | 0.370 |

**Cross-game consistency** improved slightly with per-game tiers and with the densest human chart
as the reference. Correlations of 2 s density curves:

| Comparison | Round 1 | Round 2 |
| --- | --- | --- |
| Generated games, pairwise | 0.20–0.29 | 0.18–0.37 |
| Generated vs human | 0.13–0.20 | 0.17–0.33 |

These correlations are still weak. Song-level context, now implemented with `--song-context true`,
is the next lever.

## Operations

- **VM A sat idle for about 25 minutes.** It was a fresh VM without `/content/logs`, so the chain's
  log redirects failed silently. `chain_r2.sh` now creates the directory.
- **The watchers now also alert on stalls**: no change in metrics for three checks.

## Does cross-game pre-training help Taiko when Taiko gets full training? (`chain_ft.sh`)

Both arms: identical fine-tuning (4k Taiko steps, lr 1e-4) from an 8k-step checkpoint; only the
first 8k steps differ.

| Arm | best val loss (Taiko) | F1@25 | easy | normal | hard | oni | edit | role | density err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all games → Taiko | **0.421** | 0.553 | 0.504 | 0.511 | 0.557 | 0.616 | 0.680 | 0.46 | 0.121 |
| Taiko → Taiko | 0.442 | 0.558 | 0.510 | 0.530 | 0.556 | 0.616 | 0.680 | 0.47 | 0.118 |

- **Likelihood**: cross-game pre-training improves Taiko token likelihood (validation loss 0.442 →
  0.421). This is the first positive transfer signal.
- **Generation**: F1 is unchanged (0.553 vs 0.558, within noise).
- **Taiko-only plateau**: Taiko-only F1 stays at 0.555–0.558 with 4k more steps.
- **Hypothesis**: the bottleneck is decoding (temperature 0.9 sampling) or the single-reference
  F1 metric, not what the model has learned. Next: a temperature sweep (eval only), then R3 with
  song context.

## Decoding temperature (eval only, `ft-r2-all` checkpoint, Taiko test)

| temperature | F1@25 | role | IOI hist L1 (lower = closer to human rhythm distribution) |
| --- | --- | --- | --- |
| 0.9 | 0.553 | 0.46 | **0.49** |
| **0.6** | **0.595** | **0.515** | 0.75 |
| 0.3 | 0.597 | 0.502 | 0.97 |

- **F1 was limited by decoding, not by the model.** Moving to temperature 0.6 adds 0.042 F1 and
  pushes role accuracy above 0.5 for the first time.
- **Lower temperatures cost rhythmic diversity.** The IOI distribution drifts away from the human
  one.
- **0.6 is the new default** in `generate()`, `evaluate.py` and `export_playable.py`.

## Song-level context (R3 vs R2-all; all games, 8k steps, temperature 0.9)

| Run | F1@25 | P | R | role | density err |
| --- | --- | --- | --- | --- | --- |
| R2-all | 0.546 | 0.554 | 0.547 | 0.429 | 0.150 |
| **R3-all + song context** | **0.570** | 0.547 | **0.607** | 0.443 | 0.192 |

- Song context improves Taiko F1 by 0.024, and every course improves.
- It also over-generates: recall rises and density error grows.

R3 on osu! test songs and cross-game consistency (vs R2-all):

| | mania | catch | std | osu!taiko |
| --- | --- | --- | --- | --- |
| R2-all | 0.693 | 0.405 | 0.370 | 0.560 |
| **R3** | **0.741** | **0.460** | 0.376 | 0.552 |

| Density-curve correlation | R2-all | R3 |
| --- | --- | --- |
| Taiko (generated) vs human | 0.203 | **0.303** |
| osu!taiko vs human | 0.241 | **0.307** |
| Taiko vs osu!taiko (both generated) | 0.293 | **0.370** |
| mania vs osu!taiko (both generated) | 0.368 | **0.446** |

Song context improves both quality and the song-level agreement that review point 5 targeted.

## Best recipe so far (all at temperature 0.6, Taiko test, 599 charts)

| Model | Taiko val loss | F1@25 | easy | normal | hard | oni | edit | role | density err | IOI L1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Taiko only (12k steps) | 0.442 | 0.591 | 0.541 | 0.558 | 0.590 | 0.653 | 0.703 | 0.516 | **0.111** | **0.653** |
| all games → Taiko | 0.421 | 0.595 | 0.546 | 0.565 | 0.599 | 0.653 | 0.691 | 0.515 | 0.122 | 0.750 |
| **all games + song context → Taiko** | **0.411** | **0.596** | **0.556** | 0.555 | 0.595 | **0.657** | **0.707** | **0.517** | 0.119 | 0.739 |

- **Validation loss falls consistently** with cross-game pre-training and with song context:
  0.442 → 0.421 → 0.411. The model learns Taiko better.
- **Taiko F1 at the best temperature differs by at most 0.005**, within noise. Reference-matching
  F1 no longer separates these models.
- **The Taiko-only model keeps the most human-like rhythm distribution** (lowest IOI L1).
- **Where song context helped most** is osu! (mania +0.048, catch +0.055) and song-level
  consistency (+0.10). On Taiko the gain shrinks after fine-tuning.
- **Next**: better evaluation (multi-reference or human playtests, ChartGenEval families), a
  per-game sampling temperature, and longer cross-game pre-training before concluding on
  transfer.

Playable: `runs/v1b-taiko/playable/*/urfm-best.tja` (all games + song context → Taiko, t=0.6).
Compute for round 2: about 15 CU (`urfm-l4` about 8.5 h, `urfm-l4-b` about 1.2 h).

## R4: audio-only, generalising chart generation (in progress)

Spec: [r4/SPEC.md](r4/SPEC.md). Benchmark songs (P3): Crystal Hail and three newer charts at or
beyond the training distribution's tails (銀の黎明 edit: notes, density and peak density above
every Taiko-1000 chart; Nivalis＊Anima: BPM 32–273 with 53 scroll changes; 魔宵月 edit: 40% of
notes off the 16th grid where Taiko-1000's maximum is 6%, 38 meter changes).

### Timing from audio (`beat-v1`, 5M, 8k steps, 0.84 h)

| Panel | beat F@70 | F@25 | any level F@70 | downbeat F@70 | grid fit (human map) |
| --- | --- | --- | --- | --- | --- |
| osu! test (150 songs) | 0.920 | 0.902 | 0.955 | 0.576 | 0.922 (0.978) |
| osu! test, random short tempo warps (P5) | 0.914 | 0.891 | 0.949 | 0.570 | 0.908 |
| Taiko-1000 test (136 songs) | 0.872 | 0.760 | 0.912 | 0.637 | 0.539 (0.979) |

- Short tempo changes (half a beat to a measure, ±25%) barely hurt: the model follows them.
- **Taiko charts sit 15 ms later on the same audio than osu! maps.** Median offset of the beat
  model against Taiko-1000 charts is −15.0 ms (300 training songs; test −15.6 ms), against osu!
  +0.7 ms. Grid fit on Taiko is low only because of this; maps for Taiko are now shifted by
  15 ms (`urfm.beat.decode.CONVENTION_OFFSET_US`, set from training songs).
- **Decoding matters.** Peak picking let the model switch metrical level between chunks (69
  tempo changes for a one-tempo song, jumping 152 ↔ 305 ↔ 610). Dynamic-programming tracking
  (strength minus a per-beat cost minus a squared log-tempo-change penalty), one metrical level
  per song, and a single global meter give 4 tempo segments on Crystal Hail and a grid fit of
  0.98; R3 audio-only F1 on the benchmark rose with it (Nivalis edit 0.587 → 0.666).
- Visual tempo folding (`urfm.data.canonical`) over all Taiko and osu!taiko charts: 1,249
  segments fold on a simpler grid and 641 by octave; about 89k real tempo changes are kept;
  145 charts with BPM outside 10–2000 are left alone.

### Note model, 15M (`r4a-full-15m`: conditions, texture, plan, augmentation; 8k steps)

Same protocol for every run (t=0.85, calibrated plan, seed 0; P1 = Taiko test, 599 charts;
canonical tables: `python experiments/r4/tables.py runs/r4/results`):

| P1 F1@25 | R3 (`ft-r3-song`) | r4a 15M | r4b 15M, no aug |
| --- | --- | --- | --- |
| human timing | 0.574 | **0.587** | 0.511 |
| audio timing (DP decode) | 0.554 | 0.556 | 0.488 |
| song 1.25x faster | 0.558 (−0.016) | **0.588** (+0.001) | 0.520 |
| IOI L1 on the human grid, normal / 1.25x | 0.543 / 0.591 | **0.488 / 0.492** | 0.493 / 0.524 |
| section r (density) | 0.157 | 0.145 | 0.054 |

P6 osu! test (120 songs) F1@25, same protocol: R3 (all games, then fine-tuned on Taiko) vs r4a —
catch 0.323 → 0.514, mania 0.529 → 0.706, std 0.248 → 0.400, osu!taiko 0.623 → 0.602 (−0.021,
just past the 0.02 rule; R3's Taiko fine-tune is the closer relative for this game). Rule 1
(audio ≥ R3 oracle − 0.02 = 0.554) holds at 0.556.

P1 Taiko test, 599 charts, t=0.85, calibrated plan:

| Timing | F1@25 | role acc | density err | IOI L1 (human grid) |
| --- | --- | --- | --- | --- |
| human (oracle) | 0.587 | 0.477 | 0.136 | 0.488 |
| audio (peak-picking decode) | 0.540 | | 0.150 | |
| 1.25x faster song (P4, oracle) | 0.588 | | 0.155 | |

- Playing every song 25% faster costs nothing (0.588 vs 0.587).
- **Augmentation helps in distribution too.** The same recipe without augmentation
  (`r4b-noaug-15m`, canonical tempo only) reaches P1 F1@25 0.511 (r4a 0.587), density error 0.227
  (0.136) and section density r 0.054 (0.145) at equal validation loss (1.030 vs 1.025).
  Augmentation stays (SPEC rule).
- On the benchmark with human timing, r4a beats R3 on the new charts (Nivalis edit 0.597 vs
  0.509, 銀の黎明 edit 0.609 vs 0.546 with 1175 vs 806 notes for 1507 human, 魔宵月 edit IOI L1
  0.52 vs 1.33).
- With audio timing r4a still writes the extreme new charts too sparsely; on Taiko-1000
  validation songs it does not (generated/human notes 1.04 audio vs 1.06 oracle; F1 0.592 vs
  0.660), so this is extrapolation, not a pipeline fault. Requests on a predicted grid now leave
  out notation-dependent conditions (onsets per beat, share off the 16th grid).
- Infrastructure lessons: augmentation made the loader the bottleneck (re-notation at half BPM
  left the integer-tick fast path; a linear scroll-bin search; Fraction sorting), fixed by
  `TICKS_PER_BEAT` x4, a bisect, float keys. 45M does not fit batch 64 (or 32) at 2048 decoder
  tokens on an L4; it trains at 16 x 4 accumulation.
- **Section contrast is the largest remaining gap.** Two human courses of the same song agree
  across 8-beat sections at r = 0.75 (density), 0.63 (fine share), 0.77 (rest share), 0.51 (ka
  share) (benchmark songs, hard↔oni and oni↔edit); generated charts reach 0.15–0.4. The plan is
  per 16 s window, so sub-window contrast is left to token-by-token sampling.

### 45M (`r4-45m`, 16k steps at 16 x 4, 4.7 h)

- Validation loss keeps falling well below 15M (0.907 at 10k steps vs 1.025 for r4a), yet
  **free-running generation is worse**: P1 F1@25 0.527 (r4a 0.587), edit charts at 0.77 of the
  human note count, more rests (validation songs: rest share 0.38–0.41 vs human 0.30 and r4a
  0.21), section correlations near 0. Sampling temperature (0.6–1.0) does not fix it, and
  prefill/step logits match teacher forcing to 2e-5, so it is not a decoding bug: the bigger
  decoder leans on its own previous tokens (exposure bias). Lower loss is not a better chart.
- Consequence: the predicted-grid fine-tune starts from r4a (15M), not 45M. Next round should
  attack exposure bias directly (noise or model samples in the context during training) before
  scaling up again.

### Predicted-grid fine-tune (`r4a-ft-pred`: r4a + 3k steps, 30% of visits re-notated on the audio-predicted grid)

Final comparison. P1 with audio timing, mean ± std over seeds 0–2 (599 charts each):

| Model | F1@25 | density err | IOI L1 (human grid) | section r density / fine / rest |
| --- | --- | --- | --- | --- |
| R3 | 0.553 ± 0.001 | 0.141 | 0.527 | 0.128 / 0.030 / 0.093 |
| r4a | **0.557 ± 0.001** | 0.144 | 0.510 | 0.131 / 0.026 / 0.096 |
| r4a + predicted grid | 0.554 ± 0.001 | 0.186 | **0.499** | 0.134 / 0.036 / 0.102 |

| Panel (seed 0) | R3 | r4a | r4a + predicted grid | 45M |
| --- | --- | --- | --- | --- |
| P1 human timing | 0.574 | **0.587** | 0.582 | 0.527 |
| P4 1.25x faster | 0.558 | **0.588** | 0.587 | 0.538 |
| P6 catch / mania / std / osu!taiko | 0.323 / 0.529 / 0.248 / 0.623 | 0.514 / 0.706 / 0.400 / 0.602 | **0.530 / 0.731 / 0.481 / 0.633** | 0.365 / 0.551 / 0.211 / 0.362 |

P3 benchmark songs with audio timing, 7 charts x 3 seeds:

| Model | F1@25 | note-count error | IOI L1 | section r density / rest |
| --- | --- | --- | --- | --- |
| R3 | 0.636 | **0.182** | **0.679** | 0.152 / 0.077 |
| r4a | 0.568 | 0.263 | 0.721 | 0.068 / 0.123 |
| r4a + predicted grid | **0.638** | 0.219 | 0.737 | **0.191 / 0.200** |

Against the pre-registered rules (SPEC): (1) audio F1 ≥ R3 oracle − 0.02 holds; (2) IOI L1 is
lower but section correlations are unchanged; (3) speed stress holds (no loss at 1.25x where R3
loses 0.016), but on the extreme new charts the note count is still 22% off and R4 does not beat
R3's rhythm vocabulary there; (4) every osu! game improves with the fine-tuned model; (5) play
test pending. **R4 is a better and more robust system, not yet a human-level one.**

What R4 established:
- Audio-only charting works: with the beat model, F1 matches oracle-timing R3 minus 0.02.
- Augmentation is the main generalisation lever (+0.076 P1 F1 over the same model without it,
  and no loss when songs are played faster).
- Bigger is not better yet: 45M lowers validation loss and generates worse charts.
- The remaining gap is musical structure: section contrast (r ≈ 0.13 vs 0.75 between two human
  courses) and density on extreme charts.

Next (R5): attack exposure bias (train on the model's own or noised context), a finer plan
(per 2 measures) so section contrast is decided explicitly, then scale again.

Compute for R4: about 25.9 CU (`urfm-l4` 10.98 h, `urfm-l4-b` 5.82 h). Playable charts:
`runs/r4/playable/<song>/urfm-r4-final-audio.tja` (audio only, all courses).

### Why generated charts avoid very fast notes (`experiments/r4/diag_fine.py`, validation only)

The tokenizer can place 1/8, 1/6, 1/12 and 1/16 beats; the problem is absolute speed (intervals
under ~60 ms, e.g. 16ths at 305 BPM), not the notation.

| | Taiko-1000 train | osu!taiko train |
| --- | --- | --- |
| intervals < 60 ms / < 45 ms | 1.5% / 0.2% | 5.1% / 0.6% |
| charts with ≥ 5% of intervals < 60 ms | 3.7% | 8.9% |

On the 41 validation charts that use fast notes (all osu!taiko; no Taiko-1000 validation chart
qualifies), with human timing:

- **The model knows fast passages.** Teacher-forced, an event 60 ms after its predecessor gets
  mean probability 0.88 for its beat and median 0.95 for its sub-beat token, and 97% of those
  tokens survive top-p 0.95 (96% for other events).
- **It does not start them.** Free-running, 3.2% of intervals are under 60 ms (human 17.2%) and
  the charts have 0.66 of the human note count. Top-p 1.0 barely helps (4.4%, 0.59 of the notes).

So truncation is not the cause. In free running the model rarely enters a dense passage on its
own (exposure bias plus scarce training examples) even when the density plan asks for it. Fixes
target the sequence level: sample dense windows more often in training, make the plan binding,
and reward reaching the requested (peak) density, which is a natural first RL reward.

## R5: fast passages and RL (15M only)

Same protocol (P1, audio timing, seed 0). Fast = intervals under 60 ms.

| Model | F1@25 | density err | IOI L1 | section r density | < 60 ms, oni+edit (human 2.0%) |
| --- | --- | --- | --- | --- | --- |
| r4a-ft-pred (R4 final) | 0.555 | 0.187 | 0.492 | 0.133 | 1.1% |
| `r5-sft`: + 2k steps with fast windows oversampled (`--fast-boost 4`) | 0.538 | 0.175 | 0.493 | 0.125 | 2.4% |
| `r5-rl`: + GRPO v1, 150 steps (F1 + density + fast share) | **0.576** | **0.091** | 0.706 | 0.142 | 0.5% |

- Oversampling fast windows makes the model start fast passages (human rate reached) at a cost
  of 0.017 F1.
- GRPO v1 (16 prompts x 8 samples per step, 11 s per step, 30 min) raised F1 by 0.038 over the
  fine-tune and halved the density error, but it **reward-hacked**: rhythm variety collapsed
  (IOI L1 0.49 → 0.71) and fast passages disappeared again. The fast term was a share
  difference (0.02 at most) and nothing rewarded rhythmic variety, so F1 won. v2 rewards the
  log ratio of fast-interval counts and the IOI class distance as well.
- **GRPO v2** (`r5-rl2`, rewards F1 + density + log fast-count ratio + IOI class distance, same
  150 steps): P1 audio F1 0.574, density err 0.160, IOI L1 0.528, section r 0.156; P1 with human
  timing F1 **0.607** (R4 final 0.582), section r 0.190 (0.157). Rhythm variety is kept this
  time, but fast passages still fall back to the starting level (1.1% on oni+edit) — when
  writing alone, a misplaced fast note costs F1, so the policy avoids the risk.
- Benchmark songs, audio timing, 3 seeds: R3 / R4 final / r5-rl2 = F1 0.636 / 0.638 / 0.639,
  note-count error 0.182 / 0.219 / **0.119**, IOI L1 0.679 / 0.737 / **0.645**, section r density
  0.152 / 0.191 / **0.224**, rest 0.077 / 0.200 / **0.259**.
- **The 銀の黎明 failure is notation, not speed.** The beat model gives it ~124–152 BPM (human
  305), so its 16th-note streams would be 32nds at ~150 BPM, a combination absent from training
  data. On the same audio, doubling the predicted grid makes r5-rl2 write them (edit: < 60 ms
  intervals 0% → 40%, human 51%; F1 0.576 → 0.681; oni 0.2% → 31%). On Nivalis (grid 257) fast
  passages already appear (edit 47%, human 44%).
- Choosing the octave automatically is not solved: a density-per-beat rule fitted on training
  songs is no better than keeping the predicted octave (87.9% vs 88.9% agreement with human
  notation); picking the octave whose output density best matches the request is uninformative
  because the calibrated plan forces density in every octave (it picked x1/2 for 銀). Not
  adopted. Fix belongs in training: octave re-notation that covers fast songs (so 32nds at
  150 BPM exist), or a learned notation-level choice.
- Lost data: r5-rl2's 1.25x, osu! and fast-note diagnosis outputs were not fetched before the VM
  stopped.

Compute since the R4 authorisation: about 32.6 CU of 50.

## Monotony metric (user hypothesis: a pattern unchanged for ~10 s feels dull)

`urfm.eval.monotony`: each onset is a symbol (action + interval to the next onset in beats); a
10 s window (1 s hop, at least 8 onsets) is monotone when its sequence is periodic with period
≤ 8 at ≥ 90% of positions. Reported per chart: share of monotone windows and the longest
monotone stretch. Two variants: rhythm and colour, and **rhythm only** (colours ignored).
Validated on every local chart (`experiments/r4/monotony_eval.py`, report in
`runs/r4/monotony/report.md`).

Rhythm only, share / longest stretch:

| Source | easy | normal | hard | oni | edit |
| --- | --- | --- | --- | --- | --- |
| human (6 test songs) | 10% / 16 s | 6% / 10 s | 3% / 7 s | 1% / 4 s | – |
| human (4 benchmark songs) | 6% / 12 s | 1% / 8 s | 4% / 13 s | 3% / 8 s | 3% / 5 s |
| R3 best (the chart criticised as "only 8th notes") | 51% / 40 s | 36% / 33 s | 52% / 48 s | 66% / 60 s | – |
| R4a, audio | 17% / 18 s | 7% / 14 s | 0% / 2 s | 0% / 0 s | 6% / 13 s |
| R4 final (predicted-grid fine-tune), audio | 20% / 23 s | 4% / 14 s | 3% / 6 s | 22% / 26 s | 41% / 38 s |
| R5 RL v2, audio | 14% / 17 s | 7% / 15 s | 7% / 11 s | 10% / 16 s | 41% / 40 s |
| R5 RL v2, audio, doubled grid | 7% / 15 s | 3% / 12 s | 2% / 11 s | 2% / 10 s | 0% / 0 s |

- **The metric separates what players complained about.** Human hard/oni/edit charts stay at
  1–4%; the R3 chart players found dull is at 52–66% with 40–60 s stretches.
- **10 s is the right window**, as hypothesised: at 5 s even human charts are flagged (3–4%
  share on hard/oni/edit), at 15 s real monotony is missed more often.
- **Rhythm only is the useful variant.** With colours, R4/R5 look human (0–3%), but their
  monotone windows are unbroken streams of one interval with shuffled colours: R4 final writes
  ~70 s of 16th-note stream on 銀の黎明 edit, R5 RL ~50 s of 1/3-beat stream (on a half-tempo
  grid it cannot write 1/8 beats, so it fills density with triplets). The human edit chart has
  none. F1 and IOI-distribution metrics do not see this.
- Easy charts are legitimately repetitive (human 6–10%), so compare against the human value
  of the same course.
- The predicted-grid fine-tune introduced most of this (R4a 0–6% → R4 final 22–41% on oni/edit).
- Now part of every evaluation row (`monotone_share_*`, `monotone_longest_s_*`), and a natural
  RL penalty.

### Monotony, density-calibrated (final form)

The first form had fixed thresholds, a fixed period cap of 8, and no notion of difficulty. The
final form (`urfm.eval.monotony.excess_monotone_time`):

- repetition per 10 s window, continuous: `R(w) = max_{p ≤ 16} mean_i [s_i = s_{i+p}]` (rhythm
  symbols, colours ignored);
- a monotone stretch = consecutive windows with `R ≥ 0.9`, duration `L`, median density `d`;
- allowance from humans at the same density: `q95(L | log2 d)` from 408,724 windows of
  Taiko-1000 training charts (all courses). It is 17–24 s at every density: humans hardly ever
  keep one rhythm longer than about 20 s, whatever the difficulty;
- **E_time = 60/T · Σ max(0, L − q95(L | log2 d))**: seconds per minute of monotony beyond human
  practice. Difficulty enters only through density.

Held-out human test charts (599): 0.07–0.48 s/min at every course. A window-level variant
(share of windows above the human 95th percentile of `R` at that density) is calibrated too
(held-out human 1.9–4.7% at every course) but blind where humans already reach `R = 1` (below
1.7 or above 9.5 notes/s).

E_time (s/min) / largest single excess (s):

| Source | easy | normal | hard | oni | edit |
| --- | --- | --- | --- | --- | --- |
| human, 6 test songs | 1.2 / 2 | 0.9 / 2 | 0.0 / 0 | 0.0 / 0 | – |
| human, 4 benchmark songs | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 |
| R3 best (criticised) | 13.9 / 23 | 9.7 / 16 | 21.8 / 36 | 25.8 / 42 | – |
| R4a, audio | 1.1 / 2 | 0.1 / 0 | 0.0 / 0 | 0.0 / 0 | 0.6 / 1 |
| R4 final, audio | 3.1 / 7 | 0.0 / 0 | 0.0 / 0 | 7.9 / 14 | 9.1 / 18 |
| R5 RL v2, audio | 0.1 / 0 | 0.1 / 0 | 0.8 / 2 | 2.1 / 4 | 13.3 / 22 |
| R5 RL v2, audio, doubled grid | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 | 0.0 / 0 |

Benchmark oni/edit, 3 seeds: R3 5.4 / 1.3, R4a 0.0 / 0.1, R4 final 5.5 / 14.5, R5 SFT 0.7 / 2.3,
R5 RL v2 3.1 / 10.1 s/min.

Reading: R4a (before the predicted-grid fine-tune) is the least monotone model; the fine-tune
and RL both add long constant streams on dense charts, and the half-tempo grid causes them
(doubling the grid removes them). Now in every evaluation row (`monotony_s_per_min_*`).

### Variation penalties, phase 1 (multi-scale, cross-dimensional; `urfm.eval.variation`)

Reward hypothesis proposed by the user: penalise lack of variation, never reward variation;
short-term variation may come from either dimension (space = what is hit, time = rhythm),
long-term stagnation accumulates per dimension. Implemented with these choices:

- units of 1/2/4/8 beats of the chart's own grid, whichever lasts closest to 2 s (notation
  octave does not matter); bars are avoided because predicted downbeats are unreliable;
- short-term `V_x`: normalised Gini-Simpson diversity inside the last 4 units (time: interval
  classes; space: actions, normalised by Taiko's two colours), `V_any = 1-(1-V_a)(1-V_g)`,
  `P_short = [0.3 - V_any]_+²`;
- novelty `U_x`: distance to the most similar of the previous 8 units (Jaccard of onset
  positions; normalised edit distance of actions);
- debt `D_x ← e^{-Δt/60s} D_x + Δt(1-U_x)`, `P_long = Σ_x [D_x/Q_x - 1]_+²`,
  `Q_x = 20 s · (1 + α V_x)`, α = 1; cross-dimension compensation (β, κ) **off**: in Taiko,
  shuffled colours are cheap and the model's colours are near chance (role accuracy ~0.47), so
  letting colour variety extend rhythm tolerance would reward exactly the observed failure.

Stress test, 120 s synthetic charts (score; seconds until the first penalty):

| Chart | score | first penalty | E_time |
| --- | --- | --- | --- |
| all don, 16ths | 2.07 | 1.6 s | 48.9 |
| don-ka alternating, 16ths | 1.01 | 25.6 s | 48.9 |
| evolving colours, 16ths | 0.99 | 25.6 s | 48.9 |
| alternating colours, rhythm phrases vary | 0.09 | 54.4 s | 0.0 |
| random colours and rhythm | 0.00 | never | 0.0 |
| one repeated unit, one note changed every 3 units | 0.17 | 43.2 s | 48.0 |
| ABAB units | 0.11 | 49.6 s | 51.0 |
| all don with 40% of notes deleted | 0.83 | 27.2 s | 0.0 |

Real hard/oni/edit charts: human 0.02, R3 best 0.58, R4 final 0.17, R5 RL v2 0.19, R5 RL v2 on
the doubled grid (1 song) 0.06.

- Intended order holds where the design can see it: constant streams are penalised soonest,
  colour variation delays but does not remove the penalty, small edits and ABAB do not clear
  the debt, varied rhythm is barely penalised, humans stay near zero.
- Unlike `E_time`, repeating a rich pattern (ABAB, one-note edits) is penalised much less than
  a constant stream, and only after ~45 s.
- Known gaps, by design: random variation scores 0 (needs a coherence term), note deletion
  escapes the time debt (needs the density term). Evolving colours ≈ alternating colours
  because cross-dimension compensation is off.
- Next: calibrate Q0, α (and test whether β, κ > 0 is supported) on Taiko-1000 training charts.

### Variation penalties, phase 2 (calibrated on Taiko-1000 training charts)

Calibration (`experiments/r4/variation_calibrate.py`, 216k units): `Q_x` = human 95th percentile
of the debt given (density, own short-term variation, other dimension's variation); `τ_S` = human
5th percentile of `V_any` given density.

- **The first debt form was blind once calibrated.** With `D ← ρD + Δt(1−U)`, human charts reach
  31–43 s of time debt (they reuse rich bar patterns constantly), so a don-ka 16th stream was
  penalised only after 70 s and models scored like humans (0.01 vs 0.00).
- **Debt gated by plainness:** `D_x ← e^{-Δt/60}D_x + Δt(1−V_x)(1−U_x)` grows only while the
  recent window is plain *and* repeats recent units. Held-out human test charts: mean penalty
  0.008–0.018 at every course.
- **Cross-dimension compensation exists but is small:** with a plain rhythm (V_time in the lowest
  third), the human time-debt allowance rises with colour variety at low density (density bin −1:
  7.6 → 22.1 s; bin 0: 18.3 → 24.3 s; bin 1: 21.2 → 25.6 s) and barely at higher density (bin 3:
  23.4 → 24.0 s). The conditional table carries it; no hand-set β/κ.

Calibrated scores (120 s synthetic; first penalty):

| Chart | score | first penalty | E_time |
| --- | --- | --- | --- |
| all don, 16ths | 1.38 | 1.6 s | 48.9 |
| don-ka alternating, 16ths | 0.14 | 44.8 s | 48.9 |
| evolving colours, 16ths | 0.14 | – | 48.9 |
| alternating colours, rhythm phrases vary | 0.00 | never | 0.0 |
| random colours and rhythm | 0.00 | never | 0.0 |
| one repeated unit, one note changed every 3 units | 0.02 | 65.6 s | 48.0 |
| ABAB units | 0.00 | 118 s | 51.0 |
| all don, 40% deleted | 1.54 | 1.6 s | 0.0 |

Real hard/oni/edit charts: human 0.00, R3 best 0.55, R4 final 0.07, R5 RL v2 0.08 (E_time: 0 /
23.8 / 5.4 / 4.7). Same ranking as `E_time`; the variation penalty additionally separates
constant streams from repeated rich patterns, and colour alternation from all-don. Random
variation still scores 0: the RL reward must keep F1/onset alignment and density terms.

### Time-based stagnation penalty: human check (2026-10-10)

- Design: three debts (rhythm unchanged, colours unchanged, both), each grows 1 s per second of stagnation and repays at 2 s per second of variation; penalty `([D-start]_+/(limit-start))^2` with start/limit 10/20 s (rhythm, colours) and 5/10 s (both). A 2 s don burst + 2 s ka burst inside varied music peaks at 0.5 s debt (no penalty); all-don is first penalised after 5–8 s.
- Taiko-1000 train charts: peak rhythm debt q95 12.3 s, q99 25.8 s; both-unchanged q95 4.1 s.
- Density calibration (q95 episode peak per log2/2 density bin): rhythm 13.2 s at 1–1.4 notes/s down to 6.6 s at 2.8–4 notes/s; every bin is below the design limit, so calibration (which may only loosen) changes nothing. Kept fixed limits; table stored as evidence in `variation_calibration.json`.
- Held-out test charts (fixed limits), share with any penalty / mean: edit 0% / 0, oni 3.5% / 0.003, hard 2.8% / 0.008, normal 9% / 0.0002, easy 31% / 0.019. Generated hard–edit charts: R3 13.7, R4 final 2.2, RL v2 1.3.
- Evidence: `runs/r4/monotony/stagnation_check.json`.

## R6: song planner, structure memory, notation-invariant fine rhythm, then RL (started 2026-10-10)

Budget 50 CU (new authorisation), model ≤ 20M. Spec: `experiments/r6/SPEC.md` (design evidence from
`experiments/r6/analysis.py` → `runs/r6/analysis.json`).

- Human-data analysis (Taiko-1000; ~0.3 CU): two human courses agree across 8-beat sections at
  density r 0.63–0.70; a linear ridge model from per-section audio already reaches 0.54 (current
  models 0.13–0.22); structure alignment (chart vs audio self-similarity) is 0.30 for held-out
  human charts (null 0.055) and ≈ 0.04 for R4 / RL v2 on the benchmark; 32nds and finer are 0.5% of
  canonical Taiko-1000 onsets; the beat model returns half the human tempo for 8% of test songs.
  (The predicted-grid subdivision census in that run used a too-loose tolerance and is not
  reported; fixed in the script.)
- User decisions: F1 against one human chart is a diagnostic, not a success criterion; reward
  placement precision (a note that exists must sit exactly on its place) separately from
  existence (weak F1@25 weight).
- Size correction: the R4 "15m" model is 20.3M parameters in total (preset names count only the
  Transformer core). R6 = decoder with one song-encoder layer (18.53M) + planner (1.07M) = 19.6M.
- VM smoke (300 steps, full R6 recipe): 191 samples/s, data wait 0.1%, GPU 100%, val loss 1.10.
- Launched: `r6-base` (40k steps, VM A); VM B: planner `r6-plan` (10k steps), then ablations
  `r6-a10k`, `r6-noplan`, `r6-noref`, `r6-oct15` (10k steps each).
- Infrastructure (2026-10-10 night): the first full-length run was loader-bound (60–100 samples/s,
  GPU 33%) although workers sat idle -- the DataLoader returns batches in worker order and each
  worker yields in bursts of 8 sorted batches, so the slowest worker stalled all. Fixed with
  `in_order=False`, per-song caching of the onset envelope and band sums, and
  `--windows-per-visit 4` (same expected windows per chart, a quarter of the per-visit work):
  193 samples/s, data wait 0.1%, GPU 100%. Both Colab runtimes were then lost at ~23:05 UTC; the
  trainer now resumes from `latest.pt` on the Hub (`--resume true`, optimizer state included) and
  evaluation results are uploaded as they finish. `r6-base` resumed from step 2000; the planner
  is the step-9000 checkpoint (of 10k) saved before the loss.

### R6 results (2026-10-11; canonical tables: `python experiments/r6/tables.py runs/r6/results` → `runs/r6/tables.md`)

P1 Taiko test (599 charts), audio timing; mean ± std over seeds 0–2 where given. F1 is a diagnostic.

| Model | F1@25 | place ≤10 ms | dens err | IOI L1 | sec r dens / fine / rest | struct (human 0.274) | rec 32nd+ | E_time (human 0.26) | stag (human 0.007) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RL v2 (`r5-rl2`) | 0.575 | 0.693 | 0.154 | 0.515 | 0.157 / 0.042 / 0.115 | 0.066 | 0.249 | 0.66 | 0.61 |
| `r6-base` (40k) | 0.563 | 0.682 | 0.106 | 0.450 | 0.261 / 0.160 / 0.172 | 0.116 | 0.217 | 1.89 | 0.47 |
| `r6-rl` (all rewards) | 0.577 | 0.687 | 0.106 | 0.469 | 0.285 / 0.188 / 0.183 | 0.130 | 0.231 | 1.97 | 0.69 |
| `r6-rl-nostruct` | 0.576 | 0.690 | 0.117 | 0.445 | 0.280 / 0.180 / 0.181 | 0.117 | 0.245 | 1.06 | 0.30 |
| `r6-rl` + planner v2, decoder t=1.0 (seed 0) | 0.570 | 0.693 | 0.113 | **0.409** | 0.277 / 0.180 / 0.179 | 0.123 | 0.217 | 0.72 | **0.24** |

Ablations (10k steps, seed 0): without the planner section r 0.196 (0.224), E_time 2.59 (1.69),
stag 1.52 (0.74); without the memory no clear change (struct 0.079 vs 0.076); with the R4 octave
policy, on a half-BPM notation (P7) F1 0.570 vs 0.595, 16th recall 0.055 vs 0.125, E_time 4.2 vs
2.4. Plan temperature: 0 → section r 0.273 but stag 5.4; 0.3 → 0.264 / 2.8; 0.7 → 0.224 / 0.74;
1.0 (planner v2, r6-rl) → 0.236 / 0.17.

Benchmark (4 songs, hard–edit, audio timing, 3 seeds, decoder t=0.85): RL v2 / r6-base / r6-rl /
r6-rl + planner v2 = F1 0.622 / 0.641 / 0.645 / **0.650**, IOI L1 0.576 / 0.462 / 0.479 / 0.472,
section r density 0.209 / 0.339 / 0.403 / **0.420**, rest 0.200 / 0.333 / 0.395 / 0.378, structure
0.052 / 0.134 / 0.156 / 0.147 (human 0.192), 24th recall 0.211 / 0.306 / 0.321 / 0.273, 32nd+
recall 0.209 / 0.270 / 0.183 / 0.217.

Findings:
- The planner is the main lever for section contrast (ablation), and the octave policy for
  notation invariance (P7: r6-base loses 0.005 F1 at half-BPM notation).
- Planner targets v1 were wrong: exact grid matching sent nearly every osu! or predicted-grid
  section to the finest class, so on audio grids the planner never asked for 32nds. Fixed with a
  1/96-beat tolerance (`r6-plan2`); on 銀の黎明 edit it now plans 32nds for half the sections.
- The structure reward works (+0.013 structure, ~3σ) but over-repeats: E_time 1.97 vs 1.06 without
  it. The window-level stagnation reward never fires (debt rarely builds in 16 s after human
  context), so nothing balances it.
- Monotony at decoder temperature 0.85 was mostly a sampling effect: t=1.0 cuts E_time 1.63 → 0.72
  and stagnation 0.54 → 0.24 at the same plan, other metrics ~unchanged.
- Still unsolved: 銀の黎明 edit (< 60 ms 3% vs human 51%). On the half-speed grid the decoder meets
  each section's density with steady 16ths; human 32nd bursts plus rests are not expressible by a
  3.2 s mean density. Next: a per-section peak-density (or burst-share) plan value.
- Compute: ~35 CU of the 50 (two runtimes lost overnight; resume-from-Hub added).
- Playable: `runs/r4/playable/<song>/urfm-r6-audio.tja` (r6-rl + planner v2, t=1.0, audio timing).

### R6.1: extended section plan (2026-10-11)

Candidate plan values checked on human data first (`experiments/r6/analysis_plan2.py` →
`runs/r6/analysis_plan2.json`; keep if two courses agree r > 0.3 and audio predicts r > 0.2):
peak density (agreement 0.76, audio 0.69), repetition of an earlier section (0.86, 0.51),
rhythm variety (0.66, 0.59), colour-switch rate (0.53, 0.47), long-note share (0.67/0.49, 0.21).
All five kept (`SPEAK SREP SVAR SSW SLONG` after `SSUB`; planner v3 with ten heads). Decoder
`r6x-base` = r6-base + 15k steps with the extended plan (warm start, vocabulary grown).

P1 audio, 3 seeds, decoder t=1.0, plan t=0.7 (matched baseline r6-base + planner v2):

| Model | F1@25 | dens err | IOI L1 | sec r dens / fine / rest | struct | rec 32nd+ | E_time (human 0.26) | stag (human 0.007) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r6-base + plan v2 | 0.546 | 0.099 | 0.385 | 0.264 / 0.160 / 0.177 | 0.106 | 0.212 | 0.48 | 0.090 |
| r6x-base + plan v3 | 0.541 | **0.081** | **0.375** | **0.280 / 0.183 / 0.188** | 0.102 | **0.231** | **0.19** | **0.045** |

P7 (half-BPM notation, t=1.0): E_time 0.70 → 0.45, stag 0.26 → 0.15, IOI L1 0.409 → 0.388,
32nd+ recall 0.113 → 0.140, F1 0.581 → 0.573. Benchmark (t=1.0): mixed -- rest r 0.314 → 0.385,
density r 0.370 → 0.384, structure 0.112 → 0.126, but fine r 0.260 → 0.184 and 32nd+ recall
0.257 → 0.206. 銀の黎明 edit < 60 ms: 0.03 (t=0.85) → 0.12–0.13 with t=1.0 for both models (human
0.51): sampling temperature, not the peak value, made most of the difference. Caveat: r6x had
15k more training steps than its baseline. Monotony is now at human level on P1 (E_time 0.19 vs
human 0.26). Compute ~43 of 50 CU. Playable: `runs/r4/playable/<song>/urfm-r6x-audio.tja`.

### R6.2: song-level GRPO (2026-10-11, `urfm.train.rl_song`)

Each sample writes 6 consecutive windows (~96 s) reading its own context and memory; rewards on
the span: placement 1.0, F1@25 0.25, rhythm 0.5, plan 0.5, structure 0.5, stagnation 1.0, E_time
0.25 (stagnation and E_time now fire: e.g. step 12 E_time −5.0 s/min on some songs). 4 songs × 4
samples per step, ~20 s/step; crashed at step 175 (a human context with two events on one onset;
fixed), best validation checkpoint step 150 (validation is noisy: 16 songs × 1 sample).

| P1 audio, 3 seeds, t=1.0 | F1@25 | place ≤10 ms | dens err | IOI L1 | sec r dens / fine / rest | struct | E_time | stag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r6x-base | 0.541 | 0.681 | 0.081 | 0.375 | 0.280 / 0.183 / 0.188 | 0.102 | 0.19 | 0.045 |
| r6x-rl | 0.556 | 0.686 | 0.095 | 0.377 | 0.293 / 0.182 / 0.192 | 0.101 | 0.22 | 0.060 |

P7 (half-BPM notation): F1 0.573 → 0.585, placement median 0.53 → 0.38 ms, section r
0.308/0.158/0.255 → 0.313/0.175/0.269, 32nd+ recall 0.140 → 0.149, E_time 0.45 → 0.38, stag 0.15
→ 0.07. Modest net gain (P1 monotony slightly up but still at human level); `r6x-rl` is now the
app's default decoder. Compute ~46 of 50 CU.

App (`urfm.app`): Taiko and the four osu! modes, a section-plan editor (analyze & plan → edit →
generate), per-stage timing; osu! presets from train charts (`app/data/osu_defaults.json`).
