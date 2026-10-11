# R6 — Section plan, structure memory, notation-invariant fine rhythm, then RL

Budget: 50 CU (Colab L4, 1.54 CU/h). Model size cap: 20M parameters. Benchmark songs (Crystal
Hail + `runs/newcharts/`) stay eval-only.

## Evidence the design rests on (`experiments/r6/analysis.py`, `runs/r6/analysis.json`)

| Finding | Number | Consequence |
| --- | --- | --- |
| Section contrast ceiling: two human courses of one song, 8-beat sections (138 test songs) | density r 0.63–0.70, rest 0.57–0.64, fine 0.47–0.50, ka 0.38–0.46 | target for section r |
| A *linear* ridge model from per-section audio descriptors (within-song z) predicts human sections | density r 0.54 (oni 0.58, edit 0.66), rest 0.46, fine 0.29 | audio determines most of the contrast; current models (0.13–0.22) are below a linear baseline, so the plan must be explicit and song-level |
| Structure alignment (`urfm.eval.structure`: chart SSM vs audio SSM, units ≈ 2.5 s) | human test 0.30 (null 0.055, 98–100% of charts above null); R4 / RL v2 on the benchmark ≈ 0.04 (human there 0.11–0.24) | a valid, discriminative reward |
| Subdivisions in Taiko-1000 canonical notation | 16th 14.7%, 12th 2.5%, 24th 0.6%, 32nd 0.08%, 48th 0.17%, 64th 0.19%; 5/7-tuplets 0.04% | the model never sees 32nds/64ths; tuplets are noise |
| Beat-model octave (predicted / canonical tempo, Taiko test) | x1 75%, x0.5 8%, x2 3.6%, x1.5 2.9%, x0.67 1.5%, other 9% | ~1 song in 8 arrives at half or 1.5x notation; the model must write any subdivision on any grid |
| R5 diagnosis | RL v2 wrote 42% quintuplets on 魔宵月 edit; 銀 edit at half grid has 0% < 60 ms (human 51%) | mask tuplets; notation invariance |

## System (≤ 20M)

1. **Song planner** (new, ~3M, encoder-only Transformer over the whole song).
   - Sections: `k` beats, `k` the power of two nearest 3.2 s at the chart's median tempo (8
     beats at 150 BPM). The same rule on any notation octave gives the same seconds.
   - Input per section: beat-synchronous audio descriptors (texture flux now at 48 positions per
     beat, so 64ths are visible; 4 bands; 32 mel bands) + continuous conditions + game.
   - Output per section (categorical): density (16 log bins), fine share (5), rest share (5), ka
     share (5, Taiko family), finest subdivision class (none, 4, 6, 8, 12, 16 per beat).
   - Trained on human charts with the same augmentation as the decoder; this is where section
     contrast is decided, from the whole song rather than a 16 s window.
2. **Chart decoder** (15m preset, 15.8M; 45M showed lower loss but worse generation, so no
   growth here).
   - Plan tokens are interleaved: a `SEC` group (the five plan values) at every section start,
     right before the notes it governs. Teacher-forced from the human chart in training (10% of
     values perturbed so the decoder tolerates planner errors); sampled by the planner at
     inference.
   - **Structure memory**: besides the 32-beat context, the window gets the events of the
     earlier section whose audio is most similar (cosine of section descriptors, at least 2
     sections back, similarity above the song's 80th percentile), up to 48 events, as a `REF`
     block. The decoder can repeat a pattern when the music repeats (cf. Museformer's
     structure-related bars, retrieval augmentation).
   - **Exposure bias**: context and `REF` are corrupted in 50% of samples (drop 10% of events,
     flip 10% of colours, shift 5% by one grid step), since the model reads its own output at
     inference (scheduled-sampling-style noise).
3. **Fine rhythm on any grid**.
   - BPM-octave re-notation p 0.15 → 0.5, factor 1/2 (60%), 2 (25%), 1/4 (15%) within 40–420
     BPM. Halving turns the 14.7% of 16ths into 32nds, 24ths into 48ths, 32nds into 64ths;
     quartering gives 64ths from 16ths. Expected share of 32nd-or-finer notation in training:
     ~0.5% → ~6%.
   - Predicted-grid re-notation p 0.3 → 0.5 (the grid inference sees, octave and 3:2 errors
     included).
   - Charts with ≥ 2% onsets at 6+ per beat or under 60 ms: sampling weight x2.
   - Grammar masks 5/7-tuplets for `taiko` and `osu_taiko` (human 0.04%).
4. **Data**: unchanged from R4 (Taiko-1000 + osu! all modes, song-level splits), canonical
   tempo, speed and local warps as R4.

## As built (2026-10-10, before any reportable run)

- **Size.** The R4 "15m" model is 20.3M parameters in total (the preset name counts only the
  Transformer core; the song encoder, positions and stem add the rest). R6 uses one song-encoder
  layer instead of two: decoder 18.53M + planner 1.07M (d 192, 2 layers) = **19.6M** (the 5M beat
  model is shared with R4 and unchanged).
- **Plan placement.** Section plans form one block right after `SEP`
  (`SBEAT SDENS SFINE SREST SKA SSUB` per section; `SBEAT` = section start in beats after the
  window start), not interleaved with the notes; the grammar stays a small state machine.
- **Planner input.** The planner uses its own section descriptors (onset envelope folded onto
  one beat at 48 phases, so 64ths are visible); the decoder's texture context is unchanged.
- **Tuplets.** Taiko-family positions snap to the 1/48 and 1/32 grids only and the grammar masks
  5/7-tuplets, so they can neither be learned nor sampled.
- **Fine-passage weighting** reuses R5's `--fast-boost` at 2 (windows by share of < 60 ms
  intervals, charts with any x1.5).
- **Stagnation reward** uses the fixed design limits (10/20 s; both dimensions 5/10 s), not
  density-dependent ones.
- **Ablations** are four 10k-step runs including `r6-a10k` (full recipe at 10k steps), so each
  ablation is compared at an equal schedule.

## Training

| Stage | Run | Steps | Est. |
| --- | --- | --- | --- |
| Planner | `r6-plan` | 10k | 0.5 h |
| Decoder (warm start from `r4a-full-15m` where shapes match) | `r6-base` | 40k | 4 h |
| Ablations, 10k steps each | `r6-a10k` (full), `r6-noplan` (window plan as R4), `r6-noref`, `r6-oct15` (R4 octave policy) | 4 x 10k | 3.7 h |
| GRPO from `r6-base` | `r6-rl` (all rewards), `r6-rl-nostruct` (without structure) | 300 each | 2.5 h |

GRPO (16 prompts x 8 samples, window-level, KL 0.05 to `r6-base`, lr 1e-5), human context, `REF`
and plan. Rewards:

| Term | Weight | Definition |
| --- | --- | --- |
| placement precision | 1.0 | over generated notes with a human onset within ±75 ms: mean exp(−½(Δ/12 ms)²). *Whether* a note exists is uncertain, but a note that is there must sit exactly on its place |
| existence | 0.25 | onset F1@25 vs human (strict tolerance, low weight: one human chart is one valid answer, not the answer) |
| plan adherence | 0.5 | per section: −|log density ratio|, + finest-subdivision class match |
| rhythm | 0.5 | −IOI class L1 vs human |
| structure | 1.0 | correlation of the window's units' rows in the chart SSM with the same rows of the audio SSM, other units from the human chart |
| stagnation | 0.5 | −penalty, debt carried in from the context; start = human q95 by density clamped to 8–12 s, limit 2x start |

Reward-hacking watch every 25 steps on 64 validation windows: IOI L1, fast share, subdivision
shares, rest share; stop and keep the last good checkpoint if any leaves its human range.

## Evaluation (metric contract v2; P1–P6 unchanged)

- P1 Taiko test (599 charts), audio timing, seeds 0–2: F1@25, **placement precision** (median
  |Δ| of generated notes within ±75 ms of a human onset, share within 10 / 25 ms), density
  error, IOI L1, section r
  (density / fine / rest / ka), **structure alignment** (score and score − null),
  **subdivision recall** (share of human onsets at 8, 12, 16 per beat on the human grid matched
  within 25 ms), stagnation penalty, E_time, tuplet share.
- P3 benchmark (4 songs, 7 charts, 3 seeds); 銀 edit < 60 ms share.
- **P7 notation stress** (new): P1 with human timing re-notated x1/2 and x2.
- P6 osu! test (cross-game claim).

Decision rules (pre-registered). Onset F1 against one human chart is **not** a success
criterion: many charts fit a song, so F1 penalises valid alternatives and rewards copying. It
is a sanity check only.

Success (all of 1–5):

1. Section r density ≥ 0.45, rest ≥ 0.35, fine ≥ 0.25.
2. Structure alignment ≥ 0.15 (human 0.30).
3. P7: subdivision recall at 8+ per beat ≥ 0.3 and the same at x1/2 notation within 10% of x1;
   銀 edit < 60 ms ≥ 30% with audio timing.
4. Placement: median |Δ| of near-matched notes ≤ RL v2's and share within 10 ms ≥ RL v2's.
5. Guardrails: tuplet share 0, stagnation and E_time within human q95, IOI L1 ≤ 0.53, density
   error ≤ RL v2 + 0.03.

Diagnostics (never a failure on their own; a large move triggers a look at the charts):

- P1 audio F1@25/50: a drop of more than 0.05 below RL v2 (0.574) means "check timing and
  alignment", not "worse model".
- Play test: blind pairwise R6 vs RL v2 on the benchmark charts by the user; this outranks
  every automatic metric when they disagree.

Ablations attribute each gain: planner → rule 1, `REF` → rule 2, octave policy → rule 3.

## Budget

| Item | CU |
| --- | --- |
| Analysis (done) and VM smoke tests | 1.5 |
| Planner + base decoder | 7 |
| Four ablations | 6 |
| Two RL runs | 4 |
| Evaluations (about 9 x P1 3 seeds + P3 + P7) | 6 |
| Reserve (failures, reruns) | 10 |
| **Total** | **~35 of 50** |
