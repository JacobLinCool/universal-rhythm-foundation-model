# Design decisions and evidence (generation system, R4–R6.2)

Why the generation system is built the way it is. Each section states the choice, the evidence
that led to it, and what remains open. Numbers come from `experiments/LOG.md`; canonical tables
from `python experiments/r6/tables.py runs/r6/results`.

Panels used below: **P1** = Taiko-1000 test (599 charts) with timing from audio, 3 seeds;
**P7** = P1 with the human timing re-notated at half BPM (same sound); **benchmark** = four
newer, harder Taiko charts held out from all training.

## 1. What counts as a good chart

**Onset F1 is a diagnostic, not a target.** One human chart is one valid answer among many;
F1 against it penalises valid alternatives and rewards copying. It is reported, and a large
drop triggers a look at timing, but no decision rests on it.

**Placement and existence are separated.** Whether a note should exist at a moment is
uncertain; a note that exists must sit exactly on its place. Placement precision (offset to the
nearest human onset within ±75 ms, scored `exp(-½(Δ/12 ms)²)`) is a primary metric and reward;
F1 at a strict 25 ms tolerance keeps a small weight as the existence anchor. A wider F1
tolerance was rejected because it rewards approximately placed notes.

**Musical metrics, each validated on human charts first:**

| Metric | Definition | Human vs model when adopted |
| --- | --- | --- |
| Section contrast | Correlation with the human chart of per-section density, fine share, rest share, ka share | Two human courses of one song agree at r 0.63–0.70; R5 reached 0.16 |
| Structure alignment | Correlation of the chart's self-similarity matrix with the audio's (units ≈ 2.5 s) | Human 0.30 (circular-shift null 0.055; 98–100% of charts above it); R5 0.07 |
| Excess monotone time (`E_time`) | Seconds per minute of rhythm-repetitive stretches beyond the human 95th percentile at the same density | Held-out humans 0.07–0.48; R3 10–26 |
| Stagnation | Time-based debt when neither rhythm nor colour changes, with a grace period (10 s, 5 s when both are static) and a ramp | 2 s don + 2 s ka bursts inside varied music never penalised; all-don penalised after 5–8 s; human 0.007 |
| Subdivision recall | Share of human onsets at 16th, 24th, 32nd+ positions matched within 25 ms | Exposes notation failures that density metrics hide |

## 2. Song planner (section contrast)

**Choice.** A separate bidirectional model plans every ~3.2 s section from the whole song; the
decoder only realises the plan.

**Evidence.**
- A *linear* ridge model from per-section audio already predicts human section density at
  r = 0.54; R4/R5 decoders reached 0.13–0.16. Contrast is determined by the audio but was left
  to token-by-token sampling inside 16 s windows that never see the whole song.
- Ablation (10k steps): without the planner, section r falls 0.224 → 0.196 and monotony rises
  (E_time 1.69 → 2.59, stagnation 0.74 → 1.52).
- Sections are a power of two of beats nearest 3.2 s, so half- and double-tempo notations get
  sections of the same length in seconds.

**Plan values.** Each candidate was kept only if (a) two human courses of the same song agree on
it (r > 0.3, i.e. it is a property of the song, not of the charter) and (b) audio predicts it
(r > 0.2):

| Value | Course agreement | Audio r | Kept |
| --- | --- | --- | --- |
| Density, fine share, rest share | 0.47–0.70 | 0.29–0.54 | R6 |
| Ka share | 0.38–0.46 | not measured | R6 |
| Finest subdivision | not measured | not measured | R6 (needed for notation, §3) |
| Peak density (2 s) | 0.76 | 0.69 | R6.1 |
| Repetition of an earlier section | 0.86 | 0.51 | R6.1 |
| Rhythm variety | 0.66 | 0.59 | R6.1 |
| Colour-switch rate | 0.53 | 0.47 | R6.1 |
| Long-note share | 0.49–0.67 | 0.21 | R6.1 |

Adding the R6.1 values cut monotony to human level on P1 (E_time 0.48 → 0.19, stagnation 0.090 →
0.045) and raised section contrast (0.264 → 0.280), at equal decoder temperature (the extended
model also had 15k more training steps).

**Pitfall found.** The first planner targets matched subdivisions exactly; osu! charts (stored
in ms) and charts re-notated on the beat model's grid are a few ms off, so nearly every such
section fell into the finest class and the planner never asked for 32nds on an audio grid. A
1/96-beat tolerance fixed it (on 銀の黎明 edit the planner now plans 32nds for half the
sections).

## 3. Notation invariance (fine rhythm on any grid)

**Choice.** Re-notate half of the training charts at ½ (60%), 2 (25%) or ¼ (15%) of their BPM,
and half of them on the beat model's predicted grid; mask 5- and 7-tuplets for Taiko-family
games.

**Evidence.**
- 32nds and finer were 0.5% of canonical Taiko-1000 onsets, so a model trained on canonical
  notation writes fast passages only on fast grids; the beat model returns half the human tempo
  for 8% of test songs. Halving turns 16ths into 32nds, 24ths into 48ths, 32nds into 64ths.
- P7 ablation (half-BPM notation): with the R4 policy F1 0.570, 16th recall 0.055, E_time 4.2;
  with the new policy 0.595, 0.125, 2.4. The full model loses only 0.005 F1 at half-BPM notation.
- Tuplets were 0.04% of human onsets, but RL v2 wrote 42% quintuplets on one chart (reward
  hacking through an unused corner of the vocabulary); positions now snap to the 1/48 and 1/32
  grids only and the grammar masks tuplets.

**Open.** On a half-tempo grid the decoder meets a section's density with steady 16ths instead
of human 32nd bursts with rests (銀の黎明 edit: < 60 ms intervals 13% vs human 51%, up from 3%).
The peak-density plan value helped little; the next step is to make bursts binding (a burst
share or within-section shape).

## 4. Structure memory

**Choice.** Each window also sees the events of the earlier section whose audio is most similar
(cosine of section descriptors, above the song's 80th percentile), so the decoder can repeat a
pattern where the music repeats; at inference it is the model's own earlier output.

**Evidence.** Weak so far: the ablation without memory changed structure alignment by 0.003 on
P1. It is kept because it costs little and gives the repetition plan value something to act on;
whether it matters should be re-tested with the repetition plan in place.

## 5. Exposure bias and sampling

- A 45M decoder reached lower validation loss than 15M but generated worse charts (F1 0.527 vs
  0.587, section correlations near 0): it leaned on its own previous tokens. Model size stayed
  at ~20M (decoder 18.5M with one song-encoder layer + planner 1.1M), and half the training
  samples have their context and memory corrupted (drop 10%, swap don/ka 10%, shift 5% by a 16th).
- Decoder temperature 0.85 made charts monotone; 1.0 cut E_time 1.63 → 0.72 and stagnation 0.54
  → 0.24 with other metrics unchanged. Plan temperature 0.7 balances contrast against variety
  (plan temperature 0 gave the highest section r, 0.27, but stagnation 5.4).

## 6. Reinforcement learning

**Window-level GRPO (R5, R6).** 16 prompts × 8 samples of single 16 s windows after human
context, KL to the starting model.
- v1 (F1 + density + fast share) reward-hacked: rhythm variety collapsed (IOI L1 0.49 → 0.71).
  v2 added rhythm distance and a log-ratio fast-note term.
- R6 added placement, plan adherence, structure and stagnation. Structure alignment rose
  (+0.013, ~3σ) but charts became more repetitive (E_time 1.06 without the structure term vs
  1.97 with it): the window-level stagnation term never fired, because a 16 s window after human
  context rarely builds debt.

**Song-level GRPO (R6.2, `urfm.train.rl_song`).** Each sample writes six consecutive windows
(~96 s) reading its own output as context and memory; rewards are computed on the whole span:

| Term | Weight | Why |
| --- | --- | --- |
| Placement precision | 1.0 | Notes that exist must be exact |
| Onset F1 @ 25 ms | 0.25 | Existence anchor only |
| Rhythm (IOI class L1) | 0.5 | Without it v1 traded rhythmic variety for F1 |
| Plan adherence (per-section density) | 0.5 | The decoder should realise the planner |
| Structure alignment | 0.5 | Halved from window-level RL, where it over-repeated |
| Stagnation over the span | 1.0 | Debt can now build across windows |
| Excess monotone time | 0.25 | Direct guard against minute-long repetition |

4 songs × 4 samples per step, KL 0.05, 150 steps used. Result: P1 F1 0.541 → 0.556, section r
0.280 → 0.293, placement within 10 ms 0.681 → 0.686, monotony still at human level (E_time 0.19 →
0.22); P7 improved on every musical metric (E_time 0.45 → 0.38, stagnation 0.15 → 0.07). Modest
but net positive; it is the default decoder (`r6x-rl`).

## 7. Infrastructure lessons

- `DataLoader` yields batches in worker order; workers that emit bursts of sorted batches then
  stall the GPU behind the slowest one (60 → 193 samples/s with `in_order=False`, per-song
  feature caches and four windows per chart visit).
- Colab runtimes can vanish; training resumes from `latest.pt` on the Hub and evaluation results
  are uploaded as soon as they exist.
- Planner targets and writer semantics need their own tests: a meter change inside a measure
  shifted every later note of a written TJA by seconds until the writer cut such measures short
  (`tests/test_codec_tja.py::test_meter_change_inside_a_measure_keeps_time`).

## 8. Open problems

1. 32nd bursts on half-tempo grids (§3).
2. Section contrast (0.29) and structure alignment (0.10) are still well below human agreement
   (0.63–0.70 and 0.27).
3. The research claim -- cross-game supervision transfers -- has not yet been tested with the
   planned target-only vs mixed-training experiments (roadmap M5–M6).
