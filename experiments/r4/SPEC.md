# R4 — audio-only, generalising chart generation

## Question

Can one model, given only a song's audio, produce Taiko charts at human level and beyond the
training distribution (denser, short tempo changes, 32nd notes, meter tricks), as newer
official charts do?

## What changes (vs R3, `ft-r3-song`)

| Component | R3 | R4 |
| --- | --- | --- |
| Timing at inference | human timing map (oracle) | beat model on audio (`urfm.beat`): per-frame beat/downbeat, no smoothness prior, decoded to a piecewise tempo map |
| Visual tempo tricks | trained as tempo | folded into scroll (`urfm.data.canonical`, time-exact) |
| Conditions | TIER + DENS tokens | 7 continuous unbounded conditions (`urfm.train.conditions`): log rates, linear level/shares; linear + slow Fourier embedding |
| Window plan | WDENS | WDENS + GRID (finest subdivision) + REST (rest share) |
| Song context | 2 s mean-pooled mel | beat-synchronous texture units (onset flux at 24 positions per beat x 4 bands + coarse mel) |
| Augmentation | none | whole-song speed 0.8–1.6x, short beat-span tempo warps, BPM octave re-notation |
| Data mix | alpha 0.5 | alpha 0.5, Taiko x2 |

## Runs

| Run | Size | Steps | Augmentation | Purpose |
| --- | --- | --- | --- | --- |
| `beat-v1` | 5M | 8k | speed 0.75–1.4, local ±25% | timing model |
| `r4a-full-15m` | 15M | 8k | speed p=.5, local p=.3, octave p=.15 | R4 recipe |
| `r4b-noaug-15m` | 15M | 8k | none (canonical only) | ablation: augmentation |
| `r4-45m` | 45M | longer | best of a/b | final |

## Panels (fixed before any R4 result)

- **P1** Taiko-1000 test, 120 songs / 599 charts: oracle timing and audio timing.
- **P2** P1 songs with tempo changes, short segments or top-20% density (subset of P1 rows).
- **P3** Crystal Hail + Nivalis＊Anima + 銀の黎明か、黒の晶華か。 + 魔宵月 (oni/edit), audio
  timing; playable TJA output. Never used for training or tuning.
- **P4** P1 played 1.25x faster (reference re-timed): density beyond the training range.
- **P5** Timing only: P1 audio with random short tempo warps (known ground truth).
- **P6** osu! test (120 songs): taiko / mania / catch / std, no regression.

## Metrics

- Timing: beat F@70ms, F@25ms, any-metrical-level F@70, downbeat F@70, grid fit (share of human
  notes within 15 ms of a 1/4 or 1/3 beat of the predicted map; human map = ceiling).
- Notes vs human: onset F1@25/50; on the **human beat grid**: IOI class L1 (`ioi_ref_l1`), fine
  share, rest share, and 8-beat section correlations of density, fine share, rest share and ka
  share (`section_r_*`); measure rhythm variety; first-onset delay; note types; controls.
- Validity: grammar/mechanics issues per 100 events, writability (must stay 0 / 1.0).

## Decision rule (R4 is better than R3 when all hold)

1. P1 audio-timing F1@25 >= R3 oracle-timing F1@25 − 0.02.
2. P1 `ioi_ref_l1` lower and mean `section_r_*` higher than R3 (same timing condition).
3. P3/P4: generated density within 15% of the human chart's, and fine/rest shares closer to
   human than R3.
4. P6: no game loses more than 0.02 F1@25.
5. Blind play preference on P3 (user).

Augmentation is kept only if `r4a` beats `r4b` on P3/P4 without losing more than 0.01 P1
F1@25. Single seed for screening; the final comparison uses 3 seeds.

## Outcome (2026-10-10)

Rules 1, 4 hold; rule 2 half (IOI yes, sections no); rule 3 half (speed stress yes, extreme
density no); rule 5 pending. Augmentation kept (r4b loses 0.076 P1 F1). 45M dropped (worse
generation at lower loss). Final model: `r4a-ft-pred`. Details and tables: `experiments/LOG.md`.
