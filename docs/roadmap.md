# Roadmap

## Research claim

> Cross-game supervision lets a model learn a transferable representation of musical
> interaction. This improves chart generation for individual games and generalises to
> unseen or low-resource game mechanics.

Scientific question: **how much of a chart is decided by the music and by decisions shared
across games, and how much by one game's mechanics?** Two outcomes are both publishable:

- Transfer gains across mechanically distant families (drum ↔ lane ↔ plane ↔ ring).
- Gains that appear only between close games, which would map the boundary of a shared rhythm
  representation.

Everything below is a means to test that claim. Every milestone ends with a result that can be
checked or reported on its own.

## Architecture

```text
Native charts ──codecs──▶ Universal Chart IR (layer 1, lossless)
                               │ views.interaction / tokenizers
Audio ──shared audio encoder──▶ Shared interaction latent Z (layer 2, learned)
                               │ game-conditioned decoder D(Z, M_g, d_g, s_g)
                               ▼
                         IR events ──validator (MechanicsSpec)──▶ codecs ──▶ native charts
```

- `M_g`: mechanics spec (input space, legal actions, constraints).
- `d_g`: target difficulty.
- `s_g`: style.

Compute target: one machine with 1–8 GPUs, so models of roughly 100–300M parameters.

## Milestones

| # | Deliverable | Result it produces | Status |
| --- | --- | --- | --- |
| M0 | Repo, tooling, design specs | Reviewable IR, mechanics and codec specs | ✅ |
| M1 | IR + TJA + osu! (4 modes) codecs | Lossless round-trip report on real corpora; differential timing tests | ✅ |
| M2 | StepMania + simai codecs, Interaction View, `urfm stats`, rhythm-rs export | Four mechanics families in one IR; first cross-game statistics table | ✅ |
| M3 | Cross-game dataset | Song identity across games, global song-level split, manifest, dataset report | next |
| M4 | Game-agnostic evaluation | Spec-driven metrics plus a ChartGenEval adapter, validated with corruption probes | |
| M5 | Baselines | Target-only and naive mixed-training models per game | |
| M6 | Shared latent | Exp A (cross-game transfer), Exp B (low-resource adaptation) | |
| M7 | Breadth | Exp C (cross-game musical consistency); Beat Saber, Arcaea, BMS | |

### M3 — Dataset

**Song identity across games.** Audio fingerprints (chromaprint) plus NFKC-normalised
title/artist, merged with union-find. This generalises the Taiko-1000 clean split, which used
audio SHA plus title groups.

**One global song-level split.** The same split applies to every game, so test songs never
appear in training through another game's chart. The Taiko-1000 clean split (813/80/120
groups, seed 20260710) is inherited as-is.

**Manifest row per chart:**

- source and source revision
- codec and mechanics
- difficulty
- audio hash and audio-match confidence
- quality signals (ranked or loved, ratings)
- `license` and `redistributable`

**Storage.** Sharded IR in msgpack, with Parquet indexes. A game-balanced sampler handles
training.

**Dataset report.** Counts per game, family and difficulty, plus `urfm stats` per family.

### M4 — Evaluation

Reference-note F1 is insufficient because one song admits many valid charts. Evaluation will
cover these families of metrics:

- timing / onset alignment
- density and difficulty
- local pattern coherence
- structure (phrase and section boundaries)
- playability: constraint violation rate from mechanics specs

Taiko charts go through an adapter to ChartGenEval's `(t, class)` events. New metrics must pass
corruption probes before use.

### M5–M6 — Experiments

| Exp | Question | Design |
| --- | --- | --- |
| A | Does data from other games help a target game? | Fixed target data. Compare target-only, naive mixed and explicit shared latent. |
| B | Can a new game be learned from little data? | Pre-train on other games, then adapt with 50 / 200 / 1,000 charts using adapters only. |
| C | Do charts generated for different games agree musically? | Same audio rendered to Taiko, mania and maimai. Compare accents, density curves and phrase boundaries, not notes. |

Each experiment gets `experiments/<id>/SPEC.md` before running (see AGENTS.md).

## Data and licensing

Downloading rights are not training or redistribution rights. Known constraints:

- **osu!** (data.ppy.sh) dumps are licensed for statistical analysis; contact ppy before
  public use. Beatmap audio may be uncleared.
- **maimai and Arcaea** charts are proprietary. Fan simai charts have unclear rights.
- **StepMania packs** (Fraxtil, ITG) are redistributed informally and contain commercial
  audio.
- **Taiko-1000** is gated on Hugging Face.

The repository never contains copyrighted charts or audio. Fixtures are hand-written, and corpus
tests read local paths set through environment variables.
