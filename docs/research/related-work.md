# Related work and positioning

Surveyed 2026-10. Entries marked † have details that were not fully verified.

## Chart generation

| Work | What it does | Relation to URFM |
| --- | --- | --- |
| Dance Dance Convolution (Donahue et al., ICML 2017) | DDR step placement, then step selection (CNN-LSTM + LSTM) | Single game. Separating timing from action is echoed in our primitive/role split. |
| TaikoNation (Halina & Guzdial, 2021) | Pattern-focused Taiko generation | Single game |
| Beat-Aligned Spectrogram-to-Sequence (Yi, Lee, Lee; ISMIR 2023 LBD, arXiv:2311.13687) | Beat-aligned audio + Transformer. **Pre-training on osu!mania improves F1 on Fraxtil/ITG.** | **Closest prior cross-game transfer.** It stays within the lane family and has no explicit shared latent. |
| Mapperatorinator / osuT5 / osu-diffusion (OliBomby) | 219M Whisper-style model with one token vocabulary for **all four osu! modes**. Conditioning on mode, difficulty, mapper, year and descriptors. Diffusion refines positions. | Multi-mode but single ecosystem. Time is quantized to 10 ms, positions to a grid. It shows a shared vocabulary works within osu!. |
| BeatLearning / BEaRT (sedthh) | Intermediate format for 1/2/4-track osu! maps; 100 ms tokens | Work in progress, osu!-only, superseded by its author's pointer to newer work |
| Time-based Chart Partitioning (Hanzen, Halina, Guzdial; AIIDE 2025) | Onset detection + beat snapping + pattern matching for osu!mania | Single game; focuses on local pattern coherence |
| ITGPT (O'Malley, 2026, arXiv:2607.14148)† ; Dance Dance ConvLSTM (2025) | Transformer / ConvLSTM for DDR/ITG | Single family |
| InfernoSaber, DeepSaber, BeatSage†, Beat Weaver† | Beat Saber automappers | Single game |
| SoftChart (ours, Taiko) | Conditional Taiko generator, slot lattice, planner | Taiko baseline and source of the Taiko-1000 clean split |

## Evaluation

- **ChartGenEval** (ours, arXiv:2607.12857). Seven-axis, reference-free evaluation, validated by
  corruption probes. Currently Taiko-specific; M4 generalises it through the Interaction View.
- **MIREX 2026 Rhythm Game Chart Generation.** osu!taiko only. Automated checks plus jury and
  community votes.

## Interchange formats

| Format | Time model | Scope | Verdict |
| --- | --- | --- | --- |
| RGC Specification (v0.2.0, "not stable") | absolute ms per event; namespaced per-game schemas | intended for many games; docs incomplete | Borrow the namespacing idea |
| urc-spec | ms | vertical-scrolling lane games | Too narrow |
| ROX (rhythm-open-exchange) | µs, binary, Rust + Python FFI | mania/taiko/SM/Quaver/FNF | Too narrow; no beats |
| rgchart (Rust/WASM) | `GenericManiaChart` | mania-like only | Too narrow |
| rhythm-rs `CanonicalChart` (ours, taiko-rs) | µs ticks + tempo map, lanes/roles, explicit branches | Taiko, lane, radial | Semantics adopted; exported via `urfm.interop.rhythm_rs` |

None of them stores **exact musical position together with exact absolute time**, 2D and ring
targets, and paths. URFM's IR is designed to fill that gap.

## Parsing libraries

- StepMania uses `msdparser` (MIT) for the tag container plus our own note-data reader, because `simfile` rejects about 450 StepF2 PIU files and needs `pkg_resources`. `simfile` is kept as a dev-only timing oracle.
- TJA, osu! and simai are parsed by our own codecs, for these reasons:
  - Existing TJA parsers, including our tja-rs, emit float seconds and discard beat structure.
    tja-rs is kept as a differential-test oracle.
  - osu! libraries are either aimed at difficulty calculation (rosu-pp) or unmaintained
    (slider).
  - Simai parsers are C# (MaiLib) or have unverified licences.

## Where URFM is new

1. Transfer **across mechanically distant families**: drum, lane, plane and ring. Prior work
   covers either one ecosystem (osu! modes) or close lane games (mania → ITG).
2. An **explicit shared interaction latent**, compared against naive mixed training.
3. **Low-resource adaptation to new mechanics** through a declarative mechanics spec plus
   adapters.
4. A **lossless, exact-time IR** with measured round-trip fidelity, so the data layer itself is
   auditable.

A unified format alone, or one model serving several games, is **not** a sufficient
contribution. The claim must be supported by transfer gains in experiments A and B.
