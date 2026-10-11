# AGENTS.md — Universal Rhythm Foundation Model (URFM)

URFM asks whether charts from many rhythm games can teach a model a
**transferable representation of musical interaction**: what in a chart is
decided by the music and shared across games, and what is decided by one
game's mechanics. This file is a map; details live in `docs/`.

## Map

| Need | Read |
| --- | --- |
| Research claim, milestones, what each stage must deliver | [docs/roadmap.md](docs/roadmap.md) |
| Prior work and how URFM differs | [docs/research/related-work.md](docs/research/related-work.md) |
| Why the generation system is built this way (evidence, ablations, RL design) | [docs/research/design-decisions.md](docs/research/design-decisions.md) |
| The Universal Chart IR (time model, events, targets, controls, branches) | [docs/design/ir.md](docs/design/ir.md) |
| Game Mechanics Specs (TOML) and how to add a game | [docs/design/mechanics.md](docs/design/mechanics.md) |
| Codec protocol, fidelity levels, round-trip testing | [docs/design/codecs.md](docs/design/codecs.md) |
| Per-format mapping tables and known irreversible points | `docs/formats/*.md` |
| Colab GPU runtimes, data placement, HF Hub, efficiency rules | [docs/design/compute.md](docs/design/compute.md) |
| Tokenizer, model, training data and augmentation | [docs/design/tokenizer.md](docs/design/tokenizer.md), [docs/design/model.md](docs/design/model.md) |
| Timing from audio (beat model) and musical vs visual tempo | [docs/design/timing.md](docs/design/timing.md) |

## Layout

```text
src/urfm/ir/          lossless IR: time, timing, events, targets, controls, chart, views, validate, serialize, transform
src/urfm/mechanics/   MechanicsSpec model + built-in specs/*.toml
src/urfm/codecs/      native formats (tja, osu, stepmania, simai) + IR JSON; registry & entry points
src/urfm/compare/     semantic diff and parse->write->parse round-trip measurement
src/urfm/interop/     one-way exports (rhythm-rs CanonicalChart)
src/urfm/data/        dataset records, mel features, canonical tempo, texture and section features
src/urfm/tokenize/    IR <-> tokens (vocabulary, grammar)
src/urfm/model/       chart model (encoder-decoder), song planner
src/urfm/beat/        beat model, labels from charts, decoding to a TimingMap
src/urfm/train/       data pipeline, conditions, training, planner, generation, RL
src/urfm/eval/        metrics (incl. monotony, variation, structure alignment) and the protocol
src/urfm/app/         generation app: audio file -> playable TJA with per-stage timing (CLI + web UI)
src/urfm/cli.py       urfm inspect | convert | validate | roundtrip | stats
tests/                unit, property and per-codec fixture tests; corpus tests gated by env vars
```

## Commands

```bash
uv sync                     # install (Python >= 3.12)
uv run pytest               # all tests
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run urfm roundtrip <dir> --codec tja -v --report out.jsonl
uv run urfm stats <dirs...> --by mechanics
uv run --extra app python -m urfm.app.web        # chart generator UI at http://127.0.0.1:7860
uv run python -m urfm.app.pipeline song.mp3 --courses oni:9 hard:6 --out out/
```

## Architecture rules

1. **Layering.** `ir` has no I/O and no knowledge of formats. `mechanics` depends only on
   `ir`. Codecs are the only code that touches native formats. Models, datasets and
   evaluation must consume the IR, never native files.
2. **Lossless first.** A codec's read must keep everything its writer needs; anything without
   cross-game meaning goes to `native`. Fidelity is measured, not assumed (`urfm roundtrip`).
3. **Exact time.** Positions are `Time(beat: Fraction, residual_us: int)`. Never quantize to
   frames or fixed grids inside the IR; quantization belongs to model tokenizers.
4. **Primitive vs role.** `primitive` (hit/hold/repeat/path) is shared by all games;
   `role` (`taiko:don`, `mania:hold`) and `modifiers` are declared by the game's mechanics
   spec. Do not invent cross-game equivalences between roles in codecs.
5. **Codecs never convert between games.** Cross-game transfer is the model's job.
6. **New game = new spec + new codec.** Changing IR structs is a schema change: bump
   `SCHEMA`, note it in `docs/design/ir.md`, and expect `schema_digest()` to change.
7. Keep modules small, typed (msgspec Structs, frozen), and deterministic. Match surrounding
   style; no speculative abstractions.

## Compute rules

- Heavy work (dataset download/processing, feature extraction, training, evaluation at scale)
  runs on **Colab L4** through `scripts/colab.py`, never on the local machine.
- Large data never moves between local and Colab. The VM downloads raw corpora itself and
  publishes processed datasets and checkpoints to the **Hugging Face Hub**. Only code, small
  scripts and small results cross the local↔VM boundary.
- Pipelines must keep the GPU saturated: batched GPU feature extraction, pinned-memory
  prefetching loaders, non-blocking transfers, bf16 + `torch.compile`, length bucketing, no
  host syncs in hot loops. Record GPU utilisation and throughput in every run
  ([details](docs/design/compute.md)).
- Never print, log or commit credentials. Colab uses ADC (`--auth adc`), and the HF token is
  uploaded as a file.

## Research workflow rules

Follow the experimental-research workflow (spec → execution → raw evidence → canonical
tables → analysis → claims):

- Write `experiments/<id>/SPEC.md` before running anything reportable.
- Raw evidence (`runs/raw/`) is append-only; failed runs are data.
- Canonical tables come only from deterministic scripts; analysis outputs are regenerable.
- Metric, panel and split definitions are part of the data contract; version, don't redefine.
- **Song-level split across all games**: a song's charts from every game share one split.
- Record source, license and redistribution status for every dataset; downloading rights are
  not training or redistribution rights. Never commit copyrighted charts or audio; test
  fixtures are hand-written.
