# Universal Rhythm Foundation Model (URFM)

Charts for any song, for many rhythm games, from audio alone.

URFM separates **what the music asks the player to do** from **how a particular game asks for
it**. Charts from Taiko, osu! (all four modes), StepMania and maimai are read into one lossless
**Universal Chart IR**, and each game's controls are described by a declarative **Game Mechanics
Spec**. A single model trained on all of these games writes new charts for songs it has never
heard.

## Features

- **Audio in, playable chart out.** Tempo and bar lines come from a beat model, so no BPM or
  offset is needed. Charts are written for Taiko (`.tja`) and for osu!taiko, osu!mania (any key
  count), osu!catch and osu!standard (`.osu`).
- **Song-level structure.** A song planner reads the whole song and plans every section
  (density, peak bursts, fine notes, rests, colours, long notes, finest subdivision,
  repetition, rhythm variety). Choruses get denser than verses, and patterns repeat where the
  music repeats.
- **Editable plans.** The plan can be inspected and edited before the notes are written.
- **Any notation.** Sixteenths to sixty-fourths and triplet grids are written correctly on
  whatever tempo grid the audio yields (half or double tempo included).
- **Controllable.** Game, course and level, mean and peak density, scroll and meter activity,
  sampling temperatures and seed.
- **Lossless IR and codecs.** Positions are exact rational beats with microsecond residuals.
  TJA, `.osu`, `.sm`/`.ssc` and simai charts round-trip with measured fidelity.

## Try it

```bash
uv sync
uv run --extra app python -m urfm.app.web            # web UI at http://127.0.0.1:7860
uv run python -m urfm.app.pipeline song.mp3 --courses oni:9 hard:6 --out out/
uv run python -m urfm.app.pipeline song.mp3 --game osu_mania --keys 4 --courses hard insane
```

The web UI returns a zip with the audio and chart files, and reports the time spent in each
stage (a 2.5-minute song with three Taiko courses takes about 15 s on a laptop CPU, 90% of it in
the note decoder). Checkpoints are fetched from the Hugging Face Hub on first use.

## How it works

```text
audio ──► log-mel ──► beat model (5M) ──► timing map (tempo, bar lines)
              │                                  │
              ├──► song planner (1.1M) ──► section plans (every ~3 s, editable)
              │                                  │
              └──► chart decoder (18.5M) ◄───────┘   one 16 s window at a time:
                     audio window + song texture       conditions, section plan,
                     + its own previous notes          memory of a similar earlier section
                             │
                             ▼
                   Universal Chart IR ──► TJA / .osu writers
```

- **Beat model**: a convolutional Transformer over the log-mel. Dynamic-programming decoding
  turns its output into one consistent metrical level and meter.
- **Song planner**: a bidirectional Transformer over per-section audio descriptors (onset
  envelope folded onto one beat at 48 phases, band energies, tempo), conditioned on the game
  and the requested density.
- **Chart decoder**: an encoder–decoder Transformer. Each event is written as a shared part
  (position and primitive: hit, hold, repeat or path) followed by a game-specific part (role,
  lane or position, as declared by the mechanics spec). A grammar derived from the spec
  constrains decoding, so every output is a valid chart.

## Training

| Stage | Data | Method |
| --- | --- | --- |
| Data | Taiko-1000 (1,155 songs) and osu! (21,314 songs, four modes), 111k training charts with one song-level split across games | Charts are read into the IR. Benchmark songs are held out for evaluation only. |
| Supervised | All games, with Taiko weighted up | Next-token prediction, with section plans and memory taken from the human chart. Augmentation covers tempo warps, half, double and quarter-BPM re-notation, and re-notation on the beat model's grid. Context corruption counters exposure bias. |
| Planner | Same charts and augmentation | Per-section classification of the plan values. |
| RL | Taiko and osu!taiko | Song-level GRPO. The model writes ~96 s spans while reading its own output, and is rewarded for placement precision, rhythm distribution, plan adherence and structure alignment, with penalties for stagnation and monotony and a KL anchor to the supervised model. |

Onset F1 against one human chart is reported but not optimised, because a song admits many
valid charts. Instead, the model is rewarded when the notes it writes are placed precisely and
when the chart's structure follows the music.

## Results (held-out Taiko-1000 test, timing from audio)

| | R5 (previous) | Current |
| --- | --- | --- |
| Section contrast (density r with the human chart) | 0.16 | 0.29 |
| Structure alignment (human 0.27) | 0.07 | 0.10 |
| Rhythm distance to human (IOI L1) | 0.52 | 0.38 |
| Excess monotone time, s/min (human 0.26) | 0.66 | 0.22 |
| Onset F1 @ 25 ms | 0.575 | 0.556 |

On four newer, harder charts the R6 models reach a section contrast of 0.38–0.42 (R5: 0.21)
and a structure alignment of 0.13–0.15 (R5: 0.05, human: 0.19). Fast bursts are still
under-produced on songs that the beat model reads at half tempo. The reasons behind each design
choice, with ablations, are in [docs/research/design-decisions.md](docs/research/design-decisions.md),
and the full record is in [experiments/LOG.md](experiments/LOG.md).

## Documentation

| Topic | |
| --- | --- |
| Research claim and milestones | [docs/roadmap.md](docs/roadmap.md) |
| Design decisions and evidence | [docs/research/design-decisions.md](docs/research/design-decisions.md) |
| IR, mechanics specs, codecs | [docs/design/ir.md](docs/design/ir.md), [mechanics.md](docs/design/mechanics.md), [codecs.md](docs/design/codecs.md) |
| Tokenizer, model, timing, evaluation | [tokenizer.md](docs/design/tokenizer.md), [model.md](docs/design/model.md), [timing.md](docs/design/timing.md), [evaluation.md](docs/design/evaluation.md) |
| Compute (Colab, Hugging Face Hub) | [docs/design/compute.md](docs/design/compute.md) |

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

Adding a game: write `src/urfm/mechanics/specs/<game>.toml`, implement and register a `Codec`,
and add hand-written fixtures with a round-trip test (see
[docs/design/mechanics.md](docs/design/mechanics.md)).

## License

The code is MIT-licensed. Chart data and audio belong to their respective owners and are not
distributed here. See [docs/roadmap.md](docs/roadmap.md#data-and-licensing).
