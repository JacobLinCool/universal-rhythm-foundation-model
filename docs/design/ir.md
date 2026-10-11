# Universal Chart IR (`urfm.ir/1`)

The IR is **layer 1** of URFM. It is a lossless, normalised description of rhythm-game charts.
Layer 2 is the learned shared representation. The IR standardises the *vocabulary of
interaction*, not the note types of any game.

## Goals

1. **Lossless.** A codec can reproduce its native format from the IR (see [codecs.md](codecs.md)).
2. **Exact time.** It keeps musical positions as exact rationals and absolute times as exact
   microseconds, with no frames and no fixed grids.
3. **Game-agnostic structure, game-specific meaning.** Every game shares the shapes of
   interaction, while the actions themselves are declared per game by mechanics specs.
4. **Plain data.** The IR is made of frozen `msgspec.Struct`s, serialised to deterministic JSON
   (human-readable) or msgpack (bulk). Rationals serialise as strings like `"3/4"`.

## Object model

```text
ChartSet                      one source: a song with shared audio + timing
├── song: SongMeta            title, artist, subtitle, genre, creator, preview, extra
├── audio: AudioRef           path, sha256
├── timing: TimingMap         shared tempo map
├── provenance: Provenance    codec, source path, sha256, license
├── native: {…}               format-level data without cross-game meaning
└── charts: Chart[]
    ├── mechanics: str        id of a MechanicsSpec (taiko, osu_mania, maimai_dx, …)
    ├── difficulty            label, level, tier (0–4, cross-game coarse)
    ├── timing: TimingMap?    per-chart override (.ssc, per-difficulty .osu)
    ├── events: Event[]       player actions, sorted by time
    ├── controls: Control[]   presentation/scoring switches (scroll, gogo/kiai, barline, …)
    ├── branches: BranchSegment[]
    ├── meta: {str: str}      small chart facts (e.g. lanes = "7" for osu!mania)
    └── native: {…}
```

## Time

`Time(beat: Fraction, residual_us: int = 0)`. Its absolute time is
`round(timing.beat_to_us_exact(beat)) + residual_us`, given by `TimingMap.us(time)`.

- **Beat-native formats** (TJA, StepMania, simai) store notes on musical positions. Their
  `residual_us` is always 0 and `beat` is exact.
- **Time-native formats** (osu!) store milliseconds. `TimingMap.time_at_us` snaps each timestamp
  to the nearest grid beat and keeps the difference in `residual_us`, so both the musical
  position and the exact original timestamp are preserved.

A beat is a quarter note. Beat 0 is at `offset_us` microseconds of the audio. Negative beats are
valid: the first tempo extends backwards.

### TimingMap

| Primitive | Meaning | Sources |
| --- | --- | --- |
| `TempoChange(beat, bpm)` | BPM from `beat` on (exact rational) | everything |
| `Stop(beat, duration_us, before)` | time advances while the beat is frozen; `before=False` puts objects on the beat at the stop's start, `before=True` puts them at its end | SM `STOPS` (False), SM `DELAYS` (True), TJA `#DELAY` (True) |
| `Warp(beat, length)` | `length` beats take no time | SM `WARPS` |
| `MeterChange(beat, num, den)` | time signature; affects measure grid only | TJA `#MEASURE`, SM `TIMESIGNATURES`, osu! meter |

The map is compiled once (cached) into a piecewise-linear function with jumps. Both
`beat → µs` and `µs → beat` are exact with `Fraction`.

## Events

```text
Event(time, primitive, role, modifiers=(), end=None, target=None, path=(), count=None,
      group=None, branch=None, native={})
```

| Field | Meaning |
| --- | --- |
| `primitive` | **Shared across games**: `hit` (instant), `hold` (sustained to `end`), `repeat` (repeated actions in `[time, end]`: drumroll, balloon, spinner), `path` (moves along `path`: slider, slide, juice stream) |
| `role` | **Declared by the game's spec**, namespaced: `taiko:don`, `mania:hold`, `maimai:slide`, `sm:mine` |
| `modifiers` | sorted spec-declared flags: `big`, `break`, `ex`, `firework` |
| `target` | where the action happens (below); `None` for games without spatial input |
| `path` | `PathSegment(shape, points, start?, end?)`; `shape` is namespaced (`osu:bezier`, `maimai:^`) |
| `count` | required repetitions: balloon hits, slider slides |
| `group` | events designed as one unit: a maimai star head and its slides, each-pairs |
| `branch` | `BranchRef(segment, route)` when the event belongs to one route of a branch |

### Targets (tagged union)

| Tag | Struct | Used by |
| --- | --- | --- |
| `lane` | `Lane(index)` | osu!mania, StepMania, BMS |
| `point` | `Point2D(x, y)` in native coordinates | osu!standard, osu!catch |
| `ring` | `RingButton(index)` (0-based) | maimai buttons |
| `region` | `TouchRegion(name)` | maimai touch sensors |
| `grid` | `Grid3D(x, y, layer)` | Beat Saber (future) |

## Controls

`Control(time, kind, value, branch=None, native={})` holds anything that changes how the chart
looks or scores without being a player action. The shared kinds are `scroll`, `highlight`
(gogo / kiai), `barline` and `section`. Game-specific kinds are namespaced (`osu:sv`,
`sm:speed`).

## Branches

`BranchSegment(id, start, end?, routes, default, condition)` declares a fork. Events and controls
inside it carry a `BranchRef`. Events outside any branch have `branch=None`. Timing is shared by
all routes. `urfm.ir.views.active_events(chart, route)` linearises one path through the chart.

## Views (bridge to layer 2)

`urfm.ir.views.interaction(chartset, chart)` projects any chart into game-agnostic
`Interaction`s with these fields:

- onset (µs and beat)
- duration
- dominant primitive
- simultaneity: independent actions, where a `group` counts once
- accent: spec-declared `accent_modifiers`
- moved: whether the target changed

Non-interactive roles (mines, fakes) are excluded. `chart_stats` and `summarize` build
cross-game statistics on top of this view (`urfm stats`).

## Validation

`urfm.ir.validate.validate(chartset)` checks:

- **Structure:** sorting, `end` consistency, paths, branch references.
- **Mechanics:** the role exists, the primitive and modifiers are allowed, the target kind and
  range are valid, and `max_simultaneous` holds.

## Versioning

`SCHEMA = "urfm.ir/1"`. `urfm.ir.serialize.schema_digest()` hashes the JSON schema, so any
structural change alters the digest. Bump `SCHEMA` and record the change here.
