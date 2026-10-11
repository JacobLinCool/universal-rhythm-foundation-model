# Tokenizer (`urfm.tokenize`)

The tokenizer turns IR charts into decoder token sequences for one audio window and back. The
interface is in `src/urfm/tokenize/api.py`; the implementation is `UrfmTokenizer` and the
singleton is `build_tokenizer()`.

## Sequence

```text
prefix:  GAME_<spec> [LANES_<n>] TIER_<0..4|None> DENS_<0..15>     (R1–R3 models)
         GAME_<spec> [LANES_<n>] COND                              (condition models, R4)
body:    BOS  <context groups>  SEP  WDENS_<0..15> [GRID_<g> REST_<r>]  <window groups>  [EOS]
         BOS [REF SBEAT <groups> REFEND] <context groups> SEP (SBEAT SDENS SFINE SREST SKA SSUB)*
             <window groups> [EOS]                                 (section-plan models, R6)
group:   (POS_BEAT_b POS_SUB_s | POS_SAME) PRIM_p [DUR_BEAT DUR_SUB] [ACCENT]
         ROLE_r [MOD_m...] [target] [COUNT_c] [wait: DUR_BEAT DUR_SUB] [SHAPE point+]...
         └──────── shared by all games ─────────┘ └── game-specific (from the spec) ──┘
```

### Presentation controls

Charts are more than notes: Taiko charts change scroll speed (including gradual ramps),
abuse `#MEASURE` for bar-line effects (for example repeated `#MEASURE 1/32` with a scroll
ramp), mark GOGO sections and hide bar lines. These are generated as **control groups**
in the same stream as events:

```text
POS_BEAT POS_SUB CTRL_scroll SCROLL_<i>                   scroll speed (osu! SV shares it)
POS_BEAT POS_SUB CTRL_highlight_on | CTRL_highlight_off   gogo / kiai
POS_BEAT POS_SUB CTRL_barline_on   | CTRL_barline_off
POS_BEAT POS_SUB CTRL_meter METER_NUM_<n> METER_DEN_<d>   meter change (n 1..64, d 1..64 pow2)
```

- **Position**: control groups always carry an explicit position (never `POS_SAME`). At
  one position they come before events (meter, scroll, highlight, bar line, then
  source order) and do not change which event a following `POS_SAME` refers to.
- **Density**: they are not onsets, so they do not count toward `WDENS`.
- **Scroll values**: these use a 5 % log grid from 0.05 to 32, with 0.5, 0.75, 1, 1.25, 1.5,
  2, 3 and 4 kept exact. Quantization errors:

  | value | 0.78 | 0.82 | 0.85 | 1.31 | 1.62 | 1.92 | 2.25 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | error | 0.5 % | 0.6 % | 0.7 % | 1.3 % | 0.4 % | 2.2 % | 0.9 % |

  Non-positive or textual scroll values are dropped and counted as
  `control_value:scroll`.
- **Meter changes** come from the timing map. Only genuine changes are kept (osu!
  repeats 4/4 on every red line). They do not affect time; generation attaches them to the
  output `TimingMap`.
- **Grammar**: a control kind is legal only if the game's spec declares it:
  - `scroll` or `osu:sv` → scroll;
  - `highlight` → gogo/kiai;
  - `barline` → bar lines and meters.

  maimai therefore gets scroll only.
- **Not tokenized**: other namespaced controls (`tja:*`, `taiko:section`, `osu:red_line`)
  are counted in `DROPS` as `control:<kind>`.
- **Decoding**: `decode_full(...) -> (events, controls, meters)`. `decode` keeps returning
  events only. Scroll decodes to the game's own kind.
- **Vocabulary**: control tokens are appended at the end of the vocabulary, growing it from
  817 to 1,002. Ids of earlier tokens are unchanged, and checkpoints trained on the shorter
  vocabulary still generate (their masks are cut to the model's size).

### Conditioning: continuous conditions (R4)

`COND` is a slot: the model replaces its embedding with an embedding of seven chart
measurements (`urfm.train.conditions`), each on a scale chosen from how the quantity behaves:

| Condition | Scale | Why |
| --- | --- | --- |
| onsets/s, 90th-percentile onsets per 2 s, onsets per beat | log2 | density is perceived and grows multiplicatively; doubling tempo doubles each rate |
| official level (Taiko only) | linear | levels are designed as an even ladder |
| scroll changes, meter changes per minute | log2(1 + x) | log for many, 0 for none |
| share of onsets off the 16th grid | linear | a share |

Values are z-scored with fixed typical centres and scales and are not clipped, so a request
beyond anything in training is a larger number rather than a saturated bin. Each value `z`
enters as `[z, sin(πz/4), cos(πz/4), sin(πz/8), cos(πz/8)]` with a validity mask: the linear
term carries extrapolation, and the slow periodic terms (periods of 8 and 16 standard
deviations) cannot wrap around inside any sensible range. Training hides each condition with
probability 0.15 and all of them with probability 0.1, so any subset can be given.

### Conditioning: per-game tiers (R1–R3)

`TIER_k` means the *k*-th quintile of onset density among the training charts of the
same game (`urfm.train.tiers.TierTable`, built from the training index and stored in
checkpoints). A tier guessed from difficulty names did not mean the same thing across
games: adding osu!taiko hurt Taiko easy and normal courses in the first sprint.
`DENS` keeps the absolute density. For generation, `TierTable.density(game, tier)`
gives the typical density of a tier in a target game, so asking every game for the same
tier yields comparable *relative* difficulty.

### Section plans and structure memory (R6)

- **Sections** (`urfm.tokenize.sections`): `k` beats, `k` the power of two nearest 3.2 s at the
  chart's median tempo, so half- and double-BPM notations of the same music get sections of the
  same length in seconds. Section `i` spans beats `[i k, (i + 1) k)`.
- **Plan** (after `SEP`, one entry per section overlapping the window): `SBEAT_b` (section start
  in whole beats after the window start, 0 if it began earlier), then `SDENS` (onset density
  bin), `SFINE`, `SREST`, `SKA` (share bins with upper edges 0.05, 0.2, 0.4, 0.7) and `SSUB` (the
  finest subdivision per beat holding 90% of the onsets, a `GRID_VALUES` index; relative to the
  grid, so a half-BPM grid asks for 32nds where the human wrote 16ths). Teacher-forced in
  training (each value moved by one bin with probability 0.1); at inference the song planner
  (`urfm.train.planner`) writes it and generation forbids further plan tokens.
- **Memory** (`REF SBEAT <groups> REFEND`, before the context): the events of the earlier
  section whose audio is most similar to a section of the window (cosine of per-section audio
  descriptors, `urfm.data.sections`; at least 2 sections back and above the song's 80th
  percentile similarity), positions relative to that section's start, at most 48 events;
  `SBEAT` says where the matching window section starts. At inference it is the model's own
  earlier output, so the model can repeat a pattern where the music repeats.
- **Tuplets**: for `taiko` and `osu_taiko` (`TUPLET_FREE`) positions snap to the 1/48 and 1/32
  grids only and the grammar masks the 5/7-tuplet sub-beats (0.04% of Taiko-1000 onsets).

### Windows, context and plan token (shared by training and generation)

There is exactly one definition of each, used by the data pipeline and by the generator:

- **Windows** (`audio_windows(timing, duration_us)`): measure-aligned windows of at most
  `WINDOW_S` = 16 s tiling the *whole audio*, so intros, outros and long rests are trained
  as windows whose body is `WDENS_0 EOS`.
- **Context** (`UrfmTokenizer.context`): events in the `CONTEXT_BEATS` = 32 whole beats
  before the window, at most `CONTEXT_EVENTS` = 32 of them. Positions are relative to
  `window_beat − 32`, an integral beat, so an on-beat event has `POS_SUB_0` in the context
  exactly as in the body. The cut never starts on a `POS_SAME` group.
- **`WDENS`** (`encode_window`): the window's onset density bin, the first token after `SEP`.
  It is predicted by the model (a per-window density plan) and can be forced at generation
  (`Job.density_plan`) to control the song-level density arc.
- **`GRID`, `REST`** (R4): the rest of the window plan. `GRID_g` is the coarsest subdivision
  per beat (1, 2, 3, 4, 6, 8, 12, 16, 24 or 48) that holds 90 % of the window's onsets: 4 for
  16th notes, 3 or 6 for triplets, 8 for 32nds, 0 for an empty window. `REST_r` is the share
  of the window's beats in rests of at least one beat, in tenths. Both are optional in the
  grammar, so older models skip them. Generation samples them at the plan temperature, so a
  low note temperature cannot collapse every section onto one texture.
- **Truncation** (`encode_window(..., room)`): only at event boundaries; a truncated window
  gets no `EOS`, so cutting is never taught as an ending.

Every event the tokenizer cannot represent is counted in `DROPS[(mechanics, reason)]`
(`outside_position_range`, `unencodable:<role>`, `path_segment:<shape>`, `truncated`);
training logs these per mechanics (`metrics.jsonl` → `pipeline`), so losses concentrated in
one game or event type stay visible.

### Shared tokens (identical meaning in every game)

| Token | Meaning |
| --- | --- |
| `POS_BEAT_b` | Beat index relative to the window's start beat, b < 128. A 16 s window at 400 BPM is about 107 beats. |
| `POS_SUB_s` | Sub-beat position on the grid {k/48} ∪ {k/32} ∪ {k/5} ∪ {k/7}, 74 values. k/48 covers 192nd notes, k/32 covers simai pseudo-each, k/5 and k/7 cover tuplets. |
| `POS_SAME` | Same onset as the previous group; used for chords, each-pairs and star + slide. |
| `PRIM_p` | Interaction primitive: hit, hold, repeat or path. |
| `DUR_BEAT` + `DUR_SUB` | Duration in beats, on the same exact grid as positions. Omitted for hits. |
| `ACCENT` | The spec's first `accent_modifiers` entry: Taiko `big`, maimai `break`. |

### Game-specific tokens

| Token | Meaning |
| --- | --- |
| `ROLE_r` | Namespaced role. |
| `MOD_m` | Sorted non-accent modifiers. |
| Target | `LANE_i`, `RING_i`, `REGION_name`, or `PT_X PT_Y` on an 8 px osu! grid (±4 px error). |
| `COUNT_c` | Balloon hits or slider slides, on the grid 1..32, then 36..64 step 4, 80, 96, 128, 192, 256. |
| Wait | First slide segment start relative to the head, for maimai slides; a `DUR_BEAT DUR_SUB` pair. |
| Path | `SHAPE` followed by up to 8 control points per segment. The joint point shared with the head or the previous segment is implicit. |

### Choices

- **Positions are factorised into beat and sub-beat**, and durations reuse the same exact grid.
  This keeps the vocabulary small (801 tokens) with no loss on beat-native games.
  - An earlier single-token duration grid lost up to 47 ms on long osu! sliders and mania holds.
- **`POS_SAME`** encodes simultaneous events with one token instead of two.
- **Not tokenised:**
  - `residual_us`, the off-grid microseconds of time-native formats such as osu!.
  - Non-interactive roles such as mines and fakes.
  - `group` ids. On decode, maimai stars and slides that share an onset and a button are
    paired again.
  - Native fields.
- **Branches:** the `master` route is encoded when it exists; otherwise each segment's
  default.
- **Tier:** `Difficulty.tier` when the codec sets it. Otherwise, as for osu! version names,
  keyword matching on the label (easy … insane, extra). Otherwise `TIER_None`.
- **Density:** interactive onsets per second over the chart, in 16 bins spaced 2^(1/2.5) apart
  from 0.5/s to about 32/s.
- **Windows:** `windows(chartset, chart)` yields `(start_us, end_us)` pairs.
  - Each window starts on a measure boundary and ends at the next one, so windows tile the
    chart without overlap.
  - A window is at most `WINDOW_S` = 16 s long; measures longer than that are split.
  - The start beat is snapped to 1/48, so positions stay on simple fractions.
  - Previous-window context is up to the trainer: use `encode_groups()` and keep the last
    `CONTEXT_EVENTS` groups.

## Grammar and constrained decoding

`Grammar` (one per mechanics and lane count, cached) is a state machine over small hashable
tuples. It offers:

- `mask(state)`: a cached `np.bool_[vocab]` mask.
- `step(state, token) -> state`.
- `run(tokens)`.

It enforces:

- the group order;
- primitives and roles allowed by the spec (only interactive roles);
- accent only on roles that allow it;
- sorted, unique modifiers;
- the required target kind and its range, e.g. `LANE_0..3` for dance-single; for osu!mania
  pass `lanes=`;
- `SEP` before `EOS`.

A group ends implicitly when the next token starts a new group, so `EOS` is legal only where a
group may end. Measured cost: about 0.7 µs per step on an M-series CPU.

## Frame features

`frame_features(timing, start_us, n_frames)` returns `float32 [n_frames, 7]` aligned to mel
frames (`MEL.fps` ≈ 86.13 Hz). The columns are listed in `FRAME_FEATURES`:

| Column | Meaning |
| --- | --- |
| `beat_sin`, `beat_cos` | Phase within the beat |
| `measure_sin`, `measure_cos` | Phase within the measure |
| `log2_bpm_120` | log2(BPM / 120) |
| `beat_tick` | 1 on frames where a beat boundary falls |
| `downbeat_tick` | 1 on frames where a measure boundary falls |

Beats are computed with a vectorised float inverse of the timing map. With negative stops the
later segment wins, matching osu! red lines.

## Information loss (fixtures, `urfm.tokenize.report`)

`fixtures_report` runs `roundtrip_report` over all fixtures. Each window is encoded and decoded,
and every decoded event is compared with its source event:

- **exact:** same beat, primitive, role, modifiers, target and end beat.
- **≤1ms:** same kind, onset and end within 1 ms, and point within 6 px.

| mechanics | charts | events | encoded % | exact % | ≤1ms % | mean err ms | max err ms | max end err ms | max pt err px | path pts kept % | windows | events/win | tokens/win | tokens/win p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maimai_dx | 17 | 178 | 100 | 97.2 | 97.8 | 0.028 | 2.49 | 3.32 | 0 | 100 | 25 | 7.12 | 45.3 | 115 |
| osu_catch | 1 | 5 | 100 | 80 | 100 | 0 | 0 | 0 | 4 | 100 | 2 | 2.5 | 22 | 41.8 |
| osu_mania | 2 | 18 | 100 | 100 | 100 | 0.0833 | 0.5 | 0.5 | 0 | - | 4 | 4.5 | 23.8 | 37.9 |
| osu_std | 5 | 43 | 100 | 11.6 | 83.7 | 0.304 | 3.08 | 4.46 | 4 | 100 | 9 | 4.78 | 37.6 | 97.2 |
| osu_taiko | 2 | 20 | 100 | 100 | 100 | 0.259 | 0.857 | 0.418 | 0 | - | 4 | 5 | 24 | 46.2 |
| sm_dance_double | 1 | 19 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | 1 | 19 | 90 | 90 |
| sm_dance_routine | 1 | 6 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | 1 | 6 | 31 | 31 |
| sm_dance_single | 20 | 198 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | 26 | 7.62 | 38 | 94.8 |
| sm_pump_double | 1 | 8 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | 1 | 8 | 36 | 36 |
| sm_pump_single | 1 | 12 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | 2 | 6 | 27.5 | 33.4 |
| taiko | 20 | 125 | 100 | 100 | 100 | 0 | 0 | 0 | 0 | - | 32 | 3.91 | 17 | 58.5 |

### Remaining loss

- **Taiko (TJA) and StepMania:** lossless.
- **maimai:** charts written with `{#sec}` or BPM-dependent absolute durations land off the
  beat grid. The error is ≤ 2.5 ms at onset and ≤ 3.3 ms at end. Intermediate segment end
  times of chained slides that carry their own durations are not kept; only the first
  segment's start and the slide's end are.
- **osu!:**
  - Dropping `residual_us` costs up to about 3 ms on maps with off-grid red lines and under
    1 ms otherwise.
  - Points are quantised to 8 px.
  - Slider ends land off-grid because they are derived from pixel length and slider velocity
    (≤ 4.5 ms). The "exact %" for osu!standard is low only because of the 8 px grid.
- **Corpus scale:** these numbers come from hand-written fixtures. Run `roundtrip_report` on
  the ingest shards on Colab for corpus-scale numbers.
