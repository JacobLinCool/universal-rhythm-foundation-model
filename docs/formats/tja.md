# TJA (Taiko no Tatsujin simulators)

Codec: `urfm.codecs.tja.TjaCodec` (`id = "tja"`, `.tja`), mechanics `taiko`.
Fidelity: **L1 semantic-exact** (`read(write(read(x))) == read(x)`). Comments, blank
lines, whitespace, header spelling/order and the original text encoding are not kept.
Output is always UTF-8. Input is decoded as UTF-8 (with or without a BOM), then CP932
(Shift-JIS), then Latin-1.

## Time model

- Beat 0 is the start of the first measure. It sits at `-OFFSET` seconds of the audio:
  `TimingMap.offset_us = -OFFSET × 10⁶`. This matches tja-rs.
- A measure lasts `4 × num / den` beats (`#MEASURE num/den`, default 4/4).
- The `i`-th of `n` slots (characters) in a measure is at `measure_start + i/n × measure_beats`,
  as an exact `Fraction`. `residual_us` is always 0.
- A measure's meter is fixed when its first slot character is read, or at its `,` if it
  has no characters. A `#MEASURE` placed after notes therefore applies from the next measure.
- A command between slots `k-1` and `k` takes the beat of slot `k`. A command after the
  last slot takes the beat of the next measure.
- Each chart gets its own `TimingMap` in `Chart.timing`. `ChartSet.timing` holds the
  song-level `BPM`/`OFFSET`.

## Native ↔ IR mapping

| TJA | IR |
|---|---|
| `TITLE` `SUBTITLE` `GENRE` `MAKER` | `SongMeta.title/subtitle/genre/creator` |
| `WAVE` | `ChartSet.audio.path` |
| `DEMOSTART` (s) | `SongMeta.preview` (exact µs, snapped beat + residual) |
| other song-level headers (`SONGVOL`, `SEVOL`, `SCOREMODE`, …) | `SongMeta.extra` |
| `BPM` / `OFFSET` | `TimingMap.tempos[0]` / `offset_us` (per chart, and for the song) |
| `COURSE` (name or 0–6) | `Difficulty.label` ∈ easy/normal/hard/oni/edit/tower/dan (`Ura` → edit), `tier` 0–4 |
| `LEVEL` | `Difficulty.level` (kept raw in `meta` if it is not numeric) |
| `BALLOON`, `BALLOONNOR/EXP/MAS` | `Event.count` of balloons; unused counts go to `native.balloon_extra` |
| other course headers (`SCOREINIT`, `STYLE`, `EXAM*`, …) | `Chart.meta` |
| `#START P1` / `#START P2` | separate charts, `meta["player"] = "1"/"2"` |
| `#`-lines between `#END` and `#START` | `Chart.native.header_commands` (verbatim) |
| `1` `2` / `3` `4` | `HIT taiko:don` / `taiko:ka`, `big` for 3/4 |
| `A` `B` | `HIT taiko:don` / `taiko:ka` with `("big", "hand")` |
| `5` `6` … `8` | `REPEAT taiko:drumroll` (`big` for 6), `end` = beat of the `8` |
| `7` / `9` … `8` | `REPEAT taiko:balloon` (`kusudama` for 9), `count` from the balloon list |
| `C` `F` `G` | `taiko:mine`, `taiko:adlib`, `taiko:kadon` |
| `D` `E` `H` `I` | no IR role; kept in `native.unknown_notes` with their beat, so they still take a slot |
| `#BPMCHANGE x` | `TempoChange` |
| `#DELAY s` | `Stop(before=True, duration_us = s × 10⁶)`: a note on the delay beat is hit after the pause. Negative delays are allowed |
| `#MEASURE n/d` | `MeterChange`, recorded at measure starts where the meter changes |
| `#SCROLL x` | control `scroll` (float; non-numeric values such as complex scrolls stay strings) |
| `#GOGOSTART/END` | control `highlight` True/False |
| `#BARLINEON/OFF` | control `barline` True/False |
| `#SECTION` / `#LEVELHOLD` | controls `taiko:section` / `taiko:levelhold` |
| any other command (`#LYRIC`, `#JPOSSCROLL`, `#SUDDEN`, …) | control `tja:<name>` with the raw arguments as `value` |

A roll that is never closed gets `end` at the chart end (or at the next roll start in the
same context), with `native.implicit_end = true`. The writer then emits no `8` for it.

## Header state

Header lines form one persistent stream. Each `#START` takes a snapshot of every header
assigned so far. This applies to `BPM`, `OFFSET` and `BALLOON` too: a later course
inherits them unless it redefines them. Song-level fields come from the snapshot taken
at the first `#START`. A song-level key with a different value in a later snapshot is
stored as a per-chart override in `Chart.meta`. The writer emits a header line only when
it changes the reader's state. This keeps the output minimal and the round trip exact.

## Branch model

`#BRANCHSTART cond` opens a `BranchSegment` at the current beat. `condition` is
`{"raw", "type", "expert", "master"}`. `#N`/`#E`/`#M` each start a route from the segment
start, with the meter state restored to its value at the segment start. Events and
controls inside a route carry `BranchRef(segment, route)`.

`#BRANCHEND` sets `end`. Without it the segment is closed implicitly (`end = None`) by
the next `#BRANCHSTART` or by `#END`. After the segment, parsing continues from the end
of the last route listed.

**Timing is shared across routes.** For each route the reader collects the `BPMCHANGE`,
`DELAY` and meter entries. When every route has the same list, which is the normal
authoring practice, one copy goes into the chart `TimingMap`. When the routes differ, the
last route's list is used. Every route's own list is then kept in `native.route_timing`
and the writer restores each route from it. This keeps the round trip exact, but events
in the other routes are timed with the last route's tempo map.

Other native keys:
- `native.route_end`: routes whose length differs from the last route.
- `native.segment_end`: an implicit segment end that the writer cannot infer.
- `native.balloon_keys`: route-specific balloon lists that were present.

A `#BRANCHSTART` with no routes is kept as raw controls `tja:branchstart` and
`tja:branchend`. So is a stray `#BRANCHEND`.

`taiko:balloon` counts are consumed in file order. Inside a route, the route's own list
(`BALLOONNOR/EXP/MAS`) is used when present, otherwise the shared `BALLOON` list. The
writer rebuilds the lists in its own emission order.

## Known irreversible points

- Lost on write: comments, formatting, header order, redundant header or `#MEASURE`
  repetitions, the empty-vs-absent distinction of `BALLOON:`, the original encoding, and
  the spelling of `COURSE` and of numbers (`150.50` becomes `150.5`).
- `header:value` lines inside a chart body are ignored.
- Commands written on the same line as note characters are not supported. TJA requires
  them on their own line.
- `#NEXTSONG` (dan courses) and other multi-song constructs are kept as raw controls. Their
  timing effect (a new BPM or offset for each song) is not modelled.
- Commands placed between `#BRANCHSTART` and the first route marker are shared. When
  routes have divergent timing, such shared timing commands inside the segment are not
  re-emitted.

## Validation against tja-rs

`tests/test_codec_tja.py::test_differential_against_tja_rs` compares every note's absolute
time with tja-rs (`tja` 0.5) and requires a match within 1 µs. Known tja-rs behaviours that
are excluded or diverge:

- tja-rs ignores `BPM:`/`OFFSET:` redefined after the first course.
- tja-rs does not count letter notes as slots.
- tja-rs drops notes after `#BRANCHEND`.
- tja-rs adds `#DELAY` values cumulatively across routes.
- tja-rs carries `#MEASURE`/`#BPMCHANGE` state from one route into the next. URFM restores
  the state at the segment start. On every affected file in the corpus, this gives routes
  of identical length and identical chart end beats across all courses, which supports
  the URFM interpretation.
- tja-rs creates an extra empty chart for a duplicated `#START` line.
- tja-rs panics on some malformed branch structures.

## Round-trip results (2026-10-08)

| Corpus | Files | Charts | Errors | L2 timing-exact | L1 semantic-exact | Differential vs tja-rs |
|---|---|---|---|---|---|---|
| Fixtures (`tests/fixtures/tja`) | 14 | – | 0 | 14 | 14 | all pass (1 skipped: tja-rs panic) |
| Taiko-1000 (`JacobLinCool/taiko-1000-parsed`, raw `tja` column) | 1,155 | 5,010 | 0 | 1,155 (100%) | 1,155 (100%) | 1,140 match; 15 differ (14 route-state carry, 1 tja-rs header bug) |
| Local TJA collections (NewTaikoKit source/dani, taiko-rs songs) | 451 | – | 0 | 451 | 451 | 450 match; 1 route-state carry |

- All 1,950,218 Taiko-1000 events pass `urfm.ir.validate` with no issues.
- Parsing the whole of Taiko-1000 takes about 18 s.
- Reproduce the round trip with `URFM_CORPUS_TJA=<dir> uv run pytest tests/test_codec_tja.py -k corpus`
  or `uv run urfm roundtrip <dir> --codec tja`.
