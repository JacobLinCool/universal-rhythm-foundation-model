# StepMania `.sm` / `.ssc`

Codec: `urfm.codecs.stepmania.StepManiaCodec` (id `stepmania`, suffixes `.sm`, `.ssc`).
Fidelity: **L1 semantic** (`read(write(read(x))) == read(x)`).

The MSD tag container (`#TAG:value;`) is tokenized by
[`msdparser`](https://github.com/garcia/msdparser) (MIT), which handles the
`\\` / `:` / `;` / `//` escapes in both directions. `_msd.py` groups the tags
into a song table and per-chart tables. For `.sm`, each chart is one `#NOTES`
parameter with six components. For `.ssc`, each chart starts at a
`#NOTEDATA:;` marker. Keys are upper-cased and later duplicates win. A tag's
value is its first component, except for `ATTACKS` and `DISPLAYBPM`, whose
components are joined with `:`. These are the same rules `simfile` uses.

Note data is read and written by the codec's own tolerant tokenizer
(`_notedata.py`), which also accepts StepF2/StepP1 cells. `simfile` is not a
runtime dependency. It is used only as a dev-group timing oracle in the tests.

## Dialect

The dialect comes from the `source` suffix. With no suffix, the codec falls
back to the content: `#VERSION` or `#NOTEDATA` means `.ssc`. The dialect is
stored in `ChartSet.native["dialect"]`, and the writer keeps it. When an IR
chart set has no dialect, the writer uses `ssc` if any chart has its own
timing, and `sm` otherwise.

Decoding tries `utf-8`, `cp1252`, `cp932` and `cp949`, in that order. The winning encoding goes into `native["encoding"]` and is
reused on write.

## Timing model

| StepMania | IR |
|---|---|
| `OFFSET` (seconds; beat 0 is at `-OFFSET`) | `TimingMap.offset_us = -OFFSET * 1e6` (exact `Fraction`) |
| `BPMS` `beat=bpm` | `TempoChange(beat, bpm)` |
| `STOPS` / `FREEZES` `beat=seconds` | `Stop(beat, seconds*1e6, before=False)`; a note on the beat is hit when the stop starts |
| `DELAYS` `beat=seconds` | `Stop(beat, seconds*1e6, before=True)`; a note on the beat is hit after the delay |
| `WARPS` `beat=length` | `Warp(beat, length)`; the length is kept exact, never rounded |
| `TIMESIGNATURES` `beat=n=d` | `MeterChange(beat, n, d)` |

Beat positions of timing segments are rounded to 1/48 beat, because
StepMania stores segments by row. Values (BPMs, seconds, warp lengths) are
exact rationals parsed from their decimal strings.

* **Negative BPMs** (old-style warps) are converted the way SM5's
  `NotesLoaderSM` converts them. A segment `[b, c)` at `-x` BPM followed by
  `y` BPM becomes `Warp(b, (c-b)·(1 + y/x))`, and the negative tempo entry is
  dropped. Writing then emits an explicit `#WARPS` tag. The result is
  L2-exact, and L1 differs only in tag order. Zero BPM, or a negative BPM
  that is the last tempo entry, raises `CodecError`.
* **Negative stops** are kept as negative `Stop` durations. They are not
  converted to warps. For notes after the rewound span, the times are the
  same either way.
* **SSC split timing** follows StepMania (and simfile's `timing_source`). For `VERSION ≥ 0.7`,
  a chart that has any non-empty chart-level timing or control tag uses
  *only* its own timing, stored in `Chart.timing`. Its `OFFSET` defaults to 0.
  One addition to simfile: if such a chart has no `BPMS`, the codec takes the
  song's BPMs instead of failing.

## Controls

These are the *effective* controls: song-level ones, unless the SSC chart has
its own timing. They are stored per chart in `Chart.controls`, sorted by time.

| Tag | Control kind | `value` | `native` |
|---|---|---|---|
| `SCROLLS` `beat=ratio` | `scroll` | float | |
| `SPEEDS` `beat=ratio=duration=unit` | `sm:speed` | float ratio | `duration` (float), `unit` (0 beats / 1 seconds) |
| `FAKES` `beat=length` | `sm:fake` | float length in beats | |
| `COMBOS` `beat=hit[=miss]` | `sm:combo` | int | `miss` |
| `TICKCOUNTS` `beat=n` | `sm:tickcount` | int | |
| `LABELS` `beat=text` | `section` | str | |

The song-level control tags are written from the first chart that has no
timing of its own. If every chart has its own timing, the song-level control
tags are invisible to gameplay, so they are kept verbatim in
`native["tags"]`. A control tag that does not parse is also kept raw.

## Notes

Each cell becomes one `Event` with target `Lane(column)` and an exact beat
time (`residual_us == 0`). For routine charts, the beat restarts at 0 for
every `&` player block.

| Cell | Primitive | Role |
|---|---|---|
| `1` | hit | `sm:tap` |
| `2` … `3` | hold (`end` = tail) | `sm:hold` |
| `4` … `3` | hold | `sm:roll` |
| `M` / `L` / `F` / `K` / `A` | hit | `sm:mine` / `sm:lift` / `sm:fake` / `sm:keysound` / `sm:attack` |
| anything else (`X`, `Y`, `I`, digits `5`–`9`, stray `3`) | hit | `sm:other`, character kept in `native["char"]` |

Some information is kept in `native` fields:

| `native` key | Meaning |
|---|---|
| `keysound` / `tail_keysound` | Keysound indices `1[3]` |
| `player` | Routine player (0 or 1) |
| `cell` / `tail_cell` | StepF2 groups such as `{1\|v\|1\|0}`, stored verbatim. The base note character (first field) determines the role, so a `{2\|h\|0\|0}` … `{3\|h\|0\|0}` pair is still an `sm:hold` |
| `char` | Hold heads without a tail are `sm:tap` events (StepMania plays them as taps); `char` stores their original character |

**Writer:** each measure uses `4 × lcm(beat denominators)` rows, the fewest
rows that place every note exactly, so 12ths, 24ths, 192nds and 5-tuplets (20 rows) stay
exact. An event with a nonzero `residual_us`, which only happens in IR from
time-native formats, is snapped to a 1/192-beat grid. Negative beats raise
`CodecError`.

## Metadata

* `TITLE`, `SUBTITLE`, `ARTIST`, `GENRE`, `CREDIT` go to `SongMeta`.
* `MUSIC` goes to `AudioRef.path`.
* `SAMPLESTART` goes to `SongMeta.preview` (a `Time` with residual).
* `STEPSTYPE` sets the mechanics id `sm_<stepstype with - → _>`. There are
  34 built-in specs, covering the SM5 dance, pump, kb7, ez2, para, ds3ddx,
  bm, maniax, techno, pnm and lights-cabinet types.
* `DIFFICULTY` sets `Difficulty.label`. `tier` runs from Beginner = 0 to
  Challenge = 4; Edit has no tier.
* `METER` sets `Difficulty.level`. A non-numeric meter (`?`) gives
  `level = None`, and the raw value goes to `native["meter"]`.
* `DESCRIPTION` goes to `Chart.meta["description"]`.
* Everything else is kept verbatim: song tags in `ChartSet.native["tags"]`,
  SSC chart tags in `Chart.native["tags"]`, and `.sm` radar values and extra
  fields in `Chart.native["radarvalues"/"extradata"]`. The original tag order
  is kept in `native["tag_order"]`.

## Known irreversible points

* Comments, whitespace, row counts per measure and number formatting are not
  preserved. The writer canonicalizes them, but the IR is unchanged.
* For negative-BPM files, the tag order changes (an extra `#WARPS`).
* An SSC chart with no `#NOTES`/`#NOTES2` tag is written with an empty
  `#NOTES`. It stays L2-exact.
* A non-multi-value tag with extra `:` components keeps only its first
  component (simfile and StepMania behave the same way).
* Two notes in the same cell (for example a tail and a new head on the same
  row and column) cannot both be written; the later one wins.
* Unsupported input that raises `CodecError`: unknown stepstypes, unparseable
  timing tags, zero BPMs, and an unconvertible negative BPM.

## Timing validation

There are two kinds of check:

* **Hand-computed**: `test_hand_computed_times` derives expected
  microseconds by hand from the tags of `bpm_stops.sm`, `gimmicks.ssc` and
  `split_timing.ssc`. These cover offset, BPM changes, stop versus delay
  semantics, warps and per-chart timing. They always run.
* **Oracle**: `test_times_match_simfile_engine` compares every hittable
  note's `TimingMap.us` with simfile's `TimingEngine.time_at`, within 1 µs.
  simfile is in the dev dependency group only. Through its `fs` dependency it
  needs setuptools' `pkg_resources`, so the dev group pins `setuptools<81`.
  If simfile cannot be imported, the test is skipped with that reason. Two
  documented divergences are excluded:
  * simfile rounds warp lengths to 1/48 beat, while StepMania and URFM keep
    them exact.
  * simfile has no handling for negative BPMs.

## Corpus results (2026-10-09, `uv run urfm roundtrip <dir> --codec stepmania`)

The corpus is 1,813 simfiles: 1,399 PIU (StepF2 `.ssc`), 266 from the
`noahm/simfile-parser` packs, 108 from `Hodobox/stepmania-songs`, 20 from
`JasonMcCoy/Stepmania`, 11 from the StepMania repo, 5 from simfile's test
data, and 4 others. They were sparse-cloned from GitHub for local testing
only and are not redistributed. In total they contain 17,175 charts and
9,169,188 events.

| Metric | Files |
|---|---|
| Parsed | 1,809 (99.78%) |
| L2 timing-exact | 1,809 (99.78%) |
| L1 semantic-exact | 1,807 (99.67%) |

Notes on the remaining files:

* **Not L1-exact (2):** both are still L2-exact. One is the negative-BPM
  normalization; the other is a chart with no notes tag.
* **Errors (4):**
  * a macOS `._` resource-fork file
  * the StepMania docs' deliberately malformed `SDF/test.sm`
  * `kickbox-quadarm` (no spec)
  * a typo stepstype, `pump-double-p`
* **Timing differential:** 5,524,247 hittable notes (the first 400 per chart)
  agree with simfile within 1 µs once the two divergences above are
  accounted for.
* **Validation:** 2 files have `target.range` issues, from charts whose rows
  are wider than their stepstype.
