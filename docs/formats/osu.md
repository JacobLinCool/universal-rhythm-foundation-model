# osu! (`.osu`) codec

Codec id `osu`, module `urfm.codecs.osu`. It reads and writes file format v3–v14
for all four rulesets. One `.osu` file maps to one `ChartSet` with one `Chart`.
Grouping the difficulties of a beatmap set into one `ChartSet` is left to the
dataset layer (M3).

| Mode | Mechanics spec |
|---|---|
| 0 osu!standard | `osu_std` |
| 1 osu!taiko | `osu_taiko` |
| 2 osu!catch | `osu_catch` |
| 3 osu!mania | `osu_mania` (`Chart.meta["lanes"]` = key count) |

The parser and writer are our own. `format.py` handles text sections and
`codec.py` maps them to the IR.

## Timing model

osu! is time-native: objects carry millisecond timestamps, and every
uninherited ("red") timing point restarts the beat grid.

- **Beat 0** is the first red line. `TimingMap.offset_us` is its time.
- **Later red lines** are placed on `ceil(b)`, where `b` is the exact beat at
  which the red line falls under the previous tempo. A `Stop(before=True)` on
  that beat absorbs the difference, which is usually negative. As a result:
  - every red line sits on an integer beat;
  - measures restart there, as they do in osu!, with one `MeterChange` per red
    line;
  - the absolute time of every beat stays exact, because all arithmetic uses
    `Fraction`.
- **BPM** is `60000 / beatLength`, parsed exactly from the decimal string. The
  writer prints `60000 / bpm` back as the exact terminating decimal.
- **Object times** are snapped relative to their governing red line. The codec
  picks the simplest subdivision (1, 1/2, 1/4, 1/3, 1/8 … 1/96, plus tuplets
  1/5, 1/7, 1/9 …) within 1.5 ms. Whatever is left over is stored in
  `Time.residual_us`, so `timing.us(event.time)` equals the source milliseconds
  × 1000 exactly. osu! truncates times to whole milliseconds, so residuals up
  to about 1 ms are normal.
- **Files older than v5** are shifted by +24 ms on read and by −24 ms on write,
  matching osu!'s `EarlyVersionTimingOffset`. `PreviewTime` is not shifted.
- **Maps without a valid red line** fall back to 60 BPM from time 0, as osu!
  does. This sets `native["implicit_timing"]`, and the writer then emits no red
  line.
- **Rejected red lines** are those with a NaN, infinite or non-positive beat
  length. osu! ignores them, so the codec does too, but keeps their raw lines
  in `native["rejected_timing_points"]`.
- **Warps, and stops not on a red line,** cannot be represented in osu!. The
  writer raises `CodecError` for them.

## Timing points → controls

Every timing point becomes one control, in time order. Ties keep file order.

| Timing point | Control |
|---|---|
| red line | `osu:red_line`. The value is `None`. `native` holds the non-default sample set, sample index, volume and effects without the kiai bit, plus `fields` when the line has fewer than 8 fields. BPM and meter live in the `TimingMap`. |
| green line in std/catch | `osu:sv`. The value is the SV multiplier `-100 / beatLength`. `native["beat_length"]` keeps the raw string, which may be `NaN`. |
| green line in taiko/mania | `scroll`, with the same payload as `osu:sv`. |
| change of kiai state | `highlight` with a `bool` value, placed right after the point that switches it. |

The writer rebuilds the effects of each point from the native effects plus the
highlight state. A `highlight` control that does not fall on an existing point
gets a new SV-1 green line.

## Hit objects → events

| osu! object | Event |
|---|---|
| std circle | `osu:circle`, HIT, `Point2D(x, y)` |
| std slider | `osu:slider`, PATH. `count` = slides. `end` is computed from the pixel length, SV, `SliderMultiplier` and beat length. |
| std spinner | `osu:spinner`, REPEAT |
| taiko circle | `taiko:don`. It becomes `taiko:ka` if the hitsound has whistle or clap, and gets the `big` modifier if it has finish. |
| taiko slider | `taiko:drumroll`, REPEAT, `big` if finish |
| taiko spinner | `taiko:balloon`, REPEAT |
| catch fruit / juice stream / banana | `catch:fruit` HIT, `catch:juice_stream` PATH, `catch:banana_shower` REPEAT |
| mania note / hold | `mania:note` HIT / `mania:hold` HOLD, with `Lane(floor(x * keys / 512))` |

Slider curves become one `PathSegment` per curve type. Shapes are
`osu:bezier`, `osu:linear`, `osu:perfect` and `osu:catmull`. Unknown curve
tokens, such as lazer's B-spline degree, become `osu:other` with the raw token
in native. Points include the head, so `path[0].points[0] == target`. Bezier
red anchors stay as repeated points.

Fields kept in `native` only when they differ from the default:

- `new_combo` and `combo_skip`
- `hitsound`. For taiko it is kept only when it differs from the canonical
  encoding: ka = clap, big = finish.
- `sample`, `edge_sounds`, `edge_sets`
- the slider pixel `length` (always kept, so the writer reproduces it exactly)
- positions that carry no gameplay meaning: taiko and spinner `x`/`y` when not
  256/192, and mania `x` when it differs from the lane centre or `y` when not
  192
- for drumrolls, the raw `curve` and `slides`

## Other sections

- `AudioFilename` → `audio.path`.
- `PreviewTime` → `song.preview`, which is `None` when the value is −1.
- `Title`, `Artist` and `Creator` → `song`.
- `Version` → `difficulty.label`.
- For mania, `CircleSize` → `meta["lanes"]`.
- Every other key–value pair (General, Editor, Metadata, Difficulty, Colours)
  is kept in order as `[key, value]` lists in `Chart.native`.
- `[Events]` lines, including storyboard indentation and comments, are kept
  verbatim.
- Unknown sections go to `native["extra_sections"]`.

## Known irreversible points

These are semantic no-ops. The codec does not restore:

- the key order inside `[General]` and `[Metadata]`, because structured keys
  are written first or last;
- `//` comments outside `[Events]`;
- blank lines and the original number formatting, e.g. `500` versus `500.0`;
- the relative order of a red and a green line at the same millisecond when
  the IR has no `osu:red_line` controls, i.e. a chart built from scratch;
- the `TimingMap` structure when it was not produced by this codec. Charts
  built from scratch round-trip at L2 (exact timing), not L1.

`difficulty.level` is left empty, because the star rating needs a difficulty
calculator.

## Fidelity (2026-10-08)

| Corpus | Files | Events | Errors | L2 timing-exact | L1 semantic-exact |
|---|---|---|---|---|---|
| `tests/fixtures/osu` (handwritten, all modes) | 10 | 86 | 0 | 10 | 10 |
| Real osu!taiko maps (`music visual dynamics/data`, 191 difficulties) | 191 | 151,616 | 0 | 191 | 191 |
| osu!lazer test resources (80 std, 5 taiko, 4 mania, 1 catch; edge cases such as NaN points, header-only files, multi-segment sliders, v3–v14) | 90 | 13,990 | 0 | 90 | 90 |

On all 291 files, the absolute time of every event equals the source
milliseconds × 1000. The residuals are distributed as follows:

| Residual | Events |
|---|---|
| 0 | 25,239 |
| 1 ms or less | 138,623 |
| 1–1.5 ms | 1,858 |
| over 1.5 ms (fallback 1/192 snap) | 1 |

To reproduce:

```
URFM_CORPUS_OSU=<dir> uv run pytest tests/test_codec_osu.py -k corpus
uv run urfm roundtrip <dir> --codec osu -v
```

Real osu!mania and osu!catch maps are still under-represented: only lazer's
sample maps cover them. Extend the corpus from data.ppy.sh dumps or
project-riz/osu-beatmaps once M3 data access is settled.
