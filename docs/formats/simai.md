# simai (`maidata.txt`) ↔ URFM IR

Codec: `urfm.codecs.simai.SimaiCodec` (id `simai`, mechanics `maimai_dx`, fidelity `SEMANTIC`).
Sources used to pin the syntax: the simai wiki by Celeca (w.atwiki.jp/simai, pages 25 and 1003),
SimaiSharp (tokenizer/deserializer), MajdataEdit `SimaiProcess.cs`, PySimaiParser.

## File structure

| simai | IR |
|---|---|
| `&title=`, `&artist=`, `&des=`, `&genre=` | `SongMeta.title / artist / creator / genre` |
| `&first=` (seconds) | `TimingMap.offset_us` (exact decimal → µs) |
| `&inote_N=` (non-empty) | one `Chart` per N, in file order; `Chart.native["slot"] = N` |
| `&lv_N=` | `Chart.meta["level"]` (raw text) and `Difficulty.level` (`13+` → 13.5, `13.7` → 13.7) |
| `&des_N=` | `Chart.meta["designer"]` |
| N = 1..7 | `Difficulty.label` Easy, Basic, Advanced, Expert, Master, Re:Master, Utage; `tier` 0,1,2,3,4,4,– |
| any other `&key=value` (incl. `&wholebpm`, empty `&lv_N` of absent charts) | `SongMeta.extra[key]` |

A field's value runs until the next line that starts with `&`. Input is decoded as UTF-8 (BOM
allowed), falling back to Shift-JIS. The first chart's tempo map becomes `ChartSet.timing`; a
chart whose tempo map differs keeps its own `Chart.timing`.

## Note stream and time

| simai | IR |
|---|---|
| `(bpm)` | `TempoChange` at the current tick (exact decimal BPM). Missing initial `(bpm)` falls back to `&wholebpm`. |
| `{div}` | each `,` advances `4/div` beats (pure formatting; not stored) |
| `{#sec}` | each `,` advances `sec` seconds = `sec·bpm/60` beats at the current BPM |
| `,` | tick separator; notes before it sit on the tick's exact beat |
| `E` | end of chart (anything after is ignored) |
| `<HS*x>` | `Control(kind="scroll", value=x)` |
| `\|\| comment` | dropped |

All beats are exact `Fraction`s with `residual_us == 0`. Second-based lengths (holds, slide
waits and lengths) are computed in exact µs from the note's tick using the BPM at that tick (simai
semantics: a hold's `[div:n]` is a duration in seconds, not a beat span), and their end is mapped
back to an exact beat through the finished tempo map. A hold that crosses a tempo change therefore
ends on the correct beat.

## Notes

| simai | IR event |
|---|---|
| `1`..`8` | `maimai:tap`, `HIT`, `RingButton(0..7)` |
| `A1`..`E8`, `C`, `C1`, `C2` | `maimai:touch`, `HIT`, `TouchRegion(name)` (name kept as written) |
| `…h`, `…h[dur]`, `…[dur]` | `maimai:hold` / `maimai:touch_hold`, `HOLD`; `end = start + dur`; bare `h` → `end == time` |
| `b` `x` `m` `f` | modifiers `break`, `ex`, `mine`, `firework` (sorted) |
| `$`, `$$` on a tap | role `maimai:star`; `native["appearance"] = "$" / "$$"` |
| `@` | `native["appearance"] = "@"` |
| `56` (adjacent locations) or `5/6` | separate events on the same tick (written back as `5/6`) |
| `` a`b`c `` (pseudo-each) | part *k* is placed `k/32` beat after the tick (1/128 note, as MajdataEdit/PySimaiParser); `native["pseudo"] = k` |

Hold durations: `div:n`, `bpm#div:n`, `#sec`, `bpm#sec`. The original text is kept in
`native["dur"]` so the writer reproduces it.

## Slides

A slide note `1b-5[4:1]*>7[8:3]b` becomes:

* one **head** event `maimai:star` (`HIT`, `RingButton(0)`, head modifiers such as `break`/`ex`),
* one **slide** event per `*`-joined path: `maimai:slide`, `PATH`, `time` = head tick,
  `end` = end of the motion, `target` = start button, `modifiers = ("break",)` when the slide
  itself is a break (`b` after the slide operator/duration),
* all sharing one `group` id (per chart, counting from 0).

Each operator in a chain is one `PathSegment(shape="maimai:<op>", points=(from, *vertices))`;
`V` takes two vertices (`1V35`). Operators: `- ^ < > v V p q pp qq s z w`.

* `segment.start` (first segment only) = head time + wait.
* `segment.end` is set for every segment that has its own duration bracket; with a single
  bracket (`1-4q7-2[2:1]`) only the last segment has an end, because how simai distributes the
  time over segments depends on path geometry that the IR does not encode.
* Duration brackets: `div:n`, `bpm#div:n`, `bpm#sec`, `#sec`, `wait##sec`, `wait##div:n`.
  The wait (default: one beat at the current BPM) is taken from the slide's first bracket, as in
  MajdataEdit/SimaiSharp. Raw text is kept in `segment.native["dur"]`.
* Headless slides `1?-5[4:1]` / `1!-5[4:1]` produce no star event; every slide of the note has
  `native["head"] = "?" | "!"`.

## Writer

* Emits `&title`, `&artist`, `&des`, `&genre`, `&first`, `extra` fields, then per chart
  `&lv_N`, `&des_N`, `&inote_N`.
* Every event, tempo change and `scroll` control becomes a tick on its exact beat; `{div}` is
  chosen per run of gaps as the least common multiple of the divisions each gap needs (runs break
  when the LCM would exceed 64), so positions are always exact. `{#sec}` is never emitted.
* Durations use the preserved `native["dur"]` text; without it they are derived from event times
  as `[div:n]` at the tick's BPM (smallest exact `div`), falling back to `[#sec]`; a non-default
  slide wait is written as `[wait##…]`.
* Rejects: events with `residual_us != 0`, stops/warps, per-chart `&first`, controls other than
  `scroll`, non-ring/touch targets, modifiers outside `break/ex/mine/firework`.

## Irreversible points (text-level; the IR round-trip is exact)

* Comments, whitespace/line layout, `{div}` choices and `{#sec}` notation are not preserved;
  `{#sec}` ticks are re-expressed with an equivalent `{div}`.
* `56` is rewritten as `5/6`; numbers are normalised (`150.00` → `150`, `&first=1.250` → `1.25`).
* `&first_N`, `&demo_*` and other keys survive only as opaque `extra` text.

## Known semantic choices / divergences

* `{#sec}`: we advance `sec` seconds per comma and keep the real BPM for later `[div:n]`
  durations; SimaiSharp also re-derives the tempo (`60/sec`) for durations in that region.
* Pseudo-each offset follows MajdataEdit/PySimaiParser (1/32 beat); SimaiSharp keeps the notes
  simultaneous and only marks them as "broken each".
* Hold notes that also start slides (`1h[4:1]-5[4:1]`) and slides starting on touch regions are
  rejected.

## Verification

* 12 handwritten fixtures (`tests/fixtures/simai/`) cover every construct above; all are L1
  semantic-exact and pass `validate()` (`tests/test_codec_simai.py`, 49 tests).
* Community corpus (Neskol/Maichart-Converts, 1,806 `maidata.txt`, 7,095 charts, 3.17 M events;
  used locally only, not redistributed): **L1 1806/1806, L2 1806/1806**, 0 parse errors, 0
  validation issues. With all preserved `dur` text removed, the writer's derived durations still
  round-trip µs-exact on 1806/1806 files.
* Differential test against PySimaiParser on the same corpus: identical onset sets on all 7,095
  charts (|Δt| ≤ 50 µs, the 4-decimal rounding of the comparison); on a 300-file sample, hold
  and slide lengths agree within 1 µs; over the full corpus, slide waits and lengths match after
  rounding to 0.1 ms except for 324 of 283,929 slides, all spot-checked cases being values on
  the 4th-decimal rounding boundary (integer-µs vs float). PySimaiParser itself misreads the
  terminal `E` as a touch note and counts each `*` slide as a separate note; the comparison
  accounts for both.
