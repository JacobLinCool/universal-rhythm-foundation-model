# Evaluation (v0)

`urfm.eval` compares a generated chart with the reference chart for the same song. Everything
it measures is computed from the IR, so the same harness works for every game. Reference-note
metrics alone are not enough, because one song admits many valid charts. These metrics are the
fast-iteration panel, and M4 adds reference-free metrics.

```python
from urfm.eval import evaluate_pair, aggregate, write_report

row = evaluate_pair(gen_set, gen_chart, ref_set, ref_chart, target_density=6.0)
write_report(rows, "eval.jsonl", "eval.md")  # JSONL per chart + Markdown table per mechanics
```

## Metrics

| Key | Definition |
| --- | --- |
| `precision@T`, `recall@T`, `f1@T` (T = 25, 50 ms) | One-to-one greedy matching of sorted onsets |
| `mean_abs_offset_ms@25` | Mean absolute offset of matched pairs |
| `density_gen`, `density_ref`, `density_abs_err`, `density_rel_err` | Onsets per second; see below |
| `density_target`, `target_abs_err`, `target_rel_err` | Error against the requested density, when one is given |
| `issues`, `issues_per_100` | `urfm.ir.validate` issues (structure plus mechanics spec) per 100 generated events |
| `writable` | The game's codec can write the chart and read it back; see below |
| `primitive_ratio_l1` | L1 distance between primitive shares (hit, hold, repeat, path) |
| `chord_ratio_gen/ref/diff` | Share of onsets with more than one independent action |
| `ioi_hist_l1` | L1 distance between inter-onset-interval histograms in beats (`urfm.ir.views` bins) |
| `taiko_role_acc@25` | Taiko only: agreement of `don` / `ka` / `big` on onsets matched within 25 ms |
| `cge.*` | Taiko only, when `chartgeneval` is importable: its `compute_all_families` profile; non-finite values are dropped |
| `lane_entropy_gen/ref` | Lane games: normalised entropy of lane usage |
| `jack_ratio_gen/ref` | Lane games: share of lane uses repeating the previous onset's lane |

Notes on the definitions:

- **Onsets** are the interactive onsets of the game-agnostic Interaction View: mines, fakes and
  other non-interactive roles are excluded, branches follow their default route, and grouped
  events count once.
- **Matching** is a single pass over both sorted lists. A reference onset takes the nearest
  unmatched generated onset within the tolerance.
- **Density** for the generated and reference charts uses the same denominator: the union
  span of both charts' onsets.
- **Writable** maps each game to its codec: `taiko` → `tja`, `osu_*` → `osu`, `sm_*` →
  `stepmania`, `maimai_dx` → `simai`.
- **Taiko classes** are `don`, `ka`, `don_big` and `ka_big`. `chartgeneval` is not a
  dependency; install it alongside, e.g. `uv run --with ~/Documents/GitHub/ChartGenEval`.

`aggregate(rows)` averages every numeric field per generated mechanics. `None` and NaN values
are ignored, and `charts` counts the rows.

## Example

`evaluate_pair` on the `basic.tja` fixture against itself gives:

```text
f1@25 = 1.0   f1@50 = 1.0   issues = 0   writable = True   density_abs_err = 0
taiko_role_acc@25 = 1.0   primitive_ratio_l1 = 0   ioi_hist_l1 = 0
```

The same chart shifted by 30 ms gives `f1@25 = 0` and `f1@50 = 1`.
