# Game Mechanics Specification

A mechanics spec declares, in machine-readable form, how a game (or game mode) is played. Specs:

- define which IR events are legal for a chart (`urfm.ir.validate`);
- tell game-agnostic views which modifiers are accents and which roles are non-interactive;
- are the conditioning signal `M_g` for game-conditioned decoders (roadmap M6), so a new game
  can be adapted to by providing a spec and a few examples instead of rewriting the model.

Built-in specs live in `src/urfm/mechanics/specs/*.toml`. More can be added at runtime with
`urfm.mechanics.register_directory(path)`.

## Schema

```toml
id = "maimai_dx"            # referenced by Chart.mechanics
name = "maimai DX"
family = "ring"             # drum | lane | plane | ring | grid
extends = "taiko"           # optional: documents lineage (osu_taiko extends taiko)
controls = ["section"]      # control kinds the game uses
accent_modifiers = ["break"]

[space]                     # input space
kind = "ring"               # none | lanes | plane | ring | grid
buttons = 8                 # kind-specific: lanes, width/height, buttons/regions, columns/rows
regions = ["A1", "...", "C"]

[roles."maimai:slide"]      # every legal action
primitives = ["path"]       # which IR primitives this role may use
modifiers = ["break"]
targets = ["ring"]          # allowed target tags; empty = no target
interactive = true          # false for mines/fakes
description = "..."

[constraints]
max_simultaneous = 2
min_interval_us = 0
exclusive_holds = true

[difficulty]
labels = ["Basic", "Advanced", "Expert", "Master", "Re:Master"]
level_min = 1
level_max = 15
```

## Conventions

- **Role namespaces** name the *game*, not the format: osu!taiko reuses `taiko:*` roles because
  the actions are the same, and `osu_taiko.toml` restricts them (e.g. no `hand` modifier).
- **Variable input spaces.** A spec without a fixed size (osu!mania) leaves `lanes` unset. Each
  chart then states its own size in `Chart.meta["lanes"]`.
- **Families** are coarse groupings for analysis and transfer experiments. They must not be used
  to merge roles across games.

## Adding a game

1. Write the spec TOML. Start from the closest family.
2. Write a codec ([codecs.md](codecs.md)) that emits only roles, modifiers and targets the spec
   declares.
3. Check with `urfm validate <files>`.
