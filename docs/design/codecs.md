# Codecs

A codec converts between **one native chart format** and the IR. Codecs are the only code that
touches native formats.

```python
class Codec(ABC):
    id: ClassVar[str]  # "tja"
    extensions: ClassVar[tuple[str, ...]]  # (".tja",)
    mechanics: ClassVar[frozenset[str]]  # mechanics spec ids it reads/writes
    fidelity: ClassVar[Fidelity]  # declared round-trip guarantee

    def sniff(self, path, head) -> bool: ...
    def read(self, data: bytes, *, source: str | None = None) -> ChartSet: ...
    def write(self, chartset: ChartSet) -> bytes: ...
```

## Rules

- **No cross-game conversion.** `write` accepts only charts whose `mechanics` it supports and
  raises `CodecError` otherwise. Moving a chart from one game to another is the model's job.
- **Keep what the writer needs.** Anything without cross-game meaning goes to `native` (on the
  chart set, chart, event, control or path segment). Comments and whitespace need not survive.
- **Determinism.** The same IR always produces the same bytes.

## Fidelity levels

The guarantee is defined on `read(write(read(x)))` versus `read(x)`. Checking it does not require
byte equality with the original file, which would penalise harmless formatting changes.

| Level | Name | Equal |
| --- | --- | --- |
| L2 | `TIMING` | per chart: the multiset of (µs onset, primitive, role, modifiers, target, µs end, branch) |
| L1 | `SEMANTIC` | the whole IR except `provenance`, including controls, branches and `native` |

`urfm.compare.roundtrip_bytes(codec, data)` measures both levels. `urfm roundtrip <dir>` runs it
over a corpus and can write a JSONL report.

**Differential tests** check absolute times against independent implementations:

- TJA against tja-rs.
- StepMania against `simfile`'s `TimingEngine` (dev-only oracle).

## Registry

Built-in codecs are listed in `urfm.codecs.registry.BUILTIN` and imported lazily. Third-party
codecs are registered through the entry-point group `urfm.codecs`:

```toml
[project.entry-points."urfm.codecs"]
beatsaber = "urfm_beatsaber:BeatSaberCodec"
```

## Built-in codecs

| id | Formats | Mechanics | Notes |
| --- | --- | --- | --- |
| `tja` | `.tja` | `taiko` | [formats/tja.md](../formats/tja.md) |
| `osu` | `.osu` v3–v14 | `osu_std`, `osu_taiko`, `osu_catch`, `osu_mania` | [formats/osu.md](../formats/osu.md) |
| `stepmania` | `.sm`, `.ssc` | `sm_*` | [formats/stepmania.md](../formats/stepmania.md) |
| `simai` | `maidata.txt` | `maimai_dx` | [formats/simai.md](../formats/simai.md) |
| `urfm` | `.urfm.json` | all | the IR's own JSON |

The one-way export `urfm.interop.rhythm_rs.to_canonical_chart` produces rhythm-rs
`CanonicalChart` JSON (taiko-rs), so charts can be played in that engine.
