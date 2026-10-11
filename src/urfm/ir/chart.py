"""Charts and chart sets.

A :class:`ChartSet` is what one source (a ``.tja`` file, an ``.sm`` file, a
simai ``maidata.txt``, an ``.osu`` file) describes: one song with shared timing
and audio, and one or more :class:`Chart` difficulties. A chart may override
the shared timing (``.ssc`` per-chart timing, per-difficulty ``.osu`` files).
"""

from __future__ import annotations

from typing import Any

import msgspec

from urfm.ir.controls import Control
from urfm.ir.events import Event
from urfm.ir.time import Time
from urfm.ir.timing import TimingMap

SCHEMA = "urfm.ir/1"


class BranchSegment(msgspec.Struct, frozen=True, kw_only=True):
    """A span where the chart forks into alternative routes (Taiko branches).

    Events and controls inside the span carry a :class:`~urfm.ir.events.BranchRef`
    pointing at one of ``routes``. ``condition`` describes how the route is chosen,
    in the source game's terms.
    """

    id: int
    start: Time
    end: Time | None = None
    routes: tuple[str, ...]
    default: str
    condition: dict[str, Any] = {}


class Difficulty(msgspec.Struct, frozen=True, kw_only=True):
    label: str | None = None
    """Native difficulty name (``oni``, ``Challenge``, ``Master``, ``Insane``)."""
    level: float | None = None
    """Native numeric level or star rating."""
    tier: int | None = None
    """Cross-game coarse tier 0 (easiest) .. 4 (hardest); filled by the codec when known."""


class Chart(msgspec.Struct, frozen=True, kw_only=True):
    mechanics: str
    """Id of the :class:`~urfm.mechanics.spec.MechanicsSpec` this chart is played with."""
    difficulty: Difficulty = Difficulty()
    events: tuple[Event, ...] = ()
    controls: tuple[Control, ...] = ()
    branches: tuple[BranchSegment, ...] = ()
    timing: TimingMap | None = None
    """Overrides :attr:`ChartSet.timing` for this chart."""
    meta: dict[str, str] = {}
    native: dict[str, Any] = {}


class SongMeta(msgspec.Struct, frozen=True, kw_only=True):
    title: str = ""
    artist: str = ""
    subtitle: str = ""
    genre: str = ""
    creator: str = ""
    preview: Time | None = None
    extra: dict[str, str] = {}


class AudioRef(msgspec.Struct, frozen=True, kw_only=True):
    path: str
    sha256: str | None = None


class Provenance(msgspec.Struct, frozen=True, kw_only=True):
    codec: str
    source: str | None = None
    sha256: str | None = None
    license: str | None = None


class ChartSet(msgspec.Struct, frozen=True, kw_only=True):
    schema: str = SCHEMA
    song: SongMeta = SongMeta()
    audio: AudioRef | None = None
    timing: TimingMap
    charts: tuple[Chart, ...] = ()
    provenance: Provenance | None = None
    native: dict[str, Any] = {}

    def timing_of(self, chart: Chart) -> TimingMap:
        return chart.timing or self.timing
