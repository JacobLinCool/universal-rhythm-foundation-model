"""Where an action happens: the spatial part of an event.

Targets are a tagged union so that every game declares its input space with
the same small vocabulary. Games without spatial input (Taiko) use no target.
"""

from __future__ import annotations

import msgspec


class Lane(msgspec.Struct, frozen=True, tag="lane"):
    """A discrete column: osu!mania keys, StepMania panels, BMS keys."""

    index: int


class Point2D(msgspec.Struct, frozen=True, tag="point"):
    """A position on a 2D playfield in the game's native coordinate system."""

    x: float
    y: float


class RingButton(msgspec.Struct, frozen=True, tag="ring"):
    """One of the buttons placed on a ring (maimai buttons 1-8)."""

    index: int


class TouchRegion(msgspec.Struct, frozen=True, tag="region"):
    """A named touch-sensitive region (maimai ``A1``..``E8``, ``C``)."""

    name: str


class Grid3D(msgspec.Struct, frozen=True, tag="grid"):
    """A cell of a layered grid (Beat Saber line index, layer, ...)."""

    x: int
    y: int
    layer: int = 0


Target = Lane | Point2D | RingButton | TouchRegion | Grid3D
