"""Gameplay events: the actions a player performs.

An event separates *what kind of interaction* happens (``primitive``, shared by
all games) from *which game action* realises it (``role``, declared by the
game's mechanics spec). Everything a format needs for a lossless round trip but
that has no cross-game meaning lives in ``native``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec

from urfm.ir.targets import Target
from urfm.ir.time import Time


class Primitive(StrEnum):
    HIT = "hit"
    """An instantaneous action."""
    HOLD = "hold"
    """An action sustained from ``time`` to ``end``."""
    REPEAT = "repeat"
    """Repeated actions inside ``[time, end]`` (drumrolls, balloons, mashes)."""
    PATH = "path"
    """An action that moves along ``path`` (sliders, slides, arcs)."""


class BranchRef(msgspec.Struct, frozen=True):
    """Membership of an object in one route of a :class:`~urfm.ir.chart.BranchSegment`."""

    segment: int
    route: str


class PathSegment(msgspec.Struct, frozen=True, kw_only=True):
    """One piece of a movement path.

    ``shape`` is namespaced (``osu:bezier``, ``maimai:^``) because path
    families differ between games; ``points`` are the control points in the
    game's own target space. ``start``/``end`` are set when the motion has its
    own timing distinct from the owning event (e.g. maimai slide delays).
    """

    shape: str
    points: tuple[Target, ...]
    start: Time | None = None
    end: Time | None = None
    native: dict[str, Any] = {}


class Event(msgspec.Struct, frozen=True, kw_only=True):
    time: Time
    primitive: Primitive
    role: str
    """Namespaced game action, e.g. ``taiko:don``, ``mania:note``, ``maimai:slide``."""
    modifiers: tuple[str, ...] = ()
    """Sorted game-declared modifiers, e.g. ``big``, ``break``, ``ex``."""
    end: Time | None = None
    target: Target | None = None
    path: tuple[PathSegment, ...] = ()
    count: int | None = None
    """Required repetitions: balloon hits, slider repeats."""
    group: int | None = None
    """Events sharing a group id are designed as one unit (each/chord, slide chain)."""
    branch: BranchRef | None = None
    native: dict[str, Any] = {}
