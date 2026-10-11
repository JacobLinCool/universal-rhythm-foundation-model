"""Game Mechanics Specification.

A mechanics spec is a declarative, machine-readable description of how a game
is played: its input space, which (primitive, role, modifier) combinations are
legal, and its operational constraints. Specs drive IR validation today and
are the conditioning signal ``M_g`` for game-conditioned decoders later, so a
new game can be added by writing a spec instead of changing the model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import msgspec

from urfm.ir.events import Primitive


class SpaceKind(StrEnum):
    NONE = "none"
    """No spatial choice (Taiko)."""
    LANES = "lanes"
    """Discrete columns ``0 .. lanes-1``."""
    PLANE = "plane"
    """Continuous 2D playfield ``[0, width] x [0, height]``."""
    RING = "ring"
    """Buttons on a ring plus optional named touch regions (maimai)."""
    GRID = "grid"
    """Discrete layered grid ``columns x rows`` (Beat Saber)."""


class InputSpace(msgspec.Struct, frozen=True, kw_only=True):
    kind: SpaceKind
    lanes: int | None = None
    width: float | None = None
    height: float | None = None
    buttons: int | None = None
    regions: tuple[str, ...] = ()
    columns: int | None = None
    rows: int | None = None


class RoleSpec(msgspec.Struct, frozen=True, kw_only=True):
    primitives: tuple[Primitive, ...]
    modifiers: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    """Allowed target tags (``lane``, ``point``, ``ring``, ``region``, ``grid``); empty = none."""
    interactive: bool = True
    """``False`` for objects the player must not or need not act on (mines, fakes)."""
    description: str = ""


class Constraints(msgspec.Struct, frozen=True, kw_only=True):
    max_simultaneous: int | None = None
    """Maximum number of events starting at the same instant."""
    min_interval_us: int | None = None
    """Minimum gap between consecutive event onsets (per chart)."""
    exclusive_holds: bool = False
    """A held target cannot receive another event until the hold ends."""


class DifficultyScale(msgspec.Struct, frozen=True, kw_only=True):
    labels: tuple[str, ...] = ()
    level_min: float | None = None
    level_max: float | None = None


class MechanicsSpec(msgspec.Struct, frozen=True, kw_only=True):
    id: str
    name: str
    family: str
    """Coarse mechanics family: ``drum``, ``lane``, ``plane``, ``ring``, ``grid``."""
    space: InputSpace
    roles: dict[str, RoleSpec]
    constraints: Constraints = Constraints()
    difficulty: DifficultyScale = DifficultyScale()
    controls: tuple[str, ...] = ()
    """Control kinds this game uses (shared or namespaced)."""
    accent_modifiers: tuple[str, ...] = ()
    """Modifiers that mark an emphasised action (Taiko ``big``, maimai ``break``)."""
    extends: str | None = None
    notes: str = ""
    extra: dict[str, Any] = {}

    def role(self, name: str) -> RoleSpec | None:
        return self.roles.get(name)
