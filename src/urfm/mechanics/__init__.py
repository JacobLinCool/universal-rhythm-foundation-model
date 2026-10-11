"""Declarative game mechanics specifications."""

from urfm.mechanics.registry import all_specs, get, load_spec, register, register_directory
from urfm.mechanics.spec import (
    Constraints,
    DifficultyScale,
    InputSpace,
    MechanicsSpec,
    RoleSpec,
    SpaceKind,
)

__all__ = [
    "Constraints",
    "DifficultyScale",
    "InputSpace",
    "MechanicsSpec",
    "RoleSpec",
    "SpaceKind",
    "all_specs",
    "get",
    "load_spec",
    "register",
    "register_directory",
]
