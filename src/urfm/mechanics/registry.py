"""Loading and lookup of mechanics specs.

Built-in specs ship as TOML files in ``urfm/mechanics/specs``. Additional spec
directories can be registered at runtime with :func:`register_directory`.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

import msgspec

from urfm.mechanics.spec import MechanicsSpec

_specs: dict[str, MechanicsSpec] = {}
_loaded_builtin = False


def load_spec(source: str | bytes | Path) -> MechanicsSpec:
    if isinstance(source, Path):
        source = source.read_bytes()
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return msgspec.convert(tomllib.loads(source), MechanicsSpec)


def register(spec: MechanicsSpec) -> None:
    _specs[spec.id] = spec


def register_directory(directory: Path) -> None:
    for path in sorted(directory.glob("*.toml")):
        register(load_spec(path))


def _ensure_builtin() -> None:
    global _loaded_builtin
    if _loaded_builtin:
        return
    _loaded_builtin = True
    for entry in sorted(resources.files("urfm.mechanics.specs").iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".toml"):
            register(load_spec(entry.read_text("utf-8")))


def get(spec_id: str) -> MechanicsSpec:
    _ensure_builtin()
    try:
        return _specs[spec_id]
    except KeyError:
        raise KeyError(f"unknown mechanics spec {spec_id!r}; known: {sorted(_specs)}") from None


def all_specs() -> dict[str, MechanicsSpec]:
    _ensure_builtin()
    return dict(_specs)
