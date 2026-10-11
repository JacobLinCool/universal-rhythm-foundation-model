"""Codec discovery.

Built-in codecs are imported lazily by module path. Third-party packages add
codecs through the ``urfm.codecs`` entry-point group, whose values point at
:class:`~urfm.codecs.base.Codec` subclasses.
"""

from __future__ import annotations

import importlib
from functools import cache
from importlib.metadata import entry_points
from pathlib import Path

from urfm.codecs.base import Codec

BUILTIN = {
    "tja": "urfm.codecs.tja:TjaCodec",
    "osu": "urfm.codecs.osu:OsuCodec",
    "stepmania": "urfm.codecs.stepmania:StepManiaCodec",
    "simai": "urfm.codecs.simai:SimaiCodec",
    "urfm": "urfm.codecs.native:UrfmJsonCodec",
}


def _load(target: str) -> Codec:
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)()


_failures: dict[str, Exception] = {}


@cache
def codecs() -> dict[str, Codec]:
    """All codecs that load; a broken codec is recorded in ``failures()`` instead of raising."""
    targets = dict(BUILTIN)
    targets.update({ep.name: ep.value for ep in entry_points(group="urfm.codecs")})
    found: dict[str, Codec] = {}
    for name, target in targets.items():
        try:
            found[name] = _load(target)
        except Exception as exc:
            _failures[name] = exc
    return found


def failures() -> dict[str, Exception]:
    codecs()
    return dict(_failures)


def get(codec_id: str) -> Codec:
    loaded = codecs()
    if codec_id in loaded:
        return loaded[codec_id]
    if codec_id in _failures:
        raise RuntimeError(f"codec {codec_id!r} failed to load") from _failures[codec_id]
    raise KeyError(f"unknown codec {codec_id!r}; known: {sorted(loaded)}")


def for_path(path: str | Path) -> Codec:
    path = Path(path)
    with path.open("rb") as fh:
        head = fh.read(4096)
    for codec in codecs().values():
        if codec.sniff(path, head):
            return codec
    raise KeyError(f"no codec recognises {path}")
