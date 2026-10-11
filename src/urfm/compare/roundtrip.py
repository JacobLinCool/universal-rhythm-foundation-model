"""``parse -> write -> parse`` fidelity measurement for codecs."""

from __future__ import annotations

import traceback
from pathlib import Path

import msgspec

from urfm.codecs.base import Codec
from urfm.compare.diff import Difference, semantic_differences, timing_differences


class RoundTripResult(msgspec.Struct, kw_only=True):
    source: str
    codec: str
    charts: int = 0
    events: int = 0
    timing_exact: bool = False
    semantic_exact: bool = False
    stable_write: bool = False
    """``write(read(write(x))) == write(x)``: the writer reached a fixed point."""
    error: str | None = None
    timing_diffs: list[Difference] = []
    semantic_diffs: list[Difference] = []


def roundtrip_bytes(codec: Codec, data: bytes, source: str = "<bytes>") -> RoundTripResult:
    result = RoundTripResult(source=source, codec=codec.id)
    try:
        first = codec.read(data, source=source)
        result.charts = len(first.charts)
        result.events = sum(len(c.events) for c in first.charts)
        written = codec.write(first)
        second = codec.read(written, source=source)
        result.stable_write = codec.write(second) == written
    except Exception as exc:
        result.error = "".join(traceback.format_exception_only(exc)).strip()
        return result
    result.timing_diffs = timing_differences(first, second)
    result.semantic_diffs = semantic_differences(first, second)
    result.timing_exact = not result.timing_diffs
    result.semantic_exact = not result.semantic_diffs
    return result


def roundtrip_path(codec: Codec, path: Path) -> RoundTripResult:
    return roundtrip_bytes(codec, path.read_bytes(), str(path))
