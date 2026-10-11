"""Semantic comparison of chart sets: the core of round-trip fidelity checks."""

from __future__ import annotations

from typing import Any

import msgspec

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.timing import TimingMap


class Difference(msgspec.Struct, frozen=True):
    path: str
    left: str
    right: str


def timing_key(timing: TimingMap, chart: Chart) -> list[tuple[Any, ...]]:
    """Order-independent gameplay view used for L2 (timing-exact) comparison."""
    keys = [
        (
            timing.us(ev.time),
            ev.primitive.value,
            ev.role,
            ev.modifiers,
            ev.target,
            None if ev.end is None else timing.us(ev.end),
            ev.branch,
        )
        for ev in chart.events
    ]
    return sorted(keys, key=repr)


def timing_differences(a: ChartSet, b: ChartSet, limit: int = 20) -> list[Difference]:
    out: list[Difference] = []
    if len(a.charts) != len(b.charts):
        return [Difference("charts", str(len(a.charts)), str(len(b.charts)))]
    for ci, (ca, cb) in enumerate(zip(a.charts, b.charts, strict=True)):
        ka, kb = timing_key(a.timing_of(ca), ca), timing_key(b.timing_of(cb), cb)
        if len(ka) != len(kb):
            out.append(Difference(f"charts[{ci}].events", f"{len(ka)} events", f"{len(kb)} events"))
        for ei, (x, y) in enumerate(zip(ka, kb, strict=False)):
            if x != y:
                out.append(Difference(f"charts[{ci}].events~[{ei}]", repr(x), repr(y)))
                if len(out) >= limit:
                    return out
    return out


def semantic_differences(a: Any, b: Any, path: str = "", limit: int = 20) -> list[Difference]:
    """Structural diff of two IR values, ignoring provenance."""
    out: list[Difference] = []
    _walk(a, b, path or "$", out, limit)
    return out


def _walk(a: Any, b: Any, path: str, out: list[Difference], limit: int) -> None:
    if len(out) >= limit or a == b:
        return
    if isinstance(a, msgspec.Struct) and type(a) is type(b):
        for field in a.__struct_fields__:
            if field == "provenance":
                continue
            _walk(getattr(a, field), getattr(b, field), f"{path}.{field}", out, limit)
        return
    if isinstance(a, tuple | list) and isinstance(b, tuple | list):
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _walk(x, y, f"{path}[{i}]", out, limit)
        if len(a) != len(b):
            out.append(Difference(f"{path}.length", str(len(a)), str(len(b))))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            _walk(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}", out, limit)
        return
    out.append(Difference(path, _short(a), _short(b)))


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."
