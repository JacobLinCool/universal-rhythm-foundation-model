"""Export to the ``CanonicalChart`` JSON of rhythm-rs (taiko-rs ``rhythm-chart``).

rhythm-rs is a headless rhythm engine whose chart model uses absolute
microsecond ticks. Exporting lets URFM charts -- including generated ones -- be
played and previewed directly. The export is one-way: rhythm-rs has no beat
positions, so it cannot be read back losslessly.

Supported mechanics: Taiko (``taiko``, ``osu_taiko``), lane games and maimai.
Field conventions follow ``rhythm-importer-tja`` (lanes don/kat/both, ``FLAG_BIG``,
``FLAG_BALLOON``) and ``rhythm-mode-radial``.
"""

from __future__ import annotations

import bisect
from fractions import Fraction
from typing import Any

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event, Primitive
from urfm.ir.targets import Lane, RingButton, TouchRegion
from urfm.ir.timing import US_PER_MINUTE
from urfm.mechanics import registry

FLAG_BIG = 1 << 0
FLAG_BALLOON = 1 << 1
SCROLL_SCALE = 1_000_000
LANE_DON, LANE_KAT, LANE_BOTH = 0, 1, 2
TAIKO = frozenset({"taiko", "osu_taiko"})


class ExportError(ValueError):
    pass


def to_canonical_chart(chartset: ChartSet, chart: Chart) -> dict[str, Any]:
    spec = registry.get(chart.mechanics)
    timing = chartset.timing_of(chart)

    def tick(time) -> int:
        value = timing.us(time)
        if value < 0:
            raise ExportError(
                "rhythm-rs ticks must be non-negative; trim objects before audio start"
            )
        return value

    tempo_map, last = [], None
    for change in sorted(timing.tempos, key=lambda t: t.beat):
        mpq = round(Fraction(US_PER_MINUTE) / change.bpm)
        if mpq != last:
            tempo_map.append(
                {
                    "tick": max(round(timing.beat_to_us_exact(change.beat)), 0),
                    "micros_per_quarter": mpq,
                }
            )
            last = mpq
    signatures = [
        {
            "tick": max(round(timing.beat_to_us_exact(m.beat)), 0),
            "numerator": m.numerator,
            "denominator": m.denominator,
        }
        for m in timing.meters
    ]

    if chart.mechanics in TAIKO:
        lanes = [
            {"id": LANE_DON, "name": "don", "role": "TaikoDon"},
            {"id": LANE_KAT, "name": "kat", "role": "TaikoKat"},
            {"id": LANE_BOTH, "name": "both", "role": "Generic"},
        ]
    elif spec.space.kind == "ring":
        lanes = [
            {"id": i, "name": str(i + 1), "role": "Radial"} for i in range(spec.space.buttons or 8)
        ]
    elif spec.space.kind == "lanes":
        count = spec.space.lanes or int(chart.meta.get("lanes", 0))
        lanes = [{"id": i, "name": f"lane{i}", "role": "Generic"} for i in range(count)]
    else:
        raise ExportError(f"rhythm-rs has no mode for {chart.mechanics!r}")

    route_ids = {b.id: {name: i for i, name in enumerate(b.routes)} for b in chart.branches}
    segment_ids = {b.id: i for i, b in enumerate(chart.branches)}
    scroll = _ControlLookup(chart, timing, "scroll")
    regions = {name: i for i, name in enumerate(spec.space.regions)}

    objects = []
    for ev in chart.events:
        role = spec.role(ev.role)
        if role is not None and not role.interactive:
            continue
        obj = _object(ev, chart.mechanics in TAIKO, regions)
        if obj is None:
            continue
        start = tick(ev.time)
        obj |= {
            "start_tick": start,
            "end_tick": tick(ev.end) if ev.end is not None else start,
            "branch_segment_id": None,
            "branch_route_id": 0,
        }
        value = scroll.at(ev)
        if isinstance(value, int | float) and value != 1:
            obj["scroll_scaled"] = round(value * SCROLL_SCALE)
        if ev.branch is not None:
            obj["branch_segment_id"] = segment_ids[ev.branch.segment]
            obj["branch_route_id"] = route_ids[ev.branch.segment][ev.branch.route]
        objects.append(obj)
    objects.sort(key=lambda o: o["start_tick"])
    for i, obj in enumerate(objects):
        obj["id"] = i

    events = []
    for control in chart.controls:
        if control.kind == "highlight" and control.branch is None:
            events.append(
                {"tick": tick(control.time), "kind": "GogoStart" if control.value else "GogoEnd"}
            )
        elif control.kind == "section" and isinstance(control.value, str) and control.value:
            events.append({"tick": tick(control.time), "kind": {"Marker": control.value}})
    events.sort(key=lambda e: e["tick"])

    return {
        "metadata": {
            "title": chartset.song.title,
            "subtitle": chartset.song.subtitle,
            "artist": chartset.song.artist,
            "charter": chartset.song.creator,
            "audio_path": chartset.audio.path if chartset.audio else None,
            "offset": -round(timing.offset_us),
            "difficulty_name": chart.difficulty.label,
            "difficulty_level": None
            if chart.difficulty.level is None
            else int(chart.difficulty.level),
        },
        "tempo_map": tempo_map,
        "signatures": signatures,
        "lanes": lanes,
        "branch_segments": [
            {
                "id": segment_ids[b.id],
                "default_route_id": b.routes.index(b.default),
                "route_count": len(b.routes),
                "decision_hint": None,
            }
            for b in chart.branches
        ],
        "objects": objects,
        "events": events,
    }


def _object(ev: Event, taiko: bool, regions: dict[str, int]) -> dict[str, Any] | None:
    flags = FLAG_BIG if "big" in ev.modifiers else 0
    base = {"flags": flags, "required_hits": 0, "slide_to": None}
    if taiko:
        if ev.role in ("taiko:don", "taiko:ka", "taiko:kadon"):
            lane = LANE_KAT if ev.role == "taiko:ka" else LANE_DON
            return base | {"kind": "Tap", "lane_or_region": {"Lane": lane}}
        if ev.role == "taiko:drumroll":
            return base | {"kind": "Roll", "lane_or_region": {"Lane": LANE_BOTH}}
        if ev.role == "taiko:balloon":
            return base | {
                "kind": "Roll",
                "lane_or_region": {"Lane": LANE_BOTH},
                "flags": flags | FLAG_BALLOON,
                "required_hits": ev.count or 0,
            }
        return None
    if isinstance(ev.target, Lane | RingButton):
        where = {"Lane": ev.target.index}
    elif isinstance(ev.target, TouchRegion) and ev.target.name in regions:
        return base | {"kind": "Touch", "lane_or_region": {"Region": regions[ev.target.name]}}
    else:
        return None
    if ev.primitive is Primitive.PATH:
        last = ev.path[-1].points[-1] if ev.path and ev.path[-1].points else None
        slide_to = last.index if isinstance(last, RingButton | Lane) else None
        return base | {"kind": "Slide", "lane_or_region": where, "slide_to": slide_to}
    if ev.primitive is Primitive.HOLD:
        return base | {"kind": "Hold", "lane_or_region": where}
    if ev.primitive is Primitive.REPEAT:
        return base | {"kind": "Roll", "lane_or_region": where}
    return base | {"kind": "Tap", "lane_or_region": where}


class _ControlLookup:
    """Value of the latest control of ``kind`` at an event's time, honouring branches."""

    def __init__(self, chart: Chart, timing, kind: str) -> None:
        self._by_branch: dict[Any, tuple[list[int], list[Any]]] = {}
        for control in sorted(chart.controls, key=lambda c: timing.us(c.time)):
            if control.kind != kind:
                continue
            ticks, values = self._by_branch.setdefault(control.branch, ([], []))
            ticks.append(timing.us(control.time))
            values.append(control.value)
        self._timing = timing

    def at(self, ev: Event) -> Any:
        us = self._timing.us(ev.time)
        best = None
        for branch in (None, ev.branch):
            if branch not in self._by_branch:
                continue
            ticks, values = self._by_branch[branch]
            i = bisect.bisect_right(ticks, us) - 1
            if i >= 0 and (best is None or ticks[i] >= best[0]):
                best = (ticks[i], values[i])
        return None if best is None else best[1]
