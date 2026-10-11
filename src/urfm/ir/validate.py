"""Structural and mechanics validation of IR chart sets."""

from __future__ import annotations

from collections import Counter

import msgspec

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.controls import SHARED_KINDS
from urfm.ir.events import Event, Primitive
from urfm.ir.targets import Lane, RingButton, TouchRegion
from urfm.mechanics import registry
from urfm.mechanics.spec import MechanicsSpec


class Issue(msgspec.Struct, frozen=True):
    chart: int | None
    event: int | None
    code: str
    message: str


def validate(chartset: ChartSet, *, mechanics: bool = True) -> list[Issue]:
    issues: list[Issue] = []
    if not chartset.timing.tempos:
        issues.append(Issue(None, None, "timing.empty", "timing map has no tempo"))
    for ci, chart in enumerate(chartset.charts):
        issues.extend(_validate_chart(ci, chart, mechanics=mechanics))
    return issues


def _validate_chart(ci: int, chart: Chart, *, mechanics: bool) -> list[Issue]:
    out: list[Issue] = []

    def add(ei: int | None, code: str, message: str) -> None:
        out.append(Issue(ci, ei, code, message))

    branch_routes = {b.id: set(b.routes) for b in chart.branches}
    if len(branch_routes) != len(chart.branches):
        add(None, "branch.duplicate", "duplicate branch segment id")
    for b in chart.branches:
        if b.default not in b.routes:
            add(None, "branch.default", f"segment {b.id}: default {b.default!r} not in routes")

    previous = None
    for ei, ev in enumerate(chart.events):
        if previous is not None and ev.time < previous:
            add(ei, "order", "events are not sorted by time")
        previous = ev.time
        if ev.primitive is Primitive.HIT and ev.end is not None:
            add(ei, "end.unexpected", "hit events have no end")
        if ev.primitive is not Primitive.HIT and ev.end is None:
            add(ei, "end.missing", f"{ev.primitive} events need an end")
        if ev.end is not None and ev.end < ev.time:
            add(ei, "end.before_start", "end precedes start")
        if ev.primitive is Primitive.PATH and not ev.path:
            add(ei, "path.missing", "path events need at least one path segment")
        if tuple(sorted(ev.modifiers)) != ev.modifiers:
            add(ei, "modifiers.order", "modifiers must be sorted")
        if ev.branch is not None:
            routes = branch_routes.get(ev.branch.segment)
            if routes is None or ev.branch.route not in routes:
                add(ei, "branch.ref", f"unknown branch reference {ev.branch}")

    for control in chart.controls:
        if ":" not in control.kind and control.kind not in SHARED_KINDS:
            add(None, "control.kind", f"{control.kind!r} is neither shared nor namespaced")
        if control.branch is not None:
            routes = branch_routes.get(control.branch.segment)
            if routes is None or control.branch.route not in routes:
                add(None, "branch.ref", f"control refers to unknown branch {control.branch}")

    if mechanics:
        try:
            spec = registry.get(chart.mechanics)
        except KeyError as exc:
            add(None, "mechanics.unknown", str(exc))
        else:
            for ei, ev in enumerate(chart.events):
                for code, message in _check_event(spec, chart, ev):
                    add(ei, code, message)
            out.extend(_check_constraints(ci, spec, chart))
    return out


def _check_event(spec: MechanicsSpec, chart: Chart, ev: Event) -> list[tuple[str, str]]:
    problems: list[tuple[str, str]] = []
    role = spec.role(ev.role)
    if role is None:
        return [("role.unknown", f"{ev.role!r} is not a role of {spec.id}")]
    if ev.primitive not in role.primitives:
        problems.append(("role.primitive", f"{ev.role} cannot be {ev.primitive}"))
    extra = set(ev.modifiers) - set(role.modifiers)
    if extra:
        problems.append(("role.modifier", f"{ev.role} does not allow {sorted(extra)}"))
    tag = None if ev.target is None else type(ev.target).__struct_config__.tag
    if role.targets and tag not in role.targets:
        problems.append(("target.kind", f"{ev.role} needs target in {role.targets}, got {tag}"))
    if not role.targets and tag is not None:
        problems.append(("target.unexpected", f"{ev.role} takes no target"))
    space = spec.space
    if isinstance(ev.target, Lane):
        lanes = space.lanes or int(chart.meta.get("lanes", 0)) or None
        if lanes is not None and not 0 <= ev.target.index < lanes:
            problems.append(("target.range", f"lane {ev.target.index} outside 0..{lanes - 1}"))
    elif isinstance(ev.target, RingButton):
        if space.buttons is not None and not 0 <= ev.target.index < space.buttons:
            problems.append(("target.range", f"button {ev.target.index} outside ring"))
    elif (
        isinstance(ev.target, TouchRegion) and space.regions and ev.target.name not in space.regions
    ):
        problems.append(("target.range", f"unknown touch region {ev.target.name!r}"))
    return problems


def _check_constraints(ci: int, spec: MechanicsSpec, chart: Chart) -> list[Issue]:
    out: list[Issue] = []
    limit = spec.constraints.max_simultaneous
    if limit is not None:
        counts = Counter((ev.time, ev.branch) for ev in chart.events)
        for (time, _), n in counts.items():
            if n > limit:
                out.append(Issue(ci, None, "constraint.simultaneous", f"{n} events at {time}"))
    return out


def enforce_single_action(chart: Chart, us_of) -> tuple[Chart, int]:
    """For games with ``max_simultaneous == 1`` (Taiko): drop events that start at the same
    time as, or while, a previous kept event is still being played (a drumroll occupies the
    drum until its end, and its end occupies a slot). Returns ``(chart, dropped)``.

    ``us_of`` maps a :class:`~urfm.ir.time.Time` to absolute microseconds.
    """
    spec = registry.get(chart.mechanics)
    if spec.constraints.max_simultaneous != 1:
        return chart, 0
    kept: list[Event] = []
    busy_until = None
    for ev in sorted(chart.events, key=lambda e: (e.time, e.role)):
        onset = us_of(ev.time)
        if busy_until is not None and onset <= busy_until:
            continue
        kept.append(ev)
        busy_until = us_of(ev.end) if ev.end is not None else onset
    dropped = len(chart.events) - len(kept)
    return msgspec.structs.replace(chart, events=tuple(kept)), dropped
