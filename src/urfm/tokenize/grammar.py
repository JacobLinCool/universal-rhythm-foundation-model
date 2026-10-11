"""Token grammar as a small state machine with cached vocabulary masks.

The decoder body after ``BOS`` is ``[REF SBEAT group* REFEND] group* SEP plan group* EOS``:
an optional structure-memory block (an earlier section's events and the window section it
resembles), context events, separator, the window's plan and the window events. The plan is
either ``WDENS [GRID REST]`` (R1-R5: onset density, then optionally the finest subdivision and
the share of rests) or zero or more section plans ``SBEAT SDENS SFINE SREST SKA SSUB`` (R6).
One event group is::

    (POS_BEAT POS_SUB | POS_SAME) PRIM [DUR_BEAT DUR_SUB] [ACCENT] ROLE
        [MOD...] [TARGET] [COUNT] [WAIT] [SHAPE point+ ...]

or a presentation control group ``POS_BEAT POS_SUB CTRL_<x> [SCROLL | METER_NUM METER_DEN]``
(kinds allowed by the spec's ``controls``; never after ``POS_SAME``),

where everything after ``ROLE`` is the role's *plan*, derived from the
mechanics spec: allowed non-accent modifiers (sorted, each at most once), the
target kind (``LANE``/``RING``/``REGION`` or ``PT_X PT_Y``), an optional
``COUNT`` for repeat/path roles, an optional wait (``DUR_BEAT DUR_SUB``) for
ring slides, and path segments for path roles. A group ends implicitly when
the next token starts a new group (or is ``SEP``/``EOS``).

States are small hashable tuples; :meth:`Grammar.mask` caches one boolean
vector per state, so incremental generation costs one dict lookup per step.
"""

from __future__ import annotations

from functools import cache
from typing import Any, NamedTuple

import numpy as np

from urfm.ir.events import Primitive
from urfm.mechanics import registry
from urfm.mechanics.spec import MechanicsSpec
from urfm.tokenize.vocab import (
    MAX_LANES,
    MAX_PATH_POINTS,
    SECTION_EXTRA_KINDS,
    SECTION_PLAN_KINDS,
    SHAPES_BY_SPACE,
    SUB_GRID,
    TUPLET_FREE,
    Vocab,
)

State = tuple[Any, ...]
START, END = "start", "end"
MODS, TARGET, COUNT, WAIT, PATH = range(5)


class Plan(NamedTuple):
    mods: tuple[str, ...]
    targets: tuple[str, ...]
    has_count: bool
    wait: bool
    path: bool


class Grammar:
    def __init__(self, vocab: Vocab, spec: MechanicsSpec, lanes: int | None = None) -> None:
        self.vocab = vocab
        self.spec = spec
        self.space = spec.space.kind.value
        self.lanes = spec.space.lanes or lanes or MAX_LANES
        self.accent = spec.accent_modifiers[0] if spec.accent_modifiers else None
        v = vocab
        self.k = v.kinds
        self.val = v.values
        self.prims = {p for r in spec.roles.values() if r.interactive for p in r.primitives}
        self.roles = {name: r for name, r in spec.roles.items() if r.interactive}
        self._masks: dict[State, np.ndarray] = {}
        self._plans: dict[tuple[str, str], Plan] = {}
        self._sets: dict[State, frozenset[int]] = {}
        # games that allow one action at a time (Taiko) never get a simultaneous group
        controls = set(spec.controls)
        ctrl: set[str] = set()
        if controls & {"scroll", "osu:sv"}:
            ctrl.add("scroll")
        if "highlight" in controls:
            ctrl |= {"highlight_on", "highlight_off"}
        if "barline" in controls:  # bar lines follow the meter, so meters ride along
            ctrl |= {"barline_on", "barline_off", "meter"}
        self.control_names = frozenset(ctrl)
        self.control_ids = frozenset(v.id("CTRL", c) for c in ctrl)
        single = spec.constraints.max_simultaneous == 1
        self.group_start = frozenset(v.by_kind["POS_BEAT"]) | (
            frozenset() if single else {v.id("POS_SAME")}
        )
        subs = v.by_kind["POS_SUB"]
        if spec.id in TUPLET_FREE:  # only the 1/48 and 1/32 grids
            subs = [t for t in subs if 96 % SUB_GRID[self.val[t]].denominator == 0]
        self.sub_ids = frozenset(subs)
        self.dur_sub_ids = frozenset(v.id("DUR_SUB", self.val[t]) for t in subs)
        shapes = SHAPES_BY_SPACE.get(self.space, ())
        self.shape_ids = frozenset(v.id("SHAPE", s) for s in shapes)
        self.path_point_ids = (
            frozenset(v.by_kind["PT_X"])
            if self.space == "plane"
            else frozenset(v.id("RING", i) for i in range(spec.space.buttons or 8))
        )

    # -- plans -----------------------------------------------------------------

    def plan(self, role: str, prim: str) -> Plan:
        key = (role, prim)
        cached = self._plans.get(key)
        if cached is None:
            cached = self._plans[key] = self._make_plan(role, prim)
        return cached

    def _make_plan(self, role: str, prim: str) -> Plan:
        spec = self.roles[role]
        mods = tuple(sorted(m for m in spec.modifiers if m != self.accent))
        targets = tuple(spec.targets)
        return Plan(
            mods=mods,
            targets=targets,
            has_count=prim in (Primitive.REPEAT, Primitive.PATH),
            wait=prim == Primitive.PATH and "ring" in targets,
            path=prim == Primitive.PATH,
        )

    def roles_for(self, prim: str, accent: bool) -> list[str]:
        return [
            name
            for name, r in sorted(self.roles.items())
            if prim in r.primitives and (not accent or self.accent in r.modifiers)
        ]

    # -- token sets per state ----------------------------------------------------

    def _start_tokens(self, sep) -> frozenset[int]:
        """``sep``: False (context), True (window body) or ``"ref"`` (structure memory)."""
        v = self.vocab
        if sep == "ref":
            extra = {v.id("REFEND")}
        else:
            extra = {v.id("EOS")} if sep else {v.id("SEP"), v.id("REF")}
        return self.group_start | extra

    def _target_tokens(self, tags: tuple[str, ...]) -> frozenset[int]:
        v, out = self.vocab, set()
        for tag in tags:
            if tag == "lane":
                out |= {v.id("LANE", i) for i in range(min(self.lanes, MAX_LANES))}
            elif tag == "ring":
                out |= {v.id("RING", i) for i in range(self.spec.space.buttons or 8)}
            elif tag == "region":
                out |= {v.id("REGION", g) for g in self.spec.space.regions}
            elif tag == "point":
                out |= set(v.by_kind["PT_X"])
        return frozenset(out)

    def _phase(self, plan: Plan, phase: int, local) -> tuple[frozenset[int], bool]:
        """Tokens accepted by ``phase`` at progress ``local`` and whether it may be skipped."""
        v = self.vocab
        if phase == MODS:
            ids = {v.id("MOD", m) for i, m in enumerate(plan.mods) if i > local}
            return frozenset(ids), True
        if phase == TARGET:
            if not plan.targets:
                return frozenset(), True
            if local == 1:
                return frozenset(v.by_kind["PT_Y"]), False
            return self._target_tokens(plan.targets), False
        if phase == COUNT:
            return (frozenset(v.by_kind["COUNT"]) if plan.has_count else frozenset()), True
        if phase == WAIT:
            if not plan.wait:
                return frozenset(), True
            if local == 1:
                return frozenset(v.by_kind["DUR_SUB"]), False
            return frozenset(v.by_kind["DUR_BEAT"]), True
        if not plan.path:
            return frozenset(), True
        if local[0] == "shape":
            return self.shape_ids, False
        _, n, half = local
        if half:
            return frozenset(v.by_kind["PT_Y"]), False
        ids = set(self.shape_ids) if n else set()
        if n < MAX_PATH_POINTS:
            ids |= self.path_point_ids
        return frozenset(ids), n > 0

    def tokens(self, state: State) -> frozenset[int]:
        cached = self._sets.get(state)
        if cached is not None:
            return cached
        v, kind = self.vocab, state[0]
        if kind == START:
            out = self._start_tokens(state[1])
        elif kind == END:
            out = frozenset({v.id("PAD")})
        elif kind == "wdens":  # either the window plan or section plans
            out = (
                frozenset(v.by_kind["WDENS"])
                | frozenset(v.by_kind["SBEAT"])
                | self._start_tokens(True)
            )
        elif kind == "plan":
            out = frozenset(v.by_kind["SBEAT"]) | self._start_tokens(True)
        elif kind == "splus":  # optional R6.1 extension after SSUB
            out = frozenset(v.by_kind["SPEAK"]) | frozenset(v.by_kind["SBEAT"])
            out = out | self._start_tokens(True)
        elif kind in _PLAN_NEXT:
            out = frozenset(v.by_kind[kind.upper()])
        elif kind == "refpos":
            out = frozenset(v.by_kind["SBEAT"])
        elif kind == "grid":  # optional texture plan (models predating it go straight on)
            out = frozenset(v.by_kind["GRID"]) | self._start_tokens(True)
        elif kind == "rest":
            out = frozenset(v.by_kind["REST"])
        elif kind == "sub":
            out = self.sub_ids
        elif kind == "prim":  # after an explicit position: an event or a control group
            out = frozenset(v.id("PRIM", p.value) for p in self.prims) | self.control_ids
        elif kind == "prim_same":  # after POS_SAME: events only
            out = frozenset(v.id("PRIM", p.value) for p in self.prims)
        elif kind == "scroll_val":
            out = frozenset(v.by_kind["SCROLL"])
        elif kind == "meter_num":
            out = frozenset(v.by_kind["METER_NUM"])
        elif kind == "meter_den":
            out = frozenset(v.by_kind["METER_DEN"])
        elif kind == "dur":
            out = frozenset(v.by_kind["DUR_BEAT"])
        elif kind == "dur_sub":
            out = self.dur_sub_ids
        elif kind == "acc":
            prim = state[2]
            ids = {v.id("ROLE", r) for r in self.roles_for(prim, False)}
            if self.accent and self.roles_for(prim, True):
                ids.add(v.id("ACCENT"))
            out = frozenset(ids)
        elif kind == "role":
            out = frozenset(v.id("ROLE", r) for r in self.roles_for(state[2], True))
        else:  # ("r", sep, prim, role, phase, local)
            _, sep, prim, role, phase, local = state
            plan = self.plan(role, prim)
            ids: set[int] = set()
            ended = True
            for j in range(phase, PATH + 1):
                toks, skippable = self._phase(plan, j, local if j == phase else _initial(j))
                ids |= toks
                if not skippable:
                    ended = False
                    break
            if ended:
                ids |= self._start_tokens(sep)
            out = frozenset(ids)
        self._sets[state] = out
        return out

    def mask(self, state: State) -> np.ndarray:
        cached = self._masks.get(state)
        if cached is None:
            cached = np.zeros(len(self.vocab), dtype=bool)
            cached[list(self.tokens(state))] = True
            self._masks[state] = cached
        return cached

    # -- transitions ---------------------------------------------------------------

    def initial(self) -> State:
        return (START, False)

    def step(self, state: State, token: int) -> State:
        if token not in self.tokens(state):
            raise ValueError(f"token {self.vocab.name(token)} not allowed in state {state}")
        kind, value = self.k[token], self.val[token]
        sep = state[1]
        if kind == "POS_BEAT":
            return ("sub", sep)
        if kind == "POS_SUB":
            return ("prim", sep)
        if kind == "POS_SAME":
            return ("prim_same", sep)
        if kind == "SEP":
            return ("wdens", True)
        if kind == "REF":
            return ("refpos", "ref")
        if kind == "REFEND":
            return (START, False)
        if kind == "SBEAT":
            return (START, "ref") if state[0] == "refpos" else ("sdens", True)
        if kind in SECTION_PLAN_KINDS or kind in SECTION_EXTRA_KINDS:
            return (_PLAN_NEXT[kind.lower()], True)
        if kind == "WDENS":
            return ("grid", True)
        if kind == "GRID":
            return ("rest", True)
        if kind == "REST":
            return (START, True)
        if kind == "EOS":
            return (END, True)
        if kind == "PAD":
            return state
        if kind == "CTRL":
            if value == "scroll":
                return ("scroll_val", sep)
            if value == "meter":
                return ("meter_num", sep)
            return (START, sep)
        if kind == "METER_NUM":
            return ("meter_den", sep)
        if kind in ("SCROLL", "METER_DEN"):
            return (START, sep)
        if kind == "PRIM":
            return ("acc", sep, value) if value == Primitive.HIT else ("dur", sep, value)
        if state[0] == "dur":
            return ("dur_sub", sep, state[2])
        if state[0] == "dur_sub":
            return ("acc", sep, state[2])
        if kind == "ACCENT":
            return ("role", sep, state[2])
        if kind == "ROLE":
            return ("r", sep, state[2], value, MODS, -1)
        _, _, prim, role, phase, local = state
        plan = self.plan(role, prim)
        for j in range(phase, PATH + 1):
            loc = local if j == phase else _initial(j)
            toks, skippable = self._phase(plan, j, loc)
            if token in toks:
                return ("r", sep, prim, role, *_advance(plan, j, loc, kind, value))
            if not skippable:
                break
        raise AssertionError("unreachable: token accepted by tokens() but no phase")

    def run(self, tokens: list[int]) -> State:
        state = self.initial()
        for t in tokens:
            state = self.step(state, t)
        return state


_PLAN_NEXT = {
    "sdens": "sfine",
    "sfine": "srest",
    "srest": "ska",
    "ska": "ssub",
    "ssub": "splus",
    "speak": "srep",
    "srep": "svar",
    "svar": "ssw",
    "ssw": "slong",
    "slong": "plan",
}
"""Section plan states: each accepts its own token kind, then moves to the next state."""


def _initial(phase: int):
    return {MODS: -1, TARGET: 0, COUNT: 0, WAIT: 0, PATH: ("shape",)}[phase]


def _advance(plan: Plan, phase: int, local, kind: str, value) -> tuple[int, object]:
    if phase == MODS:
        return MODS, plan.mods.index(value)
    if phase == TARGET:
        return (TARGET, 1) if kind == "PT_X" else (COUNT, 0)
    if phase == COUNT:
        return WAIT, 0
    if phase == WAIT:
        return (WAIT, 1) if kind == "DUR_BEAT" else (PATH, ("shape",))
    if kind == "SHAPE":
        return PATH, ("pts", 0, False)
    _, n, _half = local
    if kind == "PT_X":
        return PATH, ("pts", n, True)
    return PATH, ("pts", n + 1, False)


@cache
def grammar_for(vocab: Vocab, mechanics: str, lanes: int | None = None) -> Grammar:
    return Grammar(vocab, registry.get(mechanics), lanes)
