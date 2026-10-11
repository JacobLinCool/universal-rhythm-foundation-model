"""Deterministic token vocabulary derived from the mechanics specs.

Every token has a *kind* (``POS_BEAT``, ``ROLE``, ``LANE``, ...) and a *value*.
The vocabulary is a pure function of the registered specs and the constants
below, so token ids are stable as long as specs and constants do not change;
:attr:`Vocab.digest` fingerprints it for checkpoints.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from functools import lru_cache
from typing import Any

from urfm.ir.events import Primitive
from urfm.mechanics import registry

# -- grids ---------------------------------------------------------------------

MAX_BEATS = 128
"""Positions are ``beat index (0..MAX_BEATS-1) + sub-beat``; 16 s at 400 BPM is ~107 beats."""

SUB_GRID: tuple[Fraction, ...] = tuple(
    sorted(
        {Fraction(k, 48) for k in range(48)}
        | {Fraction(k, 32) for k in range(32)}
        | {Fraction(k, 5) for k in range(5)}
        | {Fraction(k, 7) for k in range(7)}
    )
)
"""Sub-beat positions: 1/48 (192nd notes) plus 1/32 (simai pseudo-each) and 5/7-tuplets."""

COUNT_GRID: tuple[int, ...] = (
    *range(1, 33),
    *range(36, 65, 4),
    80,
    96,
    128,
    192,
    256,
)
"""Repetition counts (balloon hits, slider slides)."""

DENS_EDGES: tuple[float, ...] = tuple(0.5 * 2 ** (k / 2.5) for k in range(16))
"""Lower edges of the 16 onsets-per-second bins (0.5 .. ~32 /s)."""

PT_STEP = 8
PT_X_RANGE = (-64, 576)
PT_Y_RANGE = (-64, 448)
"""osu! playfield coordinates are quantized to an 8 px grid (max error 4 px per axis)."""

MAX_LANES = 18
MAX_RING = 8
MAX_PATH_POINTS = 8
"""Control points kept per path segment (first ``MAX_PATH_POINTS - 1`` and the last)."""

OSU_SHAPES = ("osu:bezier", "osu:linear", "osu:perfect", "osu:catmull", "osu:other")
MAIMAI_SHAPES = tuple(
    f"maimai:{op}" for op in ("pp", "qq", "-", "^", "<", ">", "v", "V", "p", "q", "s", "z", "w")
)
SHAPES_BY_SPACE = {"plane": OSU_SHAPES, "ring": MAIMAI_SHAPES}

SCROLL_COMMON = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)


def _scroll_grid() -> tuple[float, ...]:
    """Scroll multipliers 0.05 .. 32 on a 5% log grid, with common exact values pinned."""
    logs = [0.05 * 1.05**k for k in range(round(math.log(640) / math.log(1.05)) + 1)]
    kept = [x for x in logs if all(abs(x / c - 1) > 0.015 for c in SCROLL_COMMON)]
    return tuple(sorted({*(round(x, 4) for x in kept), *SCROLL_COMMON}))


SCROLL_GRID: tuple[float, ...] = _scroll_grid()
"""Scroll-speed values (TJA ``#SCROLL``, osu! SV); nearest-in-log quantization (≤2.5% error)."""

METER_NUMS: tuple[int, ...] = tuple(range(1, 65))
METER_DENS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
"""Time signatures ``num/den`` (TJA ``#MEASURE`` 1/32 .. 64/1)."""

GRID_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 48)
"""Window plan ``GRID_g``: the finest subdivision (per beat) a window's onsets need -- 4 for
16th notes, 3 or 6 for triplets, 8 for 32nds; 0 for an empty window."""
REST_BINS = 11
"""Window plan ``REST_r``: share of the window's beats in rests of at least one beat, in
tenths (0 .. 10)."""

CONTROL_TOKENS = (
    "scroll",
    "highlight_on",
    "highlight_off",
    "barline_on",
    "barline_off",
    "meter",
)
"""``CTRL_*`` values: a control group is ``POS CTRL_<x> [value tokens]``."""

SPECIALS = ("PAD", "BOS", "SEP", "EOS", "POS_SAME", "ACCENT")

SHARE_BINS = 5
"""Section plan share bins (``urfm.tokenize.sections.SHARE_EDGES``)."""
SECTION_PLAN_KINDS = ("SDENS", "SFINE", "SREST", "SKA", "SSUB")
SECTION_EXTRA_KINDS = ("SPEAK", "SREP", "SVAR", "SSW", "SLONG")
"""R6.1 extension after ``SSUB`` (optional in the grammar, so R6 models skip it)."""
"""R6 section plan: ``SBEAT_b`` (section start in whole beats after the window start; 0 for a
section that began earlier) then one token of each kind (``urfm.tokenize.sections``)."""

TUPLET_FREE = frozenset({"taiko", "osu_taiko"})
"""Games whose grammar forbids the 5- and 7-tuplet sub-beats (0.04% of Taiko-1000 onsets)."""


def nearest(grid: tuple[Any, ...], value: Any) -> int:
    """Index of the grid element closest to ``value`` (ties go to the smaller element)."""
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if grid[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and abs(grid[lo - 1] - value) <= abs(grid[lo] - value):
        return lo - 1
    return lo


PLAIN_SUBS: tuple[int, ...] = tuple(i for i, f in enumerate(SUB_GRID) if 96 % f.denominator == 0)
"""``SUB_GRID`` indices on the 1/48 and 1/32 grids (no 5- or 7-tuplets)."""
_PLAIN_GRID = tuple(SUB_GRID[i] for i in PLAIN_SUBS)


@lru_cache(maxsize=65536)
def _snap_sub(frac: Fraction, plain: bool = False) -> tuple[int, int]:
    """``(carry, SUB_GRID index)`` for a fractional beat in ``[0, 1)``; fractions repeat a
    lot (simple musical subdivisions), so this is cached. ``plain`` snaps to
    :data:`PLAIN_SUBS` only (games in ``TUPLET_FREE``)."""
    sub = PLAIN_SUBS[nearest(_PLAIN_GRID, frac)] if plain else nearest(SUB_GRID, frac)
    if abs(1 - frac) < abs(SUB_GRID[sub] - frac):
        return 1, 0
    return 0, sub


def split_beats(value: Fraction, plain: bool = False) -> tuple[int, int]:
    """``value`` (>= 0 beats) as ``(whole beats, SUB_GRID index)``, clamped to ``MAX_BEATS``."""
    whole = math.floor(value)
    carry, sub = _snap_sub(value - whole, plain)
    whole += carry
    if whole >= MAX_BEATS:
        return MAX_BEATS - 1, len(SUB_GRID) - 1
    return max(whole, 0), sub


def dens_bin(onsets_per_s: float) -> int:
    if onsets_per_s <= DENS_EDGES[0]:
        return 0
    return min(int(math.log2(onsets_per_s / 0.5) * 2.5), len(DENS_EDGES) - 1)


class Vocab:
    def __init__(self) -> None:
        specs = registry.all_specs()
        roles = sorted({r for s in specs.values() for r in s.roles})
        modifiers = sorted(
            {m for s in specs.values() for r in s.roles.values() for m in r.modifiers}
        )
        regions = sorted({g for s in specs.values() for g in s.space.regions})
        self.tokens: list[tuple[str, Any]] = []
        add = self.tokens.append
        for name in SPECIALS:
            add((name, None))
        for spec_id in sorted(specs):
            add(("GAME", spec_id))
        for n in range(1, MAX_LANES + 1):
            add(("LANES", n))
        for t in range(5):
            add(("TIER", t))
        add(("TIER", None))
        for d in range(len(DENS_EDGES)):
            add(("DENS", d))
        for d in range(len(DENS_EDGES)):
            add(("WDENS", d))
        for b in range(MAX_BEATS):
            add(("POS_BEAT", b))
        for i in range(len(SUB_GRID)):
            add(("POS_SUB", i))
        for p in Primitive:
            add(("PRIM", p.value))
        for b in range(MAX_BEATS):
            add(("DUR_BEAT", b))
        for i in range(len(SUB_GRID)):
            add(("DUR_SUB", i))
        for r in roles:
            add(("ROLE", r))
        for m in modifiers:
            add(("MOD", m))
        for i in range(MAX_LANES):
            add(("LANE", i))
        for i in range(MAX_RING):
            add(("RING", i))
        for g in regions:
            add(("REGION", g))
        for i in range(len(range(*PT_X_RANGE, PT_STEP)) + 1):
            add(("PT_X", i))
        for i in range(len(range(*PT_Y_RANGE, PT_STEP)) + 1):
            add(("PT_Y", i))
        for i in range(len(COUNT_GRID)):
            add(("COUNT", i))
        for shape in (*OSU_SHAPES, *MAIMAI_SHAPES):
            add(("SHAPE", shape))
        # presentation controls (appended last so earlier token ids stay stable)
        for name in CONTROL_TOKENS:
            add(("CTRL", name))
        for i in range(len(SCROLL_GRID)):
            add(("SCROLL", i))
        for n in METER_NUMS:
            add(("METER_NUM", n))
        for d in METER_DENS:
            add(("METER_DEN", d))
        # R4: continuous-condition slot and window texture plan
        add(("COND", None))
        for g in GRID_VALUES:
            add(("GRID", g))
        for r in range(REST_BINS):
            add(("REST", r))
        # R6: section plan and structure memory
        for b in range(MAX_BEATS):
            add(("SBEAT", b))
        for d in range(len(DENS_EDGES)):
            add(("SDENS", d))
        for kind in ("SFINE", "SREST", "SKA"):
            for i in range(SHARE_BINS):
                add((kind, i))
        for i in range(len(GRID_VALUES)):
            add(("SSUB", i))
        add(("REF", None))
        add(("REFEND", None))
        # R6.1: section plan extension
        for d in range(len(DENS_EDGES)):
            add(("SPEAK", d))
        for i in range(4):
            add(("SREP", i))
        for i in range(6):
            add(("SVAR", i))
        for kind in ("SSW", "SLONG"):
            for i in range(SHARE_BINS):
                add((kind, i))
        self.ids: dict[tuple[str, Any], int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.kinds: list[str] = [k for k, _ in self.tokens]
        self.values: list[Any] = [v for _, v in self.tokens]
        self.by_kind: dict[str, list[int]] = {}
        for i, kind in enumerate(self.kinds):
            self.by_kind.setdefault(kind, []).append(i)
        self.digest = hashlib.sha256(repr(self.tokens).encode()).hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.tokens)

    def id(self, kind: str, value: Any = None) -> int:
        return self.ids[(kind, value)]

    def name(self, token: int) -> str:
        kind, value = self.tokens[token]
        return kind if value is None else f"{kind}_{value}"


def pt_x_bin(x: float) -> int:
    return _pt_bin(x, PT_X_RANGE)


def pt_y_bin(y: float) -> int:
    return _pt_bin(y, PT_Y_RANGE)


def _pt_bin(v: float, rng: tuple[int, int]) -> int:
    n = len(range(*rng, PT_STEP))
    return max(0, min(n, round((v - rng[0]) / PT_STEP)))


def pt_x_value(i: int) -> float:
    return float(PT_X_RANGE[0] + i * PT_STEP)


def pt_y_value(i: int) -> float:
    return float(PT_Y_RANGE[0] + i * PT_STEP)


_SCROLL_LOGS = tuple(math.log(v) for v in SCROLL_GRID)


def scroll_bin(value: float) -> int:
    """Nearest SCROLL_GRID index in log space (``value`` must be > 0)."""
    return nearest(_SCROLL_LOGS, math.log(value))
