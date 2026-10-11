"""Multi-scale, cross-dimensional variation penalties (a reward hypothesis, not yet validated
against players).

A chart is cut into **units** of ``k`` beats of its own grid, ``k`` in {1, 2, 4, 8} chosen so a
unit lasts about two seconds (so half- or double-tempo notation gives the same units). Each
unit has two patterns:

* time ``M_g``: the set of onset positions in the unit (1/48 beat);
* space ``M_a``: the sequence of actions (don, ka, big, roll...).

Distances are in ``[0, 1]``: Jaccard distance of position sets, normalised edit distance of
action sequences. Units with fewer than two onsets are rests: they add no debt.

* **Short-term variation** ``V_x(b)``: diversity inside the last ``short_units`` units --
  normalised Gini-Simpson index of interval classes (time) or actions (space, normalised by
  Taiko's two colours, so don-ka alternation is 1 and all-don is 0). Either dimension may supply
  it: ``V_any = 1 - (1 - V_a)(1 - V_g)``, ``P_short = [tau_s - V_any]_+^2``.
* **Novelty** ``U_x(b)``: distance to the most similar of the previous ``memory_units`` units,
  so an ABAB alternation varies locally (``V`` high) but brings nothing new (``U`` ~ 0).
* **Stagnation debt** per dimension (calibrated form, ``trace``):
  ``D_x(b) = exp(-dt/H) D_x(b-1) + dt (1 - V_x(b)) (1 - U_x(b))`` (seconds): it grows only
  while the recent window is plain *and* repeats recent units. A first form without the
  ``(1 - V_x)`` factor was blind once calibrated: human charts reuse rich bar patterns so often
  that their unit-repetition debt reaches 31-43 s. (The uncalibrated ``variation`` keeps that
  first form for comparison.)
* ``P_long = sum_x [D_x / Q_x - 1]_+^2`` with ``Q_x = Q0 (1 + alpha V_x + beta V_other +
  kappa U_other)``; the compensation coefficients start at zero and are to be estimated from
  human charts.

Chart score: time-weighted means of the penalties (``score = (1/T) sum dt (lam_s P_short +
lam_l P_long)``), plus the components for diagnosis.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cache

import numpy as np

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.views import active_events


@dataclass(frozen=True)
class VariationParams:
    target_unit_s: float = 2.0
    short_units: int = 4
    memory_units: int = 8
    tau_s: float = 0.3
    horizon_s: float = 60.0
    q0_time: float = 20.0
    q0_space: float = 20.0
    alpha: float = 1.0
    beta: float = 0.0
    kappa: float = 0.0
    lam_s: float = 1.0
    lam_l: float = 1.0


@dataclass
class Unit:
    start_s: float
    duration_s: float
    positions: frozenset[int]
    actions: tuple[str, ...]
    intervals: tuple[int, ...] = ()
    """Interval classes (1/48 beat) between consecutive onsets inside the unit."""

    @property
    def rest(self) -> bool:
        return len(self.actions) < 2


def units(chartset: ChartSet, chart: Chart, target_s: float = 2.0) -> list[Unit]:
    timing = chartset.timing_of(chart)
    events = sorted(active_events(chart), key=lambda e: e.time)
    if not events:
        return []
    beats = [e.time.beat for e in events]
    first, last = math.floor(beats[0]), math.ceil(beats[-1]) + 1
    beat_s = (
        float(np.median(np.diff(timing.us_array([Fraction(b) for b in range(first, last + 1)]))))
        / 1e6
    )
    k = min((1, 2, 4, 8), key=lambda n: abs(n * beat_s - target_s))
    start = first - first % k
    out: list[Unit] = []
    i = 0
    b = start
    while b <= beats[-1]:
        end = b + k
        pos, acts = set(), []
        while i < len(events) and events[i].time.beat < end:
            e = events[i]
            p = round((e.time.beat - b) * 48)
            if p not in pos:
                pos.add(p)
                acts.append(e.role.split(":")[-1] + ("+" if e.modifiers else ""))
            i += 1
        t0 = float(timing.beat_to_us_exact(Fraction(b))) / 1e6
        t1 = float(timing.beat_to_us_exact(Fraction(end))) / 1e6
        ordered = sorted(pos)
        gaps = tuple(b2 - a2 for a2, b2 in itertools.pairwise(ordered))
        out.append(Unit(t0, max(t1 - t0, 1e-3), frozenset(pos), tuple(acts), gaps))
        b = end
    return out


def d_time(a: Unit, b: Unit) -> float:
    union = a.positions | b.positions
    return 1.0 - len(a.positions & b.positions) / len(union) if union else 0.0


def d_space(a: Unit, b: Unit) -> float:
    x, y = a.actions, b.actions
    if not x and not y:
        return 0.0
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1] / max(len(x), len(y))


DIST = {"time": d_time, "space": d_space}
SPACE_KINDS = 2
"""Normalisation of spatial diversity: Taiko has two colours (big notes and rolls are rare)."""


def diversity(items: list, kinds: int) -> float:
    """Normalised Gini-Simpson index: 0 = one symbol only, 1 = as even as ``kinds`` allows."""
    if len(items) < 2:
        return 0.0
    _, counts = np.unique(np.asarray([str(x) for x in items]), return_counts=True)
    probs = counts / counts.sum()
    gini = 1.0 - float((probs**2).sum())
    k = min(kinds, len(items))
    return min(gini / (1 - 1 / k), 1.0) if k > 1 else 0.0


def _vbin(v: float) -> int:
    return 0 if v < 1 / 3 else 1 if v < 2 / 3 else 2


class Calibration:
    """Thresholds estimated from human charts.

    ``q[x][(density bin, own V bin, other V bin)]``: 95th percentile of the human debt ``D_x``
    in that condition (density = notes per second over the short window, log2 bins of 0.5;
    V bins thirds). ``tau[density bin]``: 5th percentile of human ``V_any``. Missing cells fall
    back to coarser conditions (density and own V, then density).
    """

    def __init__(self, table: dict) -> None:
        self.table = table

    @staticmethod
    def dbin(d: float) -> int:
        return int(np.floor(np.log2(max(d, 1e-3)) / 0.5))

    @classmethod
    def build(cls, records: list[dict], q: float = 0.95, min_count: int = 50) -> Calibration:
        table: dict = {"q": {"time": {}, "space": {}}, "tau": {}}
        for x, other in (("time", "space"), ("space", "time")):
            cells: dict[str, list[float]] = {}
            for r in records:
                db, vo, vt = cls.dbin(r["density"]), _vbin(r[f"v_{x}"]), _vbin(r[f"v_{other}"])
                for key in (f"{db},{vo},{vt}", f"{db},{vo}", f"{db}"):
                    cells.setdefault(key, []).append(r[f"d_{x}"])
            table["q"][x] = {
                k: float(np.quantile(v, q)) for k, v in cells.items() if len(v) >= min_count
            }
        taus: dict[str, list[float]] = {}
        for r in records:
            taus.setdefault(str(cls.dbin(r["density"])), []).append(r["v_any"])
        table["tau"] = {
            k: float(np.quantile(v, 1 - q)) for k, v in taus.items() if len(v) >= min_count
        }
        return cls(table)

    def q(self, x: str, density: float, v_own: float, v_other: float) -> float:
        cells = self.table["q"][x]
        db = self.dbin(density)
        for key in (f"{db},{_vbin(v_own)},{_vbin(v_other)}", f"{db},{_vbin(v_own)}", f"{db}"):
            if key in cells:
                return cells[key]
        nearest = min((int(k) for k in cells if "," not in k), key=lambda b: abs(b - db))
        return cells[str(nearest)]

    def tau(self, density: float) -> float:
        bins = [int(k) for k in self.table["tau"]]
        db = min(bins, key=lambda b: abs(b - self.dbin(density)))
        return self.table["tau"][str(db)]


def trace(chartset: ChartSet, chart: Chart, p: VariationParams | None = None) -> list[dict]:
    """Per played unit: duration, local density, V, U and debt per dimension (independent of
    the thresholds, so it serves both calibration and scoring)."""
    p = p or VariationParams()
    us = units(chartset, chart, p.target_unit_s)
    debt = {"time": 0.0, "space": 0.0}
    out: list[dict] = []
    recent: list[Unit] = []
    for u in us:
        rho = math.exp(-u.duration_s / p.horizon_s)
        if u.rest:
            for x in debt:
                debt[x] *= rho
            continue
        rec: dict = {"start": u.start_s, "dt": u.duration_s}
        window = [*recent[-(p.short_units - 1) :], u] if p.short_units > 1 else [u]
        gaps = [g for w in window for g in w.intervals]
        acts = [a for w in window for a in w.actions]
        rec["v_time"], rec["v_space"] = diversity(gaps, 13), diversity(acts, SPACE_KINDS)
        rec["v_any"] = 1 - (1 - rec["v_space"]) * (1 - rec["v_time"])
        for x, dist in DIST.items():
            novelty = min((dist(u, r) for r in recent), default=1.0)
            # debt grows only while the window is plain *and* repeats recent units: humans
            # reuse rich bar patterns all the time, which is not what makes a chart dull
            debt[x] = rho * debt[x] + u.duration_s * (1.0 - rec[f"v_{x}"]) * (1.0 - novelty)
            rec[f"u_{x}"], rec[f"d_{x}"] = novelty, debt[x]
        span = sum(w.duration_s for w in window)
        rec["density"] = len(acts) / max(span, 1e-3)
        out.append(rec)
        recent = [*recent, u][-p.memory_units :]
    return out


def calibrated_variation(chartset: ChartSet, chart: Chart, cal: Calibration) -> dict:
    """Penalties with human-calibrated thresholds: ``P_short = [tau(d) - V_any]_+^2``,
    ``P_long = sum_x [D_x / Q_x(d, V_x, V_other) - 1]_+^2``; time-weighted means."""
    rows = trace(chartset, chart)
    if len(rows) < 4:
        return {"score": None}
    total = s_short = 0.0
    s_long = {"time": 0.0, "space": 0.0}
    first = None
    t0 = rows[0]["start"]
    for r in rows:
        ps = max(0.0, cal.tau(r["density"]) - r["v_any"]) ** 2
        pl = {}
        for x, other in (("time", "space"), ("space", "time")):
            q = cal.q(x, r["density"], r[f"v_{x}"], r[f"v_{other}"])
            pl[x] = max(0.0, r[f"d_{x}"] / max(q, 1e-3) - 1) ** 2
            s_long[x] += r["dt"] * pl[x]
        s_short += r["dt"] * ps
        if first is None and (ps > 0 or any(pl.values())):
            first = r["start"] + r["dt"] - t0
        total += r["dt"]
    return {
        "score": (s_short + s_long["time"] + s_long["space"]) / total,
        "p_short": s_short / total,
        "p_long_time": s_long["time"] / total,
        "p_long_space": s_long["space"] / total,
        "first_penalty_s": first,
    }


@cache
def default_calibration() -> Calibration:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "data" / "variation_calibration.json"
    return Calibration(json.loads(path.read_text()))


def variation(chartset: ChartSet, chart: Chart, p: VariationParams | None = None) -> dict:
    """Uncalibrated (fixed-threshold) penalties, see the module docstring."""
    p = p or VariationParams()
    us = units(chartset, chart, p.target_unit_s)
    played = [u for u in us if not u.rest]
    if len(played) < 4:
        return {"score": None}
    debt = {"time": 0.0, "space": 0.0}
    hist: dict[str, list[float]] = {"time": [], "space": []}  # consecutive-unit distances
    nov: dict[str, list[float]] = {"time": [], "space": []}
    total_t = s_short = s_long = 0.0
    s_long_dim = {"time": 0.0, "space": 0.0}
    max_debt = {"time": 0.0, "space": 0.0}
    prev: Unit | None = None
    recent: list[Unit] = []
    first_penalty: float | None = None
    t0 = us[0].start_s
    for u in us:
        rho = math.exp(-u.duration_s / p.horizon_s)
        if u.rest:
            for x in debt:
                debt[x] *= rho
            continue
        for x, dist in DIST.items():
            if prev is not None:
                hist[x].append(dist(prev, u))
            novelty = min((dist(u, r) for r in recent), default=1.0)
            nov[x].append(novelty)
            debt[x] = rho * debt[x] + u.duration_s * (1.0 - novelty)
            max_debt[x] = max(max_debt[x], debt[x])
        window = [*recent[-(p.short_units - 1) :], u] if p.short_units > 1 else [u]
        gaps = [g for w in window for g in w.intervals]
        acts = [a for w in window for a in w.actions]
        v = {"time": diversity(gaps, 13), "space": diversity(acts, SPACE_KINDS)}
        u_recent = {x: float(np.mean(nov[x][-p.short_units :])) for x in DIST}
        v_any = 1 - (1 - v["space"]) * (1 - v["time"])
        p_short = max(0.0, p.tau_s - v_any) ** 2
        p_long = 0.0
        for x, other, q0 in (("time", "space", p.q0_time), ("space", "time", p.q0_space)):
            q = q0 * (1 + p.alpha * v[x] + p.beta * v[other] + p.kappa * u_recent[other])
            term = max(0.0, debt[x] / q - 1) ** 2
            p_long += term
            s_long_dim[x] += u.duration_s * term
        if first_penalty is None and (p_short > 0 or p_long > 0):
            first_penalty = u.start_s + u.duration_s - t0
        s_short += u.duration_s * p_short
        s_long += u.duration_s * p_long
        total_t += u.duration_s
        prev = u
        recent = [*recent, u][-p.memory_units :]
    return {
        "score": (p.lam_s * s_short + p.lam_l * s_long) / total_t,
        "p_short": s_short / total_t,
        "p_long": s_long / total_t,
        "p_long_time": s_long_dim["time"] / total_t,
        "p_long_space": s_long_dim["space"] / total_t,
        "max_debt_time_s": max_debt["time"],
        "max_debt_space_s": max_debt["space"],
        "units": len(played),
        "first_penalty_s": first_penalty,
    }


# -- time-based stagnation (grace period, then a ramp to the human limit) ---------------------

STAG_KINDS = ("time", "space", "any")


@dataclass(frozen=True)
class StagnationParams:
    payback: float = 2.0
    """Seconds of debt repaid per second of variety (a single change repays little)."""
    start: tuple[tuple[str, float], ...] = (("time", 10.0), ("space", 10.0), ("any", 5.0))
    """Grace period (s) before any penalty."""
    limit: tuple[tuple[str, float], ...] = (("time", 20.0), ("space", 20.0), ("any", 10.0))
    """Debt (s) at which the penalty reaches 1 (overridden by the human calibration)."""


class StagnationLimits:
    """Human limits by density: 95th percentile of the peak debt of each stagnation episode
    (a maximal run of positive debt) in Taiko-1000 training charts, density = median notes per
    second of the episode, log2 bins of 0.5."""

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self.table = table

    @classmethod
    def build(cls, episodes: list[tuple[str, float, float]], q: float = 0.95, min_count=30):
        by: dict[str, dict[int, list[float]]] = {k: {} for k in STAG_KINDS}
        for kind, peak, d in episodes:
            by[kind].setdefault(Calibration.dbin(d), []).append(peak)
        return cls(
            {
                k: {
                    str(b): float(np.quantile(v, q))
                    for b, v in sorted(bins.items())
                    if len(v) >= min_count
                }
                for k, bins in by.items()
            }
        )

    def limit(self, kind: str, density: float) -> float:
        cells = self.table[kind]
        db = Calibration.dbin(density)
        nearest = min((int(b) for b in cells), key=lambda b: abs(b - db))
        return cells[str(nearest)]


def stagnation_trace(chartset: ChartSet, chart: Chart, sp: StagnationParams | None = None):
    """Per played unit: the three debts after the unit (seconds) and the local density."""
    sp = sp or StagnationParams()
    debt = dict.fromkeys(STAG_KINDS, 0.0)
    out = []
    for r in trace(chartset, chart):
        s = {x: (1 - r[f"v_{x}"]) * (1 - r[f"u_{x}"]) for x in ("time", "space")}
        s["any"] = s["time"] * s["space"]
        for k in STAG_KINDS:
            debt[k] = max(0.0, debt[k] + r["dt"] * (s[k] - sp.payback * (1 - s[k])))
        out.append({"start": r["start"], "dt": r["dt"], "density": r["density"], **debt})
    return out


def stagnation_episodes(rows: list[dict]) -> list[tuple[str, float, float]]:
    """``(kind, peak debt, median density)`` for every run of positive debt."""
    out = []
    for k in STAG_KINDS:
        run: list[dict] = []
        for r in [*rows, None]:
            if r is not None and r[k] > 0:
                run.append(r)
            elif run:
                out.append(
                    (k, max(x[k] for x in run), float(np.median([x["density"] for x in run])))
                )
                run = []
    return out


def stagnation_penalty(
    chartset: ChartSet,
    chart: Chart,
    limits: StagnationLimits | None = None,
    sp: StagnationParams | None = None,
) -> dict:
    """``p_k = ([D_k - start_k]_+ / (limit_k - start_k))^2`` per unit, time-weighted mean of the
    sum over kinds; also the first time any penalty appears and the peak debts."""
    sp = sp or StagnationParams()
    start, fixed = dict(sp.start), dict(sp.limit)
    rows = stagnation_trace(chartset, chart, sp)
    if len(rows) < 4:
        return {"score": None}
    total = 0.0
    acc = dict.fromkeys(STAG_KINDS, 0.0)
    first = None
    t0 = rows[0]["start"]
    for r in rows:
        any_pen = False
        for k in STAG_KINDS:
            # human calibration may only loosen the design limit, never tighten it
            lim = max(fixed[k], limits.limit(k, r["density"])) if limits else fixed[k]
            p = (max(0.0, r[k] - start[k]) / (lim - start[k])) ** 2
            acc[k] += r["dt"] * p
            any_pen |= p > 0
        if any_pen and first is None:
            first = r["start"] + r["dt"] - t0
        total += r["dt"]
    return {
        "score": sum(acc.values()) / total,
        **{f"p_{k}": acc[k] / total for k in STAG_KINDS},
        **{f"peak_{k}_s": max(r[k] for r in rows) for k in STAG_KINDS},
        "first_penalty_s": first,
    }


@cache
def default_stagnation_limits() -> StagnationLimits | None:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "data" / "variation_calibration.json"
    data = json.loads(path.read_text())
    return StagnationLimits(data["stagnation"]) if "stagnation" in data else None


def window_stagnation(
    chartset: ChartSet, chart: Chart, from_s: float, sp: StagnationParams | None = None
) -> float:
    """Mean over the units starting at or after ``from_s`` of the summed stagnation penalty
    (debt carried in from everything before): the RL guardrail for one generated window."""
    sp = sp or StagnationParams()
    start, limit = dict(sp.start), dict(sp.limit)
    rows = [r for r in stagnation_trace(chartset, chart, sp) if r["start"] >= from_s - 1e-6]
    if not rows:
        return 0.0
    total = sum(r["dt"] for r in rows)
    acc = sum(
        r["dt"] * (max(0.0, r[k] - start[k]) / (limit[k] - start[k])) ** 2
        for r in rows
        for k in STAG_KINDS
    )
    return acc / max(total, 1e-6)
