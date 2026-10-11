"""Section plans as an editable table (readable units in, plan bins out)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from urfm.ir.timing import TimingMap
from urfm.tokenize.sections import REP_BINS, SHARE_EDGES, VAR_BINS, SectionPlan, share_bin
from urfm.tokenize.vocab import DENS_EDGES, GRID_VALUES, dens_bin

COLUMNS = (
    "section",
    "start (s)",
    "notes/s",
    "peak notes/s",
    "fine %",
    "rest %",
    "ka %",
    "colour switch %",
    "long notes %",
    "finest grid (per beat)",
    "repeat (0 new … 3 repeat)",
    "rhythm variety (0–5)",
)
_SHARE_MID = (0.025, 0.125, 0.3, 0.55, 0.85)
assert len(_SHARE_MID) == len(SHARE_EDGES) + 1


def rate_of(b: int) -> float:
    """Typical notes per second of a density bin."""
    return 0.0 if b == 0 else 0.5 * 2 ** ((b + 0.5) / 2.5)


def to_rows(plans: dict[int, SectionPlan], timing: TimingMap, k: int) -> list[list]:
    rows = []
    for i in sorted(plans):
        p = plans[i]
        start = float(timing.beat_to_us_exact(Fraction(i * k))) / 1e6
        if start < 0:
            continue
        rows.append(
            [
                i,
                round(start, 2),
                round(rate_of(p.dens), 2),
                round(rate_of(p.peak), 2) if p.peak is not None else None,
                round(100 * _SHARE_MID[p.fine]),
                round(100 * _SHARE_MID[p.rest]),
                round(100 * _SHARE_MID[p.ka]),
                round(100 * _SHARE_MID[p.sw]) if p.sw is not None else None,
                round(100 * _SHARE_MID[p.long]) if p.long is not None else None,
                GRID_VALUES[p.sub],
                p.rep,
                p.var,
            ]
        )
    return rows


def _num(x, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(v) else v


def from_rows(rows: list[list], original: dict[int, SectionPlan]) -> dict[int, SectionPlan]:
    """Edited rows back to plans; sections missing from the table keep their plan."""
    out = dict(original)
    for row in rows:
        try:
            i = int(row[0])
        except (TypeError, ValueError):
            continue
        p = original.get(i)
        if p is None:
            continue

        def share(col: int, cur: int | None, row=row) -> int | None:
            if cur is None:
                return None
            return share_bin(_num(row[col], 100 * _SHARE_MID[cur]) / 100)

        grid = _num(row[9], GRID_VALUES[p.sub])
        sub = min(range(len(GRID_VALUES)), key=lambda j: abs(GRID_VALUES[j] - grid))
        out[i] = SectionPlan(
            section=i,
            dens=min(dens_bin(_num(row[2], rate_of(p.dens))), len(DENS_EDGES) - 1),
            fine=share(4, p.fine) or 0,
            rest=share(5, p.rest) or 0,
            ka=share(6, p.ka) or 0,
            sub=sub,
            peak=None if p.peak is None else min(dens_bin(_num(row[3], rate_of(p.peak))), 15),
            rep=None if p.rep is None else int(np.clip(_num(row[10], p.rep), 0, REP_BINS - 1)),
            var=None if p.var is None else int(np.clip(_num(row[11], p.var), 0, VAR_BINS - 1)),
            sw=share(7, p.sw),
            long=share(8, p.long),
        )
    return out
