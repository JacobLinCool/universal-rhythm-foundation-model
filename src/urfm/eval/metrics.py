"""Reference-based metrics for generated charts.

Every metric works on the IR, so it applies to any game: onsets come from the
game-agnostic interaction view, validity from the mechanics spec, writability
from the game's codec. Game-specific extras (Taiko roles, lane usage) are added
when they apply. ``chartgeneval`` is used for Taiko when it is importable.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event
from urfm.ir.targets import Lane
from urfm.ir.validate import validate
from urfm.ir.views import active_events, chart_stats, interaction

TOLERANCES_MS = (25, 50)
TAIKO = frozenset({"taiko", "osu_taiko"})


def codec_for(mechanics: str) -> str | None:
    if mechanics == "taiko":
        return "tja"
    if mechanics.startswith("osu_"):
        return "osu"
    if mechanics.startswith("sm_"):
        return "stepmania"
    if mechanics == "maimai_dx":
        return "simai"
    return None


# -- onset matching ----------------------------------------------------------


def match_onsets(ref: np.ndarray, gen: np.ndarray, tol_us: int) -> list[tuple[int, int]]:
    """One-to-one greedy matching of two sorted onset arrays within ``tol_us``.

    Returns index pairs ``(ref_index, gen_index)``.
    """
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < len(ref) and j < len(gen):
        d = int(gen[j]) - int(ref[i])
        if abs(d) <= tol_us:
            # prefer the closer of gen[j] / gen[j+1] for ref[i]
            if j + 1 < len(gen) and abs(int(gen[j + 1]) - int(ref[i])) < abs(d):
                j += 1
                continue
            pairs.append((i, j))
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return pairs


def _prf(n_match: int, n_ref: int, n_gen: int) -> tuple[float, float, float]:
    p = n_match / n_gen if n_gen else (1.0 if not n_ref else 0.0)
    r = n_match / n_ref if n_ref else (1.0 if not n_gen else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# -- per-chart helpers --------------------------------------------------------


def _onsets(cs: ChartSet, chart: Chart) -> np.ndarray:
    return np.array([x.onset_us for x in interaction(cs, chart)], dtype=np.int64)


def _events_by_onset(cs: ChartSet, chart: Chart) -> dict[int, list[Event]]:
    timing = cs.timing_of(chart)
    out: dict[int, list[Event]] = {}
    for ev in active_events(chart):
        out.setdefault(timing.us(ev.time), []).append(ev)
    return out


def _taiko_label(events: list[Event]) -> str:
    ev = events[0]
    color = {"taiko:don": "don", "taiko:ka": "ka"}.get(ev.role, ev.role.split(":")[-1])
    return f"{color}_big" if "big" in ev.modifiers else color


def _lane_stats(cs: ChartSet, chart: Chart) -> dict[str, float] | None:
    by_onset = _events_by_onset(cs, chart)
    lanes_per_onset = [
        {ev.target.index for ev in evs if isinstance(ev.target, Lane)}
        for _, evs in sorted(by_onset.items())
    ]
    lanes_per_onset = [s for s in lanes_per_onset if s]
    if not lanes_per_onset:
        return None
    counts = Counter(lane for s in lanes_per_onset for lane in s)
    total = sum(counts.values())
    n_lanes = max(int(chart.meta.get("lanes", 0)), max(counts) + 1, 2)
    probs = np.array([counts.get(i, 0) / total for i in range(n_lanes)])
    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum() / math.log(n_lanes))
    jacks = sum(len(a & b) for a, b in pairwise(lanes_per_onset))
    return {"lane_entropy": entropy, "jack_ratio": jacks / total}


def _validity(cs: ChartSet, chart: Chart) -> tuple[int, bool | None]:
    single = ChartSet(
        song=cs.song, timing=cs.timing, charts=(chart,), audio=cs.audio, native=cs.native
    )
    issues = len(validate(single))
    codec_id = codec_for(chart.mechanics)
    if codec_id is None:
        return issues, None
    try:
        from urfm.codecs import registry

        codec = registry.get(codec_id)
        codec.check_writable(single)
        again = codec.read(codec.write(single))
        writable = len(again.charts) == 1
    except Exception:
        writable = False
    return issues, writable


def _chartgeneval(cs: ChartSet, chart: Chart, duration_s: float) -> dict[str, Any] | None:
    try:
        from chartgeneval.metrics import compute_all_families  # type: ignore[import-not-found]
        from chartgeneval.metrics.base import make_ctx  # type: ignore[import-not-found]
    except ImportError:
        return None
    timing = cs.timing_of(chart)
    events = [
        (us / 1e6, _taiko_label(evs))
        for us, evs in sorted(_events_by_onset(cs, chart).items())
        if _taiko_label(evs) in ("don", "ka", "don_big", "ka_big")
    ]
    bpm = float(timing.bpm_at(timing.tempos[0].beat)) if timing.tempos else None
    course = (chart.difficulty.label or "").lower() or None
    try:
        out = compute_all_families(events, make_ctx(course=course, bpm=bpm, duration=duration_s))
    except Exception as exc:
        return {"error": repr(exc)[:200]}
    return {
        k: v
        for k, v in out.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
    }


# -- public API ---------------------------------------------------------------


def evaluate_pair(
    gen: ChartSet,
    gen_chart: Chart,
    ref: ChartSet,
    ref_chart: Chart,
    *,
    target_density: float | None = None,
) -> dict[str, Any]:
    """Compare one generated chart to its reference; returns a flat dict of metrics."""
    g_on, r_on = _onsets(gen, gen_chart), _onsets(ref, ref_chart)
    row: dict[str, Any] = {
        "mechanics": gen_chart.mechanics,
        "ref_mechanics": ref_chart.mechanics,
        "difficulty": gen_chart.difficulty.label,
        "n_gen": len(g_on),
        "n_ref": len(r_on),
    }
    pairs_25: list[tuple[int, int]] = []
    for tol in TOLERANCES_MS:
        pairs = match_onsets(r_on, g_on, tol * 1000)
        if tol == 25:
            pairs_25 = pairs
        p, r, f = _prf(len(pairs), len(r_on), len(g_on))
        row |= {f"precision@{tol}": p, f"recall@{tol}": r, f"f1@{tol}": f}
    if pairs_25:
        offsets = np.array([g_on[j] - r_on[i] for i, j in pairs_25]) / 1000
        row["mean_abs_offset_ms@25"] = float(np.abs(offsets).mean())

    # density over the union span of both charts
    both = np.concatenate([g_on, r_on])
    span = (both.max() - both.min()) / 1e6 if len(both) > 1 else 0.0
    g_d = len(g_on) / span if span else 0.0
    r_d = len(r_on) / span if span else 0.0
    row |= {
        "density_gen": g_d,
        "density_ref": r_d,
        "density_abs_err": abs(g_d - r_d),
        "density_rel_err": abs(g_d - r_d) / r_d if r_d else None,
    }
    if target_density is not None:
        row |= {
            "density_target": target_density,
            "target_abs_err": abs(g_d - target_density),
            "target_rel_err": abs(g_d - target_density) / target_density
            if target_density
            else None,
        }

    issues, writable = _validity(gen, gen_chart)
    n_events = len(gen_chart.events)
    row |= {
        "issues": issues,
        "issues_per_100": 100 * issues / n_events if n_events else 0.0,
        "writable": writable,
    }

    gs, rs = chart_stats(gen, gen_chart), chart_stats(ref, ref_chart)
    prims = set(gs.primitive_ratio) | set(rs.primitive_ratio)
    bins = set(gs.ioi_beat_hist) | set(rs.ioi_beat_hist)
    row |= {
        "primitive_ratio_l1": sum(
            abs(gs.primitive_ratio.get(k, 0.0) - rs.primitive_ratio.get(k, 0.0)) for k in prims
        ),
        "chord_ratio_gen": gs.chord_ratio,
        "chord_ratio_ref": rs.chord_ratio,
        "chord_ratio_diff": gs.chord_ratio - rs.chord_ratio,
        "ioi_hist_l1": sum(
            abs(gs.ioi_beat_hist.get(k, 0.0) - rs.ioi_beat_hist.get(k, 0.0)) for k in bins
        ),
    }

    if gen_chart.mechanics in TAIKO and ref_chart.mechanics in TAIKO:
        g_ev, r_ev = _events_by_onset(gen, gen_chart), _events_by_onset(ref, ref_chart)
        hits = [
            _taiko_label(r_ev[int(r_on[i])]) == _taiko_label(g_ev[int(g_on[j])])
            for i, j in pairs_25
        ]
        row["taiko_role_acc@25"] = fmean(hits) if hits else None
        cge = _chartgeneval(gen, gen_chart, span)
        if cge is not None:
            row |= {f"cge.{k}": v for k, v in cge.items()}

    gm, rm = structure(gen, gen_chart), structure(ref, ref_chart)
    for key in ("measure_rhythm_variety", "rhythm_change_rate", "ioi_classes"):
        row[f"{key}_gen"], row[f"{key}_ref"] = gm.get(key), rm.get(key)
        if gm.get(key) is not None and rm.get(key) is not None:
            row[f"{key}_diff"] = gm[key] - rm[key]
    if gm.get("first_onset_s") is not None and rm.get("first_onset_s") is not None:
        row["first_onset_delay_s"] = gm["first_onset_s"] - rm["first_onset_s"]
    row["note_type_l1"] = _l1(gm.get("note_types", {}), rm.get("note_types", {}))
    row["controls_gen"], row["controls_ref"] = gm.get("controls", 0), rm.get("controls", 0)

    row |= rhythm_in_reference_beats(gen, gen_chart, ref, ref_chart)
    row |= placement(g_on, r_on)
    row |= subdivision_recall(g_on, ref, ref_chart)
    row["tuplet_share_gen"] = tuplet_share(gen_chart)
    from urfm.eval.monotony import default_references, excess_monotone_time

    _, stretch_ref = default_references()
    for name, (c_s, c) in (("gen", (gen, gen_chart)), ("ref", (ref, ref_chart))):
        m = excess_monotone_time(c_s, c, stretch_ref)
        row[f"monotony_s_per_min_{name}"] = m["excess_s_per_min"]
        row[f"monotony_max_s_{name}"] = m["max_excess_s"]
    from urfm.eval.variation import stagnation_penalty

    for name, (c_s, c) in (("gen", (gen, gen_chart)), ("ref", (ref, ref_chart))):
        row[f"stagnation_{name}"] = stagnation_penalty(c_s, c)["score"]

    g_lane, r_lane = _lane_stats(gen, gen_chart), _lane_stats(ref, ref_chart)
    if g_lane and r_lane:
        row |= {f"{k}_gen": v for k, v in g_lane.items()}
        row |= {f"{k}_ref": v for k, v in r_lane.items()}
    return row


PLACE_WINDOW_MS = 75.0
PLACE_SIGMA_MS = 12.0


def placement(g_on: np.ndarray, r_on: np.ndarray) -> dict[str, Any]:
    """How exactly notes sit where a human put one: over generated onsets with a human onset
    within ``PLACE_WINDOW_MS``, the median offset, the shares within 10 and 25 ms, and the mean
    of ``exp(-(d / PLACE_SIGMA_MS)^2 / 2)`` (the RL placement reward). Whether a note should
    exist at all is left to F1."""
    if not len(g_on) or not len(r_on):
        return {}
    d = _nearest_gap(g_on, r_on) / 1000
    near = d[d <= PLACE_WINDOW_MS]
    if not len(near):
        return {"placement_near_share": 0.0}
    return {
        "placement_near_share": float(len(near) / len(d)),
        "placement_median_ms": float(np.median(near)),
        "placement_within_10ms": float((near <= 10).mean()),
        "placement_within_25ms": float((near <= 25).mean()),
        "placement_score": float(np.exp(-0.5 * (near / PLACE_SIGMA_MS) ** 2).mean()),
    }


def _nearest_gap(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from each of ``x`` to the nearest of ``ref`` (non-empty)."""
    r = np.sort(np.asarray(ref, np.float64))
    idx = np.searchsorted(r, x)
    left = r[np.clip(idx - 1, 0, len(r) - 1)]
    right = r[np.clip(idx, 0, len(r) - 1)]
    return np.minimum(np.abs(x - left), np.abs(x - right))


SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16)


def _subdivision(beat: Fraction) -> int:
    frac = beat % 1
    return next((d for d in SUBDIVISIONS if (frac * d).denominator == 1), 0)


def subdivision_recall(g_on: np.ndarray, ref: ChartSet, ref_chart: Chart) -> dict[str, Any]:
    """Share of the human onsets at each subdivision per beat (on the human chart's canonical
    grid: 4 = 16ths, 6 = 24ths, 8 = 32nds, 12 = 48ths, 16 = 64ths) that a generated onset
    matches within 25 ms; ``recall_fine8`` pools 8, 12 and 16."""
    from urfm.data.canonical import canonicalize
    from urfm.tokenize.tokenizer import playable_events

    cs, chart = canonicalize(ref, ref_chart)
    timing = cs.timing_of(chart)
    by: dict[int, list[float]] = {}
    for e in playable_events(chart):
        if e.time.residual_us:
            continue
        by.setdefault(_subdivision(e.time.beat), []).append(float(timing.us(e.time)))
    out: dict[str, Any] = {}
    pooled = [0, 0]
    for d, us in by.items():
        if d < 4:
            continue
        u = np.unique(np.asarray(us))
        hit = _nearest_gap(u, g_on) <= 25_000 if len(g_on) else np.zeros(len(u), bool)
        out[f"recall_sub{d}"] = float(hit.mean())
        out[f"ref_share_sub{d}"] = len(u)
        if d >= 8:
            pooled[0] += int(hit.sum())
            pooled[1] += len(u)
    if pooled[1]:
        out["recall_fine8"] = pooled[0] / pooled[1]
    return out


def tuplet_share(chart: Chart) -> float:
    """Share of playable events on 5- or 7-tuplet positions."""
    from urfm.tokenize.tokenizer import playable_events

    events = playable_events(chart)
    if not events:
        return 0.0
    bad = sum(
        1
        for e in events
        if (e.time.beat % 1).denominator % 5 == 0 or (e.time.beat % 1).denominator % 7 == 0
    )
    return bad / len(events)


IOI_CLASSES = tuple(
    Fraction(n, d)
    for n, d in ((1, 8), (1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (1, 1), (3, 2), (2, 1))
)


def structure(cs: ChartSet, chart: Chart) -> dict[str, Any]:
    """What a player notices beyond onset agreement (game-agnostic).

    - ``first_onset_s``: when the chart starts.
    - ``ioi_classes``: rhythmic vocabulary, the number of inter-onset interval classes (in beats)
      that each cover at least 2% of intervals.
    - ``measure_rhythm_variety``: distinct per-measure onset patterns / measures.
    - ``rhythm_change_rate``: share of consecutive measures whose onset pattern differs.
    - ``note_types``: share of each (role, accent) among interactive events.
    - ``controls``: number of presentation controls (scroll, highlight, barline...).
    """
    timing = cs.timing_of(chart)
    events = sorted(active_events(chart), key=lambda e: e.time)
    if not events:
        return {"controls": len(chart.controls)}
    onsets = sorted({e.time.beat for e in events})
    classes: dict[str, int] = {}
    for a, b in pairwise(onsets):
        gap = b - a
        best = min(IOI_CLASSES, key=lambda c: abs(c - gap))
        key = str(best) if abs(best - gap) <= Fraction(1, 96) else "other"
        classes[key] = classes.get(key, 0) + 1
    total = max(sum(classes.values()), 1)
    starts = timing.measure_starts(onsets[-1] + 4)
    sigs: list[tuple[int, ...]] = []
    j = 0
    for m0, m1 in pairwise(starts):
        sig = []
        while j < len(onsets) and onsets[j] < m1:
            if onsets[j] >= m0:
                sig.append(round((onsets[j] - m0) * 48))
            j += 1
        if sig:
            sigs.append(tuple(sig))
    kinds: dict[str, int] = {}
    for e in events:
        key = e.role + ("+accent" if e.modifiers else "")
        kinds[key] = kinds.get(key, 0) + 1
    return {
        "first_onset_s": timing.seconds(events[0].time),
        "ioi_classes": sum(1 for v in classes.values() if v / total >= 0.02),
        "measure_rhythm_variety": len(set(sigs)) / max(len(sigs), 1),
        "rhythm_change_rate": sum(a != b for a, b in pairwise(sigs)) / max(len(sigs) - 1, 1),
        "note_types": {k: v / len(events) for k, v in kinds.items()},
        "controls": len(chart.controls),
    }


SECTION_BEATS = 8
FINE_GAP = Fraction(1, 4)
"""Intervals of at most a 16th note (plus tolerance) count as fine rhythm."""


def _ref_beats(timing, us: np.ndarray) -> np.ndarray:
    from urfm.tokenize.tokenizer import beats_at_us

    return beats_at_us(timing, us.astype(np.float64))


def _ioi_shares(beats: np.ndarray) -> dict[str, float]:
    gaps = np.diff(beats)
    if len(gaps) == 0:
        return {}
    classes = np.array([float(c) for c in IOI_CLASSES])
    idx = np.abs(gaps[:, None] - classes[None]).argmin(1)
    ok = np.abs(gaps - classes[idx]) <= 1 / 24
    keys = np.where(ok, idx, -1)
    out: dict[str, float] = {}
    for k in keys:
        name = str(IOI_CLASSES[k]) if k >= 0 else "other"
        out[name] = out.get(name, 0.0) + 1 / len(gaps)
    return out


def _section_features(
    beats: np.ndarray, labels: list[str], edges: np.ndarray, edge_s: np.ndarray
) -> np.ndarray:
    """Per section ``[edges[i], edges[i+1])`` (reference beats): onsets/s, share of fine
    intervals, share of the section in rests of >= 1 beat, share of ``ka`` hits."""
    feats = []
    for (a, b), dur in zip(pairwise(edges), np.diff(edge_s), strict=True):
        sel = (beats >= a) & (beats < b)
        n = int(sel.sum())
        sb = beats[sel]
        gaps = np.diff(np.concatenate([[a], sb, [b]])).astype(np.float64)
        rest = float(gaps[gaps >= 1.0].sum() / (b - a))
        inner = np.diff(sb)
        fine = float((inner <= float(FINE_GAP) + 1 / 48).mean()) if len(inner) else 0.0
        lab = [labels[i] for i in np.nonzero(sel)[0]]
        ka = sum(1 for x in lab if x.startswith("ka")) / n if n else 0.0
        feats.append((n / max(float(dur), 1e-3), fine, rest, ka))
    return np.array(feats, dtype=np.float64).reshape(-1, 4)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rhythm_in_reference_beats(
    gen: ChartSet, gen_chart: Chart, ref: ChartSet, ref_chart: Chart
) -> dict[str, Any]:
    """Rhythm compared on the *reference's* beat grid, so it is fair when the generated chart
    uses another (e.g. audio-predicted) timing map or another notation octave.

    - ``ioi_ref_l1``: L1 distance of inter-onset-interval class shares;
    - ``fine_share_*`` / ``rest_share_*``: share of intervals of a 16th or less, share of the
      charted span in rests of at least one beat;
    - ``section_r_*``: correlation with the reference across 8-beat sections of density,
      fine share, rest share and (Taiko) ka share: do sections contrast like the human's?
    """
    timing = ref.timing_of(ref_chart)
    g_us = np.array(sorted(_events_by_onset(gen, gen_chart)), dtype=np.float64)
    r_us = np.array(sorted(_events_by_onset(ref, ref_chart)), dtype=np.float64)
    if len(g_us) < 2 or len(r_us) < 2:
        return {}
    g_b, r_b = _ref_beats(timing, g_us), _ref_beats(timing, r_us)
    out: dict[str, Any] = {"ioi_ref_l1": _l1(_ioi_shares(g_b), _ioi_shares(r_b))}
    lo = np.floor(r_b[0] / SECTION_BEATS) * SECTION_BEATS
    hi = np.ceil((r_b[-1] + 1e-9) / SECTION_BEATS) * SECTION_BEATS
    edges = np.arange(lo, hi + SECTION_BEATS / 2, SECTION_BEATS)
    if len(edges) < 2:
        return out
    g_ev, r_ev = _events_by_onset(gen, gen_chart), _events_by_onset(ref, ref_chart)
    g_lab = [_taiko_label(g_ev[int(u)]) for u in g_us]
    r_lab = [_taiko_label(r_ev[int(u)]) for u in r_us]
    edge_s = timing.us_array(list(edges)) / 1e6
    gf = _section_features(g_b, g_lab, edges, edge_s)
    rf = _section_features(r_b, r_lab, edges, edge_s)
    for i, name in enumerate(("density", "fine", "rest", "ka")):
        if name == "ka" and gen_chart.mechanics not in TAIKO:
            continue
        out[f"section_r_{name}"] = _corr(gf[:, i], rf[:, i])
    for name, us in (("gen", g_us), ("ref", r_us)):
        out[f"lt60ms_{name}"] = float((np.diff(us) < 60_000).mean())
    for name, b in (("gen", g_b), ("ref", r_b)):
        gaps = np.diff(b)
        out[f"fine_share_{name}"] = float((gaps <= float(FINE_GAP) + 1 / 48).mean())
        out[f"rest_share_{name}"] = float(gaps[gaps >= 1].sum() / max(b[-1] - b[0], 1e-9))
    return out


def _l1(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


AGGREGATE_KEYS = (
    "f1@25",
    "f1@50",
    "precision@25",
    "recall@25",
    "density_rel_err",
    "target_rel_err",
    "issues_per_100",
    "writable",
    "primitive_ratio_l1",
    "ioi_hist_l1",
    "taiko_role_acc@25",
    "note_type_l1",
    "first_onset_delay_s",
    "measure_rhythm_variety_gen",
    "measure_rhythm_variety_ref",
    "rhythm_change_rate_gen",
    "rhythm_change_rate_ref",
    "controls_gen",
    "controls_ref",
    "ioi_ref_l1",
    "section_r_density",
    "section_r_fine",
    "section_r_rest",
    "section_r_ka",
    "fine_share_gen",
    "fine_share_ref",
    "rest_share_gen",
    "rest_share_ref",
    "lt60ms_gen",
    "lt60ms_ref",
    "monotony_s_per_min_gen",
    "monotony_s_per_min_ref",
    "monotony_max_s_gen",
    "placement_median_ms",
    "placement_within_10ms",
    "placement_score",
    "recall_sub4",
    "recall_sub6",
    "recall_fine8",
    "tuplet_share_gen",
    "stagnation_gen",
    "stagnation_ref",
    "structure_score_gen",
    "structure_score_ref",
    "structure_null_gen",
)


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean of numeric metrics per generated mechanics (``None`` values ignored)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["mechanics"], []).append(row)
    out: dict[str, dict[str, float]] = {}
    for mech, items in sorted(groups.items()):
        summary: dict[str, float] = {"charts": float(len(items))}
        keys = sorted({k for r in items for k, v in r.items() if _numeric(v)})
        for key in keys:
            values = [float(r[key]) for r in items if _numeric(r.get(key))]
            if values:
                summary[key] = fmean(values)
        out[mech] = summary
    return out


def _numeric(v: Any) -> bool:
    return isinstance(v, int | float) and not (isinstance(v, float) and math.isnan(v))


def write_report(rows: list[dict[str, Any]], path_jsonl: str | Path, path_md: str | Path) -> None:
    with Path(path_jsonl).open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    table = aggregate(rows)
    lines = ["# Evaluation report", ""]
    cols = ["charts", *[k for k in AGGREGATE_KEYS if any(k in v for v in table.values())]]
    lines.append("| mechanics | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for mech, values in table.items():
        cells = [_fmt(values.get(c)) for c in cols]
        lines.append(f"| {mech} | " + " | ".join(cells) + " |")
    Path(path_md).write_text("\n".join(lines) + "\n")


def _fmt(v: float | None) -> str:
    if v is None:
        return "–"
    return f"{v:.0f}" if float(v).is_integer() and abs(v) >= 1 else f"{v:.3f}"
