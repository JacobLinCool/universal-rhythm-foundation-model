"""URFM tokenizer: IR charts <-> token sequences (see ``urfm.tokenize.api``)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache

import numpy as np

from urfm.data.features import MEL
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.controls import Control
from urfm.ir.events import Event, PathSegment, Primitive
from urfm.ir.targets import Lane, Point2D, RingButton, Target, TouchRegion
from urfm.ir.time import Time
from urfm.ir.timing import MeterChange, TimingMap, _compile
from urfm.mechanics import registry
from urfm.tokenize.api import CONTEXT_EVENTS, WINDOW_S
from urfm.tokenize.grammar import Grammar, State, grammar_for
from urfm.tokenize.sections import SectionPlan
from urfm.tokenize.vocab import (
    COUNT_GRID,
    DENS_EDGES,
    GRID_VALUES,
    MAX_BEATS,
    MAX_PATH_POINTS,
    METER_DENS,
    METER_NUMS,
    REST_BINS,
    SCROLL_GRID,
    SUB_GRID,
    TUPLET_FREE,
    Vocab,
    dens_bin,
    nearest,
    pt_x_bin,
    pt_x_value,
    pt_y_bin,
    pt_y_value,
    scroll_bin,
    split_beats,
)

FRAME_FEATURES = (
    "beat_sin",
    "beat_cos",
    "measure_sin",
    "measure_cos",
    "log2_bpm_120",
    "beat_tick",
    "downbeat_tick",
)
"""Columns of :meth:`UrfmTokenizer.frame_features`."""

PREFERRED_ROUTE = "master"
CONTEXT_BEATS = 32
REF_EVENTS = 48
"""Events kept in the structure-memory block (R6)."""
"""Context covers whole beats ``[window_start - CONTEXT_BEATS, window_start)``; an integral
origin keeps context positions on the same musical grid as the window body."""

DROPS: Counter[tuple[str, str]] = Counter()
"""Events silently unrepresentable by the tokenizer, by ``(mechanics, reason)``; read with
:func:`take_drops`. Training logs these so pipeline losses stay visible."""


def take_drops() -> dict[str, int]:
    out = {f"{m}/{r}": n for (m, r), n in sorted(DROPS.items())}
    DROPS.clear()
    return out


# Coarse cross-game tier 0..4 from free-text difficulty names (osu! version names, etc.).
_TIER_WORDS = (
    (re.compile(r"beginner|easy|kantan|cup|novice"), 0),
    (re.compile(r"normal|futsuu|basic|salad|light"), 1),
    (re.compile(r"hard|muzukashii|advanced|platter|medium"), 2),
    (re.compile(r"insane|oni|expert|rain|another|challenge"), 3),
    (re.compile(r"extra|extreme|ura|edit|overdose|master|expert\+|lunatic|deluge"), 4),
)


TICKS_PER_BEAT = 34_594_560
"""2^8·3^3·5·7·11·13: every SUB_GRID step and the 1/192 osu! snap are whole ticks, also after
re-notation at half BPM (1/384), so positions of most charts are exact integers (others use
the Fraction path)."""


@dataclass(frozen=True)
class ControlItem:
    """A presentation change in token order: a chart control or a meter change."""

    time: Time
    rank: tuple[int, int]
    """Tie-break among controls on the same beat: meters, scroll, highlight, barline."""
    tokens: tuple[int, ...]
    source: Control | MeterChange


_CONTROL_RANK = {"meter": 0, "scroll": 1, "osu:sv": 1, "highlight": 2, "barline": 3}


def _item_key(item: Event | ControlItem) -> tuple:
    """Token order: by beat; at one beat controls first, then events (``_event_order``).

    The float beat leads the key: rounding is monotonic, so it never reorders distinct beats,
    and ties fall back to the exact Fraction (comparing Fractions dominated sorting)."""
    beat = item.time.beat
    if isinstance(item, ControlItem):
        return (float(beat), beat, 0, item.rank)
    return (float(beat), beat, 1, *_event_order(item)[1:])


@dataclass
class CompiledChart:
    """A chart's playable events in token order, with everything a window needs
    precomputed: integer beat ticks, rounded onsets, position-independent token bodies and
    their validity. Encoding a window is then integer arithmetic and table lookups."""

    mechanics: str
    events: list[Event | ControlItem]
    """Events and control items in token order."""
    us: np.ndarray
    ticks: list[int] | None
    """Exact ``beat * TICKS_PER_BEAT``; ``None`` if some beat is not a whole tick."""
    bodies: list[list[int] | None]
    """Position-independent tokens per event, filled lazily (only sampled windows pay)."""
    valid: list[bool | None]
    allow_same: bool
    chart: Chart | None = None
    grammar: Grammar | None = None
    skipped: list[tuple[int, str]] | None = None
    """Controls in the chart the game cannot express, as ``(onset µs, reason)`` (counted)."""


def _round_us(timing: TimingMap, times: list[Time]) -> list[int]:
    if not times:
        return []
    beats = [t.beat for t in times]
    return (
        np.round(timing.us_array(beats, [t.residual_us for t in times])).astype(np.int64).tolist()
    )


@lru_cache(maxsize=65536)
def _split_ticks(rel: int, plain: bool = False) -> tuple[int, int]:
    return split_beats(Fraction(rel, TICKS_PER_BEAT), plain)


class UrfmTokenizer:
    def __init__(self) -> None:
        self.vocab = Vocab()
        v = self.vocab
        self.vocab_size = len(v)
        self.pad_id = v.id("PAD")
        self.bos_id = v.id("BOS")
        self.sep_id = v.id("SEP")
        self.eos_id = v.id("EOS")
        self._valid_cache: dict[tuple, bool] = {}

    # -- conditioning ------------------------------------------------------------------

    def prefix(self, chartset: ChartSet, chart: Chart) -> list[int]:
        return self.conditioning(
            chart.mechanics,
            tier=chart_tier(chart),
            onsets_per_s=onset_rate(chartset, chart),
            lanes=chart_lanes(chart),
        )

    def conditioning(
        self,
        mechanics: str,
        *,
        tier: int | None = None,
        onsets_per_s: float | None = None,
        lanes: int | None = None,
    ) -> list[int]:
        """Prefix tokens from explicit values (generation for any target game).

        ``onsets_per_s=None`` falls back to the median density bin.
        """
        v = self.vocab
        spec = registry.get(mechanics)
        out = [v.id("GAME", mechanics)]
        lanes = spec.space.lanes or lanes
        if spec.space.kind.value == "lanes" and lanes:
            out.append(v.id("LANES", min(lanes, 18)))
        out.append(v.id("TIER", tier))
        dens = dens_bin(onsets_per_s) if onsets_per_s is not None else len(DENS_EDGES) // 2
        out.append(v.id("DENS", dens))
        return out

    # -- events -> tokens ---------------------------------------------------------------

    def encode(self, chartset: ChartSet, chart: Chart, start_us: int, end_us: int) -> list[int]:
        return [t for group in self.encode_groups(chartset, chart, start_us, end_us) for t in group]

    def encode_groups(
        self, chartset: ChartSet, chart: Chart, start_us: int, end_us: int
    ) -> list[list[int]]:
        """Like :meth:`encode`, one token list per event (handy for context truncation)."""
        return [group for _, group in self.encode_pairs(chartset, chart, start_us, end_us)]

    def control_items(
        self, chartset: ChartSet, chart: Chart
    ) -> tuple[list[ControlItem], list[tuple[Time, str]]]:
        """Controls and meter changes the game can express, in token form, plus the ones it
        cannot (``(time, reason)``, counted in :data:`DROPS` when they fall in a window).

        Scroll-like controls (``scroll``, ``osu:sv``) share one token family; meter changes
        come from the timing map and only genuine changes are kept (osu! repeats 4/4 on every
        red line). Branch routes follow :func:`playable_events`.
        """
        v = self.vocab
        g = grammar_for(v, chart.mechanics, chart_lanes(chart))
        allowed = g.control_names
        routes = {
            b.id: PREFERRED_ROUTE if PREFERRED_ROUTE in b.routes else b.default
            for b in chart.branches
        }
        items: list[ControlItem] = []
        skipped: list[tuple[Time, str]] = []
        for i, c in enumerate(chart.controls):
            if c.branch is not None and routes.get(c.branch.segment) != c.branch.route:
                continue
            tokens: tuple[int, ...] | None = None
            reason = f"control:{c.kind}"
            if c.kind in ("scroll", "osu:sv") and "scroll" in allowed:
                value = c.value
                if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
                    tokens = (v.id("CTRL", "scroll"), v.id("SCROLL", scroll_bin(float(value))))
                else:
                    reason = "control_value:scroll"
            elif c.kind in ("highlight", "barline") and f"{c.kind}_on" in allowed:
                if isinstance(c.value, bool):
                    tokens = (v.id("CTRL", f"{c.kind}_{'on' if c.value else 'off'}"),)
                else:
                    reason = f"control_value:{c.kind}"
            if tokens is None:
                skipped.append((c.time, reason))
            else:
                items.append(ControlItem(c.time, (_CONTROL_RANK[c.kind], i), tokens, c))
        if "meter" in allowed:
            timing = chartset.timing_of(chart)
            previous = (4, 4)
            for m in sorted(timing.meters, key=lambda m: m.beat):
                value = (m.numerator, m.denominator)
                if value == previous:
                    continue
                previous = value
                if m.numerator in METER_NUMS and m.denominator in METER_DENS:
                    tokens = (
                        v.id("CTRL", "meter"),
                        v.id("METER_NUM", m.numerator),
                        v.id("METER_DEN", m.denominator),
                    )
                    items.append(ControlItem(Time(m.beat), (0, 0), tokens, m))
                else:
                    skipped.append((Time(m.beat), "meter_unrepresentable"))
        return items, skipped

    def encode_pairs(
        self,
        chartset: ChartSet,
        chart: Chart,
        start_us: int,
        end_us: int,
        origin_beat: Fraction | None = None,
        cache: dict | None = None,
    ) -> list[tuple[Event | ControlItem, list[int]]]:
        """``(item, tokens)`` for every encodable event and control with onset in
        ``[start_us, end_us)``, in token order. Positions are relative to ``origin_beat``
        (default: the beat at ``start_us``). Control groups always carry an explicit position
        and never change which event a following ``POS_SAME`` refers to."""
        timing = chartset.timing_of(chart)
        origin = window_start_beat(timing, start_us) if origin_beat is None else origin_beat
        controls, skipped = self.control_items(chartset, chart)
        candidates: list[Event | ControlItem] = [*playable_events(chart), *controls]
        # one vectorized pass instead of an exact Fraction evaluation per item; both sides
        # of the window test are rounded the same way, so boundary items stay consistent
        onsets = _round_us(timing, [x.time for x in candidates])
        items: list[Event | ControlItem] = []
        for item, us in zip(candidates, onsets, strict=True):
            if not start_us <= us < end_us:
                continue
            if not 0 <= item.time.beat - origin < MAX_BEATS:
                DROPS[chart.mechanics, "outside_position_range"] += 1
                continue
            items.append(item)
        for us, (_, reason) in zip(
            _round_us(timing, [t for t, _ in skipped]), skipped, strict=True
        ):
            if start_us <= us < end_us:
                DROPS[chart.mechanics, reason] += 1
        items.sort(key=_item_key)
        pairs: list[tuple[Event | ControlItem, list[int]]] = []
        previous = None
        for item in items:
            rel = item.time.beat - origin
            if isinstance(item, ControlItem):
                beat, sub = split_beats(rel, chart.mechanics in TUPLET_FREE)
                head = [self.vocab.id("POS_BEAT", beat), self.vocab.id("POS_SUB", sub)]
                pairs.append((item, head + list(item.tokens)))
                continue
            group = self._encode_event(chart, item, rel, same=rel == previous, cache=cache)
            if group:
                pairs.append((item, group))
                previous = rel
            elif rel == previous:  # a second action at once in a single-action game
                DROPS[chart.mechanics, "simultaneous"] += 1
            else:
                DROPS[chart.mechanics, f"unencodable:{item.role}"] += 1
        return pairs

    def compile(self, chartset: ChartSet, chart: Chart) -> CompiledChart:
        """Precompute a chart for fast repeated window encoding (training)."""
        timing = chartset.timing_of(chart)
        g = grammar_for(self.vocab, chart.mechanics, chart_lanes(chart))
        controls, skipped = self.control_items(chartset, chart)
        items: list[Event | ControlItem] = sorted(
            [*playable_events(chart), *controls], key=_item_key
        )
        us = np.asarray(_round_us(timing, [x.time for x in items]), dtype=np.int64)
        scaled = [x.time.beat * TICKS_PER_BEAT for x in items]
        exact = all(x.denominator == 1 for x in scaled)
        v = self.vocab
        return CompiledChart(
            mechanics=chart.mechanics,
            events=items,
            us=us,
            ticks=[int(x) for x in scaled] if exact else None,
            bodies=[list(x.tokens) if isinstance(x, ControlItem) else None for x in items],
            valid=[True if isinstance(x, ControlItem) else None for x in items],
            allow_same=v.id("POS_SAME") in g.group_start,
            chart=chart,
            grammar=g,
            skipped=list(
                zip(
                    _round_us(timing, [t for t, _ in skipped]), [r for _, r in skipped], strict=True
                )
            ),
        )

    def _compiled_body(self, cc: CompiledChart, i: int) -> tuple[list[int], bool]:
        body = cc.bodies[i]
        if body is None:
            assert cc.chart is not None and cc.grammar is not None
            v = self.vocab
            event = cc.events[i]
            assert isinstance(event, Event)
            body = cc.bodies[i] = self._event_body(cc.chart, cc.grammar, event)
            probe = [v.id("POS_BEAT", 0), v.id("POS_SUB", 0)]
            cc.valid[i] = self._valid(cc.grammar, probe + body)
        return body, bool(cc.valid[i])

    def _compiled_groups(
        self, cc: CompiledChart, start_us: int, end_us: int, origin: Fraction
    ) -> list[tuple[Event | ControlItem, list[int]]]:
        """Integer-tick equivalent of :meth:`encode_pairs` (same tokens, same drops)."""
        assert cc.ticks is not None
        v, mech = self.vocab, cc.mechanics
        o = int(origin * TICKS_PER_BEAT)
        limit = MAX_BEATS * TICKS_PER_BEAT
        same_id, beat_ids, sub_ids = v.id("POS_SAME"), v.by_kind["POS_BEAT"], v.by_kind["POS_SUB"]
        plain = mech in TUPLET_FREE
        for us, reason in cc.skipped or ():
            if start_us <= us < end_us:
                DROPS[mech, reason] += 1
        out, previous = [], None
        for i in np.nonzero((cc.us >= start_us) & (cc.us < end_us))[0].tolist():
            rel = cc.ticks[i] - o
            if not 0 <= rel < limit:
                DROPS[mech, "outside_position_range"] += 1
                continue
            item = cc.events[i]
            if isinstance(item, ControlItem):
                beat, sub = _split_ticks(rel, plain)
                out.append((item, [beat_ids[beat], sub_ids[sub], *item.tokens]))
                continue
            same = rel == previous
            body, valid = self._compiled_body(cc, i)
            ok = valid and (cc.allow_same or not same)
            if not ok:
                reason = "simultaneous" if same else f"unencodable:{item.role}"
                DROPS[mech, reason] += 1
                continue
            if same:
                head = [same_id]
            else:
                beat, sub = _split_ticks(rel, plain)
                head = [beat_ids[beat], sub_ids[sub]]
            out.append((cc.events[i], head + body))
            previous = rel
        return out

    def _window_pairs(
        self,
        chartset: ChartSet,
        chart: Chart,
        start_us: int,
        end_us: int,
        origin: Fraction | None,
        cache: dict | None,
        compiled: CompiledChart | None,
    ) -> list[tuple[Event | ControlItem, list[int]]]:
        if compiled is not None and compiled.ticks is not None:
            if origin is None:
                origin = window_start_beat(chartset.timing_of(chart), start_us)
            return self._compiled_groups(compiled, start_us, end_us, origin)
        return self.encode_pairs(chartset, chart, start_us, end_us, origin, cache)

    def context(
        self,
        chartset: ChartSet,
        chart: Chart,
        start_us: int,
        cache: dict | None = None,
        compiled: CompiledChart | None = None,
    ) -> list[int]:
        """Context tokens for a window starting at ``start_us`` -- the single definition
        shared by training and generation.

        Events in the ``CONTEXT_BEATS`` whole beats before the window, the last
        ``CONTEXT_EVENTS`` of them, positioned relative to ``window_beat - CONTEXT_BEATS``.
        The cut never starts on a ``POS_SAME`` group (it would refer to a dropped event).
        """
        timing = chartset.timing_of(chart)
        origin = window_start_beat(timing, start_us) - CONTEXT_BEATS
        origin_us = round(timing.beat_to_us_exact(origin))
        pairs = self._window_pairs(chartset, chart, origin_us, start_us, origin, cache, compiled)
        groups = [g for _, g in pairs]
        cut = max(len(groups) - CONTEXT_EVENTS, 0)
        same = self.vocab.id("POS_SAME")
        while cut < len(groups) and groups[cut][0] == same:
            cut += 1
        return [t for g in groups[cut:] for t in g]

    def encode_window(
        self,
        chartset: ChartSet,
        chart: Chart,
        start_us: int,
        end_us: int,
        room: int,
        cache: dict | None = None,
        compiled: CompiledChart | None = None,
        sections: list[SectionPlan] | None = None,
        k: int | None = None,
    ) -> tuple[list[int], bool]:
        """Window body ``WDENS GRID REST group*`` (or, with ``sections`` and their length
        ``k`` in beats, the section plans instead of the window plan) within ``room`` tokens,
        cut only at event boundaries.

        Returns ``(tokens, truncated)``; dropped events are counted in :data:`DROPS`.
        """
        pairs = self._window_pairs(chartset, chart, start_us, end_us, None, cache, compiled)
        timing = chartset.timing_of(chart)
        if sections is not None:
            assert k is not None
            out = self.plan_tokens(sections, window_start_beat(timing, start_us), k)
        else:
            onsets = {(x.time.beat, x.time.residual_us) for x, _ in pairs if isinstance(x, Event)}
            rate = len(onsets) / max((end_us - start_us) / 1e6, 1e-3)
            beats = list({b for b, _ in onsets})
            span = (window_start_beat(timing, start_us), window_start_beat(timing, end_us))
            out = [
                self.vocab.id("WDENS", dens_bin(rate)),
                self.vocab.id("GRID", window_grid(beats)),
                self.vocab.id("REST", window_rest(beats, *span)),
            ]
        for i, (_, group) in enumerate(pairs):
            if len(out) + len(group) > room:
                DROPS[chart.mechanics, "truncated"] += len(pairs) - i
                return out, True
            out += group
        return out, False

    def plan_tokens(self, sections: list[SectionPlan], window_beat: Fraction, k: int) -> list[int]:
        """``SBEAT SDENS SFINE SREST SKA SSUB`` per section, in order."""
        v, out = self.vocab, []
        for p in sections:
            rel = min(max(math.floor(p.section * k - window_beat), 0), MAX_BEATS - 1)
            out += [
                v.id("SBEAT", rel),
                v.id("SDENS", p.dens),
                v.id("SFINE", p.fine),
                v.id("SREST", p.rest),
                v.id("SKA", p.ka),
                v.id("SSUB", p.sub),
            ]
            if p.peak is not None:
                out += [
                    v.id("SPEAK", p.peak),
                    v.id("SREP", p.rep),
                    v.id("SVAR", p.var),
                    v.id("SSW", p.sw),
                    v.id("SLONG", p.long),
                ]
        return out

    def ref_block(
        self,
        chartset: ChartSet,
        chart: Chart,
        section: int,
        target: int,
        window_beat: Fraction,
        k: int,
        max_events: int = REF_EVENTS,
        cache: dict | None = None,
        compiled: CompiledChart | None = None,
    ) -> list[int]:
        """``REF SBEAT group* REFEND``: the events of section ``section`` (positions relative
        to its start, at most ``max_events``), and where the window section ``target`` it
        resembles starts, in whole beats after the window start."""
        timing = chartset.timing_of(chart)
        origin = Fraction(section * k)
        a = round(timing.beat_to_us_exact(origin))
        b = round(timing.beat_to_us_exact(origin + k))
        pairs = self._window_pairs(chartset, chart, a, b, origin, cache, compiled)
        groups = [g for x, g in pairs if isinstance(x, Event)][:max_events]
        same = self.vocab.id("POS_SAME")
        while groups and groups[0][0] == same:
            groups = groups[1:]
        rel = min(max(math.floor(target * k - window_beat), 0), MAX_BEATS - 1)
        v = self.vocab
        return [v.id("REF"), v.id("SBEAT", rel), *(t for g in groups for t in g), v.id("REFEND")]

    def window_density(self, onsets_per_s: float) -> int:
        """``WDENS`` token for a requested local density (forcing the plan at generation)."""
        return self.vocab.id("WDENS", dens_bin(onsets_per_s))

    def _encode_event(
        self, chart: Chart, ev: Event, rel: Fraction, same: bool, cache: dict | None = None
    ) -> list[int]:
        v = self.vocab
        g = grammar_for(v, chart.mechanics, chart_lanes(chart))
        out: list[int] = []
        if same:
            out.append(v.id("POS_SAME"))
        else:
            beat, sub = split_beats(rel, chart.mechanics in TUPLET_FREE)
            if rel >= MAX_BEATS:
                return []
            out += [v.id("POS_BEAT", beat), v.id("POS_SUB", sub)]
        # everything after the position depends only on the event: cache it per event
        body = cache.get(id(ev)) if cache is not None else None
        if body is None:
            body = self._event_body(chart, g, ev)
            if cache is not None:
                cache[id(ev)] = body
        out += body
        return out if self._valid(g, out) else []

    def _event_body(self, chart: Chart, g: Grammar, ev: Event) -> list[int]:
        v = self.vocab
        out: list[int] = []
        out.append(v.id("PRIM", ev.primitive.value))
        if ev.primitive is not Primitive.HIT:
            span = (ev.end.beat - ev.time.beat) if ev.end is not None else Fraction(0)
            out += self._span(span, g.spec.id in TUPLET_FREE)
        if g.accent and g.accent in ev.modifiers:
            out.append(v.id("ACCENT"))
        out.append(v.id("ROLE", ev.role))
        plan = g.plan(ev.role, ev.primitive.value)
        out += [v.id("MOD", m) for m in plan.mods if m in ev.modifiers]
        if plan.targets and ev.target is not None:
            out += self._target(ev.target)
        if plan.has_count and ev.count is not None:
            out.append(v.id("COUNT", nearest(COUNT_GRID, ev.count)))
        if plan.path:
            first = ev.path[0] if ev.path else None
            if plan.wait and first is not None and first.start is not None:
                out += self._span(
                    max(first.start.beat - ev.time.beat, Fraction(0)), g.spec.id in TUPLET_FREE
                )
            for i, seg in enumerate(ev.path):
                points = list(seg.points)
                joint = ev.target if i == 0 else ev.path[i - 1].points[-1]
                if points and points[0] == joint:
                    points = points[1:]
                if len(points) > MAX_PATH_POINTS:
                    points = [*points[: MAX_PATH_POINTS - 1], points[-1]]
                if not points or ("SHAPE", seg.shape) not in v.ids:
                    DROPS[chart.mechanics, f"path_segment:{seg.shape}"] += 1
                    continue
                out.append(v.id("SHAPE", seg.shape))
                for p in points:
                    out += self._target(p)
        return out

    def _span(self, beats: Fraction, plain: bool = False) -> list[int]:
        whole, sub = split_beats(beats, plain)
        return [self.vocab.id("DUR_BEAT", whole), self.vocab.id("DUR_SUB", sub)]

    def _target(self, target: Target) -> list[int]:
        v = self.vocab
        if isinstance(target, Lane):
            return [v.id("LANE", target.index)]
        if isinstance(target, RingButton):
            return [v.id("RING", target.index)]
        if isinstance(target, TouchRegion):
            return [v.id("REGION", target.name)]
        if isinstance(target, Point2D):
            return [v.id("PT_X", pt_x_bin(target.x)), v.id("PT_Y", pt_y_bin(target.y))]
        raise ValueError(f"unsupported target {target!r}")

    def _valid(self, g: Grammar, group: list[int]) -> bool:
        # validity never depends on the position tokens, so cache on the rest of the group
        skip = 1 if group and self.vocab.kinds[group[0]] == "POS_SAME" else 2
        key = (id(g), tuple(group[skip:]))
        cached = self._valid_cache.get(key)
        if cached is not None:
            return cached
        state: State = ("start", True)
        try:
            for t in group:
                state = g.step(state, t)
            ok = self.eos_id in g.tokens(state)
        except (ValueError, KeyError):
            ok = False
        if len(self._valid_cache) < 200_000:
            self._valid_cache[key] = ok
        return ok

    # -- tokens -> events ---------------------------------------------------------------

    def decode(
        self,
        timing: TimingMap,
        mechanics: str,
        tokens: list[int],
        start_us: int,
        lanes: int | None = None,
    ) -> list[Event]:
        """Decode the window events (the part after ``SEP`` when present).

        ``lanes`` is accepted for interface symmetry; lane ranges are enforced by the
        grammar at generation time, not re-checked here.
        """
        return self.decode_full(timing, mechanics, tokens, start_us, lanes)[0]

    def decode_full(
        self,
        timing: TimingMap,
        mechanics: str,
        tokens: list[int],
        start_us: int,
        lanes: int | None = None,
    ) -> tuple[list[Event], list[Control], list[MeterChange]]:
        """Events, presentation controls and meter changes of a window.

        Scroll tokens decode to the game's own kind (``scroll``, or ``osu:sv`` for games
        that only declare that); malformed groups are skipped.
        """
        v = self.vocab
        if self.sep_id in tokens:
            tokens = tokens[tokens.index(self.sep_id) + 1 :]
        start_beat = window_start_beat(timing, start_us)
        spec_controls = set(registry.get(mechanics).controls)
        scroll_kind = "scroll" if "scroll" in spec_controls else "osu:sv"
        events: list[Event] = []
        controls: list[Control] = []
        meters: list[MeterChange] = []
        rel: Fraction | None = None
        for group in _split_groups(v, tokens):
            if v.kinds[group[0]] == "POS_SAME":
                body = group[1:]
            elif len(group) >= 2 and v.kinds[group[1]] == "POS_SUB":
                rel = v.values[group[0]] + SUB_GRID[v.values[group[1]]]
                body = group[2:]
            else:
                continue
            if rel is None or not body:
                continue
            beat = start_beat + rel
            if v.kinds[body[0]] == "CTRL":
                name, rest = v.values[body[0]], body[1:]
                kinds = [v.kinds[t] for t in rest]
                if name == "scroll" and kinds == ["SCROLL"]:
                    value = SCROLL_GRID[v.values[rest[0]]]
                    controls.append(Control(time=Time(beat), kind=scroll_kind, value=value))
                elif name == "meter" and kinds == ["METER_NUM", "METER_DEN"]:
                    meters.append(MeterChange(beat, v.values[rest[0]], v.values[rest[1]]))
                elif not rest and name.endswith(("_on", "_off")):
                    kind, _, flag = name.rpartition("_")
                    controls.append(Control(time=Time(beat), kind=kind, value=flag == "on"))
                continue
            ev = self._decode_body(mechanics, body, beat)
            if ev is not None:
                events.append(ev)
        _assign_groups(events)
        return events, controls, meters

    def _decode_body(self, mechanics: str, body: list[int], beat: Fraction) -> Event | None:
        v = self.vocab
        g = grammar_for(v, mechanics)
        try:
            state: State = ("prim", True)
            for t in body:
                state = g.step(state, t)
            if self.eos_id not in g.tokens(state):
                return None
        except ValueError:
            return None
        it = iter(body)
        prim = Primitive(v.values[next(it)])
        rest = list(it)
        end = None
        if prim is not Primitive.HIT:
            end = Time(beat + v.values[rest.pop(0)] + SUB_GRID[v.values[rest.pop(0)]])
        modifiers = []
        if rest and v.kinds[rest[0]] == "ACCENT":
            modifiers.append(g.accent)
            rest.pop(0)
        role = v.values[rest.pop(0)]
        while rest and v.kinds[rest[0]] == "MOD":
            modifiers.append(v.values[rest.pop(0)])
        plan = g.plan(role, prim.value)
        target = None
        if plan.targets and rest:
            target, rest = _take_target(v, rest)
        count = None
        if rest and v.kinds[rest[0]] == "COUNT":
            count = COUNT_GRID[v.values[rest.pop(0)]]
        wait = None
        if plan.wait and rest and v.kinds[rest[0]] == "DUR_BEAT":
            wait = v.values[rest.pop(0)] + SUB_GRID[v.values[rest.pop(0)]]
        path: list[PathSegment] = []
        previous = target
        while rest:
            shape = v.values[rest.pop(0)]
            points: list[Target] = [previous] if previous is not None else []
            while rest and v.kinds[rest[0]] != "SHAPE":
                point, rest = _take_target(v, rest)
                points.append(point)
            path.append(PathSegment(shape=shape, points=tuple(points)))
            previous = points[-1] if points else previous
        if path:
            if wait is not None:
                path[0] = PathSegment(
                    shape=path[0].shape, points=path[0].points, start=Time(beat + wait)
                )
            if plan.wait and end is not None:
                last = path[-1]
                path[-1] = PathSegment(
                    shape=last.shape, points=last.points, start=last.start, end=end
                )
        return Event(
            time=Time(beat),
            primitive=prim,
            role=role,
            modifiers=tuple(sorted(modifiers)),
            end=end,
            target=target,
            path=tuple(path),
            count=count,
        )

    # -- grammar ------------------------------------------------------------------------

    def grammar(self, mechanics: str, lanes: int | None = None) -> Grammar:
        return grammar_for(self.vocab, mechanics, lanes)

    def grammar_state(self, mechanics: str, tokens: list[int], lanes: int | None = None) -> State:
        """Grammar state after ``tokens``; everything up to the last ``BOS`` is skipped."""
        if self.bos_id in tokens:
            tokens = tokens[len(tokens) - tokens[::-1].index(self.bos_id) :]
        return self.grammar(mechanics, lanes).run(tokens)

    def step(self, mechanics: str, state: State, token: int, lanes: int | None = None) -> State:
        return self.grammar(mechanics, lanes).step(state, token)

    def mask(self, mechanics: str, state: State, lanes: int | None = None) -> np.ndarray:
        return self.grammar(mechanics, lanes).mask(state)

    def windows(
        self, chartset: ChartSet, chart: Chart, duration_us: int | None = None
    ) -> list[tuple[int, int]]:
        return list(windows(chartset, chart, duration_us))

    def allowed(self, mechanics: str, tokens: list[int], lanes: int | None = None) -> np.ndarray:
        return self.mask(mechanics, self.grammar_state(mechanics, tokens, lanes), lanes)

    # -- audio-aligned timing features ------------------------------------------------------

    def frame_features(self, timing: TimingMap, start_us: int, n_frames: int) -> np.ndarray:
        us = start_us + np.arange(n_frames + 1, dtype=np.float64) * (1e6 / MEL.fps)
        beats = beats_at_us(timing, us)
        meter_beats, meter_len = _meter_arrays(timing)
        mi = np.clip(np.searchsorted(meter_beats, beats, side="right") - 1, 0, None)
        measure = (beats - meter_beats[mi]) / meter_len[mi]
        tempo_beats, bpms = _tempo_arrays(timing)
        ti = np.clip(np.searchsorted(tempo_beats, beats, side="right") - 1, 0, None)
        beat_index, measure_index = np.floor(beats), np.floor(measure)
        tau = 2 * np.pi
        feats = np.stack(
            [
                np.sin(tau * beats[:-1]),
                np.cos(tau * beats[:-1]),
                np.sin(tau * measure[:-1]),
                np.cos(tau * measure[:-1]),
                np.log2(bpms[ti[:-1]] / 120.0),
                (np.diff(beat_index) > 0).astype(np.float64),
                (np.diff(measure_index) > 0).astype(np.float64),
            ],
            axis=1,
        )
        return feats.astype(np.float32)


# -- helpers ---------------------------------------------------------------------------


@cache
def build_tokenizer() -> UrfmTokenizer:
    return UrfmTokenizer()


def chart_lanes(chart: Chart) -> int | None:
    spec = registry.get(chart.mechanics)
    if spec.space.lanes:
        return spec.space.lanes
    value = chart.meta.get("lanes")
    return int(value) if value and value.isdigit() else None


def chart_tier(chart: Chart) -> int | None:
    d = chart.difficulty
    if d.tier is not None:
        return max(0, min(4, d.tier))
    label = (d.label or "").casefold()
    tier = None
    for pattern, value in _TIER_WORDS:
        if pattern.search(label):
            tier = value
    return tier


def playable_events(chart: Chart) -> list[Event]:
    """Interactive events on the preferred branch route (``master`` when present)."""
    spec = registry.get(chart.mechanics)
    routes = {
        b.id: PREFERRED_ROUTE if PREFERRED_ROUTE in b.routes else b.default for b in chart.branches
    }
    out = []
    for ev in chart.events:
        role = spec.role(ev.role)
        if role is None or not role.interactive:
            continue
        if ev.branch is not None and routes.get(ev.branch.segment) != ev.branch.route:
            continue
        out.append(ev)
    return out


def onset_rate(chartset: ChartSet, chart: Chart) -> float:
    """Distinct interactive onsets per second over the charted span (vectorized)."""
    timing = chartset.timing_of(chart)
    events = playable_events(chart)
    if len(events) < 2:
        return 0.0
    us = timing.us_array([e.time.beat for e in events], [e.time.residual_us for e in events])
    onsets = np.unique(np.round(us))
    if len(onsets) < 2:
        return 0.0
    return float(len(onsets) / max((onsets[-1] - onsets[0]) / 1e6, 1.0))


GRID_COVER = 0.9
"""A window's ``GRID`` is the coarsest subdivision that holds this share of its onsets."""


def window_grid(beats: list[Fraction] | list[float]) -> int:
    """Finest per-beat subdivision (one of ``GRID_VALUES``) needed by 90% of the onsets."""
    if not len(beats):
        return 0
    x = np.asarray([float(b) for b in beats]) % 1.0
    grids = np.asarray(GRID_VALUES[1:], dtype=np.float64)
    scaled = x[:, None] * grids[None]
    fits = np.abs(scaled - np.round(scaled)) < 1e-6  # [onsets, grids]
    covered = fits.mean(0) >= GRID_COVER - 1e-9
    return int(grids[np.argmax(covered)]) if covered.any() else GRID_VALUES[-1]


def window_rest(beats: list[Fraction] | list[float], start: Fraction, end: Fraction) -> int:
    """Share of the window's beats in rests of at least one beat, in tenths."""
    span = float(end - start)
    if span <= 0:
        return 0
    if not len(beats):
        return REST_BINS - 1
    edges = np.concatenate([[float(start)], np.sort([float(b) for b in beats]), [float(end)]])
    gaps = np.diff(edges)
    rest = gaps[gaps >= 1 - 1e-9].sum()
    return min(round(float(rest / span) * (REST_BINS - 1)), REST_BINS - 1)


def window_start_beat(timing: TimingMap, start_us: int) -> Fraction:
    """Beat of a window start, snapped to the 1/48 grid so positions stay on simple fractions."""
    return Fraction(round(timing.us_to_beat(start_us) * 48), 48)


def audio_windows(timing: TimingMap, duration_us: int) -> list[tuple[int, int]]:
    """Measure-aligned windows tiling the whole audio ``[0, duration_us)``.

    The single window definition for training and generation: intros, outros and long
    rests produce windows whose body is only ``WDENS EOS``, so the model learns silence too.
    """
    return list(_tile(timing, 0, duration_us - 1))


def windows(
    chartset: ChartSet, chart: Chart, duration_us: int | None = None
) -> Iterator[tuple[int, int]]:
    """Measure-aligned ``(start_us, end_us)`` windows tiling the chart without overlap.

    Each window starts on a measure boundary and ends where the next one starts,
    at most ``WINDOW_S`` later; a measure longer than the window is split at
    ``WINDOW_S``. Coverage spans the first to the last playable onset (or
    ``duration_us`` when given).
    """
    timing = chartset.timing_of(chart)
    events = playable_events(chart)
    if not events:
        return
    onsets = [timing.us(ev.time) for ev in events]
    first_us = min(onsets)
    last_us = (
        duration_us
        if duration_us is not None
        else max(timing.us(ev.end or ev.time) for ev in events)
    )
    yield from _tile(timing, first_us, last_us)


def _measure_floor(timing: TimingMap, beat: Fraction) -> Fraction:
    """Start beat of the measure containing ``beat`` (O(log meters), no enumeration)."""
    meter = timing.meter_at(beat)
    length = meter.measure_beats
    return meter.beat + math.floor((beat - meter.beat) / length) * length


def _tile(timing: TimingMap, first_us: int, last_us: int) -> Iterator[tuple[int, int]]:
    """Windows of at most ``WINDOW_S`` ending on measure starts, O(number of windows).

    Measures are located arithmetically instead of being enumerated, so gimmick tempi
    (osu! red lines at millions of BPM) cannot blow up time or memory.
    """
    window_us = int(WINDOW_S * 1e6)
    first_measure = _measure_floor(timing, timing.us_to_beat(first_us))
    start = min(round(timing.beat_to_us_exact(first_measure)), first_us)
    while start <= last_us:
        limit = start + window_us
        end = round(timing.beat_to_us_exact(_measure_floor(timing, timing.us_to_beat(limit))))
        if not start < end <= limit:
            end = limit
        yield start, end
        start = end


def beats_at_us(timing: TimingMap, us: np.ndarray) -> np.ndarray:
    """Vectorized float ``us -> beat``; with negative stops the later segment wins."""
    c = _compile(timing)
    knots = np.array([float(k) for k in c.knots])
    starts = np.array(
        [float(b + i + s) for b, i, s in zip(c.base, c.inclusive, c.strict, strict=True)]
    )
    rates = np.array([float(r) for r in c.rate])
    idx = np.searchsorted(starts, us, side="right") - 1
    before = idx < 0
    idx = np.clip(idx, 0, None)
    rate = rates[idx]
    safe = np.where(rate > 0, rate, 1.0)
    beats = np.where(rate > 0, knots[idx] + (us - starts[idx]) / safe, knots[idx])
    first = knots[0] - (float(c.base[0]) - us) / float(c.first_rate)
    return np.where(before, first, beats)


def _meter_arrays(timing: TimingMap) -> tuple[np.ndarray, np.ndarray]:
    meters = sorted(timing.meters, key=lambda m: m.beat) or [timing.meter_at(Fraction(0))]
    return (
        np.array([float(m.beat) for m in meters]),
        np.array([float(m.measure_beats) for m in meters]),
    )


def _tempo_arrays(timing: TimingMap) -> tuple[np.ndarray, np.ndarray]:
    tempos = sorted(timing.tempos, key=lambda t: t.beat)
    return np.array([float(t.beat) for t in tempos]), np.array([float(t.bpm) for t in tempos])


def _event_order(ev: Event) -> tuple:
    return (ev.time.beat, ev.primitive is not Primitive.HIT, ev.role, repr(ev.target), ev.end)


def _split_groups(v: Vocab, tokens: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for t in tokens:
        kind = v.kinds[t]
        if kind in ("EOS", "PAD"):
            break
        if kind in ("WDENS", "GRID", "REST"):  # window plan tokens, not events
            continue
        if kind in ("POS_BEAT", "POS_SAME") or not groups:
            groups.append([t])
        else:
            groups[-1].append(t)
    return groups


def _take_target(v: Vocab, rest: list[int]) -> tuple[Target, list[int]]:
    kind, value = v.kinds[rest[0]], v.values[rest[0]]
    if kind == "LANE":
        return Lane(value), rest[1:]
    if kind == "RING":
        return RingButton(value), rest[1:]
    if kind == "REGION":
        return TouchRegion(value), rest[1:]
    return Point2D(pt_x_value(value), pt_y_value(v.values[rest[1]])), rest[2:]


def _assign_groups(events: list[Event]) -> None:
    """Pair maimai star heads with slides starting on the same beat and button."""
    import msgspec

    next_group = 0
    heads: dict[tuple[Fraction, Target | None], int] = {}
    for i, ev in enumerate(events):
        if ev.role == "maimai:star":
            heads[(ev.time.beat, ev.target)] = next_group
            events[i] = msgspec.structs.replace(ev, group=next_group)
            next_group += 1
    for i, ev in enumerate(events):
        if ev.role == "maimai:slide":
            group = heads.get((ev.time.beat, ev.target))
            if group is not None:
                events[i] = msgspec.structs.replace(ev, group=group)
