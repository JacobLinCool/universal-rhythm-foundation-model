"""How much information the tokenizer loses: encode -> decode against the IR."""

from __future__ import annotations

from statistics import fmean

import msgspec
import numpy as np

from urfm.ir.chart import ChartSet
from urfm.ir.events import Event
from urfm.ir.targets import Point2D
from urfm.ir.timing import TimingMap
from urfm.tokenize.tokenizer import UrfmTokenizer, build_tokenizer, playable_events, windows


class TokenStats(msgspec.Struct, kw_only=True):
    mechanics: str
    charts: int = 0
    events: int = 0
    encoded: int = 0
    exact: int = 0
    """Decoded event has the same beat, primitive, role, modifiers, target and end beat."""
    within_1ms: int = 0
    """Same primitive/role/modifiers, onset and end within 1 ms, point within 6 px."""
    timing_err_ms_max: float = 0.0
    timing_err_ms_sum: float = 0.0
    end_err_ms_max: float = 0.0
    point_err_px_max: float = 0.0
    path_points: int = 0
    path_points_kept: int = 0
    windows: int = 0
    tokens: list[int] = []
    events_per_window: list[int] = []

    def summary(self) -> dict[str, float | int | str]:
        """One table row; errors are measured in absolute time through the chart's timing."""
        n = max(self.encoded, 1)
        toks = np.array(self.tokens or [0])
        return {
            "mechanics": self.mechanics,
            "charts": self.charts,
            "events": self.events,
            "encoded %": 100 * self.encoded / max(self.events, 1),
            "exact %": 100 * self.exact / n,
            "≤1ms %": 100 * self.within_1ms / n,
            "mean err ms": self.timing_err_ms_sum / n,
            "max err ms": self.timing_err_ms_max,
            "max end err ms": self.end_err_ms_max,
            "max pt err px": self.point_err_px_max,
            "path pts kept %": (
                100 * self.path_points_kept / self.path_points if self.path_points else "-"
            ),
            "windows": self.windows,
            "events/win": fmean(self.events_per_window or [0]),
            "tokens/win": float(toks.mean()),
            "tokens/win p95": float(np.percentile(toks, 95)),
        }


def roundtrip_report(
    chartset: ChartSet,
    tokenizer: UrfmTokenizer | None = None,
    into: dict[str, TokenStats] | None = None,
) -> dict[str, TokenStats]:
    tok = tokenizer or build_tokenizer()
    stats = into if into is not None else {}
    for chart in chartset.charts:
        st = stats.setdefault(chart.mechanics, TokenStats(mechanics=chart.mechanics))
        st.charts += 1
        timing = chartset.timing_of(chart)
        st.events += len(playable_events(chart))
        for start, end in windows(chartset, chart):
            all_pairs = tok.encode_pairs(chartset, chart, start, end)
            tokens = [t for _, group in all_pairs for t in group]
            pairs = [(x, g) for x, g in all_pairs if isinstance(x, Event)]
            decoded = tok.decode(timing, chart.mechanics, tokens, start)
            st.windows += 1
            st.tokens.append(len(tokens))
            st.events_per_window.append(len(pairs))
            st.encoded += len(pairs)
            for (orig, _), dec in zip(pairs, decoded, strict=False):
                _compare(st, timing, orig, dec)
    return stats


def _compare(st: TokenStats, timing: TimingMap, a: Event, b: Event) -> None:
    err = abs(timing.us(a.time) - timing.us(b.time)) / 1000
    end_err = 0.0
    if a.end is not None and b.end is not None:
        end_err = abs(timing.us(a.end) - timing.us(b.end)) / 1000
    pt_err = 0.0
    if isinstance(a.target, Point2D) and isinstance(b.target, Point2D):
        pt_err = max(abs(a.target.x - b.target.x), abs(a.target.y - b.target.y))
    same_kind = (a.primitive, a.role, a.modifiers) == (b.primitive, b.role, b.modifiers)
    st.timing_err_ms_max = max(st.timing_err_ms_max, err)
    st.timing_err_ms_sum += err
    st.end_err_ms_max = max(st.end_err_ms_max, end_err)
    st.point_err_px_max = max(st.point_err_px_max, pt_err)
    a_end = a.end.beat if a.end else None
    b_end = b.end.beat if b.end else None
    if same_kind and a.time.beat == b.time.beat and a.target == b.target and a_end == b_end:
        st.exact += 1
    target_ok = a.target == b.target or pt_err <= 6
    if same_kind and err <= 1 and end_err <= 1 and target_ok:
        st.within_1ms += 1
    n_a = sum(len(s.points) for s in a.path)
    st.path_points += n_a
    st.path_points_kept += min(n_a, sum(len(s.points) for s in b.path))


def report_table(stats: dict[str, TokenStats]) -> str:
    rows = [s.summary() for _, s in sorted(stats.items())]
    if not rows:
        return ""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [f"{v:.3g}" if isinstance(v, float) else str(v) for v in r.values()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fixtures_report(paths_by_codec: dict[str, list[str]]) -> dict[str, TokenStats]:
    from urfm.codecs import get

    stats: dict[str, TokenStats] = {}
    for codec_id, paths in paths_by_codec.items():
        codec = get(codec_id)
        for path in paths:
            roundtrip_report(codec.read_path(path), into=stats)
    return stats
