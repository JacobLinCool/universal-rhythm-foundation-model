import glob
import itertools
from fractions import Fraction as F
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from urfm.codecs import get
from urfm.ir import MeterChange, TempoChange, TimingMap
from urfm.tokenize import FRAME_FEATURES, build_tokenizer, playable_events, windows
from urfm.tokenize.report import fixtures_report
from urfm.tokenize.vocab import Vocab

FIXTURES = {
    "tja": sorted(glob.glob("tests/fixtures/tja/*.tja")),
    "osu": sorted(glob.glob("tests/fixtures/osu/*.osu")),
    "stepmania": sorted(glob.glob("tests/fixtures/stepmania/*")),
    "simai": sorted(glob.glob("tests/fixtures/simai/*.txt")),
}

tok = build_tokenizer()
v = tok.vocab


def names(tokens):
    return [v.name(t) for t in tokens]


def test_vocab_is_deterministic():
    assert Vocab().tokens == v.tokens
    assert Vocab().digest == v.digest
    assert len(set(v.tokens)) == len(v.tokens) == tok.vocab_size


@pytest.fixture(scope="module")
def report():
    return fixtures_report(FIXTURES)


def test_beat_native_games_roundtrip_exactly(report):
    for mechanics, st in report.items():
        if mechanics.startswith("sm_") or mechanics == "taiko":
            assert st.encoded == st.events, mechanics
            assert st.exact == st.encoded, mechanics


def test_time_native_and_simai_losses_are_bounded(report):
    for mechanics, st in report.items():
        assert st.encoded == st.events, mechanics
        assert st.timing_err_ms_max < 5, mechanics
        assert st.end_err_ms_max < 5, mechanics
        assert st.point_err_px_max <= 4, mechanics
        assert st.within_1ms / st.encoded > 0.8, mechanics


def test_encoded_windows_satisfy_grammar():
    for codec_id, paths in FIXTURES.items():
        codec = get(codec_id)
        for path in paths:
            cs = codec.read_path(path)
            for chart in cs.charts:
                for start, end in windows(cs, chart):
                    body, _ = tok.encode_window(cs, chart, start, end, 10**6)
                    seq = [tok.bos_id, tok.sep_id, *body, tok.eos_id]
                    g = tok.grammar(chart.mechanics, None)
                    state = tok.grammar_state(chart.mechanics, [tok.bos_id])
                    for t in seq[1:]:
                        assert tok.mask(chart.mechanics, state)[t], (path, v.name(t), state)
                        state = g.step(state, t)


def test_allowed_rejects_illegal_tokens():
    taiko = tok.allowed(
        "taiko", [tok.bos_id, tok.sep_id, v.id("WDENS", 0), v.id("POS_BEAT", 0), v.id("POS_SUB", 0)]
    )
    assert taiko[v.id("PRIM", "hit")] and taiko[v.id("PRIM", "repeat")]
    assert not taiko[v.id("PRIM", "path")]
    after_role = tok.allowed(
        "taiko",
        [
            tok.bos_id,
            tok.sep_id,
            v.id("WDENS", 0),
            v.id("POS_BEAT", 0),
            v.id("POS_SUB", 0),
            v.id("PRIM", "hit"),
            v.id("ROLE", "taiko:don"),
        ],
    )
    assert not after_role[v.id("LANE", 0)]
    assert after_role[tok.eos_id] and after_role[v.id("POS_BEAT", 1)]
    four = [
        tok.bos_id,
        tok.sep_id,
        v.id("WDENS", 0),
        v.id("POS_BEAT", 0),
        v.id("POS_SUB", 0),
        v.id("PRIM", "hit"),
        v.id("ROLE", "sm:tap"),
    ]
    mask = tok.allowed("sm_dance_single", four)
    assert mask[v.id("LANE", 3)] and not mask[v.id("LANE", 7)]
    assert not mask[tok.eos_id]  # a lane target is required
    mania = [
        tok.bos_id,
        tok.sep_id,
        v.id("WDENS", 0),
        v.id("POS_BEAT", 0),
        v.id("POS_SUB", 0),
        v.id("PRIM", "hit"),
        v.id("ROLE", "mania:note"),
    ]
    assert tok.allowed("osu_mania", mania, lanes=4)[v.id("LANE", 3)]
    assert not tok.allowed("osu_mania", mania, lanes=4)[v.id("LANE", 4)]
    # mines are not interactive and never generated
    assert not mask[v.id("ROLE", "sm:mine")]
    with pytest.raises(ValueError):
        tok.grammar_state("taiko", [tok.bos_id, tok.eos_id])


def test_prefix_tokens():
    cs = get("osu").read_path("tests/fixtures/osu/mania_7k.osu")
    pre = names(tok.prefix(cs, cs.charts[0]))
    assert pre[:2] == ["GAME_osu_mania", "LANES_7"]
    assert pre[2] == "TIER_3" and pre[3].startswith("DENS_")
    cs = get("tja").read_path("tests/fixtures/tja/basic.tja")
    assert names(tok.prefix(cs, cs.charts[0]))[0] == "GAME_taiko"


def test_windows_tile_measures_without_overlap():
    cs = get("stepmania").read_path("tests/fixtures/stepmania/basic.sm")
    chart = cs.charts[0]
    spans = list(windows(cs, chart))
    assert spans and all(a[1] == b[0] for a, b in pairwise(spans))
    assert all(e - s <= 16_000_000 for s, e in spans)
    total = sum(len(tok.encode_groups(cs, chart, s, e)) for s, e in spans)
    assert total == len(playable_events(chart))


def test_frame_features_phase():
    timing = TimingMap(tempos=(TempoChange(F(0), F(120)),), meters=(MeterChange(F(0), 4, 4),))
    feats = tok.frame_features(timing, 0, 200)
    assert feats.shape == (200, len(FRAME_FEATURES)) and feats.dtype == np.float32
    # 120 BPM: a beat every 0.5 s; frame 0 is on a downbeat
    assert feats[0, 1] == pytest.approx(1.0) and feats[0, 3] == pytest.approx(1.0)
    assert feats[:, 4] == pytest.approx(0.0)
    fps = 22050 / 256
    beat_frames = np.flatnonzero(feats[:, 5])
    assert np.allclose(np.diff(beat_frames), 0.5 * fps, atol=1)
    downbeats = np.flatnonzero(feats[:, 6])
    assert np.allclose(np.diff(downbeats), 2.0 * fps, atol=1)


def test_decode_ignores_malformed_groups():
    timing = TimingMap.constant(120)
    good = [v.id("POS_BEAT", 1), v.id("POS_SUB", 0), v.id("PRIM", "hit"), v.id("ROLE", "taiko:don")]
    bad = [v.id("POS_BEAT", 2), v.id("POS_SUB", 0), v.id("ROLE", "taiko:don")]
    events = tok.decode(timing, "taiko", [*bad, *good, tok.eos_id], 0)
    assert len(events) == 1 and events[0].time.beat == 1


def _dense_taiko(n: int = 400):
    from fractions import Fraction

    from urfm.ir import Chart, ChartSet, Event, Primitive, Time, TimingMap

    events = tuple(
        Event(time=Time(Fraction(8) + Fraction(i, 4)), primitive=Primitive.HIT, role="taiko:don")
        for i in range(n)
    )
    return ChartSet(
        timing=TimingMap.constant(150), charts=(Chart(mechanics="taiko", events=events),)
    )


def test_context_is_shared_and_grid_aligned():
    from urfm.train.data import make_sample, prepare
    from urfm.train.generate import Job, _context

    cs = _dense_taiko()
    chart = cs.charts[0]
    timing = cs.timing
    mel = np.zeros((int(60 * 86.2), 128), np.uint8)
    prep = prepare(tok, cs, chart, mel.shape[0])
    start, end = prep.windows[3]
    sample = make_sample(tok, cs, prep, mel, (start, end), 1024)
    head = len(prep.prefix) + 1
    train_ctx = list(sample.tokens[head : sample.sep])
    job = Job(mel=mel, timing=timing, mechanics="taiko", prefix=[])
    job.events = [ev for ev in chart.events if timing.us(ev.time) < start]
    assert train_ctx == _context(tok, job, start) and train_ctx
    subs = [v.values[t] for t in train_ctx if v.kinds[t] == "POS_SUB"]
    assert subs and all(v.values[t] is not None for t in train_ctx if v.kinds[t] == "POS_SUB")
    on_beat = [s for s in subs if s == 0]
    assert on_beat, "quarter-note events on integral beats must keep POS_SUB 0 in context"


def test_truncation_at_event_boundary_is_counted():
    from urfm.tokenize import take_drops

    cs = _dense_taiko()
    chart = cs.charts[0]
    take_drops()
    body, truncated = tok.encode_window(cs, chart, 0, 30_000_000, 50)
    assert truncated and len(body) <= 50
    g = tok.grammar("taiko", None)
    state = g.step(g.step(g.initial(), tok.sep_id), body[0])
    for t in body[1:]:
        state = g.step(state, t)
    assert tok.eos_id in g.tokens(state)  # ends exactly after a complete event
    assert take_drops().get("taiko/truncated", 0) > 0


def test_audio_windows_cover_silence():
    from urfm.tokenize import audio_windows

    cs = _dense_taiko(8)
    wins = audio_windows(cs.timing, 60_000_000)
    assert wins[0][0] <= 0 and wins[-1][1] >= 60_000_000 - 1
    empty = [w for w in wins if not tok.encode(cs, cs.charts[0], *w)]
    assert empty  # windows without notes exist and are trained as WDENS EOS


def test_grammar_enforces_single_action_games():
    same = v.id("POS_SAME")
    g = tok.grammar("taiko", None)
    after_event = g.run(
        [
            tok.sep_id,
            v.id("WDENS", 3),
            v.id("POS_BEAT", 0),
            v.id("POS_SUB", 0),
            v.id("PRIM", "hit"),
            v.id("ROLE", "taiko:don"),
        ]
    )
    assert not g.mask(after_event)[same]
    m = tok.grammar("osu_mania", 4)
    after_note = m.run(
        [
            tok.sep_id,
            v.id("WDENS", 3),
            v.id("POS_BEAT", 0),
            v.id("POS_SUB", 0),
            v.id("PRIM", "hit"),
            v.id("ROLE", "mania:note"),
            v.id("LANE", 0),
        ]
    )
    assert m.mask(after_note)[same]


def test_window_starting_on_red_line_keeps_grid():
    """osu! red lines become negative stops; a window starting exactly on one must use the
    beat *after* the jump as origin, or every position in the window shifts off-grid."""
    from fractions import Fraction

    from urfm.ir import Chart, ChartSet, Event, Primitive, Stop, TempoChange, Time, TimingMap

    # 120 BPM; a red line re-anchors the grid at beat 8, 100 ms earlier than the old grid
    timing = TimingMap(
        tempos=(TempoChange(Fraction(0), Fraction(120)),),
        stops=(Stop(Fraction(8), Fraction(-100_000), before=True),),  # as the osu! codec
    )
    events = tuple(
        Event(time=Time(Fraction(8) + Fraction(i, 2)), primitive=Primitive.HIT, role="taiko:don")
        for i in range(8)
    )
    cs = ChartSet(timing=timing, charts=(Chart(mechanics="osu_taiko", events=events),))
    start = timing.us(Time(Fraction(8)))
    assert timing.us_to_beat(start) == 8
    toks = tok.encode(cs, cs.charts[0], start, start + 8_000_000)
    pos = [
        (v.values[a], v.values[b])
        for a, b in itertools.pairwise(toks)
        if v.kinds[a] == "POS_BEAT" and v.kinds[b] == "POS_SUB"
    ]
    assert pos[0] == (0, 0) and len(pos) == 8


def test_windows_survive_gimmick_tempo():
    """osu! maps use red lines at absurd BPM for visual tricks; tiling must stay O(windows)."""
    import time
    from fractions import Fraction

    from urfm.ir import MeterChange, TempoChange, TimingMap
    from urfm.tokenize import audio_windows

    timing = TimingMap(
        tempos=(
            TempoChange(Fraction(0), Fraction(180)),
            TempoChange(Fraction(64), Fraction(10**7)),
        ),
        meters=(MeterChange(Fraction(0), 4, 4), MeterChange(Fraction(64), 1, 4)),
    )
    t0 = time.perf_counter()
    wins = audio_windows(timing, 240_000_000)
    assert time.perf_counter() - t0 < 1.0
    assert wins[-1][1] >= 240_000_000 - 1 and all(e - s <= 16_000_000 for s, e in wins)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(wins))


def test_compiled_fast_path_matches_fraction_path():
    """The integer-tick training path must produce exactly the reference tokens."""
    import numpy as np

    from urfm.train.data import make_sample, prepare

    windows = 0
    for codec_id, paths in FIXTURES.items():
        codec = get(codec_id)
        for path in paths:
            cs = codec.read_path(path)
            for chart in cs.charts:
                if not chart.events:
                    continue
                mel = np.zeros((int(240 * 86.2), 128), np.uint8)
                fast = prepare(tok, cs, chart, mel.shape[0])
                ref = prepare(tok, cs, chart, mel.shape[0])
                ref.compiled = None
                for w in fast.windows:
                    a = make_sample(tok, cs, fast, mel, w, 2048)
                    b = make_sample(tok, cs, ref, mel, w, 2048)
                    assert (a.tokens == b.tokens).all() and a.sep == b.sep, (path, w)
                    windows += 1
    assert windows > 500


def _rebuild_with_controls(cs, chart):
    """encode every audio window -> decode_full -> chart with controls and meters."""
    import msgspec

    from urfm.ir import Chart
    from urfm.tokenize import audio_windows

    timing = cs.timing_of(chart)
    last = max(timing.us(e.end or e.time) for e in chart.events)
    events, controls, meters = [], [], []
    for start, end in audio_windows(timing, last + 4_000_000):
        body, truncated = tok.encode_window(cs, chart, start, end, 10**6)
        assert not truncated
        ev, ctl, met = tok.decode_full(timing, chart.mechanics, body, start)
        events += ev
        controls += ctl
        meters += met
    first = min((m.beat for m in meters), default=None)
    kept = [m for m in timing.meters if first is None or m.beat < first]
    rebuilt_timing = msgspec.structs.replace(
        timing, meters=tuple(sorted([*kept, *meters], key=lambda m: m.beat))
    )
    return Chart(
        mechanics=chart.mechanics,
        difficulty=chart.difficulty,
        events=tuple(sorted(events, key=lambda e: e.time)),
        controls=tuple(sorted(controls, key=lambda c: c.time)),
        timing=rebuilt_timing,
    )


def test_controls_and_meters_roundtrip_through_tokens_and_tja():
    """Scroll ramps, #MEASURE barline tricks, GOGO and bar lines survive tokens -> TJA."""
    from urfm.ir import ChartSet
    from urfm.tokenize.vocab import SCROLL_GRID, scroll_bin

    codec = get("tja")
    path = sorted(Path(__file__).parent.glob("fixtures/tja/scroll_ramp_measure_trick.tja"))[0]
    cs = codec.read_path(path)
    chart = cs.charts[0]
    rebuilt = _rebuild_with_controls(cs, chart)
    out = ChartSet(timing=cs.timing, charts=(rebuilt,), song=cs.song)
    back = codec.read(codec.write(out), source="rebuilt.tja")
    got = back.charts[0]

    def ctl(c):
        return [
            (
                cc.time.beat,
                cc.kind,
                SCROLL_GRID[scroll_bin(cc.value)] if cc.kind == "scroll" else cc.value,
            )
            for cc in c.controls
            if cc.kind in ("scroll", "highlight", "barline")
        ]

    # same controls per beat (order at one beat is not meaningful); scroll after quantization
    assert sorted(ctl(got), key=repr) == sorted(ctl(chart), key=repr)
    meters = [(m.beat, m.numerator, m.denominator) for m in back.timing_of(got).meters]
    assert meters == [(m.beat, m.numerator, m.denominator) for m in cs.timing_of(chart).meters]
    assert [(e.time, e.role) for e in got.events] == [(e.time, e.role) for e in chart.events]


def test_control_tokens_follow_mechanics_spec():
    ctrl = {name: v.id("CTRL", name) for name in ("scroll", "highlight_on", "barline_on", "meter")}
    pos = [tok.sep_id, v.id("WDENS", 3), v.id("POS_BEAT", 0), v.id("POS_SUB", 0)]
    taiko = tok.allowed("taiko", pos)
    assert all(taiko[i] for i in ctrl.values())
    std = tok.allowed("osu_std", pos)  # osu:sv maps to the scroll family; std has bar lines
    assert std[ctrl["scroll"]] and std[ctrl["highlight_on"]]
    mai = tok.allowed("maimai_dx", pos)  # maimai declares scroll only
    assert mai[ctrl["scroll"]] and not mai[ctrl["highlight_on"]] and not mai[ctrl["meter"]]
    same = tok.allowed("osu_mania", [*pos[:2], v.id("POS_SAME")], lanes=4)
    assert not same[ctrl["scroll"]]  # control groups always carry an explicit position


def test_compiled_path_counts_and_encodes_controls_like_fraction_path():
    import numpy as np

    from urfm.tokenize import take_drops
    from urfm.train.data import make_sample, prepare

    codec = get("tja")
    controls_seen = 0
    for path in sorted(Path(__file__).parent.glob("fixtures/tja/*.tja")):
        cs = codec.read_path(path)
        for chart in cs.charts:
            if not chart.events:
                continue
            mel = np.zeros((int(240 * 86.2), 128), np.uint8)
            fast, ref = prepare(tok, cs, chart, mel.shape[0]), prepare(tok, cs, chart, mel.shape[0])
            ref.compiled = None
            for w in fast.windows:
                take_drops()
                a = make_sample(tok, cs, fast, mel, w, 4096)
                drops_fast = take_drops()
                b = make_sample(tok, cs, ref, mel, w, 4096)
                assert (a.tokens == b.tokens).all() and drops_fast == take_drops(), (path, w)
                controls_seen += sum(v.kinds[t] == "CTRL" for t in a.tokens.tolist())
    assert controls_seen > 20
