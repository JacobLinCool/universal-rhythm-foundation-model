"""Data pipeline, trainer and generator tests with a tiny fake tokenizer (CPU, fast)."""

from __future__ import annotations

import itertools
import json
from fractions import Fraction
from pathlib import Path

import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from urfm.codecs import get
from urfm.data.features import MEL
from urfm.data.records import SongRecord, arrow_schema
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event, Primitive
from urfm.ir.serialize import to_msgpack
from urfm.ir.time import Time
from urfm.ir.timing import TimingMap
from urfm.model import ChartModel, preset
from urfm.train.data import (
    N_FRAMES,
    WindowStream,
    build_index,
    collate,
    fixed_windows,
    make_sample,
    shard_files,
)
from urfm.train.generate import Job, generate, song_windows
from urfm.train.train import TrainConfig, parse_args, train

FIXTURES = Path(__file__).parent / "fixtures" / "tja"


class FakeTokenizer:
    """Event = [POS (quarter-beat index from window start), ROLE]."""

    pad_id, bos_id, sep_id, eos_id = 0, 1, 2, 3
    game = 4
    pos0, n_pos = 5, 128
    roles = ("taiko:don", "taiko:ka")
    vocab_size = 5 + 128 + 2

    def prefix(self, chartset, chart):
        return [self.game]

    def conditioning(self, mechanics, *, tier=None, onsets_per_s=None, lanes=None):
        return [self.game]

    def encode(self, chartset, chart, start_us, end_us):
        timing = chartset.timing_of(chart)
        b0 = timing.us_to_beat(start_us)
        out = []
        for ev in chart.events:
            if start_us <= timing.us(ev.time) < end_us:
                q = min(max(round((ev.time.beat - b0) * 4), 0), self.n_pos - 1)
                role = ev.role if ev.role in self.roles else self.roles[0]
                out += [self.pos0 + q, self.pos0 + self.n_pos + self.roles.index(role)]
        return out

    def encode_groups(self, chartset, chart, start_us, end_us):
        flat = self.encode(chartset, chart, start_us, end_us)
        return [flat[i : i + 2] for i in range(0, len(flat), 2)]

    def context(self, chartset, chart, start_us, cache=None, compiled=None):
        timing = chartset.timing_of(chart)
        lo = round(timing.beat_to_us_exact(timing.us_to_beat(start_us) - 16))
        return [t for g in self.encode_groups(chartset, chart, lo, start_us)[-8:] for t in g]

    def encode_window(self, chartset, chart, start_us, end_us, room, cache=None, compiled=None):
        groups = self.encode_groups(chartset, chart, start_us, end_us)
        out: list[int] = []
        for i, g in enumerate(groups):
            if len(out) + len(g) > room:
                return out, i < len(groups)
            out += g
        return out, False

    def window_density(self, onsets_per_s):
        return self.game

    def windows(self, chartset, chart, duration_us=None):
        from urfm.tokenize import windows

        return list(windows(chartset, chart, duration_us))

    def decode(self, timing, mechanics, tokens, start_us, lanes=None):
        b0 = timing.us_to_beat(start_us)
        events = []
        for pos, role in zip(tokens[::2], tokens[1::2], strict=False):
            q = pos - self.pos0
            r = role - self.pos0 - self.n_pos
            if 0 <= q < self.n_pos and 0 <= r < len(self.roles):
                beat = Fraction(round((b0 + Fraction(q, 4)) * 48), 48)
                events.append(Event(time=Time(beat), primitive=Primitive.HIT, role=self.roles[r]))
        return events

    def allowed(self, mechanics, tokens):
        mask = np.zeros(self.vocab_size, dtype=bool)
        if len(tokens) % 2 == 0:
            mask[self.pos0 : self.pos0 + self.n_pos] = True
            mask[self.eos_id] = True
        else:
            mask[self.pos0 + self.n_pos :] = True
        return mask

    def frame_features(self, timing, start_us, n_frames):
        us = start_us + np.arange(n_frames) * (1_000_000 / MEL.fps)
        beats = np.array(
            [float(timing.us_to_beat(round(u))) for u in us[:: max(1, n_frames // 64)]]
        )
        beats = np.interp(
            np.arange(n_frames), np.arange(0, n_frames, max(1, n_frames // 64))[: len(beats)], beats
        )
        phase = 2 * np.pi * (beats % 1)
        return np.stack([np.sin(phase), np.cos(phase)], 1).astype(np.float32)


def taiko_chartset() -> ChartSet:
    t = TimingMap.constant(150)
    events = tuple(
        Event(
            time=Time(Fraction(i, 2)),
            primitive=Primitive.HIT,
            role="taiko:don" if i % 3 else "taiko:ka",
        )
        for i in range(160)
    )
    return ChartSet(timing=t, charts=(Chart(mechanics="taiko", events=events),))


def write_shards(root: Path, n_songs: int = 3) -> Path:
    rng = np.random.default_rng(0)
    cs = taiko_chartset()
    fixture = get("tja").read_path(sorted(FIXTURES.glob("*.tja"))[0])
    for split in ("train", "validation"):
        rows = []
        for i in range(n_songs):
            frames = int(40 * MEL.fps)
            mel = rng.integers(0, 256, (frames, MEL.n_mels), dtype=np.uint8)
            chartset = cs if i % 2 == 0 else fixture
            rec = SongRecord(
                song_id=f"t:{split}{i}",
                source="taiko1000",
                split=split,
                group=f"g{i}",
                audio_sha256="x",
                duration_s=40.0,
                mel_version=MEL.version,
                mel_frames=frames,
                mel=mel.tobytes(),
                chartset=to_msgpack(chartset),
                mechanics=["taiko"],
                n_charts=len(chartset.charts),
                n_events=sum(len(c.events) for c in chartset.charts),
            )
            rows.append(msgspec.structs.asdict(rec))
        d = root / "data" / "taiko1000"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=arrow_schema()),
            d / f"{split}-00000.parquet",
            row_group_size=2,
        )
    return root


@pytest.fixture
def shards(tmp_path):
    return write_shards(tmp_path)


def test_sample_layout_and_mask():
    tok = FakeTokenizer()
    cs = taiko_chartset()
    mel = np.zeros((int(40 * MEL.fps), MEL.n_mels), dtype=np.uint8)
    s = make_sample(tok, cs, cs.charts[0], mel, (4_000_000, 16_000_000), 1024)
    assert s.mel.shape == (N_FRAMES, MEL.n_mels) and s.mel.dtype == np.uint8
    assert s.tokens[0] == tok.game and s.tokens[1] == tok.bos_id
    assert s.tokens[s.sep] == tok.sep_id and s.tokens[-1] == tok.eos_id
    assert s.sep > 2  # previous-window context is present
    batch = collate([s, s], tok.pad_id)
    m = batch["loss_mask"][0]
    assert not m[: s.sep + 1].any() and m[s.sep + 1 : len(s.tokens)].all()
    assert batch["tokens"].shape[1] in (128, 256, 384, 512, 768, 1024)


def test_truncation_drops_context_then_body():
    tok = FakeTokenizer()
    cs = taiko_chartset()
    mel = np.zeros((int(40 * MEL.fps), MEL.n_mels), dtype=np.uint8)
    s = make_sample(tok, cs, cs.charts[0], mel, (4_000_000, 16_000_000), 40)
    assert s.truncated and len(s.tokens) <= 40
    assert s.sep == 2  # context dropped entirely


def test_index_balances_mechanics(shards):
    index = build_index(shard_files(shards, ["taiko1000"], "train"))
    assert index.counts()["taiko"] >= 2
    assert all(c.weight > 0 for c in index.charts)


def test_stream_yields_batches(shards):
    tok = FakeTokenizer()
    index = build_index(shard_files(shards, ["taiko1000"], "train"))
    batch = next(iter(WindowStream(index, tok, batch_size=4, max_dec_len=512, buffer_batches=2)))
    assert batch["mel"].dtype == torch.uint8 and batch["mel"].shape[0] == 4
    assert batch["feats"].shape == (4, N_FRAMES, 2)
    val = fixed_windows(build_index(shard_files(shards, None, "validation")), tok, 512, limit=5)
    assert 0 < len(val) <= 5


def test_train_smoke(shards, tmp_path):
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=5,
        warmup=2,
        workers=0,
        compile=False,
        device="cpu",
        log_every=5,
        eval_every=5,
        eval_windows=4,
        ckpt_every=5,
        out=str(tmp_path / "run"),
        max_dec_len=512,
    )
    last = train(cfg, tok=FakeTokenizer())
    assert last["step"] == 5 and np.isfinite(last["loss"]) and "val_loss" in last
    assert (tmp_path / "run" / "latest.pt").exists()
    lines = (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    assert json.loads(lines[-1])["samples_per_s"] > 0


def test_cli_overrides():
    cfg = parse_args(
        [
            "--size",
            "15m",
            "--batch-size",
            "32",
            "--compile",
            "false",
            "--sources",
            "osu",
            "taiko1000",
        ]
    )
    assert cfg.size == "15m" and cfg.batch_size == 32 and cfg.compile is False
    assert cfg.sources == ["osu", "taiko1000"]


def test_generate_is_grammar_valid_and_tiles_song():
    tok = FakeTokenizer()
    torch.manual_seed(0)
    model = ChartModel(preset("tiny", vocab_size=tok.vocab_size, frame_feat_dim=2, max_dec_len=96))
    timing = TimingMap.constant(150)
    frames = int(40 * MEL.fps)
    jobs = [
        Job(
            mel=np.zeros((frames, MEL.n_mels), np.uint8),
            timing=timing,
            mechanics="taiko",
            prefix=[tok.game],
        )
        for _ in range(2)
    ]
    wins = song_windows(timing, 40_000_000)
    assert wins[0][0] <= 0 and wins[-1][1] >= 40_000_000 - 1
    assert all(a[1] == b[0] for a, b in itertools.pairwise(wins))
    assert all(e - s <= 16_000_000 for s, e in wins)
    charts = generate(model, tok, jobs, device="cpu", temperature=1.0, top_p=1.0)
    assert len(charts) == 2
    for chart in charts:
        assert all(e.role in tok.roles for e in chart.events)
        times = [timing.us(e.time) for e in chart.events]
        assert times == sorted(times)
        assert all(0 <= t < 40_000_000 + 16_000_000 for t in times)


def test_real_tokenizer_end_to_end(shards, tmp_path):
    tokenize = pytest.importorskip("urfm.tokenize")
    tok = tokenize.build_tokenizer()
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=3,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=3,
        eval_every=3,
        eval_windows=4,
        ckpt_every=3,
        out=str(tmp_path / "run"),
        max_dec_len=512,
    )
    assert np.isfinite(train(cfg, tok=tok)["loss"])
    model = ChartModel(
        preset(
            "tiny",
            vocab_size=tok.vocab_size,
            frame_feat_dim=tok.frame_features(TimingMap.constant(120), 0, 2).shape[1],
            max_dec_len=256,
        )
    )
    cs = taiko_chartset()
    job = Job(
        mel=np.zeros((int(20 * MEL.fps), MEL.n_mels), np.uint8),
        timing=cs.timing,
        mechanics="taiko",
        prefix=tok.prefix(cs, cs.charts[0]),
    )
    (chart,) = generate(model, tok, [job], device="cpu", temperature=1.0, top_p=1.0)
    from urfm.ir.validate import validate
    from urfm.train.generate import to_chartset, write_native

    out = to_chartset(chart, cs.timing)
    assert not [i for i in validate(out) if i.code.startswith(("role", "target", "end"))]
    write_native(out, tmp_path / "gen.tja")
    assert (tmp_path / "gen.tja").read_text(encoding="utf-8").count("#START") == 1


def test_tier_table_is_per_game():
    from urfm.train.tiers import TierTable

    rates = {
        "taiko": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "osu_mania": [5.0, 10.0, 15.0, 20.0, 25.0] * 10,
    }
    table = TierTable.build(rates)
    assert table.tier("taiko", 1.0) == 0 and table.tier("taiko", 5.0) == 4
    assert table.tier("osu_mania", 5.0) == 0  # 5 onsets/s is easy mania but hard taiko
    assert table.tier("unknown", 3.0) is None
    assert table.density("osu_mania", 4) > table.density("taiko", 4)


# -- song context ------------------------------------------------------------------


def test_song_summary_and_marker():
    from urfm.train.data import SONG_BLOCK_US, SONG_MAX, SONG_POOL, song_marker, song_summary

    mel = np.zeros((SONG_POOL * 3 + 10, MEL.n_mels), np.uint8)
    mel[:SONG_POOL] = 200
    mel[SONG_POOL * 3 :] = 90
    blocks, starts = song_summary(mel)
    assert blocks.shape == (4, MEL.n_mels) and blocks.dtype == np.uint8
    assert blocks[0, 0] == 200 and blocks[1, 0] == 0 and blocks[3, 0] == 90  # partial block mean
    marker = song_marker(starts, SONG_BLOCK_US, 2 * SONG_BLOCK_US + 1)
    assert marker.tolist() == [0, 1, 1, 0]
    long = np.zeros((SONG_POOL * (SONG_MAX * 2), MEL.n_mels), np.uint8)
    blocks, starts = song_summary(long)
    assert len(blocks) == SONG_MAX and np.all(np.diff(starts) > 0)
    # a window inside a subsampling gap still marks its nearest block
    assert song_marker(starts, int(starts[5]) + SONG_BLOCK_US + 1, int(starts[6]) - 1).sum() == 1


def test_samples_carry_song_and_collate_pads():
    tok = FakeTokenizer()
    cs = taiko_chartset()
    mel = np.zeros((int(40 * MEL.fps), MEL.n_mels), dtype=np.uint8)
    s = make_sample(tok, cs, cs.charts[0], mel, (4_000_000, 16_000_000), 1024)
    assert s.song is not None and s.song_marker is not None and s.song_marker.sum() >= 1
    batch = collate([s, s], tok.pad_id)
    assert batch["song"].shape == (2, 256, MEL.n_mels) and batch["song"].dtype == torch.uint8
    assert batch["song_len"].tolist() == [len(s.song)] * 2


def test_train_smoke_with_song_context(shards, tmp_path):
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=3,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=3,
        eval_every=3,
        eval_windows=4,
        ckpt_every=3,
        out=str(tmp_path / "run"),
        max_dec_len=512,
        song_context=True,
    )
    last = train(cfg, tok=FakeTokenizer())
    assert last["step"] == 3 and np.isfinite(last["loss"]) and np.isfinite(last["val_loss"])
    state = torch.load(tmp_path / "run" / "latest.pt", weights_only=False)
    assert state["model_config"]["song_context"] is True
    assert any(k.startswith("song_") for k in state["model"])


def test_generate_with_song_context():
    tok = FakeTokenizer()
    torch.manual_seed(0)
    model = ChartModel(
        preset(
            "tiny", vocab_size=tok.vocab_size, frame_feat_dim=2, max_dec_len=96, song_context=True
        )
    )
    timing = TimingMap.constant(150)
    jobs = [
        Job(
            mel=np.zeros((int(40 * MEL.fps), MEL.n_mels), np.uint8),
            timing=timing,
            mechanics="taiko",
            prefix=[tok.game],
        )
    ]
    (chart,) = generate(model, tok, jobs, device="cpu", temperature=1.0, top_p=1.0)
    assert jobs[0].song is not None
    assert all(e.role in tok.roles for e in chart.events)


def test_init_from_checkpoint(shards, tmp_path):
    base = dict(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=2,
        eval_every=2,
        eval_windows=4,
        ckpt_every=2,
        max_dec_len=512,
    )
    train(TrainConfig(**base, steps=2, out=str(tmp_path / "a")), tok=FakeTokenizer())
    ckpt = tmp_path / "a" / "latest.pt"
    last = train(
        TrainConfig(**base, steps=2, out=str(tmp_path / "b"), init=str(ckpt)), tok=FakeTokenizer()
    )
    assert np.isfinite(last["loss"])
    with pytest.raises(ValueError):
        train(
            TrainConfig(**{**base, "size": "5m"}, steps=1, out=str(tmp_path / "c"), init=str(ckpt)),
            tok=FakeTokenizer(),
        )


def test_r4_conditions_texture_augment_end_to_end(shards, tmp_path):
    """Continuous conditions, texture song context and tempo/octave augmentation train and
    generate with the real tokenizer; plans carry GRID/REST tokens."""
    from urfm.tokenize import build_tokenizer
    from urfm.train.generate import reference_inputs
    from urfm.train.train import load_checkpoint

    tok = build_tokenizer()
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=3,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=3,
        eval_every=3,
        eval_windows=4,
        ckpt_every=3,
        out=str(tmp_path / "run"),
        max_dec_len=512,
        song_context=True,
        cond=True,
        texture=True,
        canonical=True,
        p_speed=1.0,
        p_local=1.0,
        p_octave=0.5,
    )
    assert np.isfinite(train(cfg, tok=tok)["loss"])
    model, meta = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert model.cfg.cond_dim and model.cfg.song_feat_dim
    cs = taiko_chartset()
    prefix, cond = reference_inputs(model, meta, tok, cs, cs.charts[0])
    assert tok.vocab.id("COND") in prefix and cond is not None
    job = Job(
        mel=np.zeros((int(20 * MEL.fps), MEL.n_mels), np.uint8),
        timing=cs.timing,
        mechanics="taiko",
        prefix=prefix,
        cond=cond,
        target_rate=4.0,
    )
    (chart,) = generate(model, tok, [job], device="cpu", temperature=1.0, plan="calibrated")
    assert chart.mechanics == "taiko"


def test_rl_step_runs(shards, tmp_path):
    """One GRPO step on a tiny condition model: rollouts, rewards, update, checkpoint."""
    from urfm.tokenize import build_tokenizer
    from urfm.train.rl import RLConfig, reward
    from urfm.train.rl import train as rl_train

    tok = build_tokenizer()
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=2,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=2,
        eval_every=2,
        eval_windows=4,
        ckpt_every=2,
        out=str(tmp_path / "sft"),
        max_dec_len=512,
        song_context=True,
        cond=True,
        texture=True,
        canonical=True,
    )
    train(cfg, tok=tok)
    last = rl_train(
        RLConfig(
            data=str(shards),
            sources=["taiko1000"],
            mechanics=["taiko"],
            init=str(tmp_path / "sft" / "latest.pt"),
            steps=1,
            prompts=2,
            group=2,
            micro_batch=2,
            workers=0,
            eval_every=1,
            eval_prompts=2,
            out=str(tmp_path / "rl"),
        )
    )
    assert "val_reward" in last and (tmp_path / "rl" / "best.pt").exists()
    r, parts = reward(np.array([0, 50_000]), np.array([0, 50_000]), RLConfig())
    assert parts["f1"] == 1.0 and parts["density"] == 0.0 and r == 1.0


def test_r6_section_plan_and_memory_follow_the_grammar():
    """Section plans, structure memory and context noise produce grammar-valid samples; the
    memory appears when the audio repeats; tuplet sub-beats are illegal for Taiko."""
    import random

    from urfm.tokenize import build_tokenizer
    from urfm.tokenize.vocab import SUB_GRID
    from urfm.train.data import Features, make_sample, prepare

    tok = build_tokenizer()
    rng = np.random.default_rng(1)
    block = rng.integers(0, 256, (int(8 * MEL.fps), MEL.n_mels), dtype=np.uint8)
    mel = np.concatenate([block] * 8)  # the music repeats every 8 s
    cs = taiko_chartset()
    events = tuple(
        Event(
            time=Time(Fraction(i, 4)),
            primitive=Primitive.HIT,
            role=("taiko:don", "taiko:ka")[i % 3 == 0],
        )
        for i in range(600)
    )
    cs = ChartSet(timing=cs.timing, charts=(Chart(mechanics="taiko", events=events),))
    feats = Features(sections=True, ref=True, ctx_noise=0.5)
    prep = prepare(tok, cs, cs.charts[0], mel.shape[0], features=feats)
    v, r = tok.vocab, random.Random(0)
    refs = 0
    for w in prep.windows:
        s = make_sample(tok, cs, prep, mel, w, 2048, feats, r)
        body = s.tokens.tolist()[len(prep.prefix) + 1 :]
        tok.grammar_state("taiko", body)  # raises on an illegal token
        assert v.id("SDENS", 0) - 16 < max(body)  # plan tokens present
        assert any(v.kinds[t] == "SBEAT" for t in body[body.index(tok.sep_id) :])
        refs += v.id("REF") in body
    assert refs > 0
    sub_state = tok.grammar_state("taiko", [v.id("SEP"), v.id("POS_BEAT", 0)])
    allowed = tok.mask("taiko", sub_state)
    tuplet = v.id("POS_SUB", SUB_GRID.index(Fraction(1, 5)))
    assert not allowed[tuplet] and allowed[v.id("POS_SUB", SUB_GRID.index(Fraction(1, 8)))]


def test_r6_train_smoke_and_partial_init(shards, tmp_path):
    """R6 features train; a run can warm-start from a checkpoint with another vocabulary."""
    from urfm.tokenize import build_tokenizer
    from urfm.train.train import load_checkpoint

    tok = build_tokenizer()
    base = dict(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=2,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=2,
        eval_every=2,
        eval_windows=4,
        ckpt_every=2,
        max_dec_len=1024,
        song_context=True,
        cond=True,
        texture=True,
        canonical=True,
    )
    cfg = TrainConfig(
        **base,
        out=str(tmp_path / "r6"),
        sections=True,
        ref=True,
        ctx_noise=0.5,
        p_octave=1.0,
        octave_factors=[0.5, 0.6, 2.0, 0.25, 0.25, 0.15],
    )
    assert np.isfinite(train(cfg, tok=tok)["loss"])
    model, meta = load_checkpoint(tmp_path / "r6" / "latest.pt")
    assert meta["features"]["sections"] and meta["features"]["ref"]
    from urfm.tokenize.sections import SectionPlan
    from urfm.train.generate import reference_inputs

    cs = taiko_chartset()
    prefix, cond = reference_inputs(model, meta, tok, cs, cs.charts[0])
    block = np.random.default_rng(0).integers(0, 256, (int(6 * MEL.fps), MEL.n_mels), np.uint8)
    job = Job(
        mel=np.concatenate([block] * 5),
        timing=cs.timing,
        mechanics="taiko",
        prefix=prefix,
        cond=cond,
        section_plans={i: SectionPlan(i, 8, 1, 1, 1, 4) for i in range(-2, 20)},
        section_k=8,
        use_ref=True,
    )
    (chart,) = generate(model, tok, [job], device="cpu", temperature=1.0)
    assert chart.mechanics == "taiko"
    warm = TrainConfig(
        **{**base, "max_dec_len": 1536},
        out=str(tmp_path / "warm"),
        sections=True,
        init=str(tmp_path / "r6" / "latest.pt"),
        init_partial=True,
    )
    assert np.isfinite(train(warm, tok=tok)["loss"])


def test_r6_planner_trains_and_plans(shards, tmp_path):
    """The song planner trains on augmented charts and plans every section of a song."""
    from urfm.tokenize.sections import SectionPlan
    from urfm.train.planner import PlanTrainConfig, load_planner, plan_song
    from urfm.train.planner import train as train_planner

    cfg = PlanTrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        out=str(tmp_path / "plan"),
        steps=2,
        batch_size=2,
        warmup=1,
        workers=0,
        eval_every=2,
        eval_charts=2,
        device="cpu",
    )
    last = train_planner(cfg)
    assert np.isfinite(last["loss"]) and "val_loss" in last
    model = load_planner(tmp_path / "plan" / "latest.pt")
    assert model.num_params() < 5_000_000
    cs = taiko_chartset()
    mel = np.zeros((int(30 * MEL.fps), MEL.n_mels), np.uint8)
    k, plans = plan_song(model, mel, cs.timing, "taiko")
    assert k == 8 and len(plans) >= 6
    assert all(isinstance(p, SectionPlan) and p.section == i for i, p in plans.items())


def test_r6_rl_step_runs(shards, tmp_path):
    """GRPO on a section-plan model with the R6 reward terms."""
    from urfm.tokenize import build_tokenizer
    from urfm.train.rl import RLConfig
    from urfm.train.rl import train as rl_train

    tok = build_tokenizer()
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=2,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=2,
        eval_every=2,
        eval_windows=4,
        ckpt_every=2,
        out=str(tmp_path / "sft"),
        max_dec_len=1024,
        song_context=True,
        cond=True,
        texture=True,
        canonical=True,
        sections=True,
        ref=True,
    )
    train(cfg, tok=tok)
    last = rl_train(
        RLConfig(
            data=str(shards),
            sources=["taiko1000"],
            mechanics=["taiko"],
            init=str(tmp_path / "sft" / "latest.pt"),
            steps=1,
            prompts=2,
            group=2,
            micro_batch=2,
            workers=0,
            eval_every=1,
            eval_prompts=2,
            out=str(tmp_path / "rl"),
            w_f1=0.25,
            w_density=0.0,
            w_fast=0.0,
            w_place=1.0,
            w_plan=0.5,
            w_struct=1.0,
            w_stag=0.5,
        )
    )
    assert "val_reward" in last and "struct" in last and "place" in last


def test_r61_plan_extension(shards, tmp_path):
    """Plan extras: grammar-valid samples, planner with ten heads, extended plans tokenized."""
    import random

    from urfm.tokenize import build_tokenizer
    from urfm.tokenize.sections import SectionPlan
    from urfm.train.data import Features, make_sample, prepare
    from urfm.train.planner import PlanTrainConfig, load_planner, plan_song
    from urfm.train.planner import train as train_planner

    tok = build_tokenizer()
    cs = taiko_chartset()
    mel = np.random.default_rng(0).integers(0, 256, (int(40 * MEL.fps), MEL.n_mels), np.uint8)
    feats = Features(sections=True, ref=True, plan_extra=True)
    prep = prepare(tok, cs, cs.charts[0], mel.shape[0], features=feats)
    v = tok.vocab
    for w in prep.windows:
        s = make_sample(tok, cs, prep, mel, w, 2048, feats, random.Random(0))
        body = s.tokens.tolist()[len(prep.prefix) + 1 :]
        tok.grammar_state("taiko", body)
        if any(v.kinds[t] == "SDENS" for t in body):
            assert any(v.kinds[t] == "SPEAK" for t in body)
    cfg = PlanTrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        out=str(tmp_path / "plan"),
        steps=2,
        batch_size=2,
        warmup=1,
        workers=0,
        eval_every=2,
        eval_charts=2,
        device="cpu",
        extras=True,
    )
    assert "val_acc_long" in train_planner(cfg)
    model = load_planner(tmp_path / "plan" / "latest.pt")
    _k, plans = plan_song(model, mel, cs.timing, "taiko")
    p = next(iter(plans.values()))
    assert isinstance(p, SectionPlan) and p.peak is not None and p.long is not None


def test_song_level_rl_step_runs(shards, tmp_path):
    """Song-level GRPO: multi-window rollouts on the model's own context, span rewards."""
    from urfm.tokenize import build_tokenizer
    from urfm.train.rl_song import SongRLConfig
    from urfm.train.rl_song import train as song_rl

    tok = build_tokenizer()
    cfg = TrainConfig(
        data=str(shards),
        sources=["taiko1000"],
        size="tiny",
        batch_size=4,
        steps=2,
        warmup=1,
        workers=0,
        compile=False,
        device="cpu",
        log_every=2,
        eval_every=2,
        eval_windows=4,
        ckpt_every=2,
        out=str(tmp_path / "sft"),
        max_dec_len=1024,
        song_context=True,
        cond=True,
        texture=True,
        canonical=True,
        sections=True,
        ref=True,
        plan_extra=True,
    )
    train(cfg, tok=tok)
    last = song_rl(
        SongRLConfig(
            data=str(shards),
            sources=["taiko1000"],
            mechanics=["taiko"],
            init=str(tmp_path / "sft" / "latest.pt"),
            steps=1,
            songs=1,
            group=2,
            span=1,
            micro_batch=4,
            workers=0,
            eval_every=1,
            eval_songs=1,
            out=str(tmp_path / "rl"),
        )
    )
    assert "val_reward" in last and all(k in last for k in ("stag", "etime", "struct"))
    assert (tmp_path / "rl" / "best.pt").exists()
