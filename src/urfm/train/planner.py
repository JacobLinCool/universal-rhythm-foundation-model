"""Song planner training and inference (R6; model in ``urfm.model.planner``).

    python -m urfm.train.planner --data /content/data/urfm-ingest-v0 --out runs/r6-plan

One sample is one chart: the descriptors of every section of its song and the chart's section
plans as targets. Charts get the same augmentation as the decoder (canonical tempo, predicted
grid, BPM octave, tempo warps), so the planner also learns which subdivision a passage needs on
whatever grid it is given.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import msgspec
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from urfm.data.features import MEL
from urfm.data.sections import FEATURES, MAX_SECTIONS, section_audio
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.serialize import LazyChartSet, from_msgpack
from urfm.ir.timing import TimingMap
from urfm.mechanics import registry
from urfm.model.planner import EXTRA_HEADS, PLAN_HEADS, Planner, PlannerConfig
from urfm.tokenize.sections import (
    REP_BINS,
    VAR_BINS,
    SectionPlan,
    plan_extras,
    plan_sections,
    section_beats,
)
from urfm.tokenize.vocab import DENS_EDGES, GRID_VALUES, SHARE_BINS
from urfm.train.conditions import NAMES as COND_NAMES
from urfm.train.conditions import chart_conditions
from urfm.train.data import (
    PIPELINE,
    Augment,
    Index,
    PredictedTimings,
    augment_chart,
    build_index,
    extreme_tempo,
    onset_beats,
    shard_files,
)
from urfm.train.data import (
    loader as make_loader,
)

HEAD_SIZES = (len(DENS_EDGES), SHARE_BINS, SHARE_BINS, SHARE_BINS, len(GRID_VALUES))
EXTRA_SIZES = (len(DENS_EDGES), REP_BINS, VAR_BINS, SHARE_BINS, SHARE_BINS)
KA_GAMES = frozenset({"taiko", "osu_taiko"})
BUCKET = 32


def games() -> tuple[str, ...]:
    return tuple(sorted(registry.all_specs()))


@dataclass
class PlanSample:
    feats: np.ndarray  # float32 [S, FEATURES]
    targets: np.ndarray  # int64 [S, heads]
    game: int
    cond: np.ndarray
    cond_mask: np.ndarray
    first: int
    k: int


def plan_sample(
    cs: ChartSet,
    chart: Chart,
    mel: np.ndarray,
    base_timing: TimingMap | None = None,
    rng: random.Random | None = None,
    cond_drop: float = 0.15,
    extras: bool = False,
) -> PlanSample:
    """Targets from ``chart``; audio descriptors on ``base_timing`` (the pre-warp timing whose
    beats coincide with the chart's) or the chart's own timing."""
    timing = cs.timing_of(chart)
    k = section_beats(timing)
    first, feats = section_audio(mel, base_timing or timing, k)
    beats, ka = onset_beats(cs, chart, timing)
    last = first + len(feats) - 1
    ext = None
    if extras:
        from urfm.train.data import long_spans

        ext = plan_extras(timing, beats, ka, long_spans(chart), first, last, k)
    plans = plan_sections(timing, beats, ka, first, last, k, ext)
    heads = (*PLAN_HEADS, *EXTRA_HEADS) if extras else PLAN_HEADS
    targets = np.array([[getattr(p, h) for h in heads] for p in plans], np.int64)
    if chart.mechanics not in KA_GAMES:
        targets[:, PLAN_HEADS.index("ka")] = -100
        if extras:
            targets[:, heads.index("sw")] = -100
    cond, mask = chart_conditions(cs, chart)
    mask = mask.copy()
    if rng is not None:
        for i in range(len(mask)):
            if rng.random() < cond_drop:
                mask[i] = False
    cond = np.where(mask, cond, 0.0).astype(np.float32)
    return PlanSample(feats, targets, games().index(chart.mechanics), cond, mask, first, k)


def collate(samples: list[PlanSample]) -> dict[str, torch.Tensor]:
    longest = max(len(s.feats) for s in samples)
    size = min(math.ceil(longest / BUCKET) * BUCKET, MAX_SECTIONS)
    b = len(samples)
    feats = np.zeros((b, size, FEATURES), np.float32)
    targets = np.full((b, size, samples[0].targets.shape[1]), -100, np.int64)
    mask = np.zeros((b, size), bool)
    for i, s in enumerate(samples):
        n = min(len(s.feats), size)
        feats[i, :n], targets[i, :n], mask[i, :n] = s.feats[:n], s.targets[:n], True
    return {
        "feats": torch.from_numpy(feats),
        "targets": torch.from_numpy(targets),
        "mask": torch.from_numpy(mask),
        "game": torch.tensor([s.game for s in samples]),
        "cond": torch.from_numpy(np.stack([s.cond for s in samples])),
        "cond_mask": torch.from_numpy(np.stack([s.cond_mask for s in samples])),
    }


class ChartStream(IterableDataset):
    """Augmented charts as planner samples, batched in the worker."""

    def __init__(
        self,
        index: Index,
        batch_size: int,
        augment: Augment,
        predicted: PredictedTimings | None = None,
        seed: int = 0,
        extras: bool = False,
    ) -> None:
        self.index, self.batch_size, self.augment = index, batch_size, augment
        self.predicted, self.seed, self.extras = predicted, seed, extras

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1009 + wid)
        groups = sorted({(c.file, c.row_group) for c in self.index.charts})[wid::nw]
        by_group: dict[tuple[str, int], list] = {}
        for c in self.index.charts:
            by_group.setdefault((c.file, c.row_group), []).append(c)
        buffer: list[PlanSample] = []
        while True:
            rng.shuffle(groups)
            for key in groups:
                table = pq.ParquetFile(key[0]).read_row_group(key[1], columns=["mel", "chartset"])
                mels, sets = table["mel"].to_pylist(), table["chartset"].to_pylist()
                for ref in by_group.get(key, []):
                    # charts are drawn in proportion to their window weight (song length)
                    if rng.random() >= min(ref.weight / 4, 1.0):
                        continue
                    cs = LazyChartSet(sets[ref.row]).view(ref.chart)
                    if extreme_tempo(cs.timing_of(cs.charts[0])):
                        continue
                    mel = np.frombuffer(mels[ref.row], np.uint8).reshape(-1, MEL.n_mels)
                    try:
                        pred = None
                        if self.predicted is not None and self.augment.p_predicted > 0:
                            pred = self.predicted.get(ref.file, ref.row_group, ref.row)
                        cs2, chart, _audio, plain = augment_chart(
                            cs, cs.charts[0], mel, rng, self.augment, pred
                        )
                        buffer.append(
                            plan_sample(
                                cs2,
                                chart,
                                mel,
                                plain.timing_of(plain.charts[0]),
                                rng,
                                extras=self.extras,
                            )
                        )
                    except Exception as exc:
                        PIPELINE[f"planner_error:{type(exc).__name__}"] += 1
                        continue
                    if len(buffer) >= self.batch_size * 4:
                        buffer.sort(key=lambda s: len(s.feats))
                        batches = [
                            buffer[i : i + self.batch_size]
                            for i in range(0, len(buffer) - self.batch_size + 1, self.batch_size)
                        ]
                        rng.shuffle(batches)
                        yield from (collate(b) for b in batches)
                        buffer = []


def validation_samples(
    index: Index, limit: int, seed: int = 0, extras: bool = False
) -> list[PlanSample]:
    from urfm.data.canonical import canonicalize

    rng = random.Random(seed)
    refs = list(index.charts)
    rng.shuffle(refs)
    out: list[PlanSample] = []
    for ref in refs[: limit * 2]:
        if len(out) >= limit:
            break
        t = pq.ParquetFile(ref.file).read_row_group(ref.row_group, columns=["mel", "chartset"])
        cs = from_msgpack(t["chartset"].to_pylist()[ref.row])
        chart = cs.charts[ref.chart]
        if extreme_tempo(cs.timing_of(chart)) or not chart.events:
            continue
        cs, chart = canonicalize(cs, chart)
        cs = msgspec.structs.replace(cs, charts=(chart,))
        mel = np.frombuffer(t["mel"].to_pylist()[ref.row], np.uint8).reshape(-1, MEL.n_mels)
        out.append(plan_sample(cs, chart, mel, extras=extras))
    return out


def expected_density(logits: torch.Tensor) -> torch.Tensor:
    """Expected log2 onsets/s from ``SDENS`` logits."""
    centres = torch.tensor(
        [math.log2(0.25)] + [math.log2(0.5) + (b + 0.5) / 2.5 for b in range(1, len(DENS_EDGES))],
        device=logits.device,
    )
    return (torch.softmax(logits.float(), -1) * centres).sum(-1)


@torch.no_grad()
def evaluate(model: Planner, samples: list[PlanSample], device: torch.device) -> dict:
    """Validation loss, per-head accuracy and section r of the expected density against the
    chart (mean over charts) -- comparable to the ridge baseline and to two human courses."""
    model.eval()
    losses, accs, rs = [], [], []
    for i in range(0, len(samples), 16):
        chunk = samples[i : i + 16]
        b = {k: v.to(device) for k, v in collate(chunk).items()}
        loss, acc = model.loss(
            b["feats"], b["mask"], b["game"], b["targets"], b["cond"], b["cond_mask"]
        )
        losses.append(float(loss))
        accs.append(acc.cpu().numpy())
        dens = expected_density(
            model(b["feats"], b["mask"], b["game"], b["cond"], b["cond_mask"])[0]
        ).cpu()
        for j, s in enumerate(chunk):
            n = len(s.feats)
            truth = s.targets[:n, 0].astype(np.float64)
            pred = dens[j, :n].numpy()
            active = truth > 0
            if active.sum() >= 8 and truth[active].std() > 0 and pred[active].std() > 0:
                rs.append(float(np.corrcoef(pred[active], truth[active])[0, 1]))
    model.train()
    acc = np.mean(accs, 0)
    heads = (*PLAN_HEADS, *EXTRA_HEADS)[: len(acc)]
    return {
        "val_loss": float(np.mean(losses)),
        **{f"val_acc_{h}": float(a) for h, a in zip(heads, acc, strict=True)},
        "val_section_r_density": float(np.mean(rs)) if rs else None,
    }


class PlanTrainConfig(msgspec.Struct, kw_only=True):
    data: str = "/content/data/urfm-ingest-v0"
    sources: list[str] = msgspec.field(default_factory=lambda: ["taiko1000", "osu"])
    out: str = "runs/r6-plan"
    steps: int = 10000
    batch_size: int = 32
    lr: float = 5e-4
    warmup: int = 500
    workers: int = 8
    eval_every: int = 1000
    eval_charts: int = 256
    seed: int = 0
    p_speed: float = 0.5
    p_local: float = 0.3
    p_octave: float = 0.5
    octave_factors: list[float] = msgspec.field(
        default_factory=lambda: [0.5, 0.6, 2.0, 0.25, 0.25, 0.15]
    )
    predicted_timing: str | None = None
    p_predicted: float = 0.5
    taiko_boost: float = 2.0
    hub_repo: str | None = None
    device: str = "cuda"
    d_model: int = 192
    layers: int = 2
    extras: bool = False
    """R6.1: also plan peak density, repetition, rhythm variety, colour switches, long notes."""
    """1.07M parameters; with the 18.5M decoder the system stays under 20M."""


def train(cfg: PlanTrainConfig) -> dict:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    index = build_index(
        shard_files(cfg.data, cfg.sources, "train"), boost={"taiko": cfg.taiko_boost}
    )
    val = validation_samples(
        build_index(shard_files(cfg.data, cfg.sources, "validation")),
        cfg.eval_charts,
        extras=cfg.extras,
    )
    pairs = cfg.octave_factors
    augment = Augment(
        canonical=True,
        p_speed=cfg.p_speed,
        p_local=cfg.p_local,
        p_octave=cfg.p_octave,
        octave_factors=tuple(
            (float(pairs[i]), float(pairs[i + 1])) for i in range(0, len(pairs) - 1, 2)
        ),
        p_predicted=cfg.p_predicted,
    )
    stream = ChartStream(
        index,
        cfg.batch_size,
        augment,
        PredictedTimings(cfg.predicted_timing) if cfg.predicted_timing else None,
        cfg.seed,
        cfg.extras,
    )
    loader = make_loader(stream, cfg.workers)
    pcfg = PlannerConfig(
        in_dim=FEATURES,
        games=games(),
        head_sizes=HEAD_SIZES + (EXTRA_SIZES if cfg.extras else ()),
        cond_dim=len(COND_NAMES),
        d_model=cfg.d_model,
        layers=cfg.layers,
        n_heads=max(cfg.d_model // 64, 1),
    )
    model = Planner(pcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(step: int) -> float:
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        p = (step - cfg.warmup) / max(cfg.steps - cfg.warmup, 1)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log = (out / "metrics.jsonl").open("a")
    meta = {"train": msgspec.to_builtins(cfg), "params": model.num_params()}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"params": model.num_params(), "val_charts": len(val)}), flush=True)
    best, last = float("inf"), {}
    t0, seen, wait = time.time(), 0, 0.0
    batches = iter(loader)
    for step in range(cfg.steps):
        tw = time.time()
        b = {k: v.to(device, non_blocking=True) for k, v in next(batches).items()}
        wait += time.time() - tw
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, acc = model.loss(
                b["feats"], b["mask"], b["game"], b["targets"], b["cond"], b["cond_mask"]
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += int(b["mask"].shape[0])
        if (step + 1) % 100 == 0 or step + 1 == cfg.steps:
            el = time.time() - t0
            last = {
                "step": step + 1,
                "loss": float(loss.detach()),
                **{
                    f"acc_{h}": float(a)
                    for h, a in zip((*PLAN_HEADS, *EXTRA_HEADS), acc, strict=False)
                },
                "lr": lr_at(step),
                "charts_per_s": seen / el,
                "data_wait_frac": wait / el,
            }
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                last |= evaluate(model, val, device)
                state = {"model": model.state_dict(), "config": msgspec.to_builtins(pcfg)}
                torch.save(state, out / "latest.pt")
                if last["val_loss"] < best:
                    best = last["val_loss"]
                    torch.save(state, out / "best.pt")
            log.write(json.dumps(last) + "\n")
            log.flush()
            if cfg.hub_repo and ((step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps):
                _upload(cfg.hub_repo, out)  # after the log line, so the Hub copy is complete
            print(json.dumps(last), flush=True)
            t0, seen, wait = time.time(), 0, 0.0
    return last


def _upload(repo: str, out: Path) -> None:
    from huggingface_hub import HfApi

    try:
        HfApi().upload_folder(
            repo_id=repo,
            folder_path=str(out),
            path_in_repo=out.name,
            allow_patterns=["latest.pt", "best.pt", "*.json", "*.jsonl"],
        )
    except Exception as exc:  # uploads never stop training
        print(json.dumps({"upload_error": type(exc).__name__}), flush=True)


def load_planner(path: str | Path, device: str | torch.device = "cpu") -> Planner:
    state = torch.load(path, map_location=device, weights_only=False)
    model = Planner(msgspec.convert(state["config"], PlannerConfig)).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


@torch.no_grad()
def plan_song(
    model: Planner,
    mel: np.ndarray,
    timing: TimingMap,
    mechanics: str,
    cond: tuple[np.ndarray, np.ndarray] | None = None,
    temperature: float = 0.7,
    seed: int = 0,
) -> tuple[int, dict[int, SectionPlan]]:
    """``(k, {section index: plan})`` for a song on ``timing``; values are sampled per section
    at ``temperature`` (0 = most likely)."""
    device = next(model.parameters()).device
    k = section_beats(timing)
    first, feats = section_audio(mel, timing, k)
    n_cond = model.cfg.cond_dim
    values, mask = cond if cond is not None else (np.zeros(n_cond), np.zeros(n_cond, bool))
    logits = model(
        torch.from_numpy(feats)[None].to(device),
        torch.ones((1, len(feats)), dtype=torch.bool, device=device),
        torch.tensor([model.cfg.games.index(mechanics)], device=device),
        torch.from_numpy(np.asarray(values, np.float32))[None].to(device),
        torch.from_numpy(np.asarray(mask, bool))[None].to(device),
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    picks = []
    for lg in logits:
        lg = lg[0].float().cpu()
        if temperature <= 0:
            picks.append(lg.argmax(-1))
        else:
            picks.append(
                torch.multinomial(torch.softmax(lg / temperature, -1), 1, generator=gen)[:, 0]
            )
    heads = (*PLAN_HEADS, *EXTRA_HEADS)[: len(picks)]
    out = {}
    for i in range(len(feats)):
        v = dict(zip(heads, (int(p[i]) for p in picks), strict=True))
        if mechanics not in KA_GAMES:
            v["ka"] = 0
            if "sw" in v:
                v["sw"] = 0
        out[first + i] = SectionPlan(first + i, **v)
    return k, out


def parse_args(argv: list[str] | None = None) -> PlanTrainConfig:
    ap = argparse.ArgumentParser(description=__doc__)
    defaults = PlanTrainConfig()
    for name in PlanTrainConfig.__struct_fields__:
        value = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        ap.add_argument(flag, nargs="*" if isinstance(value, list) else None, default=None)
    args = ap.parse_args(argv)
    fields = {n: getattr(args, n) for n in PlanTrainConfig.__struct_fields__}
    fields = {k: v for k, v in fields.items() if v is not None}
    return msgspec.convert(fields, PlanTrainConfig, strict=False)


if __name__ == "__main__":
    train(parse_args())
