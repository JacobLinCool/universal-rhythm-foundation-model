"""Train a chart model.

    python -m urfm.train.train --data /content/data/urfm-ingest-v0 --sources taiko1000 \
        --size 10m --batch-size 64 --steps 6000 --out /content/runs/v1-taiko

Every config field is a CLI flag (``--batch-size``, ``--compile false``...); ``--config``
loads a JSON file first. Logs JSONL metrics (loss, token accuracy, throughput,
data-wait fraction, GPU utilisation, memory) to ``<out>/metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import threading
import time
from collections import Counter
from pathlib import Path
from typing import cast

import msgspec
import torch

from urfm.data.texture import FEATURES as TEXTURE_FEATURES
from urfm.data.texture import MAX_UNITS
from urfm.ir.timing import TimingMap
from urfm.model import ChartModel, ModelConfig, preset
from urfm.tokenize import take_drops
from urfm.tokenize.api import Tokenizer
from urfm.train.conditions import NAMES as COND_NAMES
from urfm.train.data import (
    PIPELINE,
    Augment,
    Features,
    PredictedTimings,
    WindowStream,
    build_index,
    collate,
    fixed_windows,
    loader,
    shard_files,
)


class TrainConfig(msgspec.Struct, kw_only=True):
    data: str = "/content/data/urfm-ingest-v0"
    sources: list[str] = msgspec.field(default_factory=lambda: ["taiko1000"])
    mechanics: list[str] = msgspec.field(default_factory=list)
    """Empty = all mechanics present in the sources."""
    alpha: float = 0.5
    size: str = "10m"
    max_dec_len: int = 2048
    song_context: bool = False
    """Song-level encoder over the pooled whole-song mel (see ``ModelConfig.song_context``)."""
    song_layers: int = 2
    """Layers of the song-context encoder (R6 uses 1 to stay within the 20M budget)."""
    batch_size: int = 64
    grad_accum: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    warmup: int = 500
    steps: int = 6000
    clip: float = 1.0
    workers: int = 8
    compile: bool = True
    device: str = "cuda"
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1000
    eval_windows: int = 512
    ckpt_every: int = 2000
    out: str = "runs/train"
    init: str | None = None
    """Checkpoint to initialise weights from (fine-tuning); its model config must match."""
    cond: bool = False
    """Continuous chart conditions (``urfm.train.conditions``) instead of TIER/DENS tokens."""
    texture: bool = False
    """Beat-synchronous texture song context (needs ``song_context``)."""
    canonical: bool = False
    """Fold visual tempo tricks into scroll before tokenizing (``urfm.data.canonical``)."""
    p_speed: float = 0.0
    """Chance a visit plays the whole song 0.8-1.6x faster (log-uniform)."""
    p_local: float = 0.0
    """Chance a visit adds short beat spans (half a beat to a measure) at their own speed."""
    p_octave: float = 0.0
    """Chance a visit re-notates the chart at double or half BPM (same sound)."""
    predicted_timing: str | None = None
    """Directory of audio-predicted timing maps (``experiments/r4/predict_timing.py``)."""
    p_predicted: float = 0.0
    """Chance a visit re-notates the chart on its song's predicted timing map."""
    fast_boost: float = 0.0
    """Oversample windows with very fast passages (``Augment.fast_boost``)."""
    windows_per_visit: int = 1
    """Windows drawn per chart visit (``Augment.windows_per_visit``)."""
    taiko_boost: float = 1.0
    """Sampling weight multiplier for official Taiko charts in mixed runs."""
    octave_factors: list[float] = []
    """Octave re-notation choices as ``factor weight`` pairs, e.g. ``0.5 .6 2 .25 0.25 .15``
    (``Augment.octave_factors``); empty keeps the R4 rule."""
    sections: bool = False
    """R6 section plans instead of the window plan (``Features.sections``)."""
    ref: bool = False
    """R6 structure memory (``Features.ref``)."""
    plan_extra: bool = False
    """R6.1 section-plan extension (``Features.plan_extra``)."""
    ctx_noise: float = 0.0
    """Chance a sample's context and memory are corrupted (``Features.ctx_noise``)."""
    resume: bool = False
    """Continue from ``<out>/latest.pt`` (fetched from ``hub_repo`` when missing locally):
    weights, optimizer state and step. Colab runtimes can disappear mid-run."""
    init_partial: bool = False
    """``--init`` from a checkpoint with another vocabulary or decoder length: tensors of equal
    shape are copied, and the leading rows of tensors that differ only in their first
    dimension (token embeddings, positions)."""
    hub_repo: str | None = None
    """Upload checkpoints and metrics to this model repo (private) in the background."""


def load_partial(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, int]:
    """Copy ``state`` into ``model`` where shapes allow (see ``TrainConfig.init_partial``)."""
    own = model.state_dict()
    copied = rows = skipped = 0
    with torch.no_grad():
        for name, value in state.items():
            target = own.get(name)
            if target is None:
                skipped += 1
            elif target.shape == value.shape:
                target.copy_(value)
                copied += 1
            elif target.ndim == value.ndim and target.shape[1:] == value.shape[1:]:
                n = min(target.shape[0], value.shape[0])
                target[:n].copy_(value[:n])
                rows += 1
            else:
                skipped += 1
    return {
        "copied": copied,
        "row_copied": rows,
        "skipped": skipped,
        "new": len(own) - copied - rows,
    }


def load_tokenizer() -> Tokenizer:
    from urfm.tokenize import build_tokenizer  # provided by the tokenizer package

    return cast(Tokenizer, build_tokenizer())


def frame_feat_dim(tok: Tokenizer) -> int:
    return int(tok.frame_features(TimingMap.constant(120), 0, 2).shape[1])


def vocab_hash(tok: Tokenizer) -> str:
    explicit = getattr(tok, "vocab_hash", None)
    if explicit:
        return str(explicit)
    return hashlib.sha256(str(tok.vocab_size).encode()).hexdigest()[:16]


class GpuMonitor:
    """Samples GPU utilisation in a background thread (pynvml, else nvidia-smi)."""

    def __init__(self, interval: float = 1.0) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._read = self._backend()
        if self._read is not None:
            threading.Thread(target=self._run, args=(interval,), daemon=True).start()

    def _backend(self):
        if not torch.cuda.is_available():
            return None
        try:
            import pynvml  # pyright: ignore[reportMissingImports]

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            return lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            import subprocess

            def smi() -> float:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                ).stdout
                return float(out.split()[0])

            return smi

    def _run(self, interval: float) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self.samples.append(self._read())  # type: ignore[misc]
            time.sleep(interval)

    def take(self) -> float | None:
        samples, self.samples = self.samples, []
        return sum(samples) / len(samples) if samples else None


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * progress))


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "stats"}


def pipeline_report(stats: Counter[str]) -> dict[str, float]:
    """Per-mechanics truncation rates plus every drop/error count since the last log."""
    out: dict[str, float] = {}
    for key, n in stats.items():
        mech, _, what = key.partition("/")
        if what == "truncated_windows":
            out[f"{mech}/truncated_frac"] = n / max(stats.get(f"{mech}/windows", 0), 1)
        elif what != "windows":
            out[key] = n
    return dict(sorted(out.items()))


@torch.no_grad()
def evaluate(model: ChartModel, batches: list[dict], device: torch.device, amp) -> dict:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    total = torch.zeros((), device=device, dtype=torch.long)
    for batch in batches:
        b = to_device(batch, device)
        with amp():
            loss, c, n = model.loss(b)
        loss_sum += loss.float() * n
        correct += c
        total += n
    model.train()
    n = max(int(total), 1)
    return {"val_loss": float(loss_sum) / n, "val_acc": int(correct) / n}


class Uploader:
    def __init__(self, repo: str | None, out: Path) -> None:
        self.repo, self.out, self.thread = repo, out, None
        if repo:
            from huggingface_hub import HfApi

            HfApi().create_repo(repo, private=True, exist_ok=True)

    def push(self, message: str) -> None:
        if not self.repo or (self.thread and self.thread.is_alive()):
            return

        repo = self.repo

        def run() -> None:
            from huggingface_hub import HfApi

            with contextlib.suppress(Exception):
                HfApi().upload_folder(
                    repo_id=repo,
                    folder_path=str(self.out),
                    path_in_repo=self.out.name,
                    allow_patterns=["latest.pt", "best.pt", "*.json", "*.jsonl"],
                    commit_message=message,
                )

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def wait(self) -> None:
        if self.thread:
            self.thread.join()


def train(cfg: TrainConfig, tok: Tokenizer | None = None) -> dict:
    tok = tok or load_tokenizer()
    torch.manual_seed(cfg.seed)
    device = torch.device(
        cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    mech = set(cfg.mechanics) or None
    boost = {"taiko": cfg.taiko_boost}
    train_index = build_index(
        shard_files(cfg.data, cfg.sources, "train"), mech, cfg.alpha, boost=boost
    )
    val_index = build_index(shard_files(cfg.data, cfg.sources, "validation"), mech, cfg.alpha)
    if not train_index.charts:
        raise SystemExit(f"no training charts under {cfg.data} for {cfg.sources}")

    features = Features(
        cond=cfg.cond,
        texture=cfg.texture,
        sections=cfg.sections,
        ref=cfg.ref,
        ctx_noise=cfg.ctx_noise,
        plan_extra=cfg.plan_extra,
    )
    pairs = cfg.octave_factors
    augment = Augment(
        octave_factors=tuple(
            (float(pairs[i]), float(pairs[i + 1])) for i in range(0, len(pairs) - 1, 2)
        ),
        canonical=cfg.canonical,
        p_speed=cfg.p_speed,
        p_local=cfg.p_local,
        p_octave=cfg.p_octave,
        p_predicted=cfg.p_predicted,
        fast_boost=cfg.fast_boost,
        windows_per_visit=cfg.windows_per_visit,
    )
    extra: dict = {}
    if cfg.cond:
        extra |= {"cond_dim": len(COND_NAMES), "cond_token": tok.vocab.id("COND")}  # type: ignore[attr-defined]
    if cfg.texture:
        extra |= {"song_feat_dim": TEXTURE_FEATURES, "song_max": MAX_UNITS}
    mcfg = preset(
        cfg.size,
        vocab_size=tok.vocab_size,
        frame_feat_dim=frame_feat_dim(tok),
        max_dec_len=cfg.max_dec_len,
        song_context=cfg.song_context,
        song_layers=cfg.song_layers,
        **extra,
    )
    model = ChartModel(mcfg).to(device)
    if cfg.init and cfg.init_partial:
        state = torch.load(cfg.init, map_location=device, weights_only=False)
        print(json.dumps({"init_partial": load_partial(model, state["model"])}), flush=True)
    elif cfg.init:
        state = torch.load(cfg.init, map_location=device, weights_only=False)
        if msgspec.convert(state["model_config"], ModelConfig) != mcfg:
            raise ValueError(f"--init {cfg.init}: model config differs from this run's")
        model.load_state_dict(state["model"])
    params = model.num_params()
    meta = {
        "train": msgspec.to_builtins(cfg),
        "model": msgspec.to_builtins(mcfg),
        "params": params,
        "vocab_hash": vocab_hash(tok),
        "train_charts": dict(train_index.counts()),
        "val_charts": dict(val_index.counts()),
        "features": {
            "cond": cfg.cond,
            "texture": cfg.texture,
            "canonical": cfg.canonical,
            "sections": cfg.sections,
            "ref": cfg.ref,
            "plan_extra": cfg.plan_extra,
        },
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"params": params, "train_charts": meta["train_charts"]}), flush=True)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "pos" not in n)]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=cuda,
    )
    start, best_val = 0, float("inf")
    if cfg.resume:
        latest = _fetch_latest(cfg, out)
        if latest is not None:
            state = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            if "opt" in state:
                opt.load_state_dict(state["opt"])
            start, best_val = int(state["step"]), float(state.get("best_val", best_val))
            print(json.dumps({"resumed_at": start, "best_val": best_val}), flush=True)
    loss_fn = torch.compile(model.loss, dynamic=False) if cfg.compile and cuda else model.loss

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    tiers = train_index.tier_table()
    meta["tiers"] = msgspec.to_builtins(tiers)
    val_samples = fixed_windows(
        val_index,
        tok,
        cfg.max_dec_len,
        cfg.eval_windows,
        cfg.seed,
        tiers=tiers,
        features=features,
        canonical=cfg.canonical,
    )
    meta["val_pipeline"] = {
        **take_drops(),
        **{
            f"{m}/truncated_windows": sum(s.truncated for s in val_samples if s.mechanics == m)
            for m in {s.mechanics for s in val_samples}
        },
    }
    PIPELINE.clear()
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    val_batches = [
        collate(val_samples[i : i + cfg.batch_size], tok.pad_id)
        for i in range(0, len(val_samples), cfg.batch_size)
    ]
    stream = WindowStream(
        train_index,
        tok,
        cfg.batch_size,
        cfg.max_dec_len,
        cfg.seed + start,
        tiers=tiers,
        features=features,
        augment=augment,
        predicted=PredictedTimings(cfg.predicted_timing) if cfg.predicted_timing else None,
    )
    batches = iter(loader(stream, cfg.workers))
    gpu = GpuMonitor()
    uploader = Uploader(cfg.hub_repo, out)
    log = (out / "metrics.jsonl").open("a")

    acc_loss = torch.zeros((), device=device)
    acc_correct = torch.zeros((), device=device, dtype=torch.long)
    acc_tokens = torch.zeros((), device=device, dtype=torch.long)
    acc_trunc = torch.zeros((), device=device, dtype=torch.long)
    samples = wait = 0.0
    t_log = time.perf_counter()
    pipe: Counter[str] = Counter()
    last: dict = {}
    for step in range(start, cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        for _ in range(cfg.grad_accum):
            t0 = time.perf_counter()
            batch = next(batches)
            wait += time.perf_counter() - t0
            pipe.update(batch.pop("stats", {}))  # type: ignore[arg-type]
            b = to_device(batch, device)
            with amp():
                loss, correct, n = loss_fn(b)
            (loss / cfg.grad_accum).backward()
            acc_loss += loss.detach().float() * n
            acc_correct += correct
            acc_tokens += n
            acc_trunc += b["truncated"].sum()
            samples += b["tokens"].shape[0]
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_log
            n_tok = max(int(acc_tokens), 1)
            last = {
                "step": step + 1,
                "loss": float(acc_loss) / n_tok,
                "acc": int(acc_correct) / n_tok,
                "lr": opt.param_groups[0]["lr"],
                "samples_per_s": samples / dt,
                "target_tokens_per_s": n_tok / dt,
                "data_wait_frac": wait / dt,
                "truncated_frac": int(acc_trunc) / max(samples, 1),
                "gpu_util": gpu.take(),
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                "time": time.time(),
                "pipeline": pipeline_report(pipe),
            }
            pipe.clear()
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                last |= evaluate(model, val_batches, device, amp)
                if last["val_loss"] < best_val:
                    best_val = last["val_loss"]
                    save_checkpoint(out / "best.pt", model, mcfg, meta, step + 1)
            log.write(json.dumps(last) + "\n")
            log.flush()
            print(
                json.dumps(
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in last.items()}
                ),
                flush=True,
            )
            for acc in (acc_loss, acc_correct, acc_tokens, acc_trunc):
                acc.zero_()
            samples = wait = 0.0
            t_log = time.perf_counter()

        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.steps:
            save_checkpoint(
                out / "latest.pt",
                model,
                mcfg,
                meta,
                step + 1,
                {"opt": opt.state_dict(), "best_val": best_val},
            )
            uploader.push(f"step {step + 1}")
    log.close()
    uploader.wait()
    uploader.push("final")
    uploader.wait()
    return last


def save_checkpoint(
    path: Path,
    model: ChartModel,
    mcfg: ModelConfig,
    meta: dict,
    step: int,
    extra: dict | None = None,
) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": msgspec.to_builtins(mcfg),
            "meta": meta,
            "step": step,
            **(extra or {}),
        },
        tmp,
    )
    tmp.replace(path)


def _fetch_latest(cfg: TrainConfig, out: Path) -> Path | None:
    """``<out>/latest.pt`` and ``best.pt``, downloaded from ``cfg.hub_repo`` when missing."""
    latest = out / "latest.pt"
    if not latest.exists() and cfg.hub_repo:
        from huggingface_hub import hf_hub_download

        for name in ("latest.pt", "best.pt", "metrics.jsonl"):
            with contextlib.suppress(Exception):
                hf_hub_download(cfg.hub_repo, f"{out.name}/{name}", local_dir=out.parent)
    return latest if latest.exists() else None


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ChartModel, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    model = ChartModel(msgspec.convert(state["model_config"], ModelConfig))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state["meta"]


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", help="JSON file with TrainConfig fields")
    defaults = TrainConfig()
    for name in TrainConfig.__struct_fields__:
        value = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        if isinstance(value, list):
            parser.add_argument(flag, nargs="*", default=None)
        else:
            parser.add_argument(flag, default=None)
    args = parser.parse_args(argv)
    fields = json.loads(Path(args.config).read_text()) if args.config else {}
    for name in TrainConfig.__struct_fields__:
        value = getattr(args, name)
        if value is not None:
            fields[name] = value
    return msgspec.convert(fields, TrainConfig, strict=False)


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
