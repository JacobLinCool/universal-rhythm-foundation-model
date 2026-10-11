"""Train the beat model.

    python -m urfm.beat.train --data /content/data/urfm-ingest-v0 --out /content/runs/beat-v1 \\
        --steps 8000 --batch-size 32

Every config field is a CLI flag. Logs JSONL metrics (loss, chosen metrical level, data
wait, GPU utilisation) and, at each evaluation, validation loss and beat F-measures on
fixed unaugmented chunks; ``best.pt`` keeps the best validation ``beat_f70_any_level``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import msgspec
import numpy as np
import torch

from urfm.beat.data import ChunkStream, build_index, collate, fixed_chunks, loader
from urfm.beat.decode import peaks
from urfm.beat.metrics import beat_scores
from urfm.beat.model import BeatConfig, BeatModel, beat_loss
from urfm.data.features import MEL
from urfm.train.data import shard_files
from urfm.train.train import GpuMonitor, Uploader


class BeatTrainConfig(msgspec.Struct, kw_only=True):
    data: str = "/content/data/urfm-ingest-v0"
    sources: list[str] = msgspec.field(default_factory=lambda: ["taiko1000", "osu"])
    d_model: int = 256
    layers: int = 6
    heads: int = 4
    batch_size: int = 32
    lr: float = 5e-4
    min_lr_ratio: float = 0.05
    weight_decay: float = 0.01
    warmup: int = 500
    steps: int = 8000
    clip: float = 1.0
    workers: int = 8
    compile: bool = True
    seed: int = 0
    log_every: int = 50
    eval_every: int = 1000
    eval_chunks: int = 256
    out: str = "runs/beat"
    hub_repo: str | None = None


def lr_at(step: int, cfg: BeatTrainConfig) -> float:
    import math

    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    floor = cfg.lr * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model: BeatModel, batches: list[dict], device: torch.device) -> dict[str, float]:
    model.eval()
    losses, scores = [], []
    cuda = device.type == "cuda"
    for batch in batches:
        b = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = model(b["mel"])
        loss, _ = beat_loss(
            logits, b["beats"] / 2, b["downbeats"] / 2, b["mask"], model.cfg.downbeat_weight
        )
        losses.append(float(loss))
        act = torch.sigmoid(logits.float()).cpu().numpy()
        for i in range(act.shape[0]):
            mask = batch["mask"][i].numpy()
            if not mask.any():
                continue
            frames = np.nonzero(mask)[0]
            span = (frames[0] / MEL.fps, frames[-1] / MEL.fps)
            ref = np.nonzero(batch["beats"][i, 0].numpy() == 2)[0] / MEL.fps
            ref_d = np.nonzero(batch["downbeats"][i].numpy() == 2)[0] / MEL.fps
            est = peaks(act[i, :, 0])
            est_d = peaks(act[i, :, 1], 0.5)
            s = beat_scores(ref, est, span)
            s["downbeat_f70"] = beat_scores(ref_d, est_d, span)["beat_f70"]
            scores.append(s)
    model.train()
    out = {"val_loss": float(np.mean(losses))}
    for key in scores[0] if scores else ():
        out[f"val_{key}"] = float(np.mean([s[key] for s in scores]))
    return out


def train(cfg: BeatTrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_songs = build_index(shard_files(cfg.data, cfg.sources, "train"))
    val_songs = build_index(shard_files(cfg.data, cfg.sources, "validation"))
    print(
        json.dumps(
            {
                "train_songs": len(train_songs),
                "val_songs": len(val_songs),
                "index_s": round(time.time() - t0, 1),
            }
        ),
        flush=True,
    )
    mcfg = BeatConfig(d_model=cfg.d_model, layers=cfg.layers, heads=cfg.heads)
    model = BeatModel(mcfg).to(device)
    meta = {
        "train": msgspec.to_builtins(cfg),
        "model": msgspec.to_builtins(mcfg),
        "params": model.num_params(),
        "train_songs": len(train_songs),
        "val_songs": len(val_songs),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=cuda,
    )
    forward = torch.compile(model, dynamic=False) if cfg.compile and cuda else model
    val = fixed_chunks(val_songs, cfg.eval_chunks, cfg.seed)
    val_batches = [collate(val[i : i + cfg.batch_size]) for i in range(0, len(val), cfg.batch_size)]
    batches = iter(loader(ChunkStream(train_songs, cfg.batch_size, cfg.seed), cfg.workers))
    gpu = GpuMonitor()
    uploader = Uploader(cfg.hub_repo, out)
    log = (out / "metrics.jsonl").open("a")
    best = -1.0
    acc_loss = torch.zeros((), device=device)
    levels = torch.zeros(4, device=device)
    wait = 0.0
    t_log = time.perf_counter()
    last: dict = {}
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        t1 = time.perf_counter()
        batch = next(batches)
        wait += time.perf_counter() - t1
        b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            logits = forward(b["mel"])
        loss, level = beat_loss(
            logits, b["beats"] / 2, b["downbeats"] / 2, b["mask"], mcfg.downbeat_weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        acc_loss += loss.detach()
        levels += torch.bincount(level, minlength=4).float()
        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_log
            lv = (levels / levels.sum().clamp_min(1)).tolist()
            last = {
                "step": step + 1,
                "loss": float(acc_loss) / cfg.log_every,
                "level_frac": [round(x, 3) for x in lv],
                "lr": opt.param_groups[0]["lr"],
                "chunks_per_s": cfg.log_every * cfg.batch_size / dt,
                "data_wait_frac": wait / dt,
                "gpu_util": gpu.take(),
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if cuda else None,
                "time": time.time(),
            }
            if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
                last |= evaluate(model, val_batches, device)
                score = last.get("val_beat_f70_any_level", 0.0)
                if score > best:
                    best = score
                    save(out / "best.pt", model, meta, step + 1)
            log.write(json.dumps(last) + "\n")
            log.flush()
            print(
                json.dumps(
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in last.items()}
                ),
                flush=True,
            )
            acc_loss.zero_()
            levels.zero_()
            wait = 0.0
            t_log = time.perf_counter()
            if (step + 1) % cfg.eval_every == 0:
                save(out / "latest.pt", model, meta, step + 1)
                uploader.push(f"step {step + 1}")
    save(out / "latest.pt", model, meta, cfg.steps)
    log.close()
    uploader.wait()
    uploader.push("final")
    uploader.wait()
    return last


def save(path: Path, model: BeatModel, meta: dict, step: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "config": meta["model"], "meta": meta, "step": step}, tmp
    )
    tmp.replace(path)


def load(path: str | Path, device: str = "cpu") -> BeatModel:
    state = torch.load(path, map_location=device, weights_only=False)
    model = BeatModel(msgspec.convert(state["config"], BeatConfig))
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = BeatTrainConfig()
    for name in BeatTrainConfig.__struct_fields__:
        value = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        parser.add_argument(flag, nargs="*" if isinstance(value, list) else None, default=None)
    args = parser.parse_args(argv)
    fields = {n: getattr(args, n) for n in BeatTrainConfig.__struct_fields__}
    fields = {k: v for k, v in fields.items() if v is not None}
    with contextlib.suppress(KeyboardInterrupt):
        train(msgspec.convert(fields, BeatTrainConfig, strict=False))


if __name__ == "__main__":
    main()
