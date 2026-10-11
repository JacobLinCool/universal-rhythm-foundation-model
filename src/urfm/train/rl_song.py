"""Song-level GRPO (R6.2): the model writes several consecutive windows of a song, reading its
own output as context and memory, and is rewarded on the whole span.

Window-level RL (``urfm.train.rl``) scores 16 s written after human context, so nothing that
builds up over a song -- stagnation, a pattern repeated for a minute -- can be rewarded or
penalised: its stagnation term never fired, and the structure reward made charts repeat.
Here each sample is ``span`` windows (about 96 s) of one song; rewards are computed on the
generated span:

* ``place``: placement precision against the human chart (``urfm.eval.metrics.placement``);
* ``f1``: onset F1 at 25 ms (existence, small weight);
* ``rhythm``: ``-L1`` of inter-onset-interval class shares against the human span;
* ``plan``: per section, ``-|log((n_gen + 1) / (n_ref + 1))|`` against the human plan;
* ``struct``: structure alignment of the span's units within the song;
* ``stag``: ``-`` stagnation penalty over the span (debt carried across windows);
* ``etime``: ``-`` excess monotone time (s/min beyond human q95 at the same density).

Samples of one song form a GRPO group; every generated token of a sample gets its advantage.
The loss adds ``kl`` times a k3 KL estimate against the frozen starting model.

    python -m urfm.train.rl_song --init /content/runs/r6x-base/best.pt --out /content/runs/r6x-rl
"""

from __future__ import annotations

import argparse
import contextlib
import copy
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
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from urfm.data.canonical import canonicalize
from urfm.data.features import MEL
from urfm.ir.chart import Chart, ChartSet
from urfm.ir.events import Event
from urfm.ir.serialize import LazyChartSet
from urfm.tokenize.sections import SectionPlan, plan_extras, plan_sections, section_beats
from urfm.tokenize.tokenizer import beats_at_us, playable_events
from urfm.train.conditions import chart_conditions, cond_prefix
from urfm.train.data import (
    build_index,
    collate_song,
    extreme_tempo,
    long_spans,
    onset_beats,
    shard_files,
)
from urfm.train.generate import Job, generate, song_windows
from urfm.train.rl import token_logprobs
from urfm.train.train import GpuMonitor, Uploader, load_checkpoint, load_tokenizer, save_checkpoint

TERMS = ("place", "f1", "rhythm", "plan", "struct", "stag", "etime")


class SongRLConfig(msgspec.Struct, kw_only=True):
    data: str = "/content/data/urfm-ingest-v0"
    sources: list[str] = msgspec.field(default_factory=lambda: ["taiko1000", "osu"])
    mechanics: list[str] = msgspec.field(default_factory=lambda: ["taiko", "osu_taiko"])
    init: str = ""
    steps: int = 200
    songs: int = 4
    """Songs per step (one GRPO group each)."""
    group: int = 4
    """Samples per song."""
    span: int = 6
    """Consecutive windows generated per sample (~16 s each)."""
    micro_batch: int = 24
    lr: float = 1e-5
    kl: float = 0.05
    temperature: float = 1.0
    plan_temperature: float = 0.7
    w_place: float = 1.0
    w_f1: float = 0.25
    w_rhythm: float = 0.5
    w_plan: float = 0.5
    w_struct: float = 0.5
    w_stag: float = 1.0
    w_etime: float = 0.25
    workers: int = 4
    seed: int = 0
    eval_every: int = 25
    eval_songs: int = 16
    out: str = "runs/rl-song"
    hub_repo: str | None = None


@dataclass
class SongPrompt:
    cs: ChartSet
    chart: Chart
    mel: np.ndarray
    plans: dict[int, SectionPlan]
    k: int
    window_range: tuple[int, int]
    start_us: int
    end_us: int
    history: list[Event]
    """Human events before the span (the first window's context)."""
    ref_us: np.ndarray
    ref_beats: np.ndarray
    structure: object


def song_prompt(cs: ChartSet, chart: Chart, mel: np.ndarray, span: int, rng: random.Random):
    """A random span of ``span`` windows that holds notes, with the human plan and context."""
    from urfm.data.sections import section_audio
    from urfm.eval.structure import SongStructure

    timing = cs.timing_of(chart)
    wins = song_windows(timing, round(mel.shape[0] / MEL.fps * 1e6))
    events = sorted(playable_events(chart), key=lambda e: e.time)
    if len(wins) < span + 1 or len(events) < 32:
        return None
    us = np.asarray(
        timing.us_array([e.time.beat for e in events], [e.time.residual_us for e in events])
    )
    filled = [
        i
        for i in range(len(wins) - span + 1)
        if ((us >= wins[i][0]) & (us < wins[i + span - 1][1])).sum() > 16
    ]
    if not filled:
        return None
    a = rng.choice(filled)
    b = a + span
    start_us, end_us = wins[a][0], wins[b - 1][1]
    k = section_beats(timing)
    first, feats = section_audio(mel, timing, k)
    beats, ka = onset_beats(cs, chart, timing)
    last = first + len(feats) - 1
    extras = plan_extras(timing, beats, ka, long_spans(chart), first, last, k)
    plans = {p.section: p for p in plan_sections(timing, beats, ka, first, last, k, extras)}
    sel = (us >= start_us) & (us < end_us)
    ref_us = np.unique(np.round(us[sel]).astype(np.int64))
    # one event per onset: single-action games (Taiko) cannot encode a second one in context
    history, seen = [], set()
    for e, t in zip(events, us, strict=True):
        if t < start_us and e.time not in seen:
            seen.add(e.time)
            history.append(e)
    return SongPrompt(
        cs=cs,
        chart=chart,
        mel=mel,
        plans=plans,
        k=k,
        window_range=(a, b),
        start_us=start_us,
        end_us=end_us,
        history=history,
        ref_us=ref_us,
        ref_beats=beats_at_us(timing, ref_us.astype(np.float64)),
        structure=SongStructure(cs, chart, mel),
    )


class SongStream(IterableDataset):
    def __init__(self, index, span: int, seed: int) -> None:
        self.index, self.span, self.seed = index, span, seed

    def __iter__(self) -> Iterator[SongPrompt]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 7919 + wid)
        refs = self.index.charts[wid::nw] or self.index.charts
        weights = [r.weight for r in refs]
        while True:
            ref = rng.choices(refs, weights)[0]
            try:
                t = pq.ParquetFile(ref.file).read_row_group(
                    ref.row_group, columns=["mel", "chartset"]
                )
                cs = LazyChartSet(t["chartset"].to_pylist()[ref.row]).view(ref.chart)
                chart = cs.charts[0]
                if extreme_tempo(cs.timing_of(chart)):
                    continue
                cs, chart = canonicalize(cs, chart)
                cs = msgspec.structs.replace(cs, charts=(chart,))
                mel = np.frombuffer(t["mel"].to_pylist()[ref.row], np.uint8).reshape(-1, MEL.n_mels)
                p = song_prompt(cs, chart, mel, self.span, rng)
            except Exception:
                continue
            if p is not None:
                yield p


def make_jobs(model, meta, tok, prompts: list[SongPrompt], group: int) -> list[Job]:
    jobs = []
    for p in prompts:
        for _ in range(group):
            jobs.append(
                Job(
                    mel=p.mel,
                    timing=p.cs.timing_of(p.chart),
                    mechanics=p.chart.mechanics,
                    prefix=cond_prefix(tok, p.chart.mechanics),
                    cond=chart_conditions(p.cs, p.chart),
                    events=list(p.history),
                    section_plans=p.plans,
                    section_k=p.k,
                    use_ref=bool(meta.get("features", {}).get("ref")),
                    window_range=p.window_range,
                )
            )
    return jobs


def rewards(p: SongPrompt, chart: Chart, cfg: SongRLConfig) -> tuple[float, dict]:
    from urfm.eval.metrics import _ioi_shares, _l1, match_onsets, placement
    from urfm.eval.monotony import default_references, excess_monotone_time
    from urfm.eval.structure import _colour
    from urfm.eval.variation import window_stagnation

    timing = p.cs.timing_of(p.chart)
    events = [e for e in playable_events(chart) if p.start_us <= timing.us(e.time) < p.end_us]
    us = np.unique(np.asarray([timing.us(e.time) for e in events], np.int64))
    out: dict[str, float] = {}
    out["place"] = (
        float(placement(us, p.ref_us).get("placement_score", 0.0))
        if len(us) and len(p.ref_us)
        else 0.0
    )
    pairs = len(match_onsets(p.ref_us, us, 25_000))
    out["f1"] = 2 * pairs / max(len(us) + len(p.ref_us), 1)
    beats = beats_at_us(timing, us.astype(np.float64)) if len(us) else np.zeros(0)
    out["rhythm"] = -_l1(_ioi_shares(beats), _ioi_shares(p.ref_beats))
    lo_b, hi_b = float(timing.us_to_beat(p.start_us)), float(timing.us_to_beat(p.end_us))
    dens = []
    for i in range(math.floor(lo_b / p.k), math.floor(hi_b / p.k) + 1):
        a, b = max(i * p.k, lo_b), min((i + 1) * p.k, hi_b)
        n_gen = int(((beats >= a) & (beats < b)).sum())
        n_ref = int(((p.ref_beats >= a) & (p.ref_beats < b)).sum())
        dens.append(-abs(math.log((n_gen + 1) / (n_ref + 1))))
    out["plan"] = float(np.mean(dens)) if dens else 0.0
    colours = np.array([_colour(e) for e in events], np.int64)
    ev_beats = np.array([float(e.time.beat) for e in events])
    score = p.structure.window_score(ev_beats, colours, lo_b, hi_b)  # type: ignore[attr-defined]
    out["struct"] = float(score) if score is not None else 0.0
    hist = Chart(mechanics=chart.mechanics, events=(*p.history, *events))
    out["stag"] = -window_stagnation(
        ChartSet(timing=timing, charts=(hist,)), hist, p.start_us / 1e6
    )
    span = Chart(mechanics=chart.mechanics, events=tuple(events))
    _, stretch_ref = default_references()
    m = excess_monotone_time(ChartSet(timing=timing, charts=(span,)), span, stretch_ref)
    out["etime"] = -float(m["excess_s_per_min"] or 0.0)
    weights = {f"w_{t}": getattr(cfg, f"w_{t}") for t in TERMS}
    total = sum(weights[f"w_{t}"] * out[t] for t in TERMS)
    return total, out


def rollout(model, meta, tok, prompts, group, cfg, device, seed, temperature):
    jobs = make_jobs(model, meta, tok, prompts, group)
    record: list[dict] = []
    charts = generate(
        model,
        tok,
        jobs,
        temperature=temperature,
        top_p=1.0,
        device=device,
        seed=seed,
        plan="sample",
        plan_temperature=cfg.plan_temperature,
        record=record,
    )
    scored = [rewards(prompts[i // group], c, cfg) for i, c in enumerate(charts)]
    return record, scored


def window_batch(rows: list[dict], tok, device) -> dict[str, torch.Tensor]:
    seqs = [r["prompt"] + r["tokens"] for r in rows]
    length = max(len(x) for x in seqs)
    tokens = torch.full((len(rows), length), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), length), dtype=torch.bool)
    for i, (r, x) in enumerate(zip(rows, seqs, strict=True)):
        tokens[i, : len(x)] = torch.tensor(x)
        mask[i, len(r["prompt"]) : len(x)] = True
    b = {
        "mel": torch.from_numpy(np.stack([r["mel"] for r in rows])),
        "feats": torch.from_numpy(np.stack([r["feats"] for r in rows])),
        "tokens": tokens,
        "gen_mask": mask,
    }
    if all(r["song"][0] is not None for r in rows):
        b |= collate_song([r["song"] for r in rows])
    if all(r["cond"] is not None for r in rows):
        b["cond"] = torch.from_numpy(np.stack([r["cond"][0] for r in rows]))
        b["cond_mask"] = torch.from_numpy(np.stack([r["cond"][1] for r in rows]))
    return {k: v.to(device) for k, v in b.items()}


def train(cfg: SongRLConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type == "cuda"
    tok = load_tokenizer()
    model, meta = load_checkpoint(cfg.init, str(device))
    ref = copy.deepcopy(model).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    mech = set(cfg.mechanics)
    index = build_index(shard_files(cfg.data, cfg.sources, "train"), mech, 0.5)
    val_index = build_index(shard_files(cfg.data, cfg.sources, "validation"), mech, 0.5)
    prompts_iter = iter(
        DataLoader(
            SongStream(index, cfg.span, cfg.seed),
            batch_size=None,
            num_workers=cfg.workers,
            persistent_workers=cfg.workers > 0,
        )
    )
    val_iter = iter(SongStream(val_index, cfg.span, 99))
    val_prompts = [next(val_iter) for _ in range(cfg.eval_songs)]
    meta_out = {"rl_song": msgspec.to_builtins(cfg), **{k: v for k, v in meta.items() if k != "rl"}}
    (out / "config.json").write_text(json.dumps(meta_out, indent=2, default=str))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    gpu = GpuMonitor()
    uploader = Uploader(cfg.hub_repo, out)
    log = (out / "metrics.jsonl").open("a")
    best = -1e9

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    def evaluate(step: int) -> dict:
        _, scored = rollout(model, meta, tok, val_prompts, 1, cfg, device, 12345, 0.9)
        return {
            "val_reward": float(np.mean([r for r, _ in scored])),
            **{f"val_{t}": float(np.mean([d[t] for _, d in scored])) for t in TERMS},
        }

    last = evaluate(0) | {"step": 0}
    log.write(json.dumps(last) + "\n")
    log.flush()
    print(json.dumps(last), flush=True)
    for step in range(cfg.steps):
        t0 = time.perf_counter()
        prompts = [next(prompts_iter) for _ in range(cfg.songs)]
        t1 = time.perf_counter()
        record, scored = rollout(
            model,
            meta,
            tok,
            prompts,
            cfg.group,
            cfg,
            device,
            cfg.seed * 1_000_003 + step,
            cfg.temperature,
        )
        t2 = time.perf_counter()
        adv = np.zeros(len(scored))
        for s in range(cfg.songs):
            r = np.asarray([scored[s * cfg.group + g][0] for g in range(cfg.group)])
            if r.std() > 1e-6:
                adv[s * cfg.group : (s + 1) * cfg.group] = (r - r.mean()) / (r.std() + 1e-4)
        rows = [r for r in record if r["tokens"]]
        n_tokens = sum(len(r["tokens"]) for r in rows)
        model.train()
        opt.zero_grad(set_to_none=True)
        kl_sum = 0.0
        for i in range(0, len(rows), cfg.micro_batch):
            chunk = rows[i : i + cfg.micro_batch]
            b = window_batch(chunk, tok, device)
            a = torch.tensor([adv[r["job"]] for r in chunk], device=device, dtype=torch.float32)[
                :, None
            ]
            m = b["gen_mask"][:, 1:].float()
            with amp():
                logp = token_logprobs(model, b)
                with torch.no_grad():
                    ref_logp = token_logprobs(ref, b)
            diff = (ref_logp - logp).clamp(-10, 10)
            kl = torch.exp(diff) - diff - 1
            loss = ((-a * logp + cfg.kl * kl) * m).sum() / max(n_tokens, 1)
            loss.backward()
            kl_sum += float((kl.detach() * m).sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        t3 = time.perf_counter()
        last = {
            "step": step + 1,
            "reward": float(np.mean([r for r, _ in scored])),
            **{t: float(np.mean([d[t] for _, d in scored])) for t in TERMS},
            "kl": kl_sum / max(n_tokens, 1),
            "tokens": n_tokens,
            "prompt_s": t1 - t0,
            "rollout_s": t2 - t1,
            "update_s": t3 - t2,
            "gpu_util": gpu.take(),
        }
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            last |= evaluate(step + 1)
            save_checkpoint(out / "latest.pt", model, model.cfg, meta_out, step + 1)
            if last["val_reward"] > best:
                best = last["val_reward"]
                save_checkpoint(out / "best.pt", model, model.cfg, meta_out, step + 1)
        log.write(json.dumps(last) + "\n")
        log.flush()
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            uploader.push(f"song rl step {step + 1}")
        print(
            json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in last.items()}),
            flush=True,
        )
    log.close()
    uploader.wait()
    uploader.push("final")
    uploader.wait()
    return last


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    defaults = SongRLConfig()
    for name in SongRLConfig.__struct_fields__:
        value = getattr(defaults, name)
        ap.add_argument(
            "--" + name.replace("_", "-"),
            nargs="*" if isinstance(value, list) else None,
            default=None,
        )
    args = ap.parse_args(argv)
    fields = {n: getattr(args, n) for n in SongRLConfig.__struct_fields__}
    train(
        msgspec.convert(
            {k: v for k, v in fields.items() if v is not None}, SongRLConfig, strict=False
        )
    )


if __name__ == "__main__":
    main()
