"""Reinforcement learning on the model's own windows (GRPO-style).

Teacher forcing never shows the model its own mistakes: the 15M model continues fast passages
from human context with high probability yet rarely starts one when writing alone, and charts
that dense come out at two thirds of the human note count. Here the model writes whole
windows itself and is rewarded at the window level:

* ``f1``: onset F1 (25 ms) against the human chart of the window;
* ``density``: ``-|log((n_gen + 1) / (n_ref + 1))|``, how far the note count is from the
  human's (the plan's ``WDENS`` is forced to the human value, as in calibrated generation);
* ``fast``: ``-|log((f_gen + 1) / (f_ref + 1))|`` over the numbers of intervals under 60 ms,
  whether fast passages appear as often as the human wrote them;
* ``rhythm``: ``-L1`` between the inter-onset-interval class shares (in beats) of the window and
  the human's. Without it, the first run traded rhythmic variety for F1 (IOI L1 0.49 -> 0.71)
  and dropped fast passages, whose share-based penalty was too small to matter.

R6 terms (section-plan models; weights default to 0):

* ``place``: placement precision -- over generated notes with a human onset within 75 ms, the
  mean of ``exp(-(d / 12 ms)^2 / 2)``. Whether a note exists is uncertain (one human chart is
  one valid answer), but a note that is there must sit exactly on its place; F1 keeps a small
  weight as the existence anchor;
* ``plan``: per section of the window, ``-|log((n_gen + 1) / (n_ref + 1))|`` plus whether the
  finest subdivision used is the planned one;
* ``struct``: the window's rows of the chart self-similarity matrix (rest of the song from the
  human chart) correlated with the audio's (``urfm.eval.structure.SongStructure``);
* ``stag``: ``-`` the stagnation penalty of the window, debt carried in from the human chart
  before it (``urfm.eval.variation.window_stagnation``).

For each prompt (human context, conditions, audio window) ``group`` windows are sampled; each
window's advantage is its reward minus the group mean, over the group standard deviation. The
loss is the advantage-weighted token log-likelihood of the sampled windows (token-level mean)
plus ``kl`` times a k3 estimate of the KL divergence from the frozen starting model, which
keeps the policy close to charts it already writes well.

    python -m urfm.train.rl --init /content/runs/r5-sft/best.pt --out /content/runs/r5-rl
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
from types import SimpleNamespace

import msgspec
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from urfm.data.features import MEL
from urfm.eval.metrics import match_onsets
from urfm.ir.serialize import LazyChartSet
from urfm.ir.timing import TimingMap
from urfm.tokenize.api import Tokenizer
from urfm.tokenize.tokenizer import chart_lanes
from urfm.train.data import (
    FAST_MS,
    Augment,
    Features,
    Index,
    Sample,
    augment_chart,
    build_index,
    collate,
    extreme_tempo,
    make_sample,
    prepare,
    shard_files,
    window_weights,
)
from urfm.train.generate import _Grammar, _plan_tokens_mask, decode_batch
from urfm.train.train import GpuMonitor, Uploader, load_checkpoint, load_tokenizer, save_checkpoint


class RLConfig(msgspec.Struct, kw_only=True):
    data: str = "/content/data/urfm-ingest-v0"
    sources: list[str] = msgspec.field(default_factory=lambda: ["taiko1000", "osu"])
    mechanics: list[str] = msgspec.field(default_factory=lambda: ["taiko", "osu_taiko"])
    init: str = ""
    steps: int = 200
    prompts: int = 16
    """Prompts (windows) per step."""
    group: int = 8
    """Samples per prompt."""
    micro_batch: int = 32
    lr: float = 1e-5
    kl: float = 0.05
    temperature: float = 1.0
    w_f1: float = 1.0
    w_density: float = 0.5
    w_fast: float = 0.5
    w_rhythm: float = 0.5
    w_place: float = 0.0
    w_plan: float = 0.0
    w_struct: float = 0.0
    w_stag: float = 0.0
    octave_factors: list[float] = msgspec.field(default_factory=list)
    fast_boost: float = 4.0
    p_speed: float = 0.5
    p_local: float = 0.3
    p_octave: float = 0.15
    workers: int = 6
    seed: int = 0
    eval_every: int = 25
    eval_prompts: int = 64
    out: str = "runs/rl"
    hub_repo: str | None = None


@dataclass
class Prompt:
    sample: Sample
    """The human window (``tokens`` holds context, ``SEP``, plan and the human body)."""
    timing: TimingMap
    window: tuple[int, int]
    mechanics: str
    lanes: int | None
    ref_us: np.ndarray
    """Human onsets (µs, unique, sorted) in the window."""
    plan_end: int = 0
    """Index after the forced plan (``SEP`` + 1 + plan tokens); 0 = ``SEP`` + 2 (``WDENS``)."""
    sections: list | None = None
    """R6: ``(section, start beat, end beat, planned subdivision index, human onset count)``."""
    structure: object | None = None
    """R6: ``urfm.eval.structure.SongStructure`` of the song and the human chart."""
    history: object | None = None
    """R6: the human chart's events in the 40 s before the window (stagnation debt)."""
    start_beat: float = 0.0
    end_beat: float = 0.0

    @property
    def prompt(self) -> list[int]:
        """Prefix, ``BOS``, (memory,) context, ``SEP`` and the forced human plan."""
        end = self.plan_end or self.sample.sep + 2
        return self.sample.tokens[:end].tolist()


def onsets_us(tok: Tokenizer, timing: TimingMap, mechanics: str, body: list[int], start: int):
    return decode_window(tok, timing, mechanics, body, start)[0]


def decode_window(tok: Tokenizer, timing: TimingMap, mechanics: str, body: list[int], start: int):
    """``(unique onsets µs, events)`` of a generated window body."""
    events, _, _ = tok.decode_full(timing, mechanics, body, start)  # type: ignore[attr-defined]
    us = np.unique(np.asarray([timing.us(e.time) for e in events], dtype=np.int64))
    return us, events


PLAN_KINDS = frozenset({"SBEAT", "SDENS", "SFINE", "SREST", "SKA", "SSUB"})


def r6_terms(gen_us: np.ndarray, events: list, p: Prompt) -> dict[str, float]:
    """The R6 reward terms of one generated window (see the module docstring)."""
    from urfm.eval.metrics import placement
    from urfm.eval.structure import _colour
    from urfm.ir.chart import Chart, ChartSet
    from urfm.tokenize.sections import finest_grid
    from urfm.tokenize.tokenizer import beats_at_us

    out: dict[str, float] = {}
    if len(gen_us) and len(p.ref_us):
        out["place"] = float(placement(gen_us, p.ref_us).get("placement_score", 0.0))
    else:
        out["place"] = 1.0 if not len(gen_us) and not len(p.ref_us) else 0.0
    beats = beats_at_us(p.timing, gen_us.astype(np.float64)) if len(gen_us) else np.zeros(0)
    if p.sections:
        dens, subs = [], []
        for _i, a, b, sub, n_ref in p.sections:
            sel = beats[(beats >= a - 1e-6) & (beats < b - 1e-6)]
            dens.append(-abs(math.log((len(sel) + 1) / (n_ref + 1))))
            if len(sel) >= 4 and sub > 0:
                subs.append(float(finest_grid(sel) == sub))
        out["plan"] = float(np.mean(dens)) + (float(np.mean(subs)) if subs else 0.0)
    if p.structure is not None:
        colours = np.array([_colour(e) for e in events], np.int64)
        ev_beats = np.array([float(e.time.beat) for e in events])
        score = p.structure.window_score(ev_beats, colours, p.start_beat, p.end_beat)  # type: ignore[attr-defined]
        out["struct"] = float(score) if score is not None else 0.0
    if p.history is not None:
        from urfm.eval.variation import window_stagnation

        hist = p.history
        chart = Chart(mechanics=p.mechanics, events=(*hist.events, *events))  # type: ignore[attr-defined]
        cs = ChartSet(timing=p.timing, charts=(chart,))
        out["stag"] = -window_stagnation(cs, chart, p.window[0] / 1e6)
    return out


def full_reward(tok: Tokenizer, p: Prompt, body: list[int], cfg: RLConfig) -> tuple[float, dict]:
    """Base reward plus the weighted R6 terms."""
    gen_us, events = decode_window(tok, p.timing, p.mechanics, body, p.window[0])
    total, parts = reward(gen_us, p.ref_us, cfg, p.timing)
    if cfg.w_place or cfg.w_plan or cfg.w_struct or cfg.w_stag:
        extra = r6_terms(gen_us, events, p)
        weights = {"place": cfg.w_place, "plan": cfg.w_plan, "struct": cfg.w_struct}
        weights["stag"] = cfg.w_stag
        total += sum(weights[k] * v for k, v in extra.items())
        parts |= extra
    return total, parts


def fast_share(us: np.ndarray) -> float:
    if len(us) < 2:
        return 0.0
    return float((np.diff(us) < FAST_MS * 1000).mean())


def fast_count(us: np.ndarray) -> int:
    return int((np.diff(us) < FAST_MS * 1000).sum()) if len(us) > 1 else 0


def reward(
    gen_us: np.ndarray, ref_us: np.ndarray, cfg: RLConfig, timing: TimingMap | None = None
) -> tuple[float, dict]:
    pairs = len(match_onsets(ref_us, gen_us, 25_000))
    f1 = 2 * pairs / max(len(ref_us) + len(gen_us), 1) if len(ref_us) + len(gen_us) else 1.0
    density = -abs(math.log((len(gen_us) + 1) / (len(ref_us) + 1)))
    fast = -abs(math.log((fast_count(gen_us) + 1) / (fast_count(ref_us) + 1)))
    rhythm = 0.0
    if timing is not None and (len(gen_us) > 1 or len(ref_us) > 1):
        from urfm.eval.metrics import _ioi_shares, _l1
        from urfm.tokenize.tokenizer import beats_at_us

        g = _ioi_shares(beats_at_us(timing, gen_us.astype(np.float64)))
        r = _ioi_shares(beats_at_us(timing, ref_us.astype(np.float64)))
        rhythm = -_l1(g, r)
    total = cfg.w_f1 * f1 + cfg.w_density * density + cfg.w_fast * fast + cfg.w_rhythm * rhythm
    return total, {
        "f1": f1,
        "density": density,
        "fast": fast,
        "rhythm": rhythm,
        "fast_gen": fast_share(gen_us),
    }


class PromptStream(IterableDataset):
    """Training windows (augmented like supervised training, fast passages oversampled)."""

    def __init__(
        self,
        index: Index,
        tok: Tokenizer,
        features: Features,
        augment: Augment,
        max_dec_len: int,
        seed: int,
    ) -> None:
        self.index, self.tok, self.features, self.augment = index, tok, features, augment
        self.max_dec_len, self.seed = max_dec_len, seed

    def __iter__(self) -> Iterator[Prompt]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 7919 + wid)
        refs = self.index.charts[wid::nw] or self.index.charts
        weights = [r.weight for r in refs]
        tables: dict[tuple[str, int], tuple[list, list]] = {}
        while True:
            ref = rng.choices(refs, weights)[0]
            key = (ref.file, ref.row_group)
            if key not in tables:
                if len(tables) > 8:
                    tables.pop(next(iter(tables)))
                t = pq.ParquetFile(ref.file).read_row_group(
                    ref.row_group, columns=["mel", "chartset"]
                )
                tables[key] = (t["mel"].to_pylist(), t["chartset"].to_pylist())
            mels, sets = tables[key]
            try:
                cs = LazyChartSet(sets[ref.row]).view(ref.chart)
                if extreme_tempo(cs.timing_of(cs.charts[0])):
                    continue
                mel = np.frombuffer(mels[ref.row], np.uint8).reshape(-1, MEL.n_mels)
                cs, chart, audio, plain = augment_chart(cs, cs.charts[0], mel, rng, self.augment)
                prep = prepare(
                    self.tok,
                    cs,
                    chart,
                    mel.shape[0],
                    None,
                    self.features,
                    audio,
                    plain.timing_of(plain.charts[0]),
                )
                if not prep.windows:
                    continue
                w = window_weights(prep, self.augment.fast_boost)
                window = rng.choices(prep.windows, w)[0] if w else rng.choice(prep.windows)
                s = make_sample(self.tok, cs, prep, mel, window, self.max_dec_len, self.features)
            except Exception:
                continue
            if s.truncated:
                continue
            timing = cs.timing_of(chart)
            body = s.tokens[s.sep + 1 :].tolist()
            ref_us = onsets_us(self.tok, timing, chart.mechanics, body, window[0])
            p = Prompt(s, timing, window, chart.mechanics, chart_lanes(chart), ref_us)
            if self.features.sections:
                try:
                    self._r6(p, prep, cs, mel, plain)
                except Exception:
                    continue
            yield p

    def _r6(self, p: Prompt, prep, cs, mel: np.ndarray, plain) -> None:
        from urfm.eval.structure import SongStructure
        from urfm.tokenize.sections import plan_sections, window_sections
        from urfm.train.data import section_info

        kinds = self.tok.vocab.kinds  # type: ignore[attr-defined]
        toks = p.sample.tokens.tolist()
        end = p.sample.sep + 1
        while end < len(toks) and kinds[toks[end]] in PLAN_KINDS:
            end += 1
        p.plan_end = end
        timing, (w0, w1) = p.timing, p.window
        info = section_info(prep, cs, mel)
        a, b = window_sections(timing, w0, w1, info.k)
        p.start_beat, p.end_beat = float(timing.us_to_beat(w0)), float(timing.us_to_beat(w1))
        ref_beats = info.beats[(info.beats >= p.start_beat) & (info.beats < p.end_beat)]
        p.sections = []
        for plan in plan_sections(timing, info.beats, info.ka, a, b, info.k):
            lo = max(plan.section * info.k, p.start_beat)
            hi = min((plan.section + 1) * info.k, p.end_beat)
            n_ref = int(((ref_beats >= lo - 1e-6) & (ref_beats < hi - 1e-6)).sum())
            p.sections.append((plan.section, lo, hi, plan.sub, n_ref))
        # the pre-warp chart has the same beats on a timing that matches the recorded audio
        p.structure = SongStructure(plain, plain.charts[0], mel)
        lo_us = max(w0 - 40_000_000, 0)
        p.history = prep.between(lo_us, w0)


def _batch(prompts: list[Prompt], tok: Tokenizer, device) -> dict[str, torch.Tensor]:
    b = collate([p.sample for p in prompts], tok.pad_id)
    return {k: v.to(device) for k, v in b.items() if k != "stats"}


@torch.no_grad()
def rollout(model, tok, prompts: list[Prompt], group: int, temperature: float, gen):
    """``group`` windows per prompt: list (per prompt) of lists of generated token lists."""
    device = next(model.parameters()).device
    rows = [p for p in prompts for _ in range(group)]
    b = _batch(rows, tok, device)
    song = {k: b[k] for k in ("song", "song_marker", "song_len") if k in b}
    cond = model.condition(b.get("cond"), b.get("cond_mask"))
    forbid = _plan_tokens_mask(tok) if any(p.plan_end for p in rows) else None
    grammars = [
        _Grammar(
            tok,
            SimpleNamespace(mechanics=p.mechanics, lanes=p.lanes),  # type: ignore[arg-type]
            p.prompt,
            forbid if p.plan_end else None,
        )
        for p in rows
    ]
    outs = decode_batch(
        model,
        tok,
        [p.prompt for p in rows],
        grammars,
        b["mel"],
        b["feats"],
        song,
        cond,
        temperature=temperature,
        top_p=1.0,
        plan="sample",
        plan_temperature=1.0,
        gen=gen,
    )
    return [outs[i * group : (i + 1) * group] for i in range(len(prompts))]


def sequence_batch(prompts: list[Prompt], outs: list[list[int]], tok, device):
    """Teacher-forcing batch for ``prompt + generated`` with a mask on generated targets."""
    seqs = [p.prompt + o for p, o in zip(prompts, outs, strict=True)]
    length = max(len(x) for x in seqs)
    tokens = torch.full((len(seqs), length), tok.pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), length), dtype=torch.bool)
    for i, (p, x) in enumerate(zip(prompts, seqs, strict=True)):
        tokens[i, : len(x)] = torch.tensor(x)
        mask[i, len(p.prompt) : len(x)] = True
    b = _batch(prompts, tok, device)
    b["tokens"], b["gen_mask"] = tokens.to(device), mask.to(device)
    return b


def token_logprobs(model, b) -> torch.Tensor:
    song = (
        (b["song"], b["song_marker"], b["song_len"])
        if model.cfg.song_context
        else (None, None, None)
    )
    logits = model(
        b["mel"], b["feats"], b["tokens"][:, :-1], *song, b.get("cond"), b.get("cond_mask")
    )
    logp = F.log_softmax(logits.float(), -1)
    return logp.gather(-1, b["tokens"][:, 1:, None])[..., 0]  # [B, L-1]


def train(cfg: RLConfig) -> dict:
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
    feats = dict(meta.get("features") or {})
    features = Features(
        cond=bool(feats.get("cond")),
        texture=bool(feats.get("texture")),
        sections=bool(feats.get("sections")),
        ref=bool(feats.get("ref")),
        plan_extra=bool(feats.get("plan_extra")),
    )
    pairs = cfg.octave_factors
    augment = Augment(
        canonical=bool(feats.get("canonical")),
        p_speed=cfg.p_speed,
        p_local=cfg.p_local,
        p_octave=cfg.p_octave,
        octave_factors=tuple(
            (float(pairs[i]), float(pairs[i + 1])) for i in range(0, len(pairs) - 1, 2)
        ),
        fast_boost=cfg.fast_boost,
    )
    index = build_index(shard_files(cfg.data, cfg.sources, "train"), set(cfg.mechanics), 0.5)
    val_index = build_index(
        shard_files(cfg.data, cfg.sources, "validation"), set(cfg.mechanics), 0.5
    )
    stream = PromptStream(index, tok, features, augment, model.cfg.max_dec_len, cfg.seed)
    prompts_iter = iter(
        DataLoader(
            stream,
            batch_size=None,
            num_workers=cfg.workers,
            persistent_workers=cfg.workers > 0,
        )
    )
    val_stream = PromptStream(
        val_index, tok, features, Augment(canonical=augment.canonical), model.cfg.max_dec_len, 99
    )
    val_iter = iter(val_stream)
    val_prompts = [next(val_iter) for _ in range(cfg.eval_prompts)]
    meta_out = {"rl": msgspec.to_builtins(cfg), **{k: v for k, v in meta.items() if k != "rl"}}
    (out / "config.json").write_text(json.dumps(meta_out, indent=2, default=str))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    gpu = GpuMonitor()
    uploader = Uploader(cfg.hub_repo, out)
    log = (out / "metrics.jsonl").open("a")
    best = -1e9

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if cuda else contextlib.nullcontext()

    def evaluate() -> dict:
        model.eval()
        outs = rollout(model, tok, val_prompts, 1, 0.85, gen)
        parts = [full_reward(tok, p, o[0], cfg) for p, o in zip(val_prompts, outs, strict=True)]
        model.train()
        return {
            "val_reward": float(np.mean([r for r, _ in parts])),
            **{f"val_{k}": float(np.mean([d[k] for _, d in parts])) for k in parts[0][1]},
            "val_fast_ref": float(np.mean([fast_share(p.ref_us) for p in val_prompts])),
        }

    last = evaluate() | {"step": 0}
    log.write(json.dumps(last) + "\n")
    print(json.dumps(last), flush=True)
    for step in range(cfg.steps):
        t0 = time.perf_counter()
        prompts = [next(prompts_iter) for _ in range(cfg.prompts)]
        t1 = time.perf_counter()
        model.eval()
        outs = rollout(model, tok, prompts, cfg.group, cfg.temperature, gen)
        model.train()
        t2 = time.perf_counter()
        rows, gens, adv, stats = [], [], [], []
        for p, group in zip(prompts, outs, strict=True):
            scored = [full_reward(tok, p, o, cfg) for o in group]
            r = np.asarray([x for x, _ in scored])
            a = (r - r.mean()) / (r.std() + 1e-4) if r.std() > 1e-6 else np.zeros_like(r)
            rows += [p] * len(group)
            gens += group
            adv += a.tolist()
            stats += [d | {"reward": x} for x, d in scored]
        n_tokens = sum(len(g) for g in gens)
        opt.zero_grad(set_to_none=True)
        kl_sum = 0.0
        for i in range(0, len(rows), cfg.micro_batch):
            sl = slice(i, i + cfg.micro_batch)
            b = sequence_batch(rows[sl], gens[sl], tok, device)
            a = torch.tensor(adv[sl], device=device)[:, None]
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
        t3 = time.perf_counter()
        last = {
            "step": step + 1,
            "reward": float(np.mean([s["reward"] for s in stats])),
            "f1": float(np.mean([s["f1"] for s in stats])),
            "density": float(np.mean([s["density"] for s in stats])),
            "fast": float(np.mean([s["fast"] for s in stats])),
            "rhythm": float(np.mean([s["rhythm"] for s in stats])),
            "fast_gen": float(np.mean([s["fast_gen"] for s in stats])),
            **{
                k: float(np.mean([s[k] for s in stats if k in s]))
                for k in ("place", "plan", "struct", "stag")
                if any(k in s for s in stats)
            },
            "kl": kl_sum / max(n_tokens, 1),
            "tokens": n_tokens,
            "prompt_s": t1 - t0,
            "rollout_s": t2 - t1,
            "update_s": t3 - t2,
            "gpu_util": gpu.take(),
        }
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            last |= evaluate()
            if last["val_reward"] > best:
                best = last["val_reward"]
                save_checkpoint(out / "best.pt", model, model.cfg, meta_out, step + 1)
            save_checkpoint(out / "latest.pt", model, model.cfg, meta_out, step + 1)
            uploader.push(f"rl step {step + 1}")
        log.write(json.dumps(last) + "\n")
        log.flush()
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
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = RLConfig()
    for name in RLConfig.__struct_fields__:
        value = getattr(defaults, name)
        flag = "--" + name.replace("_", "-")
        parser.add_argument(flag, nargs="*" if isinstance(value, list) else None, default=None)
    args = parser.parse_args(argv)
    fields = {n: getattr(args, n) for n in RLConfig.__struct_fields__}
    train(
        msgspec.convert({k: v for k, v in fields.items() if v is not None}, RLConfig, strict=False)
    )


if __name__ == "__main__":
    main()
