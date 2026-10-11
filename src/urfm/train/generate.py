"""Window-by-window chart generation with grammar-constrained sampling.

Jobs (one target chart each) are generated in parallel: at window ``k`` every job
that still has audio left contributes one row to the batch. Each row's prompt is
``prefix + BOS + context + SEP`` where the context re-encodes the job's last
``CONTEXT_EVENTS`` generated events, exactly as in training. Prompts are
left-padded, prefilled once, then decoded token by token with a KV cache.

Grammar: if the tokenizer exposes an incremental state machine
(``grammar_init(mechanics)``, ``grammar_mask(state)``, ``grammar_step(state, token)``)
it is used; otherwise ``allowed(mechanics, tokens)`` is called per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import msgspec
import numpy as np
import torch

from urfm.data.texture import song_texture, unit_marker
from urfm.ir.chart import Chart, ChartSet, Difficulty, SongMeta
from urfm.ir.controls import Control
from urfm.ir.events import Event
from urfm.ir.timing import MeterChange, TimingMap
from urfm.ir.validate import enforce_single_action
from urfm.model import ChartModel
from urfm.tokenize import audio_windows
from urfm.tokenize.api import Tokenizer
from urfm.tokenize.sections import SectionPlan, section_beats, window_sections
from urfm.tokenize.tokenizer import DROPS, window_start_beat
from urfm.train.data import N_FRAMES, collate_song, slice_mel, song_marker, song_summary

CODEC_FOR = {"taiko": "tja", "maimai_dx": "simai"}


@dataclass
class Job:
    mel: np.ndarray
    """uint8 ``[frames, n_mels]`` of the whole song."""
    timing: TimingMap
    mechanics: str
    prefix: list[int]
    difficulty: Difficulty = field(default_factory=Difficulty)
    meta: dict[str, str] = field(default_factory=dict)
    lanes: int | None = None
    """Key count for variable-lane games (osu!mania); also stored as ``meta['lanes']``."""
    target_rate: float | None = None
    """Overall onsets/s to calibrate the density plan to (``plan="calibrated"``)."""
    density_plan: list[float] | None = None
    """Optional onsets/s per window (forces each window's ``WDENS``); ``None`` lets the
    model plan the density itself."""
    events: list[Event] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)
    meters: list[MeterChange] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)
    song: tuple[np.ndarray, np.ndarray] | None = None
    """Song units of ``mel`` and their start µs (computed once, for song-context models)."""
    cond: tuple[np.ndarray, np.ndarray] | None = None
    """Normalised conditions and mask (``urfm.train.conditions``) for condition models."""
    section_plans: dict[int, SectionPlan] | None = None
    """R6: plans by section index (``urfm.train.planner.plan_song``), forced after ``SEP``;
    ``section_k`` is their length in beats."""
    section_k: int | None = None
    use_ref: bool = False
    """R6: give each window the structure memory (an earlier generated section whose audio is
    most like one of the window's sections)."""
    similarity: tuple[int, np.ndarray, float] | None = None
    """First section index, section audio similarity and its threshold (computed once)."""
    window_range: tuple[int, int] | None = None
    """Generate only windows ``[a, b)`` of the song (song-level RL); ``events`` may hold the
    reference chart's events before window ``a`` as context."""


def song_windows(timing: TimingMap, duration_us: int) -> list[tuple[int, int]]:
    """The tokenizer's audio-wide windows -- identical to training."""
    return audio_windows(timing, duration_us)


def _chart(job: Job) -> Chart:
    if job.lanes and "lanes" not in job.meta:
        job.meta["lanes"] = str(job.lanes)
    events = sorted(job.events, key=lambda e: (e.time, e.role))
    controls = sorted(job.controls, key=lambda c: c.time)
    chart = Chart(
        mechanics=job.mechanics,
        difficulty=job.difficulty,
        events=tuple(events),
        controls=tuple(controls),
        timing=_with_meters(job.timing, job.meters),
        meta=job.meta,
    )
    chart, dropped = enforce_single_action(chart, job.timing.us)
    if dropped:
        DROPS[job.mechanics, "generated_overlap"] += dropped
    return chart


def _with_meters(timing: TimingMap, meters: list[MeterChange]) -> TimingMap | None:
    """``timing`` with generated meter changes (time is unaffected); ``None`` if there are none.

    The job's own meters before the first generated change are kept (the opening meter).
    """
    if not meters:
        return None
    first = min(m.beat for m in meters)
    kept = [m for m in timing.meters if m.beat < first]
    merged = sorted([*kept, *meters], key=lambda m: m.beat)
    return msgspec.structs.replace(timing, meters=tuple(merged))


def _context(tok: Tokenizer, job: Job, start_us: int, vocab_limit: int | None = None) -> list[int]:
    if not (job.events or job.controls or job.meters):
        return []
    cs = ChartSet(timing=job.timing, charts=(_chart(job),))
    tokens = tok.context(cs, cs.charts[0], start_us)
    if vocab_limit is None or all(t < vocab_limit for t in tokens):
        return tokens
    # an older checkpoint lacks newer tokens (e.g. controls): drop those whole groups
    kinds = tok.vocab.kinds  # type: ignore[attr-defined]
    groups: list[list[int]] = []
    for t in tokens:
        if kinds[t] in ("POS_BEAT", "POS_SAME") or not groups:
            groups.append([t])
        else:
            groups[-1].append(t)
    return [t for g in groups if all(x < vocab_limit for x in g) for t in g]


def _song_inputs(
    model: ChartModel, rows: list[tuple[Job, int]], device: torch.device
) -> dict[str, torch.Tensor]:
    """Song units with each row's window marked (texture units or pooled mel blocks)."""
    if not model.cfg.song_context:
        return {}
    texture = bool(model.cfg.song_feat_dim)
    items = []
    for j, k in rows:
        assert j.song is not None
        units, unit_us = j.song
        mark = unit_marker if texture else song_marker
        items.append((units, mark(unit_us, j.starts[k], j.ends[k])))
    return {key: v.to(device) for key, v in collate_song(items).items()}


def _song_units(model: ChartModel, job: Job) -> tuple[np.ndarray, np.ndarray]:
    if model.cfg.song_feat_dim:
        return song_texture(job.mel, job.timing)
    return song_summary(job.mel)


def _cond(model: ChartModel, jobs: list[Job], device: torch.device) -> torch.Tensor | None:
    if not model.cfg.cond_dim:
        return None
    k = model.cfg.cond_dim
    values = np.zeros((len(jobs), k), np.float32)
    mask = np.zeros((len(jobs), k), bool)
    for i, j in enumerate(jobs):
        if j.cond is not None:
            values[i], mask[i] = j.cond
    return model.condition(torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device))


class _Grammar:
    """Per-row constrained decoding state.

    Uses the tokenizer's state machine when it has one (``grammar_state(mechanics, prompt,
    lanes)`` + ``grammar(mechanics, lanes)`` with ``mask``/``step``); otherwise falls back to
    ``allowed(mechanics, generated_tokens)`` every step.
    """

    def __init__(
        self, tok: Tokenizer, job: Job, prompt: list[int], forbid: np.ndarray | None = None
    ) -> None:
        self.tok, self.mechanics = tok, job.mechanics
        self.tokens: list[int] = []
        self.forbid = forbid
        """Tokens never sampled (e.g. further plan tokens after a forced section plan)."""
        self.machine = None
        if hasattr(tok, "grammar") and hasattr(tok, "grammar_state"):
            self.machine = tok.grammar(job.mechanics, job.lanes)  # type: ignore[attr-defined]
            self.state = tok.grammar_state(job.mechanics, prompt, job.lanes)  # type: ignore[attr-defined]

    def mask(self) -> np.ndarray:
        if self.machine is not None:
            m = self.machine.mask(self.state)
        else:
            m = self.tok.allowed(self.mechanics, self.tokens)
        return m if self.forbid is None else m & ~self.forbid

    def at_plan(self) -> bool:
        """Whether the next token is the window's ``WDENS`` plan."""
        return self.machine is not None and self.state[0] == "wdens"

    def at_texture(self) -> bool:
        """Whether the next token is the window's ``GRID``/``REST`` plan."""
        return self.machine is not None and self.state[0] in ("grid", "rest")

    def push(self, token: int) -> None:
        self.tokens.append(token)
        if self.machine is not None:
            self.state = self.machine.step(self.state, token)


SILENT_RATE = 0.3
"""Windows whose expected density is below this many onsets/s stay silent."""


@torch.no_grad()
def plan_densities(
    model: ChartModel, tok: Tokenizer, jobs: list[Job], device: torch.device, batch: int = 64
) -> None:
    """Calibrated density plans: per-window expected density from the model, rescaled so the
    song's mean matches ``Job.target_rate``.

    The relative shape (which windows are dense, sparse or silent) comes from the audio; the
    level comes from the conditioning. Sampling the plan instead tends to under-shoot and,
    at low temperatures, to collapse onto the most frequent bin (silent or sparse windows).
    """
    from urfm.tokenize.vocab import DENS_EDGES

    n_bins = len(DENS_EDGES)
    rates = torch.tensor([0.0, *(0.5 * 2 ** ((b + 0.5) / 2.5) for b in range(1, n_bins))])
    wdens = torch.tensor([tok.window_density(0.0) + b for b in range(n_bins)])
    items = [(j, k) for j in jobs if j.target_rate for k in range(len(j.starts))]
    expected: dict[tuple[int, int], float] = {}
    for i in range(0, len(items), batch):
        chunk = items[i : i + batch]
        prompts = [[*j.prefix, tok.bos_id, tok.sep_id] for j, _ in chunk]
        p = max(len(x) for x in prompts)
        tokens = torch.full((len(chunk), p), tok.pad_id, dtype=torch.long)
        pad_mask = torch.zeros((len(chunk), p), dtype=torch.bool)
        for r, x in enumerate(prompts):
            tokens[r, p - len(x) :] = torch.tensor(x)
            pad_mask[r, p - len(x) :] = True
        mel = torch.from_numpy(np.stack([slice_mel(j.mel, j.starts[k]) for j, k in chunk]))
        feat = torch.from_numpy(
            np.stack(
                [
                    tok.frame_features(j.timing, j.starts[k], N_FRAMES).astype(np.float32)
                    for j, k in chunk
                ]
            )
        )
        song = _song_inputs(model, chunk, device)
        cond = _cond(model, [j for j, _ in chunk], device)
        cuda = device.type == "cuda"
        ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cuda
            else torch.autocast("cpu", enabled=False)
        )
        with ctx:
            memory, memory_mask = model.memory_for(
                mel.to(device),
                feat.to(device),
                song.get("song"),
                song.get("song_marker"),
                song.get("song_len"),
            )
            _, logits = model.prefill(
                tokens.to(device), pad_mask.to(device), memory, p + 1, memory_mask, cond
            )
        probs = torch.softmax(logits.float()[:, wdens.to(device)], -1).cpu()
        for (j, k), row in zip(chunk, probs, strict=True):
            expected[id(j), k] = float((row * rates).sum())
    for j in jobs:
        if not j.target_rate:
            continue
        e = [expected[id(j), k] for k in range(len(j.starts))]
        lengths = [(b - a) / 1e6 for a, b in zip(j.starts, j.ends, strict=True)]
        active = [(x, n) for x, n in zip(e, lengths, strict=True) if x > SILENT_RATE]
        mean = sum(x * n for x, n in active) / max(sum(n for _, n in active), 1e-6)
        scale = min(max(j.target_rate / mean, 0.5), 3.0) if mean > 0 else 1.0
        j.density_plan = [x * scale if x > SILENT_RATE else 0.0 for x in e]


def _plan(logits: torch.Tensor, mode: str, temperature: float, gen: torch.Generator | None):
    """Choose ``WDENS`` (rows are masked to WDENS tokens only).

    Low global temperatures collapse the plan onto its single most frequent bin (often the
    empty window), producing long silent stretches; the plan therefore gets its own
    temperature, or ``mean`` picks the bin nearest the distribution's expectation.
    """
    if mode == "mean":
        probs = torch.softmax(logits.float(), -1)
        ids = torch.arange(logits.shape[-1], device=logits.device)
        allowed = torch.isfinite(logits)
        first = torch.where(allowed, ids, logits.shape[-1]).min(-1).values
        bins = (ids[None, :] - first[:, None]).clamp_min(0).float()
        expected = (probs * bins).sum(-1).round().long()
        return first + expected
    return _sample(logits, temperature, 1.0, gen)


def _sample(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator | None):
    if temperature <= 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, -1)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(-1, descending=True)
        keep = sorted_p.cumsum(-1) - sorted_p < top_p
        sorted_p = sorted_p * keep
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=gen)[:, 0]


@torch.no_grad()
def decode_batch(
    model: ChartModel,
    tok: Tokenizer,
    prompts: list[list[int]],
    grammars: list[_Grammar],
    mel: torch.Tensor,
    feat: torch.Tensor,
    song: dict[str, torch.Tensor],
    cond: torch.Tensor | None,
    *,
    temperature: float,
    top_p: float,
    plan: str = "sample",
    plan_temperature: float = 1.0,
    gen: torch.Generator | None = None,
) -> list[list[int]]:
    """Decode one window per row from left-padded ``prompts`` with grammar constraints and a
    KV cache; each row's tokens are pushed into its grammar (``g.tokens``) and returned."""
    device = mel.device
    cuda = device.type == "cuda"
    max_len = model.cfg.max_dec_len
    n = len(prompts)
    p = max(len(x) for x in prompts)
    tokens = torch.full((n, p), tok.pad_id, dtype=torch.long)
    pad_mask = torch.zeros((n, p), dtype=torch.bool)
    for i, x in enumerate(prompts):
        tokens[i, p - len(x) :] = torch.tensor(x)
        pad_mask[i, p - len(x) :] = True
    tokens, pad_mask = tokens.to(device), pad_mask.to(device)
    ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if cuda
        else torch.autocast("cpu", enabled=False)
    )
    with ctx:
        memory, memory_mask = model.memory_for(
            mel, feat, song.get("song"), song.get("song_marker"), song.get("song_len")
        )
        cache, logits = model.prefill(tokens, pad_mask, memory, max_len, memory_mask, cond)
        key_mask = torch.zeros((n, max_len), dtype=torch.bool, device=device)
        key_mask[:, :p] = pad_mask
        positions = pad_mask.sum(1)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        for t in range(p, max_len):
            allowed = torch.from_numpy(np.stack([g.mask() for g in grammars])).to(device)
            # checkpoints trained on an older (shorter) vocabulary never emit newer tokens
            allowed = allowed[:, : logits.shape[-1]]
            logits = logits.float().masked_fill(~allowed, float("-inf"))
            nxt = _sample(logits, temperature, top_p, gen)
            planning = [i for i, g in enumerate(grammars) if g.at_plan()]
            if planning:  # the window's density plan (WDENS) has its own policy
                rows = torch.tensor(planning, device=device)
                nxt[rows] = _plan(logits[rows], plan, plan_temperature, gen)
            texture = [i for i, g in enumerate(grammars) if g.at_texture()]
            if texture:  # GRID/REST: sampled at the plan temperature, full distribution
                rows = torch.tensor(texture, device=device)
                nxt[rows] = _sample(logits[rows], plan_temperature, 1.0, gen)
            nxt = torch.where(done, torch.full_like(nxt, tok.eos_id), nxt)
            for i, g in enumerate(grammars):
                if not done[i]:
                    g.push(int(nxt[i]))
            done |= nxt == tok.eos_id
            if bool(done.all()) or t + 1 >= max_len:
                break
            key_mask[:, t] = True
            logits = model.step(nxt, positions, cache, t, key_mask)
            positions = positions + 1
    return [g.tokens for g in grammars]


@torch.no_grad()
def generate(
    model: ChartModel,
    tok: Tokenizer,
    jobs: list[Job],
    *,
    temperature: float = 0.6,
    top_p: float = 0.95,
    plan: str = "sample",
    plan_temperature: float = 1.0,
    device: str | torch.device = "cuda",
    seed: int | None = 0,
    record: list[dict] | None = None,
) -> list[Chart]:
    """``record`` (song-level RL) receives one dict per generated window: the job index, the
    prompt and generated tokens, and the model inputs needed to score them again."""
    device = torch.device(device)
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    use_song = model.cfg.song_context
    for job in jobs:
        duration = round(job.mel.shape[0] / _fps() * 1_000_000)
        wins = song_windows(job.timing, duration)
        job.starts, job.ends = [w[0] for w in wins], [w[1] for w in wins]
        if job.window_range is not None:
            a, b = job.window_range
            job.starts, job.ends = job.starts[a:b], job.ends[a:b]
        if use_song and job.song is None:
            job.song = _song_units(model, job)
    if plan == "calibrated":
        todo = [j for j in jobs if j.density_plan is None and j.section_plans is None]
        plan_densities(model, tok, todo, device)
    forbid = _plan_tokens_mask(tok)
    rounds = max(len(j.starts) for j in jobs)
    for k in range(rounds):
        active = [j for j in jobs if k < len(j.starts)]
        prompts, mels, feats, forbids = [], [], [], []
        for j in active:
            start = j.starts[k]
            context = _context(tok, j, start, model.cfg.vocab_size)
            ref = _memory(tok, j, start, j.ends[k]) if j.use_ref else []
            prompt = [*j.prefix, tok.bos_id, *ref, *context, tok.sep_id]
            forced = None
            if j.section_plans is not None:
                sk = j.section_k or section_beats(j.timing)
                a, b = window_sections(j.timing, start, j.ends[k], sk)
                plans = [j.section_plans[i] for i in range(a, b + 1) if i in j.section_plans]
                prompt += tok.plan_tokens(plans, window_start_beat(j.timing, start), sk)
                forced = forbid
            elif j.density_plan is not None and k < len(j.density_plan):
                prompt.append(tok.window_density(j.density_plan[k]))
            prompts.append(prompt)
            forbids.append(forced)
            mels.append(slice_mel(j.mel, start))
            feats.append(tok.frame_features(j.timing, start, N_FRAMES).astype(np.float32))
        mel = torch.from_numpy(np.stack(mels)).to(device)
        feat = torch.from_numpy(np.stack(feats)).to(device)
        song = _song_inputs(model, [(j, k) for j in active], device)
        cond = _cond(model, active, device)
        grammars = [
            _Grammar(tok, j, x, f) for j, x, f in zip(active, prompts, forbids, strict=True)
        ]
        decode_batch(
            model,
            tok,
            prompts,
            grammars,
            mel,
            feat,
            song,
            cond,
            temperature=temperature,
            top_p=top_p,
            plan=plan,
            plan_temperature=plan_temperature,
            gen=gen,
        )
        if record is not None:
            texture = bool(model.cfg.song_feat_dim)
            index = {id(j): n for n, j in enumerate(jobs)}
            for i, (j, x, g) in enumerate(zip(active, prompts, grammars, strict=True)):
                units = marker = None
                if j.song is not None:
                    units, unit_us = j.song
                    mark = unit_marker if texture else song_marker
                    marker = mark(unit_us, j.starts[k], j.ends[k])
                record.append(
                    {
                        "job": index[id(j)],
                        "prompt": list(x),
                        "tokens": list(g.tokens),
                        "mel": mels[i],
                        "feats": feats[i],
                        "song": (units, marker),
                        "cond": j.cond,
                    }
                )
        for j, g in zip(active, grammars, strict=True):
            start, end = j.starts[k], j.ends[k]
            body = [x for x in g.tokens if x != tok.eos_id]
            extra = {"lanes": j.lanes} if j.lanes is not None else {}
            decode_full = getattr(tok, "decode_full", None)
            if decode_full is not None:
                events, controls, meters = decode_full(j.timing, j.mechanics, body, start, **extra)
            else:
                events, controls, meters = (
                    tok.decode(j.timing, j.mechanics, body, start, **extra),
                    [],
                    [],
                )
            j.events += [ev for ev in events if start <= j.timing.us(ev.time) < end]
            j.controls += [c for c in controls if start <= j.timing.us(c.time) < end]
            j.meters += [
                m for m in meters if start <= round(j.timing.beat_to_us_exact(m.beat)) < end
            ]
    return [_chart(j) for j in jobs]


def _plan_tokens_mask(tok: Tokenizer) -> np.ndarray | None:
    """Plan tokens (``SBEAT``, ``WDENS``) as a vocabulary mask, for windows whose plan is
    forced; ``None`` for tokenizers without section plans."""
    kinds = getattr(getattr(tok, "vocab", None), "kinds", None)
    if kinds is None or "SBEAT" not in kinds:
        return None
    return np.array([k in ("SBEAT", "WDENS", "SPEAK") for k in kinds])


def _memory(tok: Tokenizer, job: Job, start: int, end: int) -> list[int]:
    """Structure-memory block for the window ``[start, end)`` from the job's own output."""
    from urfm.data.sections import choose_ref, ref_threshold, section_audio, similarity

    k = job.section_k or section_beats(job.timing)
    if job.similarity is None:
        first, feats = section_audio(job.mel, job.timing, k)
        sim = similarity(feats)
        job.similarity = (first, sim, ref_threshold(sim))
    first, sim, threshold = job.similarity
    a, b = window_sections(job.timing, start, end, k)
    pair = choose_ref(sim, threshold, first, a, b)
    if pair is None or not job.events:
        return []
    cs = ChartSet(timing=job.timing, charts=(_chart(job),))
    return tok.ref_block(
        cs, cs.charts[0], pair[0], pair[1], window_start_beat(job.timing, start), k
    )


def _fps() -> float:
    from urfm.data.features import MEL

    return MEL.fps


def to_chartset(chart: Chart, timing: TimingMap, song: SongMeta | None = None) -> ChartSet:
    return ChartSet(timing=timing, charts=(chart,), song=song or SongMeta())


def write_native(chartset: ChartSet, path: str | Path) -> Path:
    """Write a generated chart set with the codec matching its mechanics."""
    from urfm.codecs import get

    mech = chartset.charts[0].mechanics
    codec_id = CODEC_FOR.get(mech) or ("osu" if mech.startswith("osu_") else "stepmania")
    codec = get(codec_id)
    codec.check_writable(chartset)
    path = Path(path)
    path.write_bytes(codec.write(chartset))
    return path


def reference_inputs(
    model: ChartModel, meta: dict, tok: Tokenizer, chartset: ChartSet, chart: Chart
) -> tuple[list[int], tuple[np.ndarray, np.ndarray] | None]:
    """Prefix and conditions that ask a model for a chart like ``chart`` (evaluation against
    a human reference): continuous conditions measured on the chart (after the same
    canonical-tempo folding as in training) for condition models, tier/density tokens
    otherwise."""
    from urfm.tokenize.tokenizer import chart_lanes
    from urfm.train.conditions import chart_conditions, cond_prefix
    from urfm.train.tiers import chart_prefix, load_tiers

    if not model.cfg.cond_dim:
        return chart_prefix(tok, load_tiers(meta), chartset, chart), None
    if meta.get("features", {}).get("canonical"):
        from urfm.data.canonical import canonicalize

        chartset, chart = canonicalize(chartset, chart)
    return cond_prefix(tok, chart.mechanics, chart_lanes(chart)), chart_conditions(chartset, chart)
