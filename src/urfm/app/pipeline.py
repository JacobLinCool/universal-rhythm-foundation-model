"""Audio file -> playable charts (Taiko TJA or osu! .osu), with the time spent in each stage.

    uv run python -m urfm.app.pipeline song.mp3 --courses oni:9 hard:6 --out out/
    uv run python -m urfm.app.pipeline song.mp3 --game osu_mania --keys 4 --courses hard insane

Stages: decode the audio to a log-mel; beat tracking (network activations, then dynamic-
programming decoding into a timing map); the song planner (section plans per course, which a
caller may edit before generating); the chart decoder (all courses together, window by
window); writing the files.

Courses: Taiko uses easy/normal/hard/oni/edit with an official level; osu! modes use
easy/normal/hard/insane/expert. Density targets default to typical training charts
(``data/course_defaults.json`` for Taiko by course and level, ``data/osu_defaults.json`` for osu!
by mode and difficulty) unless given in notes per second.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np
import torch

from urfm.tokenize.sections import SectionPlan

GAMES = {
    "taiko": "Taiko (TJA)",
    "osu_taiko": "osu!taiko",
    "osu_mania": "osu!mania",
    "osu_catch": "osu!catch",
    "osu_std": "osu!standard",
}
TAIKO_COURSES = ("easy", "normal", "hard", "oni", "edit")
OSU_COURSES = ("easy", "normal", "hard", "insane", "expert")
CHECKPOINTS = {
    "decoder": ("runs/checkpoints/r6x-rl.pt", "r6x-rl/best.pt"),
    "planner": ("runs/checkpoints/r6-plan3.pt", "r6-plan3/best.pt"),
    "beat": ("runs/checkpoints/beat-v1.pt", "beat-v1/best.pt"),
}
HUB_REPO = "JacobLinCool/urfm-runs"
OSU_FALLBACK = {"easy": 1.5, "normal": 2.5, "hard": 3.5, "insane": 5.0, "expert": 7.0}
"""Notes per second when ``data/osu_defaults.json`` is missing."""


def courses(game: str) -> tuple[str, ...]:
    return TAIKO_COURSES if game == "taiko" else OSU_COURSES


@dataclass(frozen=True)
class CourseRequest:
    course: str
    level: int | None = None
    """Official level (Taiko only)."""
    rate: float | None = None
    """Mean onsets per second; ``None`` = typical for the course (and level)."""


@dataclass
class Analysis:
    audio: Path
    mel: np.ndarray
    timing: object
    duration_s: float
    stages: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class Result:
    files: dict[str, bytes]
    """File name -> content (charts only; the audio is the caller's)."""
    stages: list[tuple[str, float]] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    plans: dict[str, dict[int, SectionPlan]] = field(default_factory=dict)
    section_k: int | None = None


@cache
def _defaults(name: str) -> dict:
    path = Path(__file__).parent / "data" / name
    return json.loads(path.read_text()) if path.exists() else {}


def typical(game: str, course: str, level: int | None) -> dict:
    """Median log2 condition values of training charts like the request."""
    if game == "taiko":
        table = _defaults("course_defaults.json")[course]
        levels = [int(k) for k, v in table.items() if v["charts"] >= 10] or [int(k) for k in table]
        return table[str(min(levels, key=lambda x: abs(x - (level or 7))))]
    row = _defaults("osu_defaults.json").get(game, {}).get(course)
    if row:
        return row
    rate = OSU_FALLBACK[course]
    return {
        "log2_rate": float(np.log2(rate)),
        "log2_peak_rate": float(np.log2(rate * 1.6)),
        "log2p_scroll_per_min": 0.0,
        "log2p_meter_per_min": 0.0,
    }


def checkpoint(name: str) -> Path:
    local, remote = CHECKPOINTS[name]
    path = Path(local)
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(HUB_REPO, remote), path)
    return path


class Models:
    """The three networks, loaded once (the app keeps them between requests)."""

    def __init__(self, device: str | None = None) -> None:
        from urfm.beat.train import load as load_beat
        from urfm.tokenize import build_tokenizer
        from urfm.train.planner import load_planner
        from urfm.train.train import load_checkpoint

        t0 = time.perf_counter()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = build_tokenizer()
        self.decoder, self.meta = load_checkpoint(checkpoint("decoder"), self.device)
        self.planner = load_planner(checkpoint("planner"), self.device)
        self.beat = load_beat(str(checkpoint("beat")), self.device)
        self.load_s = time.perf_counter() - t0


@contextmanager
def stage(stages: list[tuple[str, float]], name: str):
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stages.append((name, time.perf_counter() - t0))


def analyze(audio: str | Path, models: Models) -> Analysis:
    """Decode the audio and track its beats (shared by every game and course)."""
    from urfm.beat.decode import activations, decode_beats, to_timing
    from urfm.data.features import MEL, audio_mel

    stages: list[tuple[str, float]] = []
    with stage(stages, "Decode audio → log-mel"):
        mel = audio_mel(audio)
    with stage(stages, "Beat tracking: network"):
        act = activations(models.beat, mel, models.device)
    with stage(stages, "Beat tracking: decode timing map"):
        timing = to_timing(decode_beats(act[:, 0]), act[:, 1])
    return Analysis(Path(audio), mel, timing, mel.shape[0] / MEL.fps, stages)


def _jobs(
    analysis: Analysis,
    game: str,
    requests: list[CourseRequest],
    models: Models,
    keys: int | None,
):
    from urfm.beat.decode import for_game
    from urfm.ir.chart import Difficulty
    from urfm.train.conditions import cond_prefix, conditions
    from urfm.train.generate import Job

    timing = for_game(analysis.timing, game)  # type: ignore[arg-type]
    lanes = keys if game == "osu_mania" else None
    jobs = []
    for r in requests:
        t = typical(game, r.course, r.level)
        rate = r.rate if r.rate else 2 ** t["log2_rate"]
        cond = conditions(
            rate=rate,
            peak_rate=2 ** t["log2_peak_rate"] * rate / 2 ** t["log2_rate"],
            level=float(r.level) if game == "taiko" and r.level else None,
            scroll_per_min=2 ** t["log2p_scroll_per_min"] - 1,
            meter_per_min=2 ** t["log2p_meter_per_min"] - 1,
        )
        label = r.course if game == "taiko" else r.course.capitalize()
        jobs.append(
            Job(
                mel=analysis.mel,
                timing=timing,
                mechanics=game,
                prefix=cond_prefix(models.tok, game, lanes),
                cond=cond,
                difficulty=Difficulty(label=label, level=r.level if game == "taiko" else None),
                lanes=lanes,
                meta={"lanes": str(lanes)} if lanes else {},
                target_rate=rate,
            )
        )
    return jobs


def plan(
    analysis: Analysis,
    game: str,
    requests: list[CourseRequest],
    models: Models,
    *,
    keys: int | None = 4,
    plan_temperature: float = 0.7,
    seed: int = 0,
) -> tuple[dict[str, dict[int, SectionPlan]], int | None, float]:
    """Section plans per course from the song planner, their length in beats, and seconds."""
    from urfm.eval.protocol import plan_job

    t0 = time.perf_counter()
    plans, k = {}, None
    for i, (r, job) in enumerate(
        zip(requests, _jobs(analysis, game, requests, models, keys), strict=True)
    ):
        plan_job(job, models.meta, models.planner, plan_temperature, seed * 7919 + i)
        if job.section_plans is not None:
            plans[r.course], k = job.section_plans, job.section_k
    return plans, k, time.perf_counter() - t0


def chart_song(
    audio: str | Path,
    requests: list[CourseRequest],
    models: Models,
    *,
    game: str = "taiko",
    keys: int | None = 4,
    title: str | None = None,
    temperature: float = 1.0,
    plan_temperature: float = 0.7,
    seed: int = 0,
    analysis: Analysis | None = None,
    plans: dict[str, dict[int, SectionPlan]] | None = None,
) -> Result:
    """Charts for ``requests``; ``analysis`` and (possibly edited) ``plans`` from earlier
    calls are reused when given, so only the missing stages run."""
    from urfm.codecs import get
    from urfm.ir.chart import AudioRef, ChartSet, SongMeta
    from urfm.train.generate import generate

    stages: list[tuple[str, float]] = []
    if analysis is None:
        analysis = analyze(audio, models)
        stages += analysis.stages
    jobs = _jobs(analysis, game, requests, models, keys)
    given = plans or {}
    todo = [r for r in requests if r.course not in given]
    planned: dict[str, dict[int, SectionPlan]] = dict(given)
    k = None
    if todo:
        new, k, secs = plan(
            analysis, game, todo, models, keys=keys, plan_temperature=plan_temperature, seed=seed
        )
        planned |= new
        stages.append(("Song planner (section plans)", secs))
    else:
        stages.append(("Song planner (edited plans reused)", 0.0))
    from urfm.tokenize.sections import section_beats

    for r, job in zip(requests, jobs, strict=True):
        if r.course in planned:
            job.section_plans = planned[r.course]
            job.section_k = k or section_beats(job.timing)
            job.use_ref = bool(models.meta.get("features", {}).get("ref"))
    with stage(stages, f"Chart decoder ({len(jobs)} {'course' if len(jobs) == 1 else 'courses'})"):
        charts = generate(
            models.decoder,
            models.tok,
            jobs,
            temperature=temperature,
            top_p=0.95,
            device=models.device,
            seed=seed,
            plan="sample",
            plan_temperature=plan_temperature,
        )
    name = title or Path(audio).stem
    audio_name = Path(audio).name
    files: dict[str, bytes] = {}
    with stage(stages, "Write chart files"):
        song = SongMeta(title=name, artist="URFM")
        if game == "taiko":
            cs = ChartSet(
                timing=jobs[0].timing,
                charts=tuple(charts),
                song=song,
                audio=AudioRef(path=audio_name),
            )
            files[f"{name}.tja"] = get("tja").write(cs)
        else:
            for c in charts:
                cs = ChartSet(
                    timing=jobs[0].timing, charts=(c,), song=song, audio=AudioRef(path=audio_name)
                )
                files[f"{name} [{c.difficulty.label}].osu"] = get("osu").write(cs)
    timing = jobs[0].timing
    bpms = [float(t.bpm) for t in timing.tempos]
    info = {
        "game": GAMES[game],
        "duration_s": analysis.duration_s,
        "bpm": float(np.median(bpms)) if bpms else None,
        "tempo_changes": max(len(bpms) - 1, 0),
        "device": models.device,
        "courses": [
            {
                "course": r.course,
                "level": r.level,
                "notes": len(c.events),
                "notes_per_s": len(c.events) / max(analysis.duration_s, 1e-3),
            }
            for r, c in zip(requests, charts, strict=True)
        ],
    }
    return Result(files, stages, info, planned, k)


def stage_table(result: Result, load_s: float | None = None) -> str:
    """Markdown table of the stages (and the model load, when given)."""
    total = sum(s for _, s in result.stages)
    dur = result.info.get("duration_s") or 0.0
    rows = ["| Stage | Seconds | Share |", "| --- | ---: | ---: |"]
    if load_s is not None:
        rows.append(f"| Load models (once per session) | {load_s:.2f} | – |")
    for name, s in result.stages:
        rows.append(f"| {name} | {s:.2f} | {100 * s / max(total, 1e-9):.0f}% |")
    rows.append(f"| **Total** | **{total:.2f}** | 100% |")
    if dur:
        rows.append(f"| Song length | {dur:.1f} | {total / dur:.2f}× real time |")
    return "\n".join(rows)


def parse_course(text: str, game: str) -> CourseRequest:
    """``oni:9`` / ``oni:9:6.5`` (Taiko: course, level, notes/s) or ``hard`` / ``hard:4.5``
    (osu!: difficulty, notes/s)."""
    parts = text.split(":")
    course = parts[0].lower()
    if course not in courses(game):
        raise ValueError(f"unknown course {course!r}; one of {', '.join(courses(game))}")
    if game == "taiko":
        level = (
            int(parts[1]) if len(parts) > 1 else {"easy": 3, "normal": 5, "hard": 7}.get(course, 9)
        )
        rate = float(parts[2]) if len(parts) > 2 else None
        return CourseRequest(course, level, rate)
    return CourseRequest(course, None, float(parts[1]) if len(parts) > 1 else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--game", default="taiko", choices=list(GAMES))
    ap.add_argument("--keys", type=int, default=4, help="osu!mania key count")
    ap.add_argument("--courses", nargs="+", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--plan-temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    default = (
        ["easy:3", "normal:5", "hard:7", "oni:9"]
        if args.game == "taiko"
        else ["normal", "hard", "insane"]
    )
    models = Models(args.device)
    result = chart_song(
        args.audio,
        [parse_course(c, args.game) for c in args.courses or default],
        models,
        game=args.game,
        keys=args.keys,
        temperature=args.temperature,
        plan_temperature=args.plan_temperature,
        seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.audio, out / Path(args.audio).name)
    for name, data in result.files.items():
        (out / name).write_bytes(data)
    print(stage_table(result, models.load_s))
    print(json.dumps(result.info, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
