"""Web UI: upload a song, choose a game and courses, optionally edit the section plan, and get
playable charts (zip with the audio) plus the time spent in each stage.

    uv run --extra app python -m urfm.app.web            # http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from urfm.app.pipeline import (
    GAMES,
    Analysis,
    CourseRequest,
    Models,
    Result,
    analyze,
    chart_song,
    courses,
    plan,
    stage_table,
)
from urfm.app.plans import COLUMNS, from_rows, to_rows

TAIKO_LEVELS = {"easy": 3, "normal": 5, "hard": 7, "oni": 9, "edit": 10}
ROW_NAMES = ("Easy", "Normal", "Hard", "Oni / Insane", "Edit / Expert")
_MODELS: Models | None = None


def models(device: str | None = None) -> Models:
    global _MODELS
    if _MODELS is None:
        _MODELS = Models(device)
    return _MODELS


def requests_from(game: str, course_inputs) -> list[CourseRequest]:
    out = []
    for i, name in enumerate(courses(game)):
        on, level, rate = course_inputs[3 * i : 3 * i + 3]
        if on:
            lvl = int(level) if game == "taiko" else None
            out.append(CourseRequest(name, lvl, float(rate) if rate else None))
    if not out:
        raise ValueError("pick at least one course")
    return out


def _key(audio, game, keys) -> tuple:
    return (str(audio), game, int(keys or 4) if game == "osu_mania" else None)


def info_table(result: Result) -> str:
    info = result.info
    head = (
        f"**{info['game']}** · **{info['duration_s']:.1f} s** song · detected tempo "
        f"**{info['bpm']:.1f} BPM** ({info['tempo_changes']} tempo changes) · device "
        f"`{info['device']}`\n\n"
    )
    rows = ["| Course | Level | Notes | Notes / s |", "| --- | ---: | ---: | ---: |"]
    for c in info["courses"]:
        level = c["level"] if c["level"] is not None else "–"
        rows.append(f"| {c['course']} | {level} | {c['notes']} | {c['notes_per_s']:.2f} |")
    return head + "\n".join(rows)


def analyze_and_plan(audio, game, keys, plan_temperature, seed, *course_inputs):
    """Step 1: beat tracking and section plans, shown for editing."""
    import gradio as gr

    if not audio:
        raise gr.Error("upload an audio file first")
    m = models()
    reqs = requests_from(game, course_inputs)
    analysis = analyze(audio, m)
    plans, k, secs = plan(
        analysis,
        game,
        reqs,
        m,
        keys=int(keys or 4),
        plan_temperature=float(plan_temperature),
        seed=int(seed),
    )
    state = {"key": _key(audio, game, keys), "analysis": analysis, "plans": plans, "k": k}
    stages = [*analysis.stages, ("Song planner (section plans)", secs)]
    table = "\n".join(
        ["| Stage | Seconds |", "| --- | ---: |", *(f"| {n} | {s:.2f} |" for n, s in stages)]
    )
    first = next(iter(plans), None)
    return (
        state,
        gr.update(choices=list(plans), value=first),
        rows_for(first, state),
        f"Plans ready for {', '.join(plans)}. Edit a course's table, press **Apply edits**, "
        f"then **Generate**.\n\n{table}",
    )


def rows_for(course, state):
    if not state or course not in state.get("plans", {}):
        return []
    from urfm.beat.decode import for_game

    game = state["key"][1]
    analysis: Analysis = state["analysis"]
    timing = for_game(analysis.timing, game)  # type: ignore[arg-type]
    k = state["k"]
    return to_rows(state["plans"][course], timing, k)


def apply_edits(course, table, state):
    if not state or course not in state.get("plans", {}):
        return state, "Run **Analyze & plan** first."
    rows = table.values.tolist() if hasattr(table, "values") else table
    state["plans"][course] = from_rows(rows, state["plans"][course])
    return state, f"Edits applied to **{course}**; **Generate** uses them."


def run(audio, title, game, keys, temperature, plan_temperature, seed, state, *course_inputs):
    """Step 2 (or everything): generate and bundle."""
    import gradio as gr

    if not audio:
        raise gr.Error("upload an audio file first")
    m = models()
    reqs = requests_from(game, course_inputs)
    t0 = time.perf_counter()
    reuse = state if state and state.get("key") == _key(audio, game, keys) else None
    src = Path(audio)
    work = Path(tempfile.mkdtemp(prefix="urfm-"))
    song = work / f"song{src.suffix.lower()}"
    shutil.copy(src, song)
    name = (title or src.stem).strip() or "song"
    result = chart_song(
        song,
        reqs,
        m,
        game=game,
        keys=int(keys or 4),
        title=name,
        temperature=float(temperature),
        plan_temperature=float(plan_temperature),
        seed=int(seed),
        analysis=reuse["analysis"] if reuse else None,
        plans={c: p for c, p in reuse["plans"].items() if c in {r.course for r in reqs}}
        if reuse
        else None,
    )
    bundle = work / f"{name}.zip"
    with zipfile.ZipFile(bundle, "w") as z:
        z.write(song, f"{name}/{song.name}")
        for fname, data in result.files.items():
            (work / fname).write_bytes(data)
            z.write(work / fname, f"{name}/{fname}")
    timing = stage_table(result, m.load_s)
    if reuse:
        timing += "\n\n(Audio analysis and section plans reused from **Analyze & plan**.)"
    timing += f"\n\nWall time for this request: {time.perf_counter() - t0:.2f} s"
    first = next(iter(result.files.values()))
    return str(bundle), info_table(result), timing, first.decode("utf-8", "replace")[:6000]


def build():
    import gradio as gr

    with gr.Blocks(title="URFM chart generator") as demo:
        gr.Markdown(
            "# URFM chart generator\n"
            "Upload a song and pick a game. Timing comes from the audio (beat model), sections "
            "from the song planner, notes from the chart decoder (`r6x-rl` + `r6-plan3`). "
            "**Generate** does everything; **Analyze & plan** first lets you edit the section "
            "plan. The zip holds the audio and playable chart files."
        )
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(type="filepath", label="Song (mp3 / ogg / wav / flac)")
                title = gr.Textbox(label="Title (optional)")
                with gr.Row():
                    game = gr.Dropdown(
                        [(v, k) for k, v in GAMES.items()], value="taiko", label="Game"
                    )
                    keys = gr.Number(value=4, precision=0, label="osu!mania keys")
                gr.Markdown(
                    "Courses: Taiko = Easy / Normal / Hard / Oni / Edit with an official level; "
                    "osu! = Easy / Normal / Hard / Insane / Expert (level ignored). Notes/s blank "
                    "= typical for that course."
                )
                course_inputs = []
                for i, label in enumerate(ROW_NAMES):
                    with gr.Row():
                        on = gr.Checkbox(value=i in (1, 2, 3), label=label, scale=1)
                        level = gr.Slider(
                            1,
                            10,
                            value=list(TAIKO_LEVELS.values())[i],
                            step=1,
                            label="Level (Taiko)",
                            scale=2,
                        )
                        rate = gr.Number(value=None, label="Notes/s", scale=2, precision=2)
                    course_inputs += [on, level, rate]
                with gr.Accordion("Sampling", open=False):
                    temperature = gr.Slider(
                        0.5, 1.3, value=1.0, step=0.05, label="Note temperature"
                    )
                    plan_t = gr.Slider(0.0, 1.3, value=0.7, step=0.05, label="Plan temperature")
                    seed = gr.Number(value=0, precision=0, label="Seed")
                with gr.Row():
                    plan_btn = gr.Button("1. Analyze & plan")
                    go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                bundle = gr.File(label="Playable bundle (audio + charts)")
                info = gr.Markdown()
                timing = gr.Markdown()
                with gr.Accordion("Chart preview", open=False):
                    preview = gr.Code(language=None)
        with gr.Accordion("Section plan editor", open=False):
            status = gr.Markdown("Press **1. Analyze & plan** to see the plan.")
            course = gr.Dropdown([], label="Course")
            table = gr.Dataframe(headers=list(COLUMNS), interactive=True, wrap=True)
            apply = gr.Button("Apply edits")
        plan_btn.click(
            analyze_and_plan,
            [audio, game, keys, plan_t, seed, *course_inputs],
            [state, course, table, status],
        )
        course.change(rows_for, [course, state], [table])
        apply.click(apply_edits, [course, table, state], [state, status])
        go.click(
            run,
            [audio, title, game, keys, temperature, plan_t, seed, state, *course_inputs],
            [bundle, info, timing, preview],
        )
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default=None)
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    args = ap.parse_args(argv)
    models(args.device)
    build().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
