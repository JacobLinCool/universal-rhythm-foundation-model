"""Command-line interface: ``urfm inspect | convert | validate | roundtrip | stats``."""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import msgspec

from urfm.codecs import registry
from urfm.compare.roundtrip import roundtrip_path
from urfm.ir.serialize import to_json
from urfm.ir.validate import validate
from urfm.ir.views import ChartStats, chart_stats, summarize


def _iter_files(paths: list[Path], codec_id: str | None) -> Iterator[tuple[Path, object]]:
    for root in paths:
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
        for path in files:
            if codec_id:
                codec = registry.get(codec_id)
                with path.open("rb") as fh:
                    head = fh.read(4096)
                if not codec.sniff(path, head):
                    continue
            else:
                try:
                    codec = registry.for_path(path)
                except KeyError:
                    continue
            yield path, codec


def cmd_inspect(args: argparse.Namespace) -> int:
    codec = registry.get(args.codec) if args.codec else registry.for_path(args.path)
    cs = codec.read_path(args.path)
    print(f"{cs.song.title} / {cs.song.artist}  [{codec.id}]")
    for i, chart in enumerate(cs.charts):
        timing = cs.timing_of(chart)
        roles = Counter(ev.role for ev in chart.events)
        last = max((timing.seconds(ev.end or ev.time) for ev in chart.events), default=0.0)
        print(
            f"  [{i}] {chart.mechanics:<16} {chart.difficulty.label or '-':<10} "
            f"lv={chart.difficulty.level}  events={len(chart.events)}  "
            f"length={last:.1f}s  {dict(roles)}"
        )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    source = registry.get(args.codec) if args.codec else registry.for_path(args.path)
    cs = source.read_path(args.path)
    if args.to == "urfm":
        data = to_json(cs, indent=2)
    else:
        target = registry.get(args.to)
        target.check_writable(cs)
        data = target.write(cs)
    if args.output:
        Path(args.output).write_bytes(data)
    else:
        sys.stdout.buffer.write(data)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    bad = 0
    for path, codec in _iter_files(args.paths, args.codec):
        issues = validate(codec.read_path(path))  # type: ignore[attr-defined]
        if issues:
            bad += 1
            print(f"{path}: {len(issues)} issue(s)")
            for issue in issues[: args.limit]:
                print(f"  chart={issue.chart} event={issue.event} {issue.code}: {issue.message}")
    return 1 if bad else 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    with contextlib.ExitStack() as stack:
        report = stack.enter_context(Path(args.report).open("wb")) if args.report else None
        total = _roundtrip_all(args, report)
    n = max(total["files"], 1)
    print(
        f"files={total['files']}  errors={total['error']}  "
        f"L2 timing-exact={total['timing_exact']} ({total['timing_exact'] / n:.2%})  "
        f"L1 semantic-exact={total['semantic_exact']} ({total['semantic_exact'] / n:.2%})"
    )
    return 0 if total["semantic_exact"] == total["files"] else 1


def _roundtrip_all(args: argparse.Namespace, report: BinaryIO | None) -> Counter[str]:
    total = Counter[str]()
    for path, codec in _iter_files(args.paths, args.codec):
        result = roundtrip_path(codec, path)  # type: ignore[arg-type]
        total["files"] += 1
        total["error"] += result.error is not None
        total["timing_exact"] += result.timing_exact
        total["semantic_exact"] += result.semantic_exact
        if report:
            report.write(msgspec.json.encode(result) + b"\n")
        if args.verbose and not result.semantic_exact:
            print(f"{path}: {result.error or ''}")
            for d in (result.timing_diffs or result.semantic_diffs)[:5]:
                print(f"  {d.path}: {d.left} != {d.right}")
    return total


def cmd_stats(args: argparse.Namespace) -> int:
    rows: list[ChartStats] = []
    for path, codec in _iter_files(args.paths, args.codec):
        try:
            cs = codec.read_path(path)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        rows.extend(chart_stats(cs, chart) for chart in cs.charts if chart.events)
    if args.json:
        Path(args.json).write_bytes(msgspec.json.encode(rows))
    table = summarize(rows, by=args.by)
    columns = list(next(iter(table.values()), {}))
    print("| " + args.by + " | " + " | ".join(columns) + " |")
    print("|" + "---|" * (len(columns) + 1))
    for name, values in table.items():
        cells = [f"{values[c]:.3g}" for c in columns]
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urfm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="summarise a chart file")
    p.add_argument("path", type=Path)
    p.add_argument("--codec")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("convert", help="convert a chart to IR JSON or another format")
    p.add_argument("path", type=Path)
    p.add_argument("--codec", help="source codec (default: detect)")
    p.add_argument("--to", default="urfm", help="target codec id (default: urfm JSON)")
    p.add_argument("-o", "--output")
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("validate", help="check IR structure and mechanics compliance")
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--codec")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("roundtrip", help="measure parse->write->parse fidelity")
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--codec")
    p.add_argument("--report", help="write per-file JSONL results here")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_roundtrip)

    p = sub.add_parser("stats", help="cross-game interaction statistics")
    p.add_argument("paths", type=Path, nargs="+")
    p.add_argument("--codec")
    p.add_argument("--by", choices=["family", "mechanics"], default="mechanics")
    p.add_argument("--json", help="write per-chart statistics as JSON")
    p.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
