"""Tolerant reader/writer for StepMania note data (the ``#NOTES`` body).

``simfile.notes.NoteData`` only accepts the core note alphabet. Real packs
also contain StepF2/StepP1 cells such as ``{1|v|1|0}`` and vendor letters
(``X``, ``Y``, ``I``...). This module tokenizes every row into cells so that
nothing is rejected: each cell is a single character, optionally followed by a
keysound index (``1[3]``), or a ``{...}`` group whose first field is the base
note character.
"""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import groupby
from math import lcm
from typing import NamedTuple

_CELL = re.compile(r"\{[^}]*\}?|[^\[{](?:\[(\d+)\])?")
_SPACE = re.compile(r"\s+")


class Cell(NamedTuple):
    player: int
    beat: Fraction
    column: int
    base: str
    """The note character (``1``, ``2``, ``M``...); for ``{...}`` groups, the first field."""
    raw: str | None
    """The verbatim cell when it is a ``{...}`` group, else ``None``."""
    keysound: int | None


def parse(notes: str) -> list[Cell]:
    cells: list[Cell] = []
    for player, block in enumerate(notes.split("&")):
        lines = [_SPACE.sub("", line.split("//", 1)[0]) for line in block.splitlines()]
        measures = "\n".join(line for line in lines if line).split(",")
        for m, measure in enumerate(measures):
            rows = [row for row in measure.split("\n") if row]
            for r, row in enumerate(rows):
                beat = Fraction(4 * m) + Fraction(4 * r, len(rows))
                for column, match in enumerate(_CELL.finditer(row)):
                    text = match.group(0)
                    if text == "0":
                        continue
                    if text.startswith("{"):
                        base = text[1:].rstrip("}").split("|", 1)[0] or "0"
                        cells.append(Cell(player, beat, column, base, text, None))
                    else:
                        keysound = int(match.group(1)) if match.group(1) else None
                        cells.append(Cell(player, beat, column, text[0], None, keysound))
    return cells


def serialize(cells: list[tuple[int, Fraction, int, str]], columns: int) -> str:
    """Write ``(player, beat, column, cell_text)`` tuples as note data.

    Each measure uses the fewest rows that place every note exactly (the
    least common multiple of the beat denominators, times four), matching
    ``simfile.notes.NoteData.from_notes``.
    """
    columns = max([columns, *(c + 1 for _, _, c, _ in cells)])
    cells = sorted(cells, key=lambda c: (c[0], c[1], c[2]))
    players: list[str] = []
    last_player = max((p for p, *_ in cells), default=0)
    for player in range(last_player + 1):
        mine = [c for c in cells if c[0] == player]
        by_measure = {m: list(g) for m, g in groupby(mine, key=lambda c: c[1] // 4)}
        last_measure = max(by_measure, default=0)
        measures = [_measure(by_measure.get(m, []), columns) for m in range(last_measure + 1)]
        players.append("\n,\n".join(measures))
    return "\n&\n".join(players)


def _measure(cells: list[tuple[int, Fraction, int, str]], columns: int) -> str:
    per_beat = lcm(1, *(c[1].denominator for c in cells))
    rows = [["0"] * columns for _ in range(4 * per_beat)]
    for _, beat, column, text in cells:
        rows[int(beat % 4 * per_beat)][column] = text
    return "\n".join("".join(row) for row in rows)
