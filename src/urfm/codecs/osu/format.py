"""Raw ``.osu`` file structure: sections, timing points and hit objects as text fields.

This layer only splits and joins text. It knows nothing about the IR, which keeps
the mapping logic in :mod:`urfm.codecs.osu.codec` free of string handling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

from urfm.codecs.base import CodecError

KEY_VALUE_SECTIONS = ("General", "Editor", "Metadata", "Difficulty", "Colours")
SECTION_ORDER = (
    "General",
    "Editor",
    "Metadata",
    "Difficulty",
    "Events",
    "TimingPoints",
    "Colours",
    "HitObjects",
)
_HEADER = re.compile(r"osu file format v(\d+)")
_SECTION = re.compile(r"^\[(.+)\]$")


@dataclass
class OsuFile:
    version: int = 14
    sections: dict[str, list[str]] = field(default_factory=dict)
    """Raw non-empty lines per section, in file order (comments kept for Events)."""

    def pairs(self, name: str) -> list[list[str]]:
        out: list[list[str]] = []
        for line in self.sections.get(name, []):
            if line.startswith("//"):
                continue
            key, sep, value = line.partition(":")
            if sep:
                out.append([key.strip(), value.strip()])
        return out


def decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def parse(data: bytes) -> OsuFile:
    text = decode(data)
    osu = OsuFile()
    current: str | None = None
    seen_header = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if not seen_header:
            match = _HEADER.search(line)
            if match:
                osu.version = int(match.group(1))
                seen_header = True
                continue
            seen_header = True
        match = _SECTION.match(line)
        if match:
            current = str(match.group(1))
            osu.sections.setdefault(current, [])
            continue
        if current is None:
            continue
        if current != "Events" and line.startswith("//"):
            continue
        # Events lines keep their leading indentation: storyboard nesting depends on it.
        osu.sections[current].append(raw.rstrip() if current == "Events" else line)
    if not osu.sections and not _HEADER.search(text):
        raise CodecError("not an osu! beatmap: no header or sections found")
    return osu


def serialize(osu: OsuFile) -> bytes:
    lines = [f"osu file format v{osu.version}", ""]
    names = [n for n in SECTION_ORDER if n in osu.sections]
    names += [n for n in osu.sections if n not in SECTION_ORDER]
    for name in names:
        lines.append(f"[{name}]")
        lines.extend(osu.sections[name])
        lines.append("")
    return "\r\n".join(lines).encode("utf-8")


# -- numbers -------------------------------------------------------------------


def number(text: str) -> Fraction:
    """Exact value of a decimal string such as ``-33.3333333333333`` or ``1e-3``."""
    try:
        return Fraction(text.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise CodecError(f"invalid number {text!r}") from exc


def format_number(value: Fraction | int | float) -> str:
    """Shortest exact decimal for terminating fractions, ``repr(float)`` otherwise."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        return repr(float(value))
    digits = max(twos, fives)
    scaled = value * 10**digits
    sign = "-" if scaled < 0 else ""
    whole = str(abs(scaled.numerator))
    whole = whole.rjust(digits + 1, "0")
    return f"{sign}{whole[:-digits]}.{whole[-digits:]}"
