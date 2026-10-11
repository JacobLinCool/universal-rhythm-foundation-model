"""Shared simai vocabulary: difficulty slots, decorators, slide operators, durations."""

from __future__ import annotations

from fractions import Fraction

from urfm.codecs.base import CodecError

US_PER_SECOND = 1_000_000
US_PER_MINUTE = 60_000_000

DIFFICULTY_LABELS = {
    1: "Easy",
    2: "Basic",
    3: "Advanced",
    4: "Expert",
    5: "Master",
    6: "Re:Master",
    7: "Utage",
}
DIFFICULTY_TIERS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 4}
LABEL_SLOTS = {label: slot for slot, label in DIFFICULTY_LABELS.items()}

# Two-character operators must be tried before their one-character prefixes.
SLIDE_OPS = ("pp", "qq", "-", "^", "<", ">", "v", "V", "p", "q", "s", "z", "w")
SLIDE_OP_CHARS = frozenset("-^<>vVpqszw")
SLIDE_VERTICES = {"V": 2}
"""Number of button vertices that follow an operator (default 1)."""

NOTE_MODIFIERS = {"b": "break", "x": "ex", "m": "mine", "f": "firework"}
MODIFIER_CHARS = {name: char for char, name in NOTE_MODIFIERS.items()}
APPEARANCES = ("$$", "$", "@")
HEADLESS = ("?", "!")

PSEUDO_EACH_STEP = Fraction(1, 32)
"""Beat offset between parts of a pseudo-each (backtick) group: a 1/128 note."""


def parse_number(text: str) -> Fraction:
    try:
        return Fraction(text.strip())
    except (ValueError, ZeroDivisionError):
        raise CodecError(f"invalid number {text!r}") from None


def format_number(value: Fraction) -> str:
    """Shortest exact decimal for ``value`` when one exists, else a 6-digit approximation."""
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
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    digits = max(twos, fives)
    scaled = value * 10**digits
    sign = "-" if scaled < 0 else ""
    whole, frac = divmod(abs(scaled.numerator), 10**digits)
    return f"{sign}{whole}.{str(frac).rjust(digits, '0').rstrip('0')}"


def beats_us(beats: Fraction, bpm: Fraction) -> Fraction:
    return beats * US_PER_MINUTE / bpm


def parse_ratio(text: str, bpm: Fraction) -> Fraction:
    """``div:n`` -> microseconds at ``bpm`` (``n`` notes of a ``div``-th of a measure)."""
    div, _, count = text.partition(":")
    return beats_us(4 * parse_number(count) / parse_number(div), bpm)


def parse_length(text: str, bpm: Fraction) -> Fraction:
    """A length that is either ``div:n`` (beats) or plain seconds."""
    if ":" in text:
        return parse_ratio(text, bpm)
    return parse_number(text) * US_PER_SECOND


def parse_hold_duration(raw: str, bpm: Fraction) -> Fraction:
    """Hold length in microseconds for ``div:n``, ``bpm#div:n``, ``#sec`` or ``bpm#sec``."""
    if "#" in raw:
        head, _, tail = raw.partition("#")
        if head and ":" in tail:
            return parse_ratio(tail, parse_number(head))
        return parse_length(tail, bpm)
    return parse_ratio(raw, bpm)


def parse_slide_duration(raw: str, bpm: Fraction) -> tuple[Fraction, Fraction]:
    """``(wait_us, duration_us)`` of one slide duration bracket.

    * ``div:n`` -- wait one beat, move for ``div:n`` at the current tempo.
    * ``bpm#div:n`` / ``bpm#sec`` -- wait one beat of ``bpm``, length at ``bpm`` or seconds.
    * ``#sec`` -- default wait, length in seconds.
    * ``wait##div:n`` / ``wait##sec`` -- explicit wait in seconds.
    """
    if "##" in raw:
        wait, _, length = raw.partition("##")
        return parse_number(wait) * US_PER_SECOND, parse_length(length, bpm)
    if "#" in raw:
        head, _, length = raw.partition("#")
        if not head:
            return beats_us(Fraction(1), bpm), parse_length(length, bpm)
        local = parse_number(head)
        return beats_us(Fraction(1), local), parse_length(length, local)
    return beats_us(Fraction(1), bpm), parse_ratio(raw, bpm)


def ratio_for(us: Fraction, bpm: Fraction, max_div: int = 1536) -> str | None:
    """Express ``us`` as ``div:n`` at ``bpm`` with the smallest valid ``div``, if any."""
    beats = us * bpm / US_PER_MINUTE  # length = 4 * n / div beats
    quarter = beats / 4
    if quarter < 0:
        return None
    div = quarter.denominator
    if div > max_div:
        return None
    return f"{div}:{quarter.numerator}"


def seconds_for(us: Fraction) -> str:
    return format_number(us / US_PER_SECOND)
