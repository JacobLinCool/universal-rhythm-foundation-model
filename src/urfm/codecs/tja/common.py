"""Shared vocabulary of the TJA codec."""

from __future__ import annotations

from fractions import Fraction

from urfm.ir.events import Primitive

US = 1_000_000

ROUTES = {"N": "normal", "E": "expert", "M": "master"}
ROUTE_COMMANDS = {v: k for k, v in ROUTES.items()}

COURSE_NAMES = {
    "0": "easy",
    "1": "normal",
    "2": "hard",
    "3": "oni",
    "4": "edit",
    "5": "tower",
    "6": "dan",
    "easy": "easy",
    "normal": "normal",
    "hard": "hard",
    "oni": "oni",
    "edit": "edit",
    "ura": "edit",
    "tower": "tower",
    "dan": "dan",
}
COURSE_TIER = {"easy": 0, "normal": 1, "hard": 2, "oni": 3, "edit": 4}

# Header keys that describe one course. Everything else is song-level.
COURSE_KEYS = frozenset(
    {
        "COURSE",
        "LEVEL",
        "BALLOON",
        "BALLOONNOR",
        "BALLOONEXP",
        "BALLOONMAS",
        "SCOREINIT",
        "SCOREDIFF",
        "STYLE",
        "TOTAL",
        "GAUGEINCR",
        "HIDDENBRANCH",
        "LIFE",
        "NOTESDESIGNER0",
        "NOTESDESIGNER1",
        "NOTESDESIGNER2",
        "NOTESDESIGNER3",
        "NOTESDESIGNER4",
        "EXAM1",
        "EXAM2",
        "EXAM3",
        "EXAM4",
        "EXAMGAUGE",
    }
)
BALLOON_KEYS = {
    None: "BALLOON",
    "normal": "BALLOONNOR",
    "expert": "BALLOONEXP",
    "master": "BALLOONMAS",
}

# Song headers that map to structured IR fields.
SONG_FIELDS = {"TITLE": "title", "SUBTITLE": "subtitle", "GENRE": "genre", "MAKER": "creator"}
TIMING_KEYS = frozenset({"BPM", "OFFSET"})
STRUCTURED_SONG_KEYS = frozenset({*SONG_FIELDS, "WAVE", "DEMOSTART", *TIMING_KEYS})

# note char -> (primitive, role, modifiers)
NOTES: dict[str, tuple[Primitive, str, tuple[str, ...]]] = {
    "1": (Primitive.HIT, "taiko:don", ()),
    "2": (Primitive.HIT, "taiko:ka", ()),
    "3": (Primitive.HIT, "taiko:don", ("big",)),
    "4": (Primitive.HIT, "taiko:ka", ("big",)),
    "5": (Primitive.REPEAT, "taiko:drumroll", ()),
    "6": (Primitive.REPEAT, "taiko:drumroll", ("big",)),
    "7": (Primitive.REPEAT, "taiko:balloon", ()),
    "9": (Primitive.REPEAT, "taiko:balloon", ("kusudama",)),
    "A": (Primitive.HIT, "taiko:don", ("big", "hand")),
    "B": (Primitive.HIT, "taiko:ka", ("big", "hand")),
    "C": (Primitive.HIT, "taiko:mine", ()),
    "F": (Primitive.HIT, "taiko:adlib", ()),
    "G": (Primitive.HIT, "taiko:kadon", ()),
}
NOTE_CHARS = {(role, mods): char for char, (_, role, mods) in NOTES.items()}
ROLL_END = "8"
# Slot-occupying characters without an IR role (simulator extensions); kept natively.
UNKNOWN_NOTE_CHARS = frozenset("DEHI")
SLOT_CHARS = frozenset("0" + ROLL_END) | NOTES.keys() | UNKNOWN_NOTE_CHARS

# commands mapped to shared/namespaced control kinds: name -> (kind, value-or-None)
FLAG_CONTROLS: dict[str, tuple[str, bool | None]] = {
    "GOGOSTART": ("highlight", True),
    "GOGOEND": ("highlight", False),
    "BARLINEON": ("barline", True),
    "BARLINEOFF": ("barline", False),
    "SECTION": ("taiko:section", None),
    "LEVELHOLD": ("taiko:levelhold", None),
}
FLAG_COMMANDS = {(kind, value): name for name, (kind, value) in FLAG_CONTROLS.items()}
RAW_PREFIX = "tja:"


def parse_decimal(text: str) -> Fraction:
    text = text.strip()
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return Fraction(float(text))


def format_decimal(value: Fraction) -> str:
    """Exact decimal text for ``value`` when it has a terminating expansion."""
    value = Fraction(value)
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
    if digits == 0:
        return str(value.numerator)
    scaled = abs(value.numerator) * 10**digits // value.denominator
    sign = "-" if value < 0 else ""
    text = f"{scaled:0{digits + 1}d}"
    return f"{sign}{text[:-digits]}.{text[-digits:]}".rstrip("0").rstrip(".")


def format_float(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))
