"""Musical time.

Every timed value in the IR is a :class:`Time`: an exact rational beat position
plus an integer microsecond residual. The absolute time of a ``Time`` is
``timing.beat_to_us(time.beat) + time.residual_us``.

Beat-native formats (TJA, StepMania, simai) always have ``residual_us == 0``.
Time-native formats (osu!) snap each timestamp to the nearest simple rational
beat and keep the leftover microseconds in ``residual_us``, so both the musical
position and the exact original timestamp survive.
"""

from __future__ import annotations

from fractions import Fraction

import msgspec

Beat = Fraction
"""A position or span measured in quarter-note beats, kept as an exact rational."""


class Time(msgspec.Struct, frozen=True, order=True):
    beat: Fraction
    residual_us: int = 0

    @classmethod
    def at(cls, beat: Fraction | int | str, residual_us: int = 0) -> Time:
        return cls(Fraction(beat), residual_us)

    def shifted(self, beats: Fraction | int) -> Time:
        return Time(self.beat + beats, self.residual_us)

    def __repr__(self) -> str:
        if self.residual_us:
            return f"Time({self.beat}{self.residual_us:+d}us)"
        return f"Time({self.beat})"


def snap(value: Fraction, max_denominator: int) -> Fraction:
    """Closest fraction to ``value`` whose denominator divides ``max_denominator``.

    Using divisors of a fixed resolution (instead of ``limit_denominator``) keeps
    snapped positions on musically meaningful grids such as 1/192 of a beat.
    """
    return Fraction(round(value * max_denominator), max_denominator)
