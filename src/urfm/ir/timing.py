"""Tempo map: the exact mapping between beats and absolute time.

A :class:`TimingMap` is the single source of truth for converting beat positions
to microseconds. It models every timing construct found in mainstream rhythm
games with four primitives:

* :class:`TempoChange` -- BPM from a beat onward.
* :class:`Stop` -- time advances while the beat position is frozen
  (StepMania ``STOPS``/``DELAYS``, TJA ``#DELAY``). ``before`` decides whether
  objects exactly on the stop beat are hit before (stop) or after (delay) it.
* :class:`Warp` -- a beat span that takes no time (StepMania ``WARPS``).
* :class:`MeterChange` -- time signature; it does not affect time, only the
  measure grid used by measure-based formats and by musical features.

Beat 0 is located at ``offset_us`` microseconds of the audio. Before the first
tempo change the first tempo extends backwards, so negative beats are valid.
"""

from __future__ import annotations

import bisect
from fractions import Fraction

import msgspec

from urfm.ir.time import Time

US_PER_MINUTE = 60_000_000


class TempoChange(msgspec.Struct, frozen=True):
    beat: Fraction
    bpm: Fraction


class Stop(msgspec.Struct, frozen=True):
    beat: Fraction
    duration_us: Fraction
    before: bool = False
    """``False``: objects on ``beat`` occur when the stop starts (StepMania stop).
    ``True``: objects on ``beat`` occur when the stop ends (delay, TJA ``#DELAY``)."""


class Warp(msgspec.Struct, frozen=True):
    beat: Fraction
    length: Fraction


class MeterChange(msgspec.Struct, frozen=True):
    beat: Fraction
    numerator: int
    denominator: int

    @property
    def measure_beats(self) -> Fraction:
        return Fraction(4 * self.numerator, self.denominator)


class TimingMap(msgspec.Struct, frozen=True, kw_only=True):
    offset_us: Fraction = msgspec.field(default_factory=Fraction)
    tempos: tuple[TempoChange, ...]
    stops: tuple[Stop, ...] = ()
    warps: tuple[Warp, ...] = ()
    meters: tuple[MeterChange, ...] = ()

    @classmethod
    def constant(cls, bpm: Fraction | int | str, offset_us: Fraction | int = 0) -> TimingMap:
        return cls(offset_us=Fraction(offset_us), tempos=(TempoChange(Fraction(0), Fraction(bpm)),))

    # -- conversions -------------------------------------------------------

    def beat_to_us_exact(self, beat: Fraction) -> Fraction:
        return _compile(self).beat_to_us(Fraction(beat))

    def us(self, time: Time) -> int:
        """Absolute integer microseconds of ``time``."""
        return round(self.beat_to_us_exact(time.beat)) + time.residual_us

    def seconds(self, time: Time) -> float:
        return self.us(time) / 1_000_000

    def us_to_beat(self, us: Fraction | int) -> Fraction:
        """Inverse of :meth:`beat_to_us_exact`.

        With negative stops (e.g. osu! red lines re-anchoring the grid) time is not
        monotonic and a timestamp can map to several beats; the latest is returned, which
        matches the tokenizer's vectorized ``beats_at_us``.
        """
        return _compile(self).us_to_beat(Fraction(us))

    def us_array(self, beats, residual_us=None):
        """Vectorized absolute µs (float64) for many beats.

        For slicing, density and other bulk work where float precision (≪1 µs over hours of
        audio) is enough; token positions and exact comparisons use :meth:`us`.
        """
        import numpy as np

        c = _compile(self)
        b = np.asarray([float(x) for x in beats], dtype=np.float64)
        knots, base, incl, strict, rate = c.arrays()
        i = np.searchsorted(knots, b, side="right") - 1
        j = np.clip(i, 0, None)
        on_knot = (i >= 0) & (b == knots[j])
        after = base[j] + incl[j] + strict[j] + (b - knots[j]) * rate[j]
        out = np.where(on_knot, base[j] + incl[j], after)
        before = base[0] - (knots[0] - b) * float(c.first_rate)
        out = np.where(i < 0, before, out)
        if residual_us is not None:
            out = out + np.asarray(residual_us, dtype=np.float64)
        return out

    def time_at_us(self, us: int, resolution: int = 192) -> Time:
        """Snap an absolute timestamp to a beat grid, keeping the exact residual."""
        beat = Fraction(round(self.us_to_beat(us) * resolution), resolution)
        return Time(beat, us - round(self.beat_to_us_exact(beat)))

    def bpm_at(self, beat: Fraction) -> Fraction:
        tempos = self.tempos
        i = bisect.bisect_right([t.beat for t in tempos], beat) - 1
        return tempos[max(i, 0)].bpm

    def meter_at(self, beat: Fraction) -> MeterChange:
        meters = self.meters or (MeterChange(Fraction(0), 4, 4),)
        i = bisect.bisect_right([m.beat for m in meters], beat) - 1
        return meters[max(i, 0)]

    def measure_starts(self, until: Fraction) -> list[Fraction]:
        """Beats on which measures start, from beat 0 up to ``until`` (inclusive). A meter of
        zero length (malformed source) contributes one measure start."""
        meters = self.meters or (MeterChange(Fraction(0), 4, 4),)
        starts: list[Fraction] = []
        for i, meter in enumerate(meters):
            last = i + 1 == len(meters)
            beat = meter.beat
            if meter.measure_beats <= 0:
                starts.append(beat)
                continue
            while beat <= until if last else beat < meters[i + 1].beat:
                starts.append(beat)
                beat += meter.measure_beats
        return starts


# -- compiled piecewise representation ---------------------------------------


class _Compiled:
    """Piecewise-linear beat -> time function with jumps at stops.

    ``knots[i]`` is a beat; ``base[i]`` is the time at that knot excluding stops
    located on the knot; ``rate[i]`` is microseconds per beat until the next knot.
    """

    def __init__(self, timing: TimingMap) -> None:
        if not timing.tempos:
            raise ValueError("TimingMap needs at least one tempo")
        tempos = sorted(timing.tempos, key=lambda t: t.beat)
        stops = sorted(timing.stops, key=lambda s: s.beat)
        warps = sorted(timing.warps, key=lambda w: w.beat)
        knots = sorted(
            {Fraction(0)}
            | {t.beat for t in tempos if t.beat > 0}
            | {s.beat for s in stops}
            | {w.beat for w in warps}
            | {w.beat + w.length for w in warps}
        )
        tempo_beats = [t.beat for t in tempos]

        def us_per_beat(beat: Fraction) -> Fraction:
            i = max(bisect.bisect_right(tempo_beats, beat) - 1, 0)
            bpm = tempos[i].bpm
            if bpm <= 0:
                raise ValueError(f"non-positive BPM {bpm} at beat {tempos[i].beat}")
            return US_PER_MINUTE / bpm

        def warped(beat: Fraction) -> bool:
            return any(w.beat <= beat < w.beat + w.length for w in warps)

        strict: dict[Fraction, Fraction] = {}  # stops applied to beats > knot
        inclusive: dict[Fraction, Fraction] = {}  # delays applied to beats >= knot
        for s in stops:
            target = inclusive if s.before else strict
            target[s.beat] = target.get(s.beat, Fraction(0)) + s.duration_us

        self.knots = knots
        self.rate = [Fraction(0) if warped(k) else us_per_beat(k) for k in knots]
        self.strict = [strict.get(k, Fraction(0)) for k in knots]
        self.inclusive = [inclusive.get(k, Fraction(0)) for k in knots]
        self.first_rate = us_per_beat(min(knots[0], tempos[0].beat))
        self.base = [timing.offset_us]
        for i in range(1, len(knots)):
            prev = i - 1
            span = knots[i] - knots[prev]
            self.base.append(
                self.base[prev] + self.inclusive[prev] + self.strict[prev] + span * self.rate[prev]
            )

    def arrays(self):
        """Float64 views of the knots for vectorized evaluation (cached)."""
        cached = getattr(self, "_arrays", None)
        if cached is None:
            import numpy as np

            cached = tuple(
                np.asarray([float(x) for x in xs], dtype=np.float64)
                for xs in (self.knots, self.base, self.inclusive, self.strict, self.rate)
            )
            self._arrays = cached
        return cached

    def beat_to_us(self, beat: Fraction) -> Fraction:
        i = bisect.bisect_right(self.knots, beat) - 1
        if i < 0:
            return self.base[0] - (self.knots[0] - beat) * self.first_rate
        if beat == self.knots[i]:
            return self.base[i] + self.inclusive[i]
        k = self.knots[i]
        return self.base[i] + self.inclusive[i] + self.strict[i] + (beat - k) * self.rate[i]

    def us_to_beat(self, us: Fraction) -> Fraction:
        """Latest beat whose time is ``us`` (negative stops make time non-monotonic; the
        later segment is the musically current one, e.g. after an osu! red line)."""
        n = len(self.knots)
        for i in reversed(range(n)):
            start = self.base[i] + self.inclusive[i] + self.strict[i]
            end = self.base[i + 1] if i + 1 < n else None
            if us >= start and (end is None or us < end) and self.rate[i] > 0:
                return self.knots[i] + (us - start) / self.rate[i]
            if self.base[i] <= us < start:  # inside a stop: the beat is frozen on the knot
                return self.knots[i]
        if us < self.base[0]:
            return self.knots[0] - (self.base[0] - us) / self.first_rate
        return self.knots[-1]


_COMPILED: dict[int, tuple[TimingMap, _Compiled]] = {}
_COMPILED_MAX = 512


def _compile(timing: TimingMap) -> _Compiled:
    """Compiled form of ``timing``, cached by object identity.

    Hashing a TimingMap hashes every tempo/stop Fraction, which made each conversion
    O(#timing points) for osu! maps with hundreds of red/green lines. The cache keeps a
    reference to the map, so an ``id`` cannot be reused while its entry is alive.
    """
    hit = _COMPILED.get(id(timing))
    if hit is not None and hit[0] is timing:
        return hit[1]
    compiled = _Compiled(timing)
    if len(_COMPILED) >= _COMPILED_MAX:
        _COMPILED.pop(next(iter(_COMPILED)))
    _COMPILED[id(timing)] = (timing, compiled)
    return compiled
