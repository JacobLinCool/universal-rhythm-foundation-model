"""Interface between charts and sequence models.

A training *sample* is one audio window of ``WINDOW_S`` seconds starting on a
measure boundary of one chart. The model sees:

* ``mel[frames, n_mels]`` for the window (``urfm.data.features``), plus
  ``frame_features(timing, start_us, n_frames)`` -- per-frame beat/measure phase
  and tempo from the chart's timing map (ground truth at training time,
  beat-tracker estimate at inference);
* decoder tokens: ``prefix`` conditioning tokens, ``BOS``, context (events and controls of
  the ``CONTEXT_BEATS`` whole beats before the window, see ``context``), ``SEP``, the
  window's density plan token ``WDENS``, the window's groups, ``EOS``.

A group is an event (below) or a presentation control ``POS_BEAT POS_SUB CTRL_<x> [values]``
(scroll speed, gogo/kiai on/off, bar lines on/off, meter change). At one position controls
come before events and always carry an explicit position.

Windows tile the whole audio (``audio_windows``) in both training and generation.

Each event is encoded as a fixed-order token group::

    POS_BEAT POS_SUB | PRIM [DUR] [ACCENT] | ROLE [TARGET...] [PATH...]
    ------ shared by all games ----------- | -- game specific (from spec) --

The tokenizer owns the vocabulary, the grammar, and decoding back to IR events.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Protocol

import numpy as np

from urfm.ir.chart import Chart, ChartSet
from urfm.ir.controls import Control
from urfm.ir.events import Event
from urfm.ir.timing import MeterChange, TimingMap

WINDOW_S = 16.0
"""Window length in seconds."""
CONTEXT_EVENTS = 32
"""Maximum number of previous-window events given as context."""


class Tokenizer(Protocol):
    vocab_size: int
    pad_id: int
    bos_id: int
    sep_id: int
    eos_id: int

    def prefix(self, chartset: ChartSet, chart: Chart) -> list[int]:
        """Conditioning tokens: game, input-space size, difficulty tier, density bin."""
        ...

    def conditioning(
        self,
        mechanics: str,
        *,
        tier: int | None = None,
        onsets_per_s: float | None = None,
        lanes: int | None = None,
    ) -> list[int]:
        """Prefix tokens from explicit values (any target game)."""
        ...

    def encode(self, chartset: ChartSet, chart: Chart, start_us: int, end_us: int) -> list[int]:
        """Event tokens for events with onset in ``[start_us, end_us)``, positions relative
        to the beat at ``start_us``."""
        ...

    def encode_groups(
        self, chartset: ChartSet, chart: Chart, start_us: int, end_us: int
    ) -> list[list[int]]:
        """Like :meth:`encode`, one token list per event (used to cut context to
        ``CONTEXT_EVENTS``)."""
        ...

    def context(
        self,
        chartset: ChartSet,
        chart: Chart,
        start_us: int,
        cache: dict | None = None,
        compiled: Any = None,
    ) -> list[int]:
        """Context tokens before a window (shared by training and generation)."""
        ...

    def encode_window(
        self,
        chartset: ChartSet,
        chart: Chart,
        start_us: int,
        end_us: int,
        room: int,
        cache: dict | None = None,
        compiled: Any = None,
        sections: list[Any] | None = None,
        k: int | None = None,
    ) -> tuple[list[int], bool]:
        """Window body ``WDENS group*`` (or section plans, R6) within ``room`` tokens;
        ``(tokens, truncated)``. ``cache`` (per chart) memoizes the position-independent part
        of each event."""
        ...

    def plan_tokens(self, sections: list[Any], window_beat: Fraction, k: int) -> list[int]:
        """Section plan tokens (R6) for sections of ``k`` beats."""
        ...

    def ref_block(
        self,
        chartset: ChartSet,
        chart: Chart,
        section: int,
        target: int,
        window_beat: Fraction,
        k: int,
        max_events: int = 48,
        cache: dict | None = None,
        compiled: Any = None,
    ) -> list[int]:
        """Structure-memory block ``REF SBEAT group* REFEND`` (R6)."""
        ...

    def window_density(self, onsets_per_s: float) -> int:
        """``WDENS`` token for a requested local onset density."""
        ...

    def decode(
        self,
        timing: TimingMap,
        mechanics: str,
        tokens: list[int],
        start_us: int,
        lanes: int | None = None,
    ) -> list[Event]:
        """Inverse of :meth:`encode`; invalid groups are dropped."""
        ...

    def decode_full(
        self,
        timing: TimingMap,
        mechanics: str,
        tokens: list[int],
        start_us: int,
        lanes: int | None = None,
    ) -> tuple[list[Event], list[Control], list[MeterChange]]:
        """Events plus presentation controls (scroll, highlight, barline) and meter changes."""
        ...

    def allowed(self, mechanics: str, tokens: list[int], lanes: int | None = None) -> np.ndarray:
        """Boolean mask over the vocabulary of tokens legal after ``tokens`` (grammar + spec).
        ``lanes`` overrides the spec for charts with a per-chart lane count (osu!mania)."""
        ...

    def frame_features(self, timing: TimingMap, start_us: int, n_frames: int) -> np.ndarray:
        """``float32 [n_frames, F]`` timing features aligned to mel frames."""
        ...

    def windows(
        self, chartset: ChartSet, chart: Chart, duration_us: int | None = None
    ) -> list[tuple[int, int]]:
        """Measure-aligned ``(start_us, end_us)`` windows that tile the chart without overlap."""
        ...
