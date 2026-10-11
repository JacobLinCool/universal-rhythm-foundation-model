"""Codec protocol: the only place where native chart formats are touched.

A codec converts between one native format and the IR. Codecs never convert
between games: a writer accepts only charts whose ``mechanics`` it supports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import ClassVar

from urfm.ir.chart import ChartSet


class Fidelity(IntEnum):
    """Round-trip guarantee of ``read(write(read(x))) == read(x)``. Higher is stronger."""

    NONE = 0
    TIMING = 1
    """L2: every event keeps its exact microsecond time, primitive, role and target."""
    SEMANTIC = 2
    """L1: the whole IR is identical, including controls, branches and native fields."""


class CodecError(ValueError):
    """Raised for malformed input or for charts a codec cannot represent."""


class Codec(ABC):
    id: ClassVar[str]
    """Short unique name, e.g. ``tja``."""
    extensions: ClassVar[tuple[str, ...]]
    """Lower-case file suffixes including the dot."""
    mechanics: ClassVar[frozenset[str]]
    """Mechanics spec ids this codec reads and writes."""
    fidelity: ClassVar[Fidelity] = Fidelity.SEMANTIC

    def sniff(self, path: Path, head: bytes) -> bool:
        """Whether ``path`` (with its first bytes ``head``) looks like this format."""
        return path.suffix.lower() in self.extensions

    @abstractmethod
    def read(self, data: bytes, *, source: str | None = None) -> ChartSet:
        """Parse native bytes into a chart set."""

    @abstractmethod
    def write(self, chartset: ChartSet) -> bytes:
        """Serialize a chart set; raise :class:`CodecError` if it cannot be represented."""

    def read_path(self, path: str | Path) -> ChartSet:
        path = Path(path)
        return self.read(path.read_bytes(), source=str(path))

    def check_writable(self, chartset: ChartSet) -> None:
        for chart in chartset.charts:
            if chart.mechanics not in self.mechanics:
                raise CodecError(f"{self.id} cannot write {chart.mechanics!r} charts")
