"""TJA (Taiko no Tatsujin simulator) codec."""

from __future__ import annotations

from pathlib import Path

from urfm.codecs.base import Codec, Fidelity
from urfm.codecs.tja.reader import read_tja
from urfm.codecs.tja.writer import write_tja
from urfm.ir.chart import ChartSet


class TjaCodec(Codec):
    id = "tja"
    extensions = (".tja",)
    mechanics = frozenset({"taiko"})
    fidelity = Fidelity.SEMANTIC

    def sniff(self, path: Path, head: bytes) -> bool:
        return path.suffix.lower() in self.extensions

    def read(self, data: bytes, *, source: str | None = None) -> ChartSet:
        return read_tja(data, source)

    def write(self, chartset: ChartSet) -> bytes:
        self.check_writable(chartset)
        return write_tja(chartset).encode("utf-8")


__all__ = ["TjaCodec"]
