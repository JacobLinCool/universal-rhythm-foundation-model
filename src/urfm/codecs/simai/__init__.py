"""maimai simai (``maidata.txt``) codec."""

from __future__ import annotations

from pathlib import Path

from urfm.codecs.base import Codec, Fidelity
from urfm.codecs.simai import reader, writer
from urfm.ir.chart import ChartSet


class SimaiCodec(Codec):
    id = "simai"
    extensions = (".txt",)
    mechanics = frozenset({reader.MECHANICS})
    fidelity = Fidelity.SEMANTIC

    def sniff(self, path: Path, head: bytes) -> bool:
        if path.suffix.lower() not in self.extensions:
            return False
        if not head:  # caller forced this codec without reading the file
            return True
        return b"&inote_" in head or b"&title=" in head

    def read(self, data: bytes, *, source: str | None = None) -> ChartSet:
        return reader.read(data, source)

    def write(self, chartset: ChartSet) -> bytes:
        self.check_writable(chartset)
        return writer.write(chartset)


__all__ = ["SimaiCodec"]
