"""The IR's own JSON serialization exposed as a codec."""

from __future__ import annotations

from pathlib import Path

from urfm.codecs.base import Codec, Fidelity
from urfm.ir.chart import ChartSet
from urfm.ir.serialize import from_json, to_json


class UrfmJsonCodec(Codec):
    id = "urfm"
    extensions = (".urfm.json",)
    fidelity = Fidelity.SEMANTIC

    mechanics = frozenset()
    """Empty: the IR's own format accepts every mechanics (see :meth:`check_writable`)."""

    def sniff(self, path: Path, head: bytes) -> bool:
        return path.name.lower().endswith(".urfm.json")

    def read(self, data: bytes, *, source: str | None = None) -> ChartSet:
        return from_json(data)

    def write(self, chartset: ChartSet) -> bytes:
        return to_json(chartset, indent=2)

    def check_writable(self, chartset: ChartSet) -> None:
        return None
