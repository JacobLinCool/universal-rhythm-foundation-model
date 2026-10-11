"""JSON / msgpack serialization of the IR.

Exact rationals are encoded as strings (``"3/4"``) so that serialized charts
remain lossless and human readable.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Any

import msgspec

from urfm.ir.chart import SCHEMA, AudioRef, Chart, ChartSet, Provenance, SongMeta
from urfm.ir.timing import TimingMap


def _enc_hook(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return str(obj)
    raise NotImplementedError(f"cannot encode {type(obj).__name__}")


_from_coprime = getattr(Fraction, "_from_coprime_ints", None)


def _dec_hook(tp: type, obj: Any) -> Any:
    if tp is Fraction:
        if isinstance(obj, int):
            return Fraction(obj)
        if isinstance(obj, str):
            num, sep, den = obj.partition("/")
            if not sep:
                return Fraction(obj)
            n, d = int(num), int(den)
            # encoded fractions are already in lowest terms (str(Fraction)); skip the gcd
            if _from_coprime is not None and d > 0:
                return _from_coprime(n, d)
            return Fraction(n, d)
    raise NotImplementedError(f"cannot decode {tp}")


def _schema_hook(tp: type) -> dict[str, Any]:
    if tp is Fraction:
        return {"type": "string", "pattern": r"^-?\d+(/\d+)?$", "description": "exact rational"}
    raise NotImplementedError


_json_encoder = msgspec.json.Encoder(enc_hook=_enc_hook, order="deterministic")
_json_decoder = msgspec.json.Decoder(ChartSet, dec_hook=_dec_hook)
_mp_encoder = msgspec.msgpack.Encoder(enc_hook=_enc_hook, order="deterministic")
_mp_decoder = msgspec.msgpack.Decoder(ChartSet, dec_hook=_dec_hook)


def to_json(chartset: ChartSet, *, indent: int = 0) -> bytes:
    data = _json_encoder.encode(chartset)
    return msgspec.json.format(data, indent=indent) if indent else data


def from_json(data: bytes | str) -> ChartSet:
    return _json_decoder.decode(data)


def to_msgpack(chartset: ChartSet) -> bytes:
    return _mp_encoder.encode(chartset)


def from_msgpack(data: bytes) -> ChartSet:
    return _mp_decoder.decode(data)


def json_schema() -> dict[str, Any]:
    return msgspec.json.schema(ChartSet, schema_hook=_schema_hook)


def schema_digest() -> str:
    """SHA-256 of the canonical JSON schema; changes whenever the IR shape changes."""
    canonical = msgspec.json.encode(json_schema(), order="sorted")
    return hashlib.sha256(canonical).hexdigest()


class _LazyChartSet(msgspec.Struct, frozen=True, kw_only=True):
    """``ChartSet`` with charts left as undecoded msgpack (same wire format)."""

    schema: str = SCHEMA
    song: SongMeta = SongMeta()
    audio: AudioRef | None = None
    timing: TimingMap
    charts: tuple[msgspec.Raw, ...] = ()
    provenance: Provenance | None = None
    native: dict[str, Any] = {}


_lazy_decoder = msgspec.msgpack.Decoder(_LazyChartSet, dec_hook=_dec_hook)
_chart_decoder = msgspec.msgpack.Decoder(Chart, dec_hook=_dec_hook)


class LazyChartSet:
    """Decode a chart set's shared parts once and individual charts on demand.

    A song often carries a dozen difficulties while training samples one or two of them;
    decoding only those avoids most of the msgpack/Fraction work.
    """

    def __init__(self, data: bytes) -> None:
        self._lazy = _lazy_decoder.decode(data)
        self._views: dict[int, ChartSet] = {}

    def __len__(self) -> int:
        return len(self._lazy.charts)

    def view(self, index: int) -> ChartSet:
        """A ``ChartSet`` holding only chart ``index`` (as ``charts[0]``)."""
        cached = self._views.get(index)
        if cached is None:
            lazy = self._lazy
            chart = _chart_decoder.decode(lazy.charts[index])
            cached = self._views[index] = ChartSet(
                schema=lazy.schema,
                song=lazy.song,
                audio=lazy.audio,
                timing=lazy.timing,
                charts=(chart,),
                provenance=lazy.provenance,
                native=lazy.native,
            )
        return cached


_timing_decoder = msgspec.msgpack.Decoder(TimingMap, dec_hook=_dec_hook)


def timing_to_msgpack(timing: TimingMap) -> bytes:
    return _mp_encoder.encode(timing)


def timing_from_msgpack(data: bytes) -> TimingMap:
    return _timing_decoder.decode(data)
