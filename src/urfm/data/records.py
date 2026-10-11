"""Song records: the unit stored in dataset shards.

One record per song (one audio track) and source. Shards are Parquet files on
the Hugging Face Hub at ``data/<source>/<split>-<index>.parquet``; every column
maps to a field of :class:`SongRecord`. ``chartset`` holds all charts of the
song as IR msgpack, so tokenization happens at load time and tokenizer changes
never require re-ingesting audio.
"""

from __future__ import annotations

import msgspec

SPLITS = ("train", "validation", "test")


class SongRecord(msgspec.Struct, kw_only=True):
    song_id: str
    """``<source>:<key>``; unique within the dataset."""
    source: str
    """``taiko1000``, ``osu``, ..."""
    split: str
    group: str
    """Song-level grouping key; all records of one group share a split."""
    audio_sha256: str
    title: str = ""
    artist: str = ""
    duration_s: float
    mel_version: str
    mel_frames: int
    mel: bytes
    """uint8 ``[mel_frames, n_mels]`` row-major (see :mod:`urfm.data.features`)."""
    chartset: bytes
    """IR :class:`~urfm.ir.chart.ChartSet` encoded with :func:`urfm.ir.serialize.to_msgpack`."""
    mechanics: list[str]
    n_charts: int
    n_events: int


def arrow_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("song_id", pa.string()),
            ("source", pa.string()),
            ("split", pa.string()),
            ("group", pa.string()),
            ("audio_sha256", pa.string()),
            ("title", pa.string()),
            ("artist", pa.string()),
            ("duration_s", pa.float32()),
            ("mel_version", pa.string()),
            ("mel_frames", pa.int32()),
            ("mel", pa.large_binary()),
            ("chartset", pa.large_binary()),
            ("mechanics", pa.list_(pa.string())),
            ("n_charts", pa.int32()),
            ("n_events", pa.int32()),
        ]
    )
