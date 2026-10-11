"""Model hyper-parameters and size presets."""

from __future__ import annotations

from typing import Any

import msgspec


class ModelConfig(msgspec.Struct, frozen=True, kw_only=True):
    vocab_size: int
    frame_feat_dim: int
    n_mels: int = 128
    d_model: int = 384
    enc_layers: int = 4
    dec_layers: int = 4
    n_heads: int = 6
    ff_mult: int = 4
    emb_dim: int | None = None
    """Factorized (ALBERT-style) token embedding width; ``None`` = ``d_model``."""
    max_enc_len: int = 704
    """Encoder positions after the 2x downsampling stem (16 s at 86 fps -> 690)."""
    max_dec_len: int = 2048
    dropout: float = 0.0
    memory_tokens: int = 0
    """Learned memory slots appended to the encoder output (unused by default)."""
    song_context: bool = False
    """Add a song-level encoder over the pooled whole-song mel (``urfm.train.data``) whose
    outputs extend the cross-attention memory, so each window sees the whole song."""
    song_layers: int = 2
    song_max: int = 256
    """Maximum song units (pooled mel blocks, or texture units)."""
    song_feat_dim: int = 0
    """Song unit features: 0 = pooled uint8 mel blocks (``n_mels``); otherwise float
    beat-synchronous texture features of this width (``urfm.data.texture``)."""
    cond_dim: int = 0
    """Number of continuous chart conditions (``urfm.train.conditions``); 0 = none."""
    cond_token: int = -1
    """Token id whose embedding is replaced by the condition embedding."""


# name -> (d_model, enc_layers, dec_layers, n_heads); head dim 64 (flash-attention friendly)
PRESETS: dict[str, tuple[int, int, int, int]] = {
    "tiny": (64, 1, 1, 2),
    "5m": (256, 3, 2, 4),
    "10m": (320, 4, 3, 5),
    "15m": (384, 4, 3, 6),
    "45m": (512, 6, 6, 8),
}


def preset(name: str, *, vocab_size: int, frame_feat_dim: int, **overrides) -> ModelConfig:
    d, enc, dec, heads = PRESETS[name]
    fields: dict[str, Any] = dict(d_model=d, enc_layers=enc, dec_layers=dec, n_heads=heads)
    fields.update(overrides)
    return ModelConfig(vocab_size=vocab_size, frame_feat_dim=frame_feat_dim, **fields)
