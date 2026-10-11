"""Audio feature specification shared by ingest (Colab) and training.

Log-mel spectrograms are computed once per song on the GPU and stored as
``uint8``: decibels relative to the song's maximum, clipped to ``[-DB_RANGE, 0]``
and mapped linearly to ``0..255`` (≈0.31 dB per step). This halves storage and
host->device traffic versus fp16; :func:`dequantize` restores float dB on device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    import torch


class MelSpec(msgspec.Struct, frozen=True):
    sample_rate: int = 22_050
    n_fft: int = 2_048
    hop_length: int = 256
    n_mels: int = 128
    f_min: float = 20.0
    f_max: float = 11_025.0
    db_range: float = 80.0
    version: str = "mel-v1"

    @property
    def fps(self) -> float:
        return self.sample_rate / self.hop_length


MEL = MelSpec()


def mel_transform(spec: MelSpec = MEL, device: str = "cuda"):
    import torchaudio

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=spec.sample_rate,
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
        n_mels=spec.n_mels,
        f_min=spec.f_min,
        f_max=spec.f_max,
        power=2.0,
        center=True,
    ).to(device)


def quantize(power: torch.Tensor, spec: MelSpec = MEL) -> torch.Tensor:
    """``[n_mels, T]`` mel power -> ``[T, n_mels]`` uint8 (time-major for window slicing)."""
    import torch

    db = 10.0 * torch.log10(power.clamp_min(1e-10))
    db = (db - db.amax()).clamp_min(-spec.db_range)
    return ((db + spec.db_range) * (255.0 / spec.db_range)).round().to(torch.uint8).T.contiguous()


def dequantize(q: torch.Tensor, spec: MelSpec = MEL) -> torch.Tensor:
    """uint8 ``[..., T, n_mels]`` -> float dB in ``[-db_range, 0]`` (run on device)."""
    return q.float() * (spec.db_range / 255.0) - spec.db_range


def audio_mel(path, spec: MelSpec = MEL, device: str = "cpu"):
    """uint8 log-mel ``[T, n_mels]`` of an audio file (decoded by ffmpeg, mono)."""
    import subprocess

    import numpy as np
    import torch

    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(spec.sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    wav = torch.from_numpy(np.frombuffer(pcm, np.float32).copy()).to(device)
    with torch.no_grad():
        return quantize(mel_transform(spec, device)(wav), spec).cpu().numpy()
