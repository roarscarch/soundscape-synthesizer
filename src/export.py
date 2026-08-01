"""Export soundscape audio to WAV file."""

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional


def export_wav(
    audio: np.ndarray,
    filepath: str,
    sample_rate: int = 44100,
) -> None:
    """
    Write audio data to a WAV file.

    Args:
        audio: Audio samples. Shape (n_samples,) for mono or (n_samples, 2) for stereo.
        filepath: Destination path for the WAV file.
        sample_rate: Sample rate in Hz.

    Raises:
        ValueError: If audio is not numeric or has unexpected shape.
    """
    if audio.ndim not in (1, 2):
        raise ValueError(f"audio must be 1D or 2D, got shape {audio.shape}