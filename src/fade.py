"""Audio fade-out utilities for the soundscape synthesizer."""

import numpy as np
from typing import Optional


def apply_fade_out(
    audio: np.ndarray,
    sample_rate: int,
    fade_duration: float = 5.0,
    shape: str = "linear",
) -> np.ndarray:
    """Apply a fade-out envelope to the audio array.

    Args:
        audio: Audio data as a 2D array of shape (channels, samples).
        sample_rate: Sample rate in Hz.
        fade_duration: Duration of the fade in seconds.
        shape: Fade shape, either 'linear' or 'cosine'.

    Returns:
        The audio with the fade applied in-place (returns a new array).
    """
    if audio.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got {audio.ndim}D")

    fade_samples = int(fade_duration * sample_rate)
    if fade_samples <= 0:
        return audio.copy()

    if fade_samples > audio.shape[1]:
        fade_samples = audio.shape[1]

    start = audio.shape[1] - fade_samples
    if shape == "linear":
        envelope = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    elif shape == "cosine":
        # Cosine curve from 1 to 0, smooth at both ends
        t = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float32)
        envelope = np.cos(t)
    else:
        raise ValueError(f"Unknown fade shape: {shape}")

    result = audio.copy()
    result[:, start:] *= envelope
    return result


def apply_fade_in(
    audio: np.ndarray,
    sample_rate: int,
    fade_duration: float = 2.0,
    shape: str = "linear",
) -> np.ndarray:
    """Apply a fade-in envelope to the audio array.

    Args:
        audio: Audio data as a 2D array of shape (channels, samples).
        sample_rate: Sample rate in Hz.
        fade_duration: Duration of the fade in seconds.
        shape: Fade shape, either 'linear' or 'cosine'.

    Returns:
        The audio with the fade applied in-place (returns a new array).
    """
    if audio.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got {audio.ndim}D")

    fade_samples = int(fade_duration * sample_rate)
    if fade_samples <= 0:
        return audio.copy()

    if fade_samples > audio.shape[1]:
        fade_samples = audio.shape[1]

    if shape == "linear":
        envelope = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    elif shape == "cosine":
        t = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float32)
        envelope = np.sin(t)
    else:
        raise ValueError(f"Unknown fade shape: {shape}