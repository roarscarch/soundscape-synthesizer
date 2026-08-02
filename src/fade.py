"""Audio fade-out utility for gentle soundscape endings."""

import numpy as np
from typing import Optional


def apply_fade_out(
    audio: np.ndarray,
    fade_duration: float,
    sample_rate: int = 44100,
    curve: str = "linear",
) -> np.ndarray:
    """
    Apply a fade-out envelope to the end of an audio buffer.

    Args:
        audio: Audio buffer of shape (samples,) for mono or (samples, channels) for stereo.
        fade_duration: Duration of the fade in seconds.
        sample_rate: Sample rate in Hz.
        curve: Fade curve type. Options: 'linear', 'exponential', 'cosine'.

    Returns:
        The audio buffer with the fade applied.

    Raises:
        ValueError: If fade_duration is non-positive or longer than the audio length.
        ValueError: If curve is not a known type.
    """
    if fade_duration <= 0:
        raise ValueError("fade_duration must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    num_samples = audio.shape[0]
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples > num_samples:
        raise ValueError("fade_duration cannot exceed audio duration")
    if fade_samples == 0:
        return audio.copy()

    # Envelope over the fade region
    t = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)

    if curve == "linear":
        envelope = t
    elif curve == "exponential":
        # Exponential taper: start at 1.0, end near 0.0, with a smoother tail
        # Avoid division by zero by using a small epsilon.
        epsilon = 1e-6
        envelope = (np.exp(-3.0 * np.linspace(0.0, 1.0, fade_samples)) - epsilon) / (np.exp(-3.0) - epsilon)
        envelope = np.clip(envelope, 0.0, 1.0)
    elif curve == "cosine":
        # Cosine curve: starts at 1.0, ends at 0.0, smooth both ends
        envelope = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_samples)))
    else:
        raise ValueError(f"Unknown curve type: {curve}