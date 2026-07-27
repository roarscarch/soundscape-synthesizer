"""
Core engine: deterministic wave function collapse over a 2D grid of audio grains.
Each grain is a short waveform segment. The collapse propagates constraints based on
adjacent grains to produce non-repeating, evolving soundscapes.
"""

import numpy as np
from hashlib import sha256
from typing import List, Tuple, Optional


def seed_to_int(seed: str) -> int:
    """Convert a seed string to a deterministic integer using SHA-256."""
    return int(sha256(seed.encode()).hexdigest(), 16)


class Grain:
    """A single audio grain: a short waveform segment with spectral properties."""

    def __init__(self, waveform: np.ndarray, freq_bin: int, amplitude: float):
        """
        :param waveform: 1D numpy array of audio samples (float32, range -1 to 1)
        :param freq_bin: integer representing frequency bin index (0..N-1)
        :param amplitude: overall amplitude scaling factor (0..1)
        """
        self.waveform = waveform.astype(np.float32)
        self.freq_bin = freq_bin
        self.amplitude = amplitude

    def __repr__(self) -> str:
        return f"Grain(freq_bin={self.freq_bin}, amp={self.amplitude:.3f}, len={len(self.waveform)}