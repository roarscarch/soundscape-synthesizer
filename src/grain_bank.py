"""
Grain bank: stores and selects audio grains deterministically from wave tables.
Each grain is a short audio snippet shaped by an envelope.
"""

import numpy as np
from typing import List, Optional
from hashlib import sha256


class GrainBank:
    """Manages a collection of grains derived from a biome's wave table."""

    def __init__(
        self,
        wave_table: np.ndarray,
        sample_rate: int = 44100,
        grain_duration: float = 0.1,
        seed: Optional[str] = None,
    ):
        """
        :param wave_table: 2D array of shape (num_waves, num_samples) containing base waveforms
        :param sample_rate: sample rate in Hz
        :param grain_duration: duration of each grain in seconds
        :param seed: optional seed string for deterministic grain ordering
        """
        if wave_table.ndim != 2:
            raise ValueError(f"wave_table must be 2D, got shape {wave_table.shape}")
        if wave_table.shape[0] == 0:
            raise ValueError("wave_table must contain at least one waveform")
        if grain_duration <= 0:
            raise ValueError("grain_duration must be positive")

        self.wave_table = wave_table.astype(np.float64)
        self.sample_rate = sample_rate
        self.grain_duration = grain_duration
        self.grain_length = int(sample_rate * grain_duration)

        # Seeded PRNG for deterministic grain selection
        if seed is None:
            seed = ""
        seed_bytes = seed.encode("utf-8")
        hash_digest = sha256(seed_bytes).digest()
        # Use first 4 bytes as seed for numpy's PCG64
        self._rng = np.random.Generator(np.random.PCG64(np.frombuffer(hash_digest[:8], dtype=np.uint64)[0]))

        # Precompute grains: for each waveform, generate a grain of length grain_length
        self.grains: List[np.ndarray] = []
        for wave in self.wave_table:
            grain = self._generate_grain(wave)
            self.grains.append(grain)

        # Shuffle grains deterministically
        self._indices = list(range(len(self.grains)))
        self._rng.shuffle(self._indices)
        self._index = 0

    def _generate_grain(self, waveform: np.ndarray) -> np.ndarray:
        """Generate a single grain by truncating or looping the waveform and applying an envelope."""
        # Repeat waveform to fill grain_length if needed
        repeats = int(np.ceil(self.grain_length / len(waveform)))
        tiled = np.tile(waveform, repeats)[:self.grain_length]
        # Apply a smooth attack-decay envelope (raised cosine)
        envelope = np.ones(self.grain_length)
        attack_len = min(int(self.sample_rate * 0.01), self.grain_length // 2)
        decay_len = min(int(self.sample_rate * 0.03), self.grain_length // 2)
        if attack_len > 0:
            envelope[:attack_len] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, attack_len)))
        if decay_len > 0:
            envelope[-decay_len:] = 0.5 * (1 + np.cos(np.linspace(0, np.pi, decay_len)))
        return tiled * envelope

    def next_grain(self) -> np.ndarray:
        """Return the next grain in deterministic sequence. Cycles indefinitely."""
        idx = self._indices[self._index % len(self._indices)]
        self._index += 1
        return self.grains[idx].copy()

    def reset(self) -> None:
        """Reset the grain sequence to the beginning (deterministic replay)."""
        self._index = 0

    def interpolate_grain(self, t: float) -> np.ndarray:
        """
        Return a grain that is a linear interpolation between two adjacent grains
        based on a continuous parameter t in [0, 1). This allows smooth transitions.
        """
        if not self.grains:
            raise RuntimeError("No grains available")
        # Map t to a floating index
        max_idx = len(self.grains) - 1
        float_idx = t * len(self.grains)
        idx0 = int(float_idx) % len(self.grains)
        idx1 = (idx0 + 1) % len(self.grains)
        frac = float_idx - int(float_idx)
        grain0 = self.grains[idx0]
        grain1 = self.grains[idx1]
        return (1.0 - frac) * grain0 + frac * grain1
