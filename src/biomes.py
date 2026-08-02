"""
Biome definitions: each biome provides a wave table (set of base waveforms)
and parameters that shape the soundscape.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from hashlib import sha256


class Biome:
    """A biome defines a wave table and parameters for soundscape generation."""

    def __init__(
        self,
        name: str,
        base_frequencies: List[float],
        harmonics: List[float],
        envelope_attack: float,
        envelope_decay: float,
        grain_duration: float,
        sample_rate: int = 44100,
        seed: Optional[str] = None,
        pan_spread: float = 0.5,
        volume_lfo_rate: float = 0.1,
        pan_lfo_rate: float = 0.05,
    ):
        """
        :param name: human-readable biome name
        :param base_frequencies: list of fundamental frequencies (Hz) for wave table
        :param harmonics: amplitude multipliers for harmonic series (e.g., [1.0, 0.5, 0.25])
        :param envelope_attack: attack time in seconds
        :param envelope_decay: decay time in seconds
        :param grain_duration: duration of each grain in seconds
        :param sample_rate: sample rate in Hz
        :param seed: optional seed string for deterministic wave table variation
        :param pan_spread: stereo spread factor (0.0 = mono, 1.0 = full stereo)
        :param volume_lfo_rate: rate of volume LFO in Hz
        :param pan_lfo_rate: rate of pan LFO in Hz
        """
        self.name = name
        self.base_frequencies = base_frequencies
        self.harmonics = harmonics
        self.envelope_attack = envelope_attack
        self.envelope_decay = envelope_decay
        self.grain_duration = grain_duration
        self.sample_rate = sample_rate
        self.seed = seed
        self.pan_spread = pan_spread
        self.volume_lfo_rate = volume_lfo_rate
        self.pan_lfo_rate = pan_lfo_rate

        # Precompute the wave table for this biome
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> np.ndarray:
        """
        Build a wave table by summing harmonic series for each base frequency.
        The result is normalized and shaped to (num_frequencies, num_samples).
        """
        # Use a deterministic PRNG based on the seed string if provided
        rng = np.random.default_rng(
            int(sha256((self.seed or self.name).encode()).hexdigest(), 16) % (2**32)
        )

        num_samples = int(self.sample_rate * self.grain_duration)
        table = np.zeros((len(self.base_frequencies), num_samples), dtype=np.float32)

        for i, freq in enumerate(self.base_frequencies):
            t = np.arange(num_samples) / self.sample_rate
            wave = np.zeros(num_samples, dtype=np.float32)
            for harmonic_idx, amp in enumerate(self.harmonics):
                # Add slight detune per harmonic for organic texture
                detune = rng.uniform(-0.5, 0.5) * (harmonic_idx + 1) * 0.1
                wave += amp * np.sin(2 * np.pi * (freq * (harmonic_idx + 1) + detune) * t)
            # Normalize to avoid clipping
            peak = np.max(np.abs(wave))
            if peak > 0:
                wave /= peak
            table[i] = wave

        return table

    def get_grain(self, frequency_index: int, phase: float = 0.0) -> np.ndarray:
        """
        Return a grain of audio based on the frequency index and phase offset.
        The grain is windowed with a simple attack-decay envelope.
        """
        wave = self.wave_table[frequency_index % len(self.base_frequencies)]
        # Apply phase offset (in samples)
        offset = int(phase * len(wave)) % len(wave)
        grain = np.roll(wave, offset)

        # Apply envelope
        env = np.ones_like(grain)
        attack_samples = int(self.envelope_attack * self.sample_rate)
        decay_samples = int(self.envelope_decay * self.sample_rate)
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
        if decay_samples > 0:
            env[-decay_samples:] = np.linspace(1.0, 0.0, decay_samples, dtype=np.float32)
        return grain * env

    def __repr__(self) -> str:
        return f"Biome(name={self.name!r}, freqs={self.base_frequencies})"


# Registry of available biomes
BIOME_REGISTRY: Dict[str, Biome] = {
    "forest": Biome(
        name="forest",
        base_frequencies=[110.0, 165.0, 220.0, 275.0],
        harmonics=[1.0, 0.5, 0.3, 0.15],
        envelope_attack=0.1,
        envelope_decay=0.4,
        grain_duration=2.0,
        seed="forest-seed",
        pan_spread=0.7,
        volume_lfo_rate=0.08,
        pan_lfo_rate=0.04,
    ),
    "ocean": Biome(
        name="ocean",
        base_frequencies=[55.0, 82.5, 110.0, 165.0],
        harmonics=[1.0, 0.4, 0.2, 0.1],
        envelope_attack=0.05,
        envelope_decay=0.8,
        grain_duration=3.0,
        seed="ocean-seed",
        pan_spread=0.9,
        volume_lfo_rate=0.12,
        pan_lfo_rate=0.06,
    ),
    "space": Biome(
        name="space",
        base_frequencies=[220.0, 330.0, 440.0, 550.0],
        harmonics=[1.0, 0.8, 0.5, 0.3],
        envelope_attack=0.2,
        envelope_decay=1.0,
        grain_duration=4.0,
        seed="space-seed",
        pan_spread=0.5,
        volume_lfo_rate=0.05,
        pan_lfo_rate=0.03,
    ),
}
