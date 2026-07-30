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

        # Build wave table deterministically
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> np.ndarray:
        """
        Build a wave table as a 2D array: shape (num_waves, num_samples).
        Each wave is a harmonic series shaped by envelope.
        Returns a float array with values in [-1, 1].
        """
        num_waves = len(self.base_frequencies)
        num_samples = int(self.sample_rate * self.grain_duration)
        if num_samples < 1:
            num_samples = 1

        table = np.zeros((num_waves, num_samples), dtype=np.float64)

        for i, freq in enumerate(self.base_frequencies):
            t = np.arange(num_samples) / self.sample_rate
            wave = np.zeros_like(t)
            for j, amp in enumerate(self.harmonics):
                harmonic_freq = freq * (j + 1)
                harmonic = amp * np.sin(2 * np.pi * harmonic_freq * t)
                wave += harmonic

            # Apply envelope (attack + decay)
            attack_samples = int(self.envelope_attack * self.sample_rate)
            decay_samples = int(self.envelope_decay * self.sample_rate)
            if attack_samples + decay_samples > num_samples:
                # Scale attack/decay proportionally
                total = attack_samples + decay_samples
                attack_samples = int(attack_samples * num_samples / total)
                decay_samples = num_samples - attack_samples

            envelope = np.ones(num_samples)
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
            if decay_samples > 0:
                start = num_samples - decay_samples
                envelope[start:] = np.linspace(1.0, 0.0, decay_samples)

            wave *= envelope

            # Normalize to [-1, 1]
            peak = np.max(np.abs(wave))
            if peak > 0:
                wave /= peak

            table[i] = wave

        # Apply deterministic seed variation if provided
        if self.seed is not None:
            seed_bytes = self.seed.encode("utf-8")
            hash_digest = sha256(seed_bytes).digest()
            rng = np.random.default_rng(
                int.from_bytes(hash_digest[:8], "little")
            )
            # Add subtle phase offsets to each wave
            for i in range(num_waves):
                offset = rng.uniform(0, 1) * 2 * np.pi
                t = np.arange(num_samples) / self.sample_rate
                phase_shift = np.sin(2 * np.pi * offset * t)
                # Blend original with phase-shifted version (10% mix)
                table[i] = table[i] * 0.9 + phase_shift * 0.1
                # Re-normalize
                peak = np.max(np.abs(table[i]))
                if peak > 0:
                    table[i] /= peak

        return table

    def get_grain(self, index: int) -> np.ndarray:
        """
        Get a grain from the wave table by index (cyclic).
        Returns a 1D numpy array of length num_samples.
        """
        idx = index % self.wave_table.shape[0]
        return self.wave_table[idx].copy()

    @property
    def num_waves(self) -> int:
        return self.wave_table.shape[0]

    @property
    def num_samples(self) -> int:
        return self.wave_table.shape[1]

    def __repr__(self) -> str:
        return f"Biome(name='{self.name}', waves={self.num_waves}, samples={self.num_samples}