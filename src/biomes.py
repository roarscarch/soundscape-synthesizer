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
        self.seed = seed or name
        self.pan_spread = pan_spread
        self.volume_lfo_rate = volume_lfo_rate
        self.pan_lfo_rate = pan_lfo_rate

        self._wave_table: Optional[np.ndarray] = None
        self._envelope: Optional[np.ndarray] = None

    def _generate_wave_table(self) -> np.ndarray:
        """
        Generate a deterministic wave table from the biome parameters and seed.
        Returns a 2D numpy array of shape (num_frequencies, num_samples).
        """
        num_freqs = len(self.base_frequencies)
        num_samples = int(self.grain_duration * self.sample_rate)
        wave_table = np.zeros((num_freqs, num_samples), dtype=np.float64)

        # Deterministic seed variation
        seed_bytes = self.seed.encode('utf-8')
        hash_digest = sha256(seed_bytes).hexdigest()
        seed_int = int(hash_digest[:8], 16)
        rng = np.random.default_rng(seed_int)

        for i, freq in enumerate(self.base_frequencies):
            t = np.arange(num_samples) / self.sample_rate
            wave = np.zeros(num_samples, dtype=np.float64)
            for j, amp in enumerate(self.harmonics):
                harmonic_freq = freq * (j + 1)
                phase = rng.uniform(0, 2 * np.pi)  # random phase for each harmonic
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t + phase)
            # Normalize to avoid clipping
            max_amp = np.max(np.abs(wave))
            if max_amp > 0:
                wave /= max_amp
            wave_table[i, :] = wave

        self._wave_table = wave_table
        return wave_table

    def _generate_envelope(self) -> np.ndarray:
        """
        Generate an amplitude envelope for the grain.
        Returns a 1D numpy array of shape (num_samples,).
        """
        num_samples = int(self.grain_duration * self.sample_rate)
        attack_samples = int(self.envelope_attack * self.sample_rate)
        decay_samples = int(self.envelope_decay * self.sample_rate)

        envelope = np.ones(num_samples, dtype=np.float64)

        # Attack: linear ramp up
        if attack_samples > 0:
            attack_ramp = np.linspace(0.0, 1.0, attack_samples)
            envelope[:attack_samples] = attack_ramp

        # Decay: linear ramp down
        if decay_samples > 0:
            decay_start = num_samples - decay_samples
            if decay_start < 0:
                decay_start = 0
            decay_ramp = np.linspace(1.0, 0.0, num_samples - decay_start)
            envelope[decay_start:] = decay_ramp

        self._envelope = envelope
        return envelope

    def get_wave_table(self) -> np.ndarray:
        """Return the wave table, generating it if necessary."""
        if self._wave_table is None:
            self._generate_wave_table()
        return self._wave_table

    def get_envelope(self) -> np.ndarray:
        """Return the amplitude envelope, generating it if necessary."""
        if self._envelope is None:
            self._generate_envelope()
        return self._envelope

    def get_grain(self, freq_index: int) -> np.ndarray:
        """
        Return a single grain (waveform * envelope) for the given frequency index.
        """
        wave_table = self.get_wave_table()
        envelope = self.get_envelope()
        return wave_table[freq_index] * envelope

    def __repr__(self) -> str:
        return f"Biome(name='{self.name}', freqs={len(self.base_frequencies)}, harmonics={len(self.harmonics)})"


# Predefined biome presets
BIOME_REGISTRY: Dict[str, Biome] = {}