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
        self.harmonics = np.array(harmonics, dtype=np.float32)
        self.envelope_attack = envelope_attack
        self.envelope_decay = envelope_decay
        self.grain_duration = grain_duration
        self.sample_rate = sample_rate
        self.pan_spread = pan_spread
        self.volume_lfo_rate = volume_lfo_rate
        self.pan_lfo_rate = pan_lfo_rate

        # Deterministic seed for wave table variation
        if seed is not None:
            digest = sha256(seed.encode()).hexdigest()
            self._seed_int = int(digest[:8], 16)
        else:
            self._seed_int = 0

        # Generate wave table (list of waveforms) on init
        self.wave_table: List[np.ndarray] = self._build_wave_table()

    def _build_wave_table(self) -> List[np.ndarray]:
        """
        Build a set of waveforms based on base frequencies and harmonics.
        Each waveform is a single period normalized to [-1, 1].
        """
        rng = np.random.default_rng(self._seed_int)
        table: List[np.ndarray] = []
        for freq in self.base_frequencies:
            # Number of samples for one period at this frequency
            period_samples = max(2, int(self.sample_rate / freq))
            t = np.linspace(0.0, 1.0, period_samples, endpoint=False)
            # Start with sine wave
            wave = np.sin(2.0 * np.pi * t)
            # Add harmonics with slight random detuning for organic feel
            for idx, amp in enumerate(self.harmonics):
                if idx == 0:
                    continue  # fundamental already added
                harmonic_freq = (idx + 1.0) * 1.0
                # Small detune factor (max 2%)
                detune = 1.0 + (rng.random() - 0.5) * 0.04
                phase_shift = rng.random() * 2.0 * np.pi
                harmonic = np.sin(2.0 * np.pi * harmonic_freq * detune * t + phase_shift)
                wave += amp * harmonic
            # Normalize to [-1, 1]
            max_abs = np.max(np.abs(wave))
            if max_abs > 0:
                wave = wave / max_abs
            table.append(wave.astype(np.float32))
        return table

    def get_grain_envelope(self) -> np.ndarray:
        """
        Return an amplitude envelope for a grain of duration grain_duration.
        Uses linear attack and decay.
        """
        total_samples = int(self.grain_duration * self.sample_rate)
        attack_samples = int(self.envelope_attack * self.sample_rate)
        decay_samples = int(self.envelope_decay * self.sample_rate)
        sustain_samples = total_samples - attack_samples - decay_samples
        if sustain_samples < 0:
            sustain_samples = 0
            attack_samples = total_samples // 2
            decay_samples = total_samples - attack_samples

        attack_ramp = np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
        sustain = np.ones(sustain_samples, dtype=np.float32)
        decay_ramp = np.linspace(1.0, 0.0, decay_samples, dtype=np.float32)
        envelope = np.concatenate([attack_ramp, sustain, decay_ramp])
        # Ensure length matches total_samples (due to rounding)
        if len(envelope) < total_samples:
            envelope = np.pad(envelope, (0, total_samples - len(envelope)), mode='constant', constant_values=0.0)
        elif len(envelope) > total_samples:
            envelope = envelope[:total_samples]
        return envelope

    def get_parameters(self) -> dict:
        """Return biome parameters as a dictionary for easy access."""
        return {
            'name': self.name,
            'base_frequencies': self.base_frequencies,
            'harmonics': self.harmonics.tolist(),
            'envelope_attack': self.envelope_attack,
            'envelope_decay': self.envelope_decay,
            'grain_duration': self.grain_duration,
            'sample_rate': self.sample_rate,
            'pan_spread': self.pan_spread,
            'volume_lfo_rate': self.volume_lfo_rate,
            'pan_lfo_rate': self.pan_lfo_rate,
            'wave_table_lengths': [len(w) for w in self.wave_table],
        }