"""
Biome definitions: each biome provides a wave table (set of base waveforms)
and parameters that shape the soundscape.
"""

import numpy as np
from typing import List, Dict, Tuple


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
    ):
        """
        :param name: human-readable biome name
        :param base_frequencies: list of fundamental frequencies (Hz) for wave table
        :param harmonics: amplitude multipliers for harmonic series (e.g., [1.0, 0.5, 0.25])
        :param envelope_attack: attack time in seconds
        :param envelope_decay: decay time in seconds
        :param grain_duration: duration of each grain in seconds
        :param sample_rate: sample rate in Hz
        """
        self.name = name
        self.base_frequencies = base_frequencies
        self.harmonics = np.array(harmonics, dtype=np.float32)
        self.envelope_attack = envelope_attack
        self.envelope_decay = envelope_decay
        self.grain_duration = grain_duration
        self.sample_rate = sample_rate
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> np.ndarray:
        """
        Build a 2D wave table: shape (num_frequencies, grain_length)
        Each row is a grain waveform (float32).
        """
        grain_length = int(self.grain_duration * self.sample_rate)
        t = np.linspace(0, self.grain_duration, grain_length, endpoint=False, dtype=np.float32)
        table = []
        for freq in self.base_frequencies:
            # Start with fundamental
            wave = np.sin(2 * np.pi * freq * t)
            # Add harmonics
            for idx, amp in enumerate(self.harmonics[1:], start=2):
                wave += amp * np.sin(2 * np.pi * freq * idx * t)
            # Normalize to prevent clipping
            max_abs = np.max(np.abs(wave))
            if max_abs > 0:
                wave /= max_abs
            # Apply amplitude envelope
            attack_samples = int(self.envelope_attack * self.sample_rate)
            decay_samples = int(self.envelope_decay * self.sample_rate)
            envelope = np.ones(grain_length, dtype=np.float32)
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)
            if decay_samples > 0:
                envelope[-decay_samples:] = np.linspace(1, 0, decay_samples, dtype=np.float32)
            wave *= envelope
            table.append(wave)
        return np.array(table, dtype=np.float32)

    def get_grain(self, freq_index: int, amplitude: float) -> 'Grain':
        """Return a Grain object from the wave table at the given frequency index."""
        from engine import Grain
        waveform = self.wave_table[freq_index % len(self.wave_table)].copy()
        # Apply amplitude scaling
        waveform *= amplitude
        return Grain(waveform, freq_index % len(self.wave_table), amplitude)

    def __repr__(self) -> str:
        return f"Biome(name='{self.name}', freqs={len(self.base_frequencies)}, harmonic_count={len(self.harmonics)})"


# Predefined biomes
FOREST_BIOME = Biome(
    name="forest",
    base_frequencies=[60, 80, 100, 120, 150, 180, 220],  # Low rumble, wind, leaves
    harmonics=[1.0, 0.3, 0.1, 0.05],
    envelope_attack=0.05,
    envelope_decay=0.3,
    grain_duration=0.5,
)

OCEAN_BIOME = Biome(
    name="ocean",
    base_frequencies=[40, 55, 70, 90, 110, 130, 160],  # Deep waves
    harmonics=[1.0, 0.4, 0.2, 0.1, 0.05],
    envelope_attack=0.1,
    envelope_decay=0.6,
    grain_duration=0.8,
)

SPACE_BIOME = Biome(
    name="space",
    base_frequencies=[100, 150, 200, 300, 400, 500, 700],  # Ethereal drones
    harmonics=[1.0, 0.6, 0.3, 0.15, 0.08],
    envelope_attack=0.2,
    envelope_decay=0.8,
    grain_duration=1.0,
)

BIOME_REGISTRY: Dict[str, Biome] = {
    "forest": FOREST_BIOME,
    "ocean": OCEAN_BIOME,
    "space": SPACE_BIOME,
}


def get_biome(name: str) -> Biome:
    """Return biome by name, case-insensitive."""
    key = name.lower()
    if key not in BIOME_REGISTRY:
        raise ValueError(f"Unknown biome '{name}'. Available: {list(BIOME_REGISTRY.keys())}")
    return BIOME_REGISTRY[key]
