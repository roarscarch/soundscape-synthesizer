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
        """
        self.name = name
        self.base_frequencies = base_frequencies
        self.harmonics = np.array(harmonics, dtype=np.float32)
        self.envelope_attack = envelope_attack
        self.envelope_decay = envelope_decay
        self.grain_duration = grain_duration
        self.sample_rate = sample_rate
        self.seed = seed

        self._wave_table = None

    def _generate_wave_table(self) -> np.ndarray:
        """Generate the wave table as a 2D array (num_grains x grain_length).
        Each grain is a harmonic waveform shaped by the envelope.
        """
        rng = np.random.default_rng(
            int(sha256((self.seed or self.name).encode()).hexdigest(), 16)
        )

        grain_samples = int(self.sample_rate * self.grain_duration)
        t = np.arange(grain_samples) / self.sample_rate

        # Envelope: linear attack and decay
        attack_samples = int(self.sample_rate * self.envelope_attack)
        decay_samples = int(self.sample_rate * self.envelope_decay)
        envelope = np.ones(grain_samples)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0.0, 1.0, attack_samples)
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1.0, 0.0, decay_samples)

        table = []
        for freq in self.base_frequencies:
            # Build harmonic waveform
            wave = np.zeros(grain_samples, dtype=np.float32)
            for i, amp in enumerate(self.harmonics):
                harmonic_freq = freq * (i + 1)
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
            # Normalize to avoid clipping
            max_abs = np.max(np.abs(wave))
            if max_abs > 0:
                wave /= max_abs
            wave *= envelope
            table.append(wave)

        return np.array(table, dtype=np.float32)

    @property
    def wave_table(self) -> np.ndarray:
        """Lazy-computed wave table."""
        if self._wave_table is None:
            self._wave_table = self._generate_wave_table()
        return self._wave_table

    def get_grain(self, index: int) -> np.ndarray:
        """Return a single grain waveform by index (modulo wave table size)."""
        return self.wave_table[index % len(self.wave_table)]

    def __repr__(self) -> str:
        return f"Biome(name={self.name!r}, grains={len(self.base_frequencies)})"


# --- Preset biome instances ---

FOREST_BIOME = Biome(
    name="forest",
    base_frequencies=[110.0, 146.83, 196.0, 246.94],  # A2, D3, G3, B3
    harmonics=[1.0, 0.4, 0.2, 0.1, 0.05],
    envelope_attack=0.05,
    envelope_decay=0.3,
    grain_duration=0.4,
    sample_rate=44100,
    seed="forest_preset",
)

OCEAN_BIOME = Biome(
    name="ocean",
    base_frequencies=[55.0, 110.0, 220.0],  # A1, A2, A3
    harmonics=[1.0, 0.3, 0.15, 0.08],
    envelope_attack=0.1,
    envelope_decay=0.8,
    grain_duration=0.8,
    sample_rate=44100,
    seed="ocean_preset",
)

SPACE_BIOME = Biome(
    name="space",
    base_frequencies=[65.41, 98.0, 130.81, 196.0],  # C2, G2, C3, G3
    harmonics=[1.0, 0.6, 0.3, 0.15, 0.07],
    envelope_attack=0.2,
    envelope_decay=1.5,
    grain_duration=1.0,
    sample_rate=44100,
    seed="space_preset",
)

# Registry for easy lookup
BIOME_REGISTRY: Dict[str, Biome] = {
    "forest": FOREST_BIOME,
    "ocean": OCEAN_BIOME,
    "space": SPACE_BIOME,
}
