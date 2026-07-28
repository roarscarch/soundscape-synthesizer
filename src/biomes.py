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
        self.seed = seed or name

        # Precompute wave table as a list of numpy arrays
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> List[np.ndarray]:
        """
        Build the wave table: for each base frequency, generate a waveform
        by summing harmonics with the given amplitudes, then apply an envelope.
        Returns a list of 1D numpy arrays (float32).
        """
        table = []
        num_harmonics = len(self.harmonics)
        for freq in self.base_frequencies:
            # Generate one period of the waveform
            period = int(self.sample_rate / freq)
            t = np.linspace(0, 2 * np.pi, period, endpoint=False, dtype=np.float32)
            waveform = np.zeros(period, dtype=np.float32)
            for i, amp in enumerate(self.harmonics):
                harmonic_freq = freq * (i + 1)
                waveform += amp * np.sin(harmonic_freq * t / freq)  # t already scaled to 2pi per fundamental period
            # Normalize to prevent clipping
            max_amp = np.max(np.abs(waveform))
            if max_amp > 0:
                waveform /= max_amp
            table.append(waveform)
        return table

    def get_grain_waveform(self, grain_index: int, duration_seconds: float) -> np.ndarray:
        """
        Return a waveform for a given grain index, tiling the wave table as needed.
        The grain_index determines which base waveform to use (modulo table size).
        Duration is in seconds; the returned array length = sample_rate * duration.
        """
        num_bases = len(self.wave_table)
        if num_bases == 0:
            return np.zeros(int(self.sample_rate * duration_seconds), dtype=np.float32)
        base_idx = grain_index % num_bases
        base_wave = self.wave_table[base_idx]
        # Tile to desired duration
        num_samples = int(self.sample_rate * duration_seconds)
        repeats = (num_samples // len(base_wave)) + 1
        tiled = np.tile(base_wave, repeats)[:num_samples]
        # Apply envelope
        envelope = self._envelope(num_samples)
        return (tiled * envelope).astype(np.float32)

    def _envelope(self, num_samples: int) -> np.ndarray:
        """
        Generate an amplitude envelope: linear attack, then exponential decay.
        """
        attack_samples = int(self.envelope_attack * self.sample_rate)
        decay_samples = int(self.envelope_decay * self.sample_rate)
        if attack_samples + decay_samples > num_samples:
            # Short grain: scale envelope to fit
            attack_samples = num_samples // 2
            decay_samples = num_samples - attack_samples
        env = np.ones(num_samples, dtype=np.float32)
        if attack_samples > 0:
            env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
        if decay_samples > 0:
            decay_curve = np.exp(-np.linspace(0, 4, decay_samples, dtype=np.float32))
            env[-decay_samples:] = decay_curve
        return env

    def __repr__(self) -> str:
        return f"Biome('{self.name}', {len(self.base_frequencies)} frequencies, {len(self.harmonics)} harmonics)"


# ---------------------------------------------------------------------------
# Preset biome instances
# ---------------------------------------------------------------------------

FOREST_BIOME = Biome(
    name="forest",
    base_frequencies=[80, 120, 180, 240, 360],
    harmonics=[1.0, 0.6, 0.3, 0.15, 0.05],
    envelope_attack=0.1,
    envelope_decay=0.8,
    grain_duration=1.0,
    sample_rate=44100,
    seed="forest_preset",
)

OCEAN_BIOME = Biome(
    name="ocean",
    base_frequencies=[40, 60, 90, 130, 200],
    harmonics=[1.0, 0.4, 0.2, 0.1, 0.02],
    envelope_attack=0.05,
    envelope_decay=1.5,
    grain_duration=2.0,
    sample_rate=44100,
    seed="ocean_preset",
)

SPACE_BIOME = Biome(
    name="space",
    base_frequencies=[30, 50, 70, 100, 150],
    harmonics=[1.0, 0.8, 0.6, 0.4, 0.2],
    envelope_attack=0.2,
    envelope_decay=2.0,
    grain_duration=3.0,
    sample_rate=44100,
    seed="space_preset",
)

# Registry for easy lookup by name
BIOME_REGISTRY: Dict[str, Biome] = {
    "forest": FOREST_BIOME,
    "ocean": OCEAN_BIOME,
    "space": SPACE_BIOME,
}
