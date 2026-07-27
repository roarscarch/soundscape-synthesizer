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

        # Build the wave table (list of waveforms)
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> List[np.ndarray]:
        """
        Generate a wave table from base frequencies and harmonics,
        with deterministic variation from the seed.
        """
        rng = np.random.default_rng(
            int(sha256(self.seed.encode()).hexdigest(), 16) & 0xFFFFFFFF
        )
        wave_table = []
        duration_samples = int(self.sample_rate * self.grain_duration)
        t = np.linspace(0, self.grain_duration, duration_samples, endpoint=False)

        for freq in self.base_frequencies:
            # Build harmonic series
            waveform = np.zeros(duration_samples, dtype=np.float32)
            for i, amp in enumerate(self.harmonics):
                harmonic_freq = freq * (i + 1)
                phase = rng.uniform(0, 2 * np.pi)  # deterministic random phase
                waveform += amp * np.sin(2 * np.pi * harmonic_freq * t + phase)

            # Normalize to [-1, 1]
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform /= max_val

            # Apply envelope
            envelope = self._envelope(duration_samples)
            waveform *= envelope

            wave_table.append(waveform)

        return wave_table

    def _envelope(self, num_samples: int) -> np.ndarray:
        """Create an amplitude envelope (attack-decay) for a grain."""
        attack_samples = int(self.sample_rate * self.envelope_attack)
        decay_samples = int(self.sample_rate * self.envelope_decay)
        sustain_samples = num_samples - attack_samples - decay_samples

        if sustain_samples < 0:
            # Short grain: scale envelope to fit
            total = attack_samples + decay_samples
            attack_ratio = attack_samples / total
            decay_ratio = decay_samples / total
            attack_samples = int(num_samples * attack_ratio)
            decay_samples = num_samples - attack_samples
            sustain_samples = 0

        envelope = np.ones(num_samples, dtype=np.float32)
        # Attack: linear ramp up
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        # Decay: linear ramp down
        if decay_samples > 0:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)
        return envelope

    def get_wave_table(self) -> List[np.ndarray]:
        """Return the wave table (list of waveforms)."""
        return self.wave_table

    def get_params(self) -> dict:
        """Return a dictionary of biome parameters for serialization."""
        return {
            "name": self.name,
            "base_frequencies": self.base_frequencies,
            "harmonics": self.harmonics.tolist(),
            "envelope_attack": self.envelope_attack,
            "envelope_decay": self.envelope_decay,
            "grain_duration": self.grain_duration,
            "sample_rate": self.sample_rate,
            "seed": self.seed,
        }


# Predefined biomes
BIOME_REGISTRY: Dict[str, Biome] = {}


def _register_biome(name: str, biome: Biome) -> None:
    """Register a biome in the global registry."""
    BIOME_REGISTRY[name] = biome


# Forest biome
_register_biome(
    "forest",
    Biome(
        name="forest",
        base_frequencies=[60, 120, 180, 240, 300],
        harmonics=[1.0, 0.3, 0.15, 0.08, 0.04],
        envelope_attack=0.05,
        envelope_decay=0.3,
        grain_duration=0.5,
        sample_rate=44100,
        seed="forest_default",
    ),
)

# Ocean biome
_register_biome(
    "ocean",
    Biome(
        name="ocean",
        base_frequencies=[40, 80, 160, 320],
        harmonics=[1.0, 0.5, 0.25, 0.125],
        envelope_attack=0.1,
        envelope_decay=0.4,
        grain_duration=0.8,
        sample_rate=44100,
        seed="ocean_default",
    ),
)

# Space biome
_register_biome(
    "space",
    Biome(
        name="space",
        base_frequencies=[30, 55, 110, 220, 440],
        harmonics=[1.0, 0.2, 0.1, 0.05, 0.02],
        envelope_attack=0.2,
        envelope_decay=0.6,
        grain_duration=1.0,
        sample_rate=44100,
        seed="space_default",
    ),
)

def get_biome(name: str) -> Optional[Biome]:
    """Get a biome by name from the registry."""
    return BIOME_REGISTRY.get(name)

def list_biomes() -> List[str]:
    """Return list of available biome names."""
    return list(BIOME_REGISTRY.keys())
