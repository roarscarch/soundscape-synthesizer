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
        frequency_jitter: float = 0.0,
        grain_density: float = 1.0,
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
        :param frequency_jitter: maximum random frequency offset as a fraction of base frequency (0.0 = none)
        :param grain_density: multiplier for grain spawn rate (1.0 = normal, higher = denser)
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
        self.frequency_jitter = frequency_jitter
        self.grain_density = grain_density

        # Precompute wave table from harmonics
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> np.ndarray:
        """Build a wave table from base frequencies and harmonics."""
        # Use the lowest base frequency as the fundamental
        fundamental = min(self.base_frequencies)
        # Create a table of one period of the fundamental
        table_size = int(self.sample_rate / fundamental)
        if table_size < 1:
            table_size = 1
        t = np.arange(table_size) / self.sample_rate
        table = np.zeros_like(t, dtype=np.float64)
        for i, amp in enumerate(self.harmonics, start=1):
            table += amp * np.sin(2 * np.pi * fundamental * i * t)
        # Normalize to prevent clipping
        max_abs = np.max(np.abs(table))
        if max_abs > 0:
            table = table / max_abs
        return table

    def get_frequency(self, rng: np.random.Generator) -> float:
        """Get a base frequency, optionally with jitter."""
        freq = float(rng.choice(self.base_frequencies))
        if self.frequency_jitter > 0:
            jitter = rng.uniform(-1, 1) * self.frequency_jitter * freq
            freq += jitter
        return freq

    def __repr__(self):
        return f"Biome(name={self.name!r})"


# Registry of available biomes
BIOME_REGISTRY: Dict[str, Biome] = {}


def register_biome(biome: Biome) -> None:
    """Register a biome by its name."""
    BIOME_REGISTRY[biome.name] = biome


def _make_biome(
    name: str,
    base_frequencies: List[float],
    harmonics: List[float],
    envelope_attack: float,
    envelope_decay: float,
    grain_duration: float,
    **kwargs
) -> Biome:
    """Helper to create and register a biome."""
    biome = Biome(
        name=name,
        base_frequencies=base_frequencies,
        harmonics=harmonics,
        envelope_attack=envelope_attack,
        envelope_decay=envelope_decay,
        grain_duration=grain_duration,
        **kwargs
    )
    register_biome(biome)
    return biome


# Define and register default biomes
_make_biome(
    name="forest",
    base_frequencies=[110.0, 165.0, 220.0],
    harmonics=[1.0, 0.5, 0.25],
    envelope_attack=0.2,
    envelope_decay=0.5,
    grain_duration=0.8,
    seed="forest",
    pan_spread=0.6,
    volume_lfo_rate=0.08,
    pan_lfo_rate=0.03,
    frequency_jitter=0.05,
    grain_density=1.0,
)

_make_biome(
    name="ocean",
    base_frequencies=[55.0, 82.5, 110.0],
    harmonics=[1.0, 0.6, 0.3],
    envelope_attack=0.8,
    envelope_decay=1.2,
    grain_duration=1.5,
    seed="ocean",
    pan_spread=0.8,
    volume_lfo_rate=0.05,
    pan_lfo_rate=0.02,
    frequency_jitter=0.03,
    grain_density=0.7,
)

_make_biome(
    name="space",
    base_frequencies=[220.0, 330.0, 440.0],
    harmonics=[1.0, 0.2, 0.1],
    envelope_attack=1.0,
    envelope_decay=2.0,
    grain_duration=2.5,
    seed="space",
    pan_spread=1.0,
    volume_lfo_rate=0.03,
    pan_lfo_rate=0.01,
    frequency_jitter=0.02,
    grain_density=0.5,
)

_make_biome(
    name="arctic",
    base_frequencies=[330.0, 440.0, 550.0],
    harmonics=[1.0, 0.4, 0.15],
    envelope_attack=0.5,
    envelope_decay=1.0,
    grain_duration=1.8,
    seed="arctic",
    pan_spread=0.7,
    volume_lfo_rate=0.06,
    pan_lfo_rate=0.04,
    frequency_jitter=0.04,
    grain_density=0.8,
)

_make_biome(
    name="desert",
    base_frequencies=[80.0, 120.0, 160.0],
    harmonics=[1.0, 0.3, 0.1],
    envelope_attack=0.4,
    envelope_decay=0.8,
    grain_duration=1.2,
    seed="desert",
    pan_spread=0.5,
    volume_lfo_rate=0.1,
    pan_lfo_rate=0.05,
    frequency_jitter=0.06,
    grain_density=1.2,
)
