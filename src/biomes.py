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

        # Precompute deterministic wave table based on seed
        self._rng = np.random.default_rng(self._seed_to_int())
        self.wave_table = self._build_wave_table()

    def _seed_to_int(self) -> int:
        """Convert seed string to a 32-bit integer for numpy RNG."""
        if self.seed is None:
            return 0
        return int(sha256(self.seed.encode()).hexdigest()[:8], 16)

    def _build_wave_table(self) -> np.ndarray:
        """Build a wave table by summing harmonics with deterministic phase offsets."""
        table_len = 2048
        t = np.linspace(0, 2 * np.pi, table_len, endpoint=False)
        wave = np.zeros(table_len, dtype=np.float32)
        for i, amp in enumerate(self.harmonics):
            if amp == 0:
                continue
            # Deterministic phase from seed
            phase = self._rng.uniform(0, 2 * np.pi)
            wave += amp * np.sin(i * t + phase)
        # Normalize to [-1, 1]
        max_abs = np.max(np.abs(wave))
        if max_abs > 0:
            wave = wave / max_abs
        return wave

    def get_frequency(self, base_freq: float) -> float:
        """Return a frequency with optional jitter applied."""
        if self.frequency_jitter <= 0:
            return base_freq
        offset = self._rng.uniform(-self.frequency_jitter, self.frequency_jitter) * base_freq
        return base_freq + offset

    def to_dict(self) -> Dict:
        """Serialize biome parameters to a dictionary."""
        return {
            "name": self.name,
            "base_frequencies": self.base_frequencies,
            "harmonics": self.harmonics,
            "envelope_attack": self.envelope_attack,
            "envelope_decay": self.envelope_decay,
            "grain_duration": self.grain_duration,
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "pan_spread": self.pan_spread,
            "volume_lfo_rate": self.volume_lfo_rate,
            "pan_lfo_rate": self.pan_lfo_rate,
            "frequency_jitter": self.frequency_jitter,
            "grain_density": self.grain_density,
        }

    def __repr__(self):
        return f"<Biome '{self.name}' (freqs={self.base_frequencies}, jitter={self.frequency_jitter})>"


# Standard biome registry with presets
BIOME_REGISTRY: Dict[str, Biome] = {
    "forest": Biome(
        name="forest",
        base_frequencies=[110.0, 165.0, 220.0, 330.0],
        harmonics=[1.0, 0.5, 0.25, 0.1, 0.05],
        envelope_attack=0.05,
        envelope_decay=0.4,
        grain_duration=0.8,
        seed="forest-default",
        pan_spread=0.7,
        volume_lfo_rate=0.15,
        pan_lfo_rate=0.08,
        frequency_jitter=0.05,
        grain_density=1.2,
    ),
    "ocean": Biome(
        name="ocean",
        base_frequencies=[55.0, 82.5, 110.0, 165.0],
        harmonics=[1.0, 0.3, 0.15, 0.05],
        envelope_attack=0.2,
        envelope_decay=0.8,
        grain_duration=1.5,
        seed="ocean-default",
        pan_spread=0.9,
        volume_lfo_rate=0.1,
        pan_lfo_rate=0.05,
        frequency_jitter=0.02,
        grain_density=0.8,
    ),
    "space": Biome(
        name="space",
        base_frequencies=[220.0, 440.0, 880.0],
        harmonics=[1.0, 0.5, 0.1],
        envelope_attack=0.5,
        envelope_decay=1.5,
        grain_duration=3.0,
        seed="space-default",
        pan_spread=0.5,
        volume_lfo_rate=0.05,
        pan_lfo_rate=0.02,
        frequency_jitter=0.1,
        grain_density=0.6,
    ),
}
