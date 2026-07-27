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
        self.wave_table = self._build_wave_table()

    def _build_wave_table(self) -> np.ndarray:
        """
        Build a wave table as a 2D numpy array of shape (num_frequencies, num_samples_per_grain).
        Each row is a waveform for a given base frequency, composed of harmonics with an envelope.
        """
        num_freqs = len(self.base_frequencies)
        num_samples = int(self.sample_rate * self.grain_duration)
        if num_samples == 0:
            num_samples = 1
        table = np.zeros((num_freqs, num_samples), dtype=np.float32)

        # Deterministic perturbation from seed
        seed_bytes = self.seed.encode('utf-8')
        hash_digest = sha256(seed_bytes).hexdigest()
        rng = np.random.default_rng(int(hash_digest[:8], 16))

        for i, freq in enumerate(self.base_frequencies):
            t = np.arange(num_samples, dtype=np.float32) / self.sample_rate
            # Build waveform from harmonics with small random phase offsets
            waveform = np.zeros(num_samples, dtype=np.float32)
            for j, amp in enumerate(self.harmonics):
                harmonic_freq = freq * (j + 1)
                phase_offset = rng.uniform(0, 2 * np.pi) if j > 0 else 0.0
                # Avoid frequencies above Nyquist
                if harmonic_freq > self.sample_rate / 2:
                    break
                waveform += amp * np.sin(2 * np.pi * harmonic_freq * t + phase_offset)

            # Normalize to prevent clipping
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform /= max_val

            # Apply envelope (attack-decay)
            attack_samples = int(self.envelope_attack * self.sample_rate)
            decay_samples = int(self.envelope_decay * self.sample_rate)
            envelope = np.ones(num_samples, dtype=np.float32)

            if attack_samples > 0:
                attack_curve = np.linspace(0.0, 1.0, min(attack_samples, num_samples))
                envelope[:len(attack_curve)] = attack_curve

            if decay_samples > 0:
                decay_start = max(0, num_samples - decay_samples)
                decay_curve = np.linspace(1.0, 0.0, min(decay_samples, num_samples - decay_start))
                envelope[decay_start:decay_start + len(decay_curve)] = decay_curve

            waveform *= envelope
            table[i, :] = waveform.astype(np.float32)

        return table

    def get_grain(self, freq_index: int, amplitude: float) -> np.ndarray:
        """
        Retrieve a grain waveform from the wave table at a given frequency index,
        scaled by amplitude.
        """
        if freq_index < 0 or freq_index >= self.wave_table.shape[0]:
            raise ValueError(f"freq_index {freq_index} out of range [0, {self.wave_table.shape[0] - 1}]")
        grain = self.wave_table[freq_index].copy()
        grain *= np.clip(amplitude, 0.0, 1.0)
        return grain

    def __repr__(self) -> str:
        return f"Biome(name='{self.name}', freqs={len(self.base_frequencies)}, harmonics={len(self.harmonics)})"


# Predefined biome presets
BIOME_PRESETS: Dict[str, Dict] = {
    "forest": {
        "base_frequencies": [55, 65, 77, 110, 130, 165, 220, 260, 330],  # low drone + gentle mids
        "harmonics": [1.0, 0.3, 0.15, 0.05],
        "envelope_attack": 0.1,
        "envelope_decay": 0.3,
        "grain_duration": 0.5,
    },
    "ocean": {
        "base_frequencies": [40, 50, 60, 80, 100, 120, 160, 200],  # deep, slow waves
        "harmonics": [1.0, 0.2, 0.1, 0.02],
        "envelope_attack": 0.2,
        "envelope_decay": 0.6,
        "grain_duration": 1.0,
    },
    "space": {
        "base_frequencies": [60, 80, 100, 150, 200, 300, 400, 600],  # ethereal, wide
        "harmonics": [1.0, 0.5, 0.3, 0.1],
        "envelope_attack": 0.05,
        "envelope_decay": 0.8,
        "grain_duration": 0.8,
    },
}


def load_biome(name: str, sample_rate: int = 44100, seed: Optional[str] = None) -> Biome:
    """
    Load a biome by name from the preset list.
    Raises ValueError if biome name is not found.
    """
    if name not in BIOME_PRESETS:
        raise ValueError(f"Unknown biome '{name}'. Available: {list(BIOME_PRESETS.keys())}")
    params = BIOME_PRESETS[name].copy()
    return Biome(
        name=name,
        sample_rate=sample_rate,
        seed=seed,
        **params
    )


def list_biomes() -> List[str]:
    """Return list of available biome names."""
    return list(BIOME_PRESETS.keys())
