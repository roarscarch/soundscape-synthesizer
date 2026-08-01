import numpy as np
from typing import List, Tuple, Optional


class GrainBank:
    """
    A bank of audio grains used by the soundscape engine.
    Each grain is a short waveform segment with an envelope.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        seed: Optional[str] = None,
    ):
        """
        :param sample_rate: sample rate in Hz
        :param seed: optional seed string for deterministic grain generation
        """
        self.sample_rate = sample_rate
        self.seed = seed or "default"
        self.grains: List[np.ndarray] = []
        self._rng = np.random.default_rng(abs(hash(self.seed)) % (2**32))

    def generate_grains(
        self,
        base_frequencies: List[float],
        harmonics: List[float],
        envelope_attack: float,
        envelope_decay: float,
        grain_duration: float,
        count: int = 16,
    ) -> None:
        """
        Generate a set of grains based on biome parameters.

        :param base_frequencies: list of fundamental frequencies (Hz)
        :param harmonics: amplitude multipliers for harmonic series
        :param envelope_attack: attack time in seconds
        :param envelope_decay: decay time in seconds
        :param grain_duration: duration of each grain in seconds
        :param count: number of grains to generate
        """
        self.grains.clear()
        num_samples = int(grain_duration * self.sample_rate)
        if num_samples < 1:
            num_samples = 1
        attack_samples = max(1, int(envelope_attack * self.sample_rate))
        decay_samples = max(1, int(envelope_decay * self.sample_rate))

        # Build a base envelope: linear attack then exponential decay
        env = np.ones(num_samples, dtype=np.float32)
        if attack_samples >= num_samples:
            env *= np.linspace(0.0, 1.0, num_samples, dtype=np.float32)
        else:
            env[:attack_samples] = np.linspace(
                0.0, 1.0, attack_samples, dtype=np.float32
            )
            if decay_samples >= num_samples - attack_samples:
                env[attack_samples:] *= np.linspace(
                    1.0, 0.0, num_samples - attack_samples, dtype=np.float32
                )
            else:
                decay_start = attack_samples
                decay_end = attack_samples + decay_samples
                env[decay_start:decay_end] *= np.linspace(
                    1.0, 0.0, decay_samples, dtype=np.float32
                )
                env[decay_end:] = 0.0

        # Generate each grain as a sum of harmonics with slight detune
        for _ in range(count):
            grain = np.zeros(num_samples, dtype=np.float32)
            for idx, amp in enumerate(harmonics):
                if amp <= 0:
                    continue
                freq = base_frequencies[idx % len(base_frequencies)] * (idx + 1)
                # Add slight detune for organic feel
                detune = self._rng.uniform(0.98, 1.02)
                phase = self._rng.uniform(0.0, 2 * np.pi)
                t = np.arange(num_samples, dtype=np.float32) / self.sample_rate
                grain += amp * np.sin(2 * np.pi * freq * detune * t + phase)

            # Normalize to avoid clipping
            peak = np.max(np.abs(grain))
            if peak > 0:
                grain /= peak
            grain *= env
            self.grains.append(grain.astype(np.float32))

    def get_grain(self, index: int) -> np.ndarray:
        """
        Retrieve a grain by index (modulo count).

        :param index: grain index
        :return: audio grain as float32 array
        """
        if not self.grains:
            raise ValueError("No grains generated. Call generate_grains first.")
        return self.grains[index % len(self.grains)]

    def mix_grains(
        self,
        indices: List[int],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Mix multiple grains together with optional weights.

        :param indices: list of grain indices
        :param weights: optional list of weights (same length as indices)
        :return: mixed audio as float32 array
        """
        if not self.grains:
            raise ValueError("No grains generated. Call generate_grains first.")
        if weights is None:
            weights = [1.0] * len(indices)
        if len(indices) != len(weights):
            raise ValueError("indices and weights must have the same length")

        # Use the longest grain as the buffer size
        max_len = max(len(self.grains[i % len(self.grains)]) for i in indices)
        mix = np.zeros(max_len, dtype=np.float32)
        for idx, w in zip(indices, weights):
            grain = self.get_grain(idx)
            mix[: len(grain)] += w * grain

        # Normalize to prevent clipping
        peak = np.max(np.abs(mix))
        if peak > 0:
            mix /= peak
        return mix

    def apply_lfo(
        self,
        audio: np.ndarray,
        lfo_volume: np.ndarray,
        lfo_pan: np.ndarray,
        pan_spread: float = 0.5,
    ) -> np.ndarray:
        """
        Apply volume and pan LFOs to a mono audio signal, producing stereo output.

        :param audio: mono audio as float32 array
        :param lfo_volume: volume modulation values (0.0 to 1.0), same length as audio
        :param lfo_pan: pan modulation values (-1.0 to 1.0), same length as audio
        :param pan_spread: stereo spread factor (0.0 = mono, 1.0 = full stereo)
        :return: stereo audio as float32 array of shape (n_samples, 2)
        """
        n = len(audio)
        if len(lfo_volume) != n or len(lfo_pan) != n:
            raise ValueError("LFO arrays must have the same length as audio")

        # Ensure volume stays within safe range
        vol = np.clip(lfo_volume, 0.0, 1.0)
        pan = np.clip(lfo_pan, -1.0, 1.0) * pan_spread

        # Equal-power panning
        left_gain = np.cos((pan + 1.0) * np.pi / 4.0)
        right_gain = np.sin((pan + 1.0) * np.pi / 4.0)

        left = audio * vol * left_gain
        right = audio * vol * right_gain
        return np.stack([left, right], axis=1).astype(np.float32)
