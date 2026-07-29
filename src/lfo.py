"""
Low-Frequency Oscillators for organic modulation of panning and volume.
"""

import numpy as np
from typing import Optional, Callable


class LFO:
    """A low-frequency oscillator that generates modulation signals."""

    def __init__(
        self,
        rate: float = 0.1,
        depth: float = 1.0,
        phase: float = 0.0,
        waveform: str = "sine",
        sample_rate: int = 44100,
    ):
        """
        :param rate: oscillation frequency in Hz
        :param depth: amplitude of modulation (0.0 to 1.0)
        :param phase: initial phase offset in radians
        :param waveform: waveform type: 'sine', 'triangle', 'saw', 'square'
        :param sample_rate: sample rate in Hz
        """
        self.rate = rate
        self.depth = depth
        self.phase = phase
        self.sample_rate = sample_rate
        self._phase_accumulator: float = phase

        self._waveform_funcs: dict = {
            "sine": self._sine,
            "triangle": self._triangle,
            "saw": self._saw,
            "square": self._square,
        }

        if waveform not in self._waveform_funcs:
            raise ValueError(f"Unknown waveform '{waveform}'. Choose from {list(self._waveform_funcs.keys())}")
        self._waveform = waveform

    def _sine(self, phase: float) -> float:
        return np.sin(phase)

    def _triangle(self, phase: float) -> float:
        # Normalize phase to [0, 1) then triangle
        p = phase / (2 * np.pi) % 1.0
        if p < 0.5:
            return 4.0 * p - 1.0
        else:
            return 3.0 - 4.0 * p

    def _saw(self, phase: float) -> float:
        p = phase / (2 * np.pi) % 1.0
        return 2.0 * p - 1.0

    def _square(self, phase: float) -> float:
        p = phase / (2 * np.pi) % 1.0
        return 1.0 if p < 0.5 else -1.0

    def reset(self, phase: Optional[float] = None) -> None:
        """Reset the LFO phase accumulator."""
        if phase is not None:
            self.phase = phase
        self._phase_accumulator = self.phase

    def generate(self, num_samples: int) -> np.ndarray:
        """
        Generate a block of modulation samples.

        :param num_samples: number of samples to generate
        :return: numpy array of shape (num_samples,) with values in [-depth, depth]
        """
        phase_increment = 2.0 * np.pi * self.rate / self.sample_rate
        phases = self._phase_accumulator + phase_increment * np.arange(num_samples, dtype=np.float64)
        self._phase_accumulator = phases[-1] % (2.0 * np.pi)

        # Apply waveform
        raw = np.array([self._waveform_funcs[self._waveform](p) for p in phases], dtype=np.float32)
        return raw * self.depth

    def generate_scaled(self, num_samples: int, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        """
        Generate modulation scaled to [min_val, max_val].

        :param num_samples: number of samples
        :param min_val: minimum output value
        :param max_val: maximum output value
        :return: numpy array of shape (num_samples,) in [min_val, max_val]
        """
        raw = self.generate(num_samples)
        # raw is in [-depth, depth]
        if self.depth == 0.0:
            return np.full(num_samples, (min_val + max_val) / 2.0, dtype=np.float32)
        normalized = (raw / self.depth + 1.0) / 2.0  # [0, 1]
        return normalized * (max_val - min_val) + min_val


class PanLFO(LFO):
    """LFO specialized for stereo panning modulation.
    Output is in [-1.0, 1.0] representing left(-1) to right(+1).
    """

    def __init__(
        self,
        rate: float = 0.05,
        depth: float = 1.0,
        phase: float = 0.0,
        waveform: str = "sine",
        sample_rate: int = 44100,
    ):
        super().__init__(rate, depth, phase, waveform, sample_rate)

    def generate_pan(self, num_samples: int) -> np.ndarray:
        """Generate panning values in [-1, 1]."""
        return np.clip(self.generate(num_samples), -1.0, 1.0)


class VolumeLFO(LFO):
    """LFO specialized for volume modulation.
    Output is in [0.0, 1.0] representing amplitude gain.
    """

    def __init__(
        self,
        rate: float = 0.1,
        depth: float = 0.5,
        phase: float = 0.0,
        waveform: str = "sine",
        sample_rate: int = 44100,
        min_gain: float = 0.2,
    ):
        super().__init__(rate, depth, phase, waveform, sample_rate)
        self.min_gain = min_gain

    def generate_volume(self, num_samples: int) -> np.ndarray:
        """Generate volume gain values in [min_gain, 1.0]."""
        return self.generate_scaled(num_samples, self.min_gain, 1.0)
