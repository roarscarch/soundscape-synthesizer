"""
Audio player module: handles real-time playback, panning, volume LFOs,
sleep timer, and fade-out.
"""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, Callable


class AudioPlayer:
    """Play audio streams with volume LFO, panning LFO, sleep timer, and fade-out."""

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 1024,
        channels: int = 2,
        dtype: str = 'float32',
    ):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.dtype = dtype

        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._lock = threading.Lock()

        # Volume and panning state
        self.volume = 1.0
        self.pan = 0.0  # -1 left, 0 center, 1 right

        # LFO parameters
        self.volume_lfo_rate: float = 0.0
        self.volume_lfo_depth: float = 0.0
        self.pan_lfo_rate: float = 0.0
        self.pan_lfo_depth: float = 0.0

        # Sleep timer
        self.sleep_duration: float = 0.0  # seconds, 0 = disabled
        self._sleep_start_time: float = 0.0
        self._fade_duration: float = 5.0  # fade-out duration in seconds

        # Callback to get audio data
        self._audio_source: Optional[Callable[[int], np.ndarray]] = None

        # Internal LFO phase accumulators
        self._phase_vol = 0.0
        self._phase_pan = 0.0

    def set_audio_source(self, source: Callable[[int], np.ndarray]) -> None:
        """Set a callable that returns a chunk of audio samples (n_samples,) or (n_samples, channels)."""
        self._audio_source = source

    def start(self) -> None:
        """Start audio playback."""
        if self._running:
            return
        self._running = True
        self._sleep_start_time = time.time()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop audio playback."""
        with self._lock:
            self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_sleep_timer(self, duration_seconds: float, fade_duration: float = 5.0) -> None:
        """Set a sleep timer: after duration_seconds, fade out and stop.

        :param duration_seconds: total playback time in seconds (0 = disabled)
        :param fade_duration: duration of the fade-out in seconds
        """
        self.sleep_duration = duration_seconds
        self._fade_duration = fade_duration
        if duration_seconds > 0:
            self._sleep_start_time = time.time()

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio callback: fill outdata with processed audio."""
        if status:
            print(f"Audio callback status: {status}")

        with self._lock:
            if not self._running or self._audio_source is None:
                outdata.fill(0.0)
                return

            # Get raw audio from source
            raw = self._audio_source(frames)
            if raw.ndim == 1:
                raw = np.column_stack([raw, raw])  # mono to stereo
            elif raw.shape[1] == 1:
                raw = np.column_stack([raw[:, 0], raw[:, 0]])

            # Ensure correct number of frames
            if raw.shape[0] < frames:
                raw = np.pad(raw, ((0, frames - raw.shape[0]), (0, 0)), mode='constant')
            elif raw.shape[0] > frames:
                raw = raw[:frames, :]

            # Apply volume LFO
            if self.volume_lfo_rate > 0 and self.volume_lfo_depth > 0:
                self._phase_vol += self.volume_lfo_rate * frames / self.sample_rate
                lfo_mod = 1.0 + self.volume_lfo_depth * np.sin(2 * np.pi * self._phase_vol)
                raw *= lfo_mod[:, np.newaxis]

            # Apply panning LFO
            if self.pan_lfo_rate > 0 and self.pan_lfo_depth > 0:
                self._phase_pan += self.pan_lfo_rate * frames / self.sample_rate
                pan_mod = self.pan + self.pan_lfo_depth * np.sin(2 * np.pi * self._phase_pan)
                left_gain = np.clip(0.5 * (1.0 - pan_mod), 0, 1)
                right_gain = np.clip(0.5 * (1.0 + pan_mod), 0, 1)
                raw[:, 0] *= left_gain
                raw[:, 1] *= right_gain
            else:
                # Static pan
                left_gain = np.clip(0.5 * (1.0 - self.pan), 0, 1)
                right_gain = np.clip(0.5 * (1.0 + self.pan), 0, 1)
                raw[:, 0] *= left_gain
                raw[:, 1] *= right_gain

            # Apply volume
            raw *= self.volume

            # Sleep timer and fade-out
            if self.sleep_duration > 0:
                elapsed = time.time() - self._sleep_start_time
                remaining = self.sleep_duration - elapsed
                if remaining <= 0:
                    # Fade-out finished or timer expired
                    outdata.fill(0.0)
                    self._running = False
                    return
                elif remaining < self._fade_duration:
                    fade_factor = remaining / self._fade_duration
                    raw *= fade_factor

            outdata[:] = raw.astype(self.dtype)

    def close(self) -> None:
        """Cleanly close the audio stream."""
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
