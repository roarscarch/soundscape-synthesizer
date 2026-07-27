"""
Real-time audio playback engine using sounddevice.
Manages streaming output, panning, volume LFOs, sleep timer, and fade-out.
"""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Callable, Optional


class AudioPlayer:
    """Streams generated audio grains to the audio output in real-time."""

    def __init__(self, sample_rate: int = 44100, channels: int = 2, blocksize: int = 1024):
        """
        :param sample_rate: sample rate in Hz
        :param channels: number of audio channels (1 mono, 2 stereo)
        :param blocksize: number of frames per buffer
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self._buffer = np.zeros((blocksize, channels), dtype=np.float32)
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._lock = threading.Lock()
        self._volume = 1.0
        self._pan = 0.0  # -1 left, 0 center, 1 right
        self._volume_lfo_freq = 0.1  # Hz
        self._pan_lfo_freq = 0.05
        self._lfo_phase = 0.0
        self._sleep_timer_end: Optional[float] = None
        self._fade_duration = 5.0  # seconds
        self._fade_start_time: Optional[float] = None
        self._fade_start_volume = 1.0
        self._grain_callback: Optional[Callable[[int], np.ndarray]] = None

    def set_grain_callback(self, callback: Callable[[int], np.ndarray]):
        """Set a function that returns a stereo buffer of given length (frames)."""
        self._grain_callback = callback

    def start(self):
        """Start the audio stream."""
        if self._stream is not None:
            return
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self._audio_callback,
            dtype=np.float32,
        )
        self._stream.start()

    def stop(self):
        """Stop the audio stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_volume(self, volume: float):
        """Set master volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    def set_pan(self, pan: float):
        """Set stereo pan (-1 left, 0 center, 1 right)."""
        self._pan = max(-1.0, min(1.0, pan))

    def set_volume_lfo(self, freq: float):
        """Set volume LFO frequency in Hz."""
        self._volume_lfo_freq = max(0.0, freq)

    def set_pan_lfo(self, freq: float):
        """Set pan LFO frequency in Hz."""
        self._pan_lfo_freq = max(0.0, freq)

    def set_sleep_timer(self, seconds: float):
        """Schedule a fade-out after given seconds."""
        self._sleep_timer_end = time.monotonic() + seconds

    def cancel_sleep_timer(self):
        """Cancel any active sleep timer."""
        self._sleep_timer_end = None
        self._fade_start_time = None

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Callback for sounddevice output stream."""
        if status:
            print(f"Audio callback status: {status}