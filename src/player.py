"""
Audio player: manages real-time playback of soundscape grains
with sleep timer, fade-out, panning LFO, volume LFO, and WAV export.
"""

import time
import threading
import queue
import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from hashlib import sha256

from .engine import Grain, seed_to_int


class AudioPlayer:
    """Plays audio grains in a separate thread with fade-out and sleep timer."""

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 1024,
        num_channels: int = 2,
        pan_lfo_rate: float = 0.1,
        pan_lfo_depth: float = 0.5,
        volume_lfo_rate: float = 0.05,
        volume_lfo_depth: float = 0.3,
    ):
        """
        :param sample_rate: audio sample rate in Hz
        :param block_size: samples per audio callback block
        :param num_channels: number of output channels (1 = mono, 2 = stereo)
        :param pan_lfo_rate: rate of panning LFO in Hz
        :param pan_lfo_depth: max pan offset (0..1, where 1 = full left/right swing)
        :param volume_lfo_rate: rate of volume LFO in Hz
        :param volume_lfo_depth: max volume modulation depth (0..1 fraction of current gain)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.num_channels = num_channels
        self.pan_lfo_rate = pan_lfo_rate
        self.pan_lfo_depth = pan_lfo_depth
        self.volume_lfo_rate = volume_lfo_rate
        self.volume_lfo_depth = volume_lfo_depth

        self._grain_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()  # start unpaused
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.OutputStream] = None

        # Internal state for LFOs
        self._phase_pan = 0.0
        self._phase_volume = 0.0
        self._current_grain: Optional[Grain] = None
        self._grain_pos = 0
        self._output_buffer = np.zeros((block_size, num_channels), dtype=np.float32)

        # Fade-out / sleep timer
        self._fade_start_time: Optional[float] = None
        self._fade_duration: float = 0.0
        self._sleep_end_time: Optional[float] = None
        self._volume_scale: float = 1.0
        self._export_callback: Optional[Callable] = None

    def start(self) -> None:
        """Start the audio stream and playback thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._paused.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop playback and join thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def pause(self) -> None:
        """Pause playback."""
        self._paused.clear()

    def resume(self) -> None:
        """Resume playback."""
        self._paused.set()

    def enqueue_grain(self, grain: Grain) -> None:
        """Add a grain to the playback queue."""
        self._grain_queue.put(grain)

    def set_sleep_timer(self, duration_seconds: float) -> None:
        """Set sleep timer: after duration, begin fade-out."""
        self._sleep_end_time = time.monotonic() + duration_seconds

    def set_fade_out(self, duration_seconds: float) -> None:
        """Set fade-out duration. Call before starting fade."""
        self._fade_duration = duration_seconds
        self._fade_start_time = time.monotonic()

    def set_export_callback(self, callback: Callable) -> None:
        """Register callback invoked when export is requested (e.g., via signal)."""
        self._export_callback = callback

    def export_current(self, seed: str) -> None:
        """Export currently buffered output to WAV using the export module.
        This is a placeholder; actual implementation in export.py."""
        if self._export_callback:
            self._export_callback(seed)

    def _run(self) -> None:
        """Main loop: fill output buffer and write to stream."""
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.num_channels,
                blocksize=self.block_size,
                callback=self._audio_callback,
            )
            self._stream.start()

            while not self._stop_event.is_set():
                self._paused.wait()  # block if paused
                self._update_grain()
                self._update_lfos()
                self._check_timer()
                self._check_fade()
                time.sleep(0.01)  # yield control
        except Exception as e:
            print(f"Audio player error: {e}")
            self._stop_event.set()

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Callback for sounddevice stream."""
        if status:
            print(f"Audio callback status: {status}