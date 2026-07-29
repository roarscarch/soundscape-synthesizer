"""
Audio player with real-time panning, volume LFOs, sleep timer, and fade-out.
"""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, Callable


class AudioPlayer:
    """
    Plays audio streams with real-time effects: panning LFO, volume LFO,
    configurable sleep timer with fade-out.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        block_size: int = 1024,
        pan_lfo_rate: float = 0.1,
        pan_lfo_depth: float = 0.5,
        volume_lfo_rate: float = 0.05,
        volume_lfo_depth: float = 0.2,
    ):
        """
        :param sample_rate: audio sample rate in Hz
        :param block_size: number of frames per callback
        :param pan_lfo_rate: frequency of panning LFO in Hz
        :param pan_lfo_depth: maximum pan deviation from center (0..1)
        :param volume_lfo_rate: frequency of volume LFO in Hz
        :param volume_lfo_depth: depth of volume modulation (0..1)
        """
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.pan_lfo_rate = pan_lfo_rate
        self.pan_lfo_depth = pan_lfo_depth
        self.volume_lfo_rate = volume_lfo_rate
        self.volume_lfo_depth = volume_lfo_depth

        self._stream: Optional[sd.OutputStream] = None
        self._audio_source: Optional[Callable[[int], np.ndarray]] = None
        self._running = False
        self._lock = threading.Lock()

        # Sleep timer state
        self._sleep_duration: Optional[float] = None
        self._sleep_start_time: Optional[float] = None
        self._fade_duration: float = 5.0  # seconds
        self._fade_start_time: Optional[float] = None
        self._on_fade_complete: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None

        # Current sample position (for LFO phase)
        self._sample_pos = 0

    def set_audio_source(self, source: Callable[[int], np.ndarray]) -> None:
        """Set the callback that provides audio frames (num_samples -> stereo ndarray)."""
        self._audio_source = source

    def set_sleep_timer(
        self,
        duration: float,
        fade_duration: float = 5.0,
        on_complete: Optional[Callable] = None,
    ) -> None:
        """
        Schedule the player to stop after `duration` seconds, with a linear fade-out.
        :param duration: total time in seconds before stop
        :param fade_duration: length of fade-out in seconds
        :param on_complete: callback invoked when fade-out completes
        """
        with self._lock:
            self._sleep_duration = duration
            self._fade_duration = fade_duration
            self._sleep_start_time = time.time()
            self._fade_start_time = None
            self._on_fade_complete = on_complete

    def set_on_stop(self, callback: Optional[Callable]) -> None:
        """Set a callback invoked when the player stops (after fade-out)."""
        self._on_stop = callback

    def start(self) -> None:
        """Start the audio stream."""
        if self._stream is not None:
            return
        if self._audio_source is None:
            raise RuntimeError("No audio source set. Call set_audio_source first.")

        self._running = True
        self._sample_pos = 0
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=2,
            callback=self._callback,
            dtype=np.float32,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio stream immediately."""
        with self._lock:
            self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        cb = self._on_stop
        if cb:
            cb()

    def is_running(self) -> bool:
        """Return True if the audio stream is active."""
        with self._lock:
            return self._running

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Audio callback: fetch frames, apply LFOs and fade-out."""
        if status:
            print(f"Audio callback status: {status}", file=__import__('sys').stderr)

        # Check sleep timer and fade-out state
        with self._lock:
            if not self._running:
                outdata.fill(0)
                return

            current_time = time.time()
            fade_gain = 1.0
            if self._sleep_start_time is not None and self._sleep_duration is not None:
                elapsed = current_time - self._sleep_start_time
                if elapsed >= self._sleep_duration:
                    # Start fade-out if not already started
                    if self._fade_start_time is None:
                        self._fade_start_time = current_time
                    fade_elapsed = current_time - self._fade_start_time
                    if fade_elapsed >= self._fade_duration:
                        # Fade-out complete
                        self._running = False
                        outdata.fill(0)
                        # Schedule stop on main thread or callback
                        if self._on_fade_complete:
                            self._on_fade_complete()
                        return
                    else:
                        fade_gain = 1.0 - (fade_elapsed / self._fade_duration)
                else:
                    # Not yet time to fade
                    pass

        # Get audio from source
        if self._audio_source is None:
            outdata.fill(0)
            return

        try:
            audio = self._audio_source(frames)
        except Exception as e:
            print(f"Audio source error: {e}