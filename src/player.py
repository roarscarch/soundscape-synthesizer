"""
Audio player module for Soundscape Synthesizer.
Provides non-blocking, callback-based audio streaming using sounddevice.
"""

import numpy as np
import sounddevice as sd
from typing import Optional, Callable
from threading import Event


class AudioPlayer:
    """
    Plays audio via a callback mechanism for real-time, infinite streaming.
    Supports fade-in/out and sleep timer.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 1024,
        channels: int = 2,
        callback: Optional[Callable[[int], np.ndarray]] = None,
    ):
        """
        :param sample_rate: sample rate in Hz
        :param blocksize: number of frames per callback block
        :param channels: number of audio channels (1=mono, 2=stereo)
        :param callback: function that takes block size and returns audio array (blocksize, channels)
        """
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.callback = callback
        self._stream: Optional[sd.OutputStream] = None
        self._stop_event = Event()
        self._sleep_timer: Optional[float] = None
        self._fade_duration: float = 0.0
        self._fade_start_time: float = 0.0
        self._is_fading: bool = False
        self._current_time: float = 0.0

    def start(self) -> None:
        """Start the audio stream."""
        if self._stream is not None:
            return
        self._stop_event.clear()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            callback=self._audio_callback,
            dtype=np.float32,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop the audio stream gracefully."""
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_sleep_timer(self, seconds: float, fade_duration: float = 5.0) -> None:
        """
        Set a sleep timer. After `seconds` of playback, the player will fade out
        and stop.
        :param seconds: total playback time before stop (seconds)
        :param fade_duration: length of fade-out (seconds)
        """
        self._sleep_timer = seconds
        self._fade_duration = fade_duration
        self._current_time = 0.0
        self._is_fading = False

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info, status
    ) -> None:
        """
        Internal callback called by sounddevice for each audio block.
        """
        if self._stop_event.is_set():
            outdata.fill(0.0)
            return

        if self.callback is not None:
            audio = self.callback(frames)
        else:
            audio = np.zeros((frames, self.channels), dtype=np.float32)

        # Ensure correct shape
        if audio.shape != (frames, self.channels):
            audio = np.resize(audio, (frames, self.channels))

        # Apply fade-out if sleep timer is active
        if self._sleep_timer is not None and self._sleep_timer > 0:
            self._current_time += frames / self.sample_rate
            if self._current_time >= self._sleep_timer - self._fade_duration:
                if not self._is_fading:
                    self._is_fading = True
                    self._fade_start_time = self._current_time
                fade_progress = (
                    self._current_time - self._fade_start_time
                ) / self._fade_duration
                fade_gain = max(0.0, 1.0 - fade_progress)
                audio = audio * fade_gain
                if fade_gain <= 0.0:
                    self.stop()

        outdata[:] = audio.astype(np.float32)

    @property
    def is_active(self) -> bool:
        """Check if the stream is active."""
        return self._stream is not None and self._stream.active
