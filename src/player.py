"""
Audio player module for real-time soundscape playback with sleep timer and fade-out.
"""

import numpy as np
import sounddevice as sd
import time
import threading
from typing import Optional, Callable


class AudioPlayer:
    """Manages real-time audio playback with fade-out and sleep timer."""

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 1024,
        sleep_timer: float = 0.0,
        fade_duration: float = 5.0,
    ):
        """
        :param sample_rate: sample rate in Hz
        :param blocksize: number of frames per callback block
        :param sleep_timer: total play duration in seconds (0 = infinite)
        :param fade_duration: fade-out duration in seconds
        """
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.sleep_timer = sleep_timer
        self.fade_duration = fade_duration
        self._stream: Optional[sd.OutputStream] = None
        self._callback: Optional[Callable[[int], np.ndarray]] = None
        self._start_time: float = 0.0
        self._stop_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None

    def start(
        self,
        callback: Callable[[int], np.ndarray],
        blocking: bool = True,
    ) -> None:
        """
        Start audio playback.

        :param callback: function that takes number of frames and returns audio array
        :param blocking: if True, blocks until playback ends; else runs in background
        """
        self._callback = callback
        self._start_time = time.time()
        self._stop_event.clear()

        def audio_callback(outdata: np.ndarray, frames: int, time_info, status):
            if status:
                print(f"Audio callback status: {status}