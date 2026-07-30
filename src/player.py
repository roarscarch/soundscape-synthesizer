"""
Audio player with real-time streaming, sleep timer, and fade-out.
"""

import asyncio
import logging
import numpy as np
import sounddevice as sd
import time
from typing import Optional, Callable

from .lfo import LFO
from .meter import AudioMeter

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Manages audio output stream with LFO processing, sleep timer, and fade-out."""

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 1024,
        channels: int = 2,
        meter: Optional[AudioMeter] = None,
        sleep_timer: float = 0.0,
        fade_duration: float = 5.0,
    ):
        """
        :param sample_rate: sample rate in Hz
        :param blocksize: number of frames per callback
        :param channels: number of audio channels (default 2 for stereo)
        :param meter: optional AudioMeter instance for level monitoring
        :param sleep_timer: duration in seconds after which player auto-stops (0 = no timer)
        :param fade_duration: duration of fade-out in seconds
        """
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.channels = channels
        self.meter = meter
        self.sleep_timer = sleep_timer
        self.fade_duration = fade_duration

        self.stream: Optional[sd.OutputStream] = None
        self.is_playing = False
        self.callback_fn: Optional[Callable[[int], np.ndarray]] = None

        # LFOs for organic movement
        self.volume_lfo = LFO(rate=0.1, min_val=0.6, max_val=1.0, waveform='sine')
        self.pan_lfo = LFO(rate=0.05, min_val=-1.0, max_val=1.0, waveform='sine')

        # Fade state
        self._fade_start_time: Optional[float] = None
        self._fade_start_volume: float = 1.0
        self._player_start_time: Optional[float] = None

        self._stop_event = asyncio.Event()

    def set_callback(self, callback: Callable[[int], np.ndarray]):
        """Set the audio generation callback."""
        self.callback_fn = callback

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self.callback_fn is None:
            outdata.fill(0)
            return

        # Generate audio block
        audio = self.callback_fn(frames)
        if audio.ndim == 1:
            audio = np.column_stack((audio, audio))  # mono to stereo

        # Apply volume LFO
        vol = self.volume_lfo.next()
        audio = audio * vol

        # Apply pan LFO
        pan = self.pan_lfo.next()
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))
        audio[:, 0] *= left_gain
        audio[:, 1] *= right_gain

        # Apply fade-out if active
        if self._fade_start_time is not None:
            elapsed = time.time() - self._fade_start_time
            if elapsed >= self.fade_duration:
                audio.fill(0)
                self._stop_event.set()
            else:
                fade_factor = 1.0 - (elapsed / self.fade_duration)
                audio *= fade_factor

        # Update meter
        if self.meter is not None:
            self.meter.update(audio)

        outdata[:] = audio[:frames]

    async def start(self):
        """Start audio playback asynchronously."""
        if self.is_playing:
            logger.warning("Player already running")
            return

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=self.channels,
            callback=self._audio_callback,
            dtype='float32',
        )
        self.stream.start()
        self.is_playing = True
        self._player_start_time = time.time()
        logger.info("Audio player started")

        # Wait for stop event or sleep timer
        if self.sleep_timer > 0:
            await asyncio.sleep(self.sleep_timer)
            await self.fade_out()

        await self._stop_event.wait()
        await self.stop()

    async def fade_out(self):
        """Initiate fade-out sequence."""
        if self._fade_start_time is not None:
            return  # already fading
        self._fade_start_time = time.time()
        logger.info(f"Fade-out started (duration: {self.fade_duration}