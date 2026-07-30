import time
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Callable

from .engine import SoundscapeEngine
from .lfo import LFO
from .meter import AudioLevelMeter


class AudioPlayer:
    """Plays audio from a SoundscapeEngine with real-time LFO modulation and optional meter."""

    def __init__(
        self,
        engine: SoundscapeEngine,
        sample_rate: int = 44100,
        blocksize: int = 1024,
        volume_lfo_rate: float = 0.1,
        volume_lfo_depth: float = 0.3,
        pan_lfo_rate: float = 0.05,
        pan_lfo_depth: float = 0.4,
        meter: Optional[AudioLevelMeter] = None,
    ):
        self.engine = engine
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.volume_lfo = LFO(rate=volume_lfo_rate, depth=volume_lfo_depth, waveform='sine')
        self.pan_lfo = LFO(rate=pan_lfo_rate, depth=pan_lfo_depth, waveform='sine')
        self.meter = meter
        self._stream: Optional[sd.OutputStream] = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self):
        """Start audio playback in a background thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            blocksize=self.blocksize,
            callback=self._audio_callback,
            dtype='float32',
        )
        self._stream.start()

    def stop(self):
        """Stop audio playback gracefully."""
        self._running = False
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        if self._stop_event.is_set():
            outdata.fill(0)
            return

        # Generate stereo block from engine
        block = self.engine.generate_block(frames)
        if block is None:
            outdata.fill(0)
            return

        # Apply volume LFO (gain modulation)
        vol_mod = self.volume_lfo.get_value()
        # vol_mod is in [-1, 1], map to [1 - depth, 1]
        vol_gain = 1.0 - (self.volume_lfo.depth * (1.0 - vol_mod)) / 2.0
        block *= vol_gain

        # Apply pan LFO (balance between left and right)
        pan_mod = self.pan_lfo.get_value()
        # pan_mod in [-1, 1], map to left/right gains
        left_gain = np.clip(1.0 - pan_mod * self.pan_lfo.depth, 0.0, 1.0)
        right_gain = np.clip(1.0 + pan_mod * self.pan_lfo.depth, 0.0, 1.0)
        block[:, 0] *= left_gain
        block[:, 1] *= right_gain

        # Update meter if available
        if self.meter is not None:
            self.meter.update(block)

        outdata[:] = block

    def is_running(self) -> bool:
        return self._running

    def set_volume_lfo_rate(self, rate: float):
        self.volume_lfo.rate = rate

    def set_pan_lfo_rate(self, rate: float):
        self.pan_lfo.rate = rate

    def set_volume_lfo_depth(self, depth: float):
        self.volume_lfo.depth = depth

    def set_pan_lfo_depth(self, depth: float):
        self.pan_lfo.depth = depth
