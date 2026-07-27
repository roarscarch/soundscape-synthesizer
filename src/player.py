"""
Audio player: real-time playback, volume LFO, panning LFO, sleep timer, fade-out, and WAV export.
"""

import numpy as np
import sounddevice as sd
import wave
import threading
import time
from typing import Optional, Callable


class AudioPlayer:
    """Handles real-time audio playback with volume/pan LFOs, sleep timer, fade-out, and export."""

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        blocksize: int = 1024,
        volume_lfo_rate: float = 0.1,
        volume_lfo_depth: float = 0.3,
        pan_lfo_rate: float = 0.05,
        pan_lfo_depth: float = 0.5,
    ):
        """
        :param sample_rate: sample rate in Hz
        :param channels: number of output channels (1 or 2)
        :param blocksize: number of frames per callback
        :param volume_lfo_rate: LFO rate for volume modulation (Hz)
        :param volume_lfo_depth: depth of volume LFO (0..1)
        :param pan_lfo_rate: LFO rate for pan modulation (Hz)
        :param pan_lfo_depth: depth of pan LFO (0..1)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.volume_lfo_rate = volume_lfo_rate
        self.volume_lfo_depth = volume_lfo_depth
        self.pan_lfo_rate = pan_lfo_rate
        self.pan_lfo_depth = pan_lfo_depth

        self._stream: Optional[sd.OutputStream] = None
        self._source_callback: Optional[Callable[[int], np.ndarray]] = None
        self._running = False
        self._start_time: float = 0.0
        self._sleep_duration: Optional[float] = None
        self._fade_duration: float = 5.0
        self._fade_start_time: Optional[float] = None
        self._exporting = False
        self._export_buffer: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def set_source(self, callback: Callable[[int], np.ndarray]) -> None:
        """Set the callback that generates audio frames.

        :param callback: function taking number of frames and returning (frames, channels) array
        """
        self._source_callback = callback

    def start(self) -> None:
        """Start playback."""
        if self._running:
            return
        if self._source_callback is None:
            raise RuntimeError("No source callback set. Call set_source() first.")

        self._running = True
        self._start_time = time.time()
        self._fade_start_time = None
        self._export_buffer = None

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            callback=self._audio_callback,
            dtype=np.float32,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop playback."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_sleep_timer(self, duration_seconds: float) -> None:
        """Set a sleep timer: after duration_seconds, begin fade-out and stop.

        :param duration_seconds: playback duration before fade-out begins
        """
        self._sleep_duration = duration_seconds

    def set_fade_duration(self, duration_seconds: float) -> None:
        """Set the duration of the fade-out (in seconds). Default 5.0."""
        self._fade_duration = max(0.1, duration_seconds)

    def start_export(self) -> None:
        """Begin capturing audio to internal buffer for later export."""
        self._exporting = True
        self._export_buffer = np.zeros((0, self.channels), dtype=np.float32)

    def stop_export(self, filepath: str) -> None:
        """Stop capturing and write the internal buffer to a WAV file.

        :param filepath: path to output .wav file
        """
        self._exporting = False
        if self._export_buffer is None or len(self._export_buffer) == 0:
            raise RuntimeError("No audio data captured. Call start_export() before playing.")

        buffer = self._export_buffer
        # Ensure float32, scale to int16
        buffer = np.clip(buffer, -1.0, 1.0)
        int_buffer = (buffer * 32767).astype(np.int16)

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(int_buffer.tobytes())

        print(f"Exported {len(buffer) / self.sample_rate:.1f}s to {filepath}")

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """Sounddevice callback: fill output buffer with processed audio."""
        if status:
            print(f"Audio callback status: {status}")

        if not self._running:
            outdata.fill(0)
            return

        # Get source audio
        audio = self._source_callback(frames)
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]  # mono to 2D
        if audio.shape[1] < self.channels:
            # Duplicate mono to stereo
            audio = np.repeat(audio, self.channels, axis=1)
        elif audio.shape[1] > self.channels:
            audio = audio[:, :self.channels]

        # Apply volume LFO
        elapsed = time.time() - self._start_time
        lfo_phase = elapsed * self.volume_lfo_rate * 2 * np.pi
        volume_mod = 1.0 - self.volume_lfo_depth * 0.5 * (1.0 + np.sin(lfo_phase))
        audio *= volume_mod

        # Apply pan LFO (if stereo)
        if self.channels == 2:
            pan_phase = elapsed * self.pan_lfo_rate * 2 * np.pi
            pan = 0.5 + self.pan_lfo_depth * 0.5 * np.sin(pan_phase)  # 0..1
            audio[:, 0] *= (1.0 - pan) * 2.0  # left
            audio[:, 1] *= pan * 2.0          # right

        # Check sleep timer
        if self._sleep_duration is not None:
            if self._fade_start_time is None:
                if elapsed >= self._sleep_duration:
                    self._fade_start_time = elapsed
            else:
                fade_elapsed = elapsed - self._fade_start_time
                if fade_elapsed >= self._fade_duration:
                    # Fade complete, stop
                    outdata.fill(0)
                    self.stop()
                    return
                fade_gain = 1.0 - (fade_elapsed / self._fade_duration)
                audio *= fade_gain

        # Export capture
        if self._exporting and self._export_buffer is not None:
            with self._lock:
                self._export_buffer = np.vstack([self._export_buffer, audio])

        outdata[:] = audio[:frames, :]

    @property
    def is_running(self) -> bool:
        return self._running

    def __del__(self) -> None:
        self.stop()
