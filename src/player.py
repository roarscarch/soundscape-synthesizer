"""
Real-time audio playback engine using sounddevice.
Manages streaming output, panning, volume LFOs, sleep timer, fade-out, and WAV export.
"""

import numpy as np
import sounddevice as sd
import threading
import time
import wave
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
        # Recording buffer for export
        self._recording_buffer: list = []
        self._recording = False

    def set_grain_callback(self, callback: Callable[[int], np.ndarray]):
        """Set the callback that provides the next block of audio grains.

        The callback receives the number of frames requested and returns
        a numpy array of shape (frames, channels) with float32 values in [-1, 1].
        """
        self._grain_callback = callback

    def start(self):
        """Start audio playback in a background thread."""
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
        """Stop audio playback."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def start_recording(self):
        """Begin recording audio for later export."""
        self._recording_buffer = []
        self._recording = True

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the accumulated audio as a 2D array.

        Returns:
            numpy array of shape (total_frames, channels), dtype float32.
        """
        self._recording = False
        if not self._recording_buffer:
            return np.zeros((0, self.channels), dtype=np.float32)
        return np.concatenate(self._recording_buffer, axis=0)

    def export_wav(self, filepath: str, audio: Optional[np.ndarray] = None):
        """Export audio to a WAV file.

        If audio is None, the current recording buffer is used (must have been
        started and stopped via start_recording/stop_recording).

        :param filepath: path to output .wav file
        :param audio: optional numpy array of shape (frames, channels) float32
        """
        if audio is None:
            audio = self.stop_recording()
        if audio.size == 0:
            raise ValueError("No audio data to export.")
        # Ensure the array is contiguous and in the correct range
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def set_volume(self, volume: float):
        """Set master volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))

    def set_pan(self, pan: float):
        """Set stereo panning (-1 left, 0 center, 1 right)."""
        self._pan = max(-1.0, min(1.0, pan))

    def set_volume_lfo(self, freq: float):
        """Set volume LFO frequency in Hz."""
        self._volume_lfo_freq = max(0.0, freq)

    def set_pan_lfo(self, freq: float):
        """Set pan LFO frequency in Hz."""
        self._pan_lfo_freq = max(0.0, freq)

    def set_sleep_timer(self, seconds: float):
        """Set a timer to automatically stop playback after given seconds."""
        self._sleep_timer_end = time.time() + seconds

    def set_fade_duration(self, seconds: float):
        """Set the fade-out duration in seconds."""
        self._fade_duration = max(0.0, seconds)

    def _audio_callback(self, outdata: np.ndarray, frames: int, times, status):
        """Internal callback for sounddevice stream."""
        if status:
            print(f"Audio callback status: {status}")

        # Check sleep timer
        if self._sleep_timer_end is not None and time.time() >= self._sleep_timer_end:
            if self._fade_start_time is None:
                self._fade_start_time = time.time()
                self._fade_start_volume = self._volume
            elapsed = time.time() - self._fade_start_time
            if elapsed >= self._fade_duration:
                outdata.fill(0)
                self.stop()
                return
            else:
                fade_factor = 1.0 - (elapsed / self._fade_duration)
                self._volume = self._fade_start_volume * fade_factor

        # Get audio from callback
        if self._grain_callback is not None:
            try:
                audio = self._grain_callback(frames)
            except Exception as e:
                print(f"Grain callback error: {e}