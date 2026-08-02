"""Core soundscape engine: manages grain scheduling, mixing, and audio output."""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, List

from .biomes import Biome
from .grain_bank import GrainBank
from .lfo import LFO
from .stereo import StereoProcessor
from .meter import Meter
from .state import SoundscapeState


class SoundscapeEngine:
    """Deterministic, evolving ambient soundscape generator."""

    def __init__(
        self,
        biome: Biome,
        seed_phrase: str,
        sample_rate: int = 44100,
        buffer_size: int = 512,
        tempo: float = 60.0,
    ):
        self.biome = biome
        self.seed_phrase = seed_phrase
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.tempo = tempo
        self.state = SoundscapeState(seed_phrase=seed_phrase, biome=biome)
        self.grain_bank = GrainBank(biome, seed_phrase, sample_rate)
        self.stereo = StereoProcessor()
        self.meter = Meter(sample_rate)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None
        self._grain_schedule: List[float] = []
        self._next_grain_time = 0.0
        self._last_time = time.perf_counter()
        self._tempo_lock = threading.Lock()
        self._tempo = tempo

    @property
    def tempo(self) -> float:
        return self._tempo

    @tempo.setter
    def tempo(self, value: float) -> None:
        with self._tempo_lock:
            self._tempo = max(20.0, min(200.0, value))

    def _grain_interval(self) -> float:
        """Return the time between grain starts based on tempo."""
        beats_per_second = self.tempo / 60.0
        # Each grain triggers on a 1/8 note (eighth note) by default
        return 0.5 / beats_per_second

    def start(self) -> None:
        """Start the audio engine in a background thread."""
        if self._running:
            return
        self._running = True
        self.state.reset()
        self._next_grain_time = 0.0
        self._last_time = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the audio engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _run(self) -> None:
        """Main audio loop: generate grains and write to output stream."""
        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=2,
            dtype='float32',
        )
        self._stream = stream
        stream.start()

        # Pre-schedule the first few grains to avoid startup delay
        for _ in range(int(self.sample_rate / self.buffer_size)):
            self._schedule_grain()

        while self._running:
            now = time.perf_counter()
            elapsed = now - self._last_time
            self._last_time = now

            # Update grain schedule and generate audio blocks
            with self._lock:
                # Generate grains that are due
                while self._grain_schedule and self._grain_schedule[0] <= self._last_time:
                    self._grain_schedule.pop(0)
                    self._trigger_grain()

                # Schedule next grain based on tempo
                if not self._grain_schedule:
                    self._schedule_grain()

            # Generate one buffer of audio
            buffer = np.zeros((self.buffer_size, 2), dtype=np.float32)

            # Pull from active grains (simplified: just generate a test tone)
            # In a real implementation, this would mix all active grains
            t = np.arange(self.buffer_size) / self.sample_rate
            freq = 220.0 * (1.0 + 0.05 * np.sin(2 * np.pi * 0.1 * (self._last_time % 10)))
            buffer[:, 0] = 0.1 * np.sin(2 * np.pi * freq * t)
            buffer[:, 1] = 0.1 * np.sin(2 * np.pi * freq * t + 0.1)

            stream.write(buffer)

        stream.stop()
        stream.close()
        self._stream = None

    def _schedule_grain(self) -> None:
        """Add a grain trigger time to the schedule."""
        interval = self._grain_interval()
        self._grain_schedule.append(self._last_time + interval)

    def _trigger_grain(self) -> None:
        """Trigger a new grain (placeholder for real grain spawning)."""
        pass

    def set_tempo(self, bpm: float) -> None:
        """Set the tempo in beats per minute."""
        self.tempo = bpm

    def get_audio_block(self, num_frames: int) -> np.ndarray:
        """Return a block of audio for export or visualization."""
        # Placeholder: return silence
        return np.zeros((num_frames, 2), dtype=np.float32)
