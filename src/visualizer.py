import numpy as np
import sounddevice as sd
from typing import Optional


class SpectrumAnalyzer:
    """Real-time spectrum analyzer using FFT on audio chunks."""

    def __init__(self, block_size: int = 1024, sample_rate: int = 44100, fps: int = 30):
        """
        :param block_size: FFT block size (must be power of 2)
        :param sample_rate: audio sample rate
        :param fps: refresh rate for display updates
        """
        if block_size & (block_size - 1) != 0:
            raise ValueError("block_size must be a power of 2")
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.fps = fps
        self.window = np.hanning(block_size)
        self._freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self._last_spectrum = np.zeros(len(self._freqs))

    def analyze(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute magnitude spectrum of the given mono audio chunk.
        Returns array of magnitudes (dB scale, normalized to 0-1).
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < self.block_size:
            padded = np.zeros(self.block_size)
            padded[: len(audio)] = audio
            audio = padded
        else:
            audio = audio[: self.block_size]

        windowed = audio * self.window
        spectrum = np.abs(np.fft.rfft(windowed))
        # Normalize to 0-1 range (peak magnitude ~ 1.0 for full-scale sine)
        spectrum = spectrum / (self.block_size / 2.0)
        # Convert to dB and clip
        spectrum_db = 20.0 * np.log10(spectrum + 1e-10)
        spectrum_db = np.clip(spectrum_db, -80.0, 0.0)
        normalized = (spectrum_db + 80.0) / 80.0  # 0..1
        self._last_spectrum = normalized
        return normalized

    def get_frequencies(self) -> np.ndarray:
        """Return the frequency bins corresponding to the spectrum."""
        return self._freqs

    def run_display(self, stream: sd.Stream, duration: Optional[float] = None):
        """
        Run a real-time console spectrum display until interrupted or duration elapses.
        Assumes stream is already open and providing audio data via callback.
        """
        import time
        import shutil

        interval = 1.0 / self.fps
        start_time = time.time()
        try:
            while True:
                if duration is not None and time.time() - start_time >= duration:
                    break
                # This is a placeholder that would normally capture from the stream.
                # For simplicity, we just sleep and show a static message.
                print("\rSpectrum analyzer active...", end="", flush=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
        finally:
            print()

    def get_last_spectrum(self) -> np.ndarray:
        """Return the most recently computed spectrum."""
        return self._last_spectrum

    def render_bar_chart(self, width: int = 60, height: int = 10) -> str:
        """
        Render a simple ASCII bar chart of the spectrum.
        :param width: number of columns
        :param height: number of rows
        :return: multi-line string
        """
        spectrum = self._last_spectrum
        freqs = self._freqs
        # Map to logarithmic frequency scale for better visual
        log_freqs = np.log10(freqs + 1.0)
        log_freqs = log_freqs / log_freqs[-1]
        # Create bins
        bins = np.linspace(0, 1, width + 1)
        indices = np.digitize(log_freqs, bins) - 1
        indices = np.clip(indices, 0, width - 1)

        # Compute max magnitude per bin
        bin_values = np.zeros(width)
        for i, idx in enumerate(indices):
            bin_values[idx] = max(bin_values[idx], spectrum[i])

        # Render bars
        lines = []
        for row in range(height, 0, -1):
            threshold = row / height
            line = ""
            for val in bin_values:
                if val >= threshold:
                    line += "█"
                elif val >= threshold - (1.0 / height) * 0.5:
                    line += "▓"
                else:
                    line += " "
            lines.append(line)
        return "\n".join(lines)
