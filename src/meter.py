"""
Real-time audio level meter using RMS and peak detection.
Displays a simple ASCII bar meter in the terminal.
"""

import numpy as np
import sys
import time
from typing import Optional


class LevelMeter:
    """Monitors audio buffer RMS and peak levels, and prints a bar meter."""

    def __init__(self, refresh_rate: float = 10.0, max_width: int = 40):
        """
        :param refresh_rate: how many times per second to update the display
        :param max_width: maximum number of characters for the bar
        """
        self.refresh_rate = refresh_rate
        self.max_width = max_width
        self.last_print = 0.0
        self.peak_left = 0.0
        self.peak_right = 0.0

    def update(self, buffer: np.ndarray, sample_rate: int) -> None:
        """
        Calculate and display levels from an audio buffer.
        Buffer shape: (samples, channels).
        """
        now = time.monotonic()
        if now - self.last_print < 1.0 / self.refresh_rate:
            return
        self.last_print = now

        if buffer.ndim == 1:
            buffer = buffer.reshape(-1, 1)

        channels = buffer.shape[1]
        for ch in range(channels):
            channel_data = buffer[:, ch]
            rms = np.sqrt(np.mean(channel_data ** 2))
            peak = np.max(np.abs(channel_data))
            if ch == 0:
                self.peak_left = max(self.peak_left * 0.95, peak)
            elif ch == 1:
                self.peak_right = max(self.peak_right * 0.95, peak)

        # Scale to 0..1
        rms_left = np.sqrt(np.mean(buffer[:, 0] ** 2)) if channels >= 1 else 0.0
        rms_right = np.sqrt(np.mean(buffer[:, 1] ** 2)) if channels >= 2 else rms_left

        # Convert to dB scale (clamp at -60 dB)
        def level_to_db(val: float) -> float:
            if val <= 1e-6:
                return -60.0
            return max(-60.0, 20.0 * np.log10(val))

        db_left = level_to_db(rms_left)
        db_right = level_to_db(rms_right)

        # Map -60..0 dB to 0..max_width characters
        def db_to_bar(db: float) -> int:
            normalized = (db + 60.0) / 60.0
            normalized = max(0.0, min(1.0, normalized))
            return int(normalized * self.max_width)

        left_bar_len = db_to_bar(db_left)
        right_bar_len = db_to_bar(db_right)

        left_bar = "#" * left_bar_len + "-" * (self.max_width - left_bar_len)
        right_bar = "#" * right_bar_len + "-" * (self.max_width - right_bar_len)

        sys.stdout.write(f"\rL: |{left_bar}| {db_left:+.1f} dB  R: |{right_bar}| {db_right:+.1f} dB")
        sys.stdout.flush()

    def clear(self) -> None:
        """Clear the meter display line."""
        sys.stdout.write("\r" + " " * (self.max_width * 2 + 30))
        sys.stdout.write("\r")
        sys.stdout.flush()
