"""
Sleep timer and fade-out utilities for soundscape playback.
"""

import time
from typing import Optional


class SleepTimer:
    """Counts down and triggers fade-out after a specified duration."""

    def __init__(self, duration: float = 0.0, fade_duration: float = 5.0):
        """
        :param duration: total sleep timer duration in seconds (0 = disabled)
        :param fade_duration: fade-out duration in seconds
        """
        self.duration = duration
        self.fade_duration = fade_duration
        self.start_time: Optional[float] = None
        self._elapsed = 0.0
        self._active = False

    def start(self) -> None:
        """Start the timer, if duration is set."""
        if self.duration > 0:
            self.start_time = time.monotonic()
            self._elapsed = 0.0
            self._active = True
        else:
            self._active = False

    def stop(self) -> None:
        """Stop the timer and reset state."""
        self._active = False
        self.start_time = None
        self._elapsed = 0.0

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def elapsed(self) -> float:
        """Elapsed seconds since start (0 if not active)."""
        if not self._active or self.start_time is None:
            return self._elapsed
        return time.monotonic() - self.start_time

    @property
    def remaining(self) -> float:
        """Seconds left before fade-out begins (0 if disabled or done)."""
        if not self._active:
            return 0.0
        remaining = self.duration - self.elapsed
        return max(0.0, remaining)

    @property
    def is_fade_out_time(self) -> bool:
        """True when the timer has elapsed and fade-out should start."""
        return self._active and self.remaining <= 0.0

    def fade_gain(self, current_time: float) -> float:
        """
        Returns a gain multiplier for fade-out.

        :param current_time: current monotonic time (seconds)
        :return: 1.0 if no fade-out active, otherwise ramps to 0.0.
        """
        if not self.is_fade_out_time:
            return 1.0
        fade_elapsed = current_time - (self.start_time + self.duration)
        if fade_elapsed >= self.fade_duration:
            return 0.0
        # linear fade-out
        return max(0.0, 1.0 - fade_elapsed / self.fade_duration)
