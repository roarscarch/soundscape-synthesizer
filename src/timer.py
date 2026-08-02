"""Sleep timer functionality for scheduled playback stop."""

import threading
import time
from typing import Callable, Optional


class SleepTimer:
    """Runs a callback after a specified duration, with optional fade-out."""

    def __init__(
        self,
        duration: float,
        callback: Callable[[], None],
        fade_duration: float = 5.0,
        check_interval: float = 0.1,
    ):
        """
        :param duration: seconds until the timer fires
        :param callback: function to call when the timer completes
        :param fade_duration: seconds before the end to start fade-out (0 to disable)
        :param check_interval: polling interval for the timer thread
        """
        self.duration = duration
        self.callback = callback
        self.fade_duration = fade_duration
        self.check_interval = check_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the timer in a background thread."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Timer already running")
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Cancel the timer if it is running."""
        self._stop_event.set()

    def remaining(self) -> float:
        """Return seconds remaining until timer fires (0 if not started)."""
        if self._start_time is None:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return max(0.0, self.duration - elapsed)

    def _run(self) -> None:
        """Background thread loop."""
        if self._start_time is None:
            return
        end_time = self._start_time + self.duration
        fade_start = end_time - self.fade_duration

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= end_time:
                self.callback()
                break
            if self.fade_duration > 0 and now >= fade_start:
                # Optionally notify about fade-out phase; here we just sleep until the end.
                pass
            time.sleep(self.check_interval)

    def is_running(self) -> bool:
        """Return True if the timer thread is active."""
        return self._thread is not None and self._thread.is_alive()
