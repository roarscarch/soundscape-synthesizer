import numpy as np
from typing import Optional


class MidSideProcessor:
    """Mid-side stereo processing to widen or narrow the stereo image.

    The mid-side transform converts a stereo signal into a mid channel
    (the sum of left and right) and a side channel (the difference).
    By scaling the side channel we can widen or narrow the perceived
    stereo spread without affecting the mono compatibility of the signal.
    """

    def __init__(self, width: float = 1.0):
        """
        :param width: Stereo width factor. 1.0 = original, 2.0 = wider,
                      0.0 = mono, negative = swapped sides.
        """
        self.width = float(width)

    def process(self, stereo: np.ndarray) -> np.ndarray:
        """
        Apply mid-side width adjustment to a stereo signal.

        :param stereo: Input audio with shape (samples, 2).
        :return: Processed stereo audio with same shape.
        """
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("Input must be a stereo array with shape (samples, 2)")

        # Convert to float32 if not already
        original_dtype = stereo.dtype
        audio = stereo.astype(np.float32, copy=True)

        # Mid-side decomposition
        mid = (audio[:, 0] + audio[:, 1]) * 0.5
        side = (audio[:, 0] - audio[:, 1]) * 0.5

        # Apply width factor to the side channel
        side = side * self.width

        # Recombine into left/right
        left = mid + side
        right = mid - side

        # Clip to safe range to avoid distortion (though clipping may occur later)
        out = np.stack([left, right], axis=1)

        # Convert back to original dtype if integer
        if np.issubdtype(original_dtype, np.integer):
            out = np.clip(out, -1.0, 1.0)
            info = np.iinfo(original_dtype)
            out = (out * info.max).astype(original_dtype)
        else:
            out = out.astype(original_dtype, copy=False)

        return out

    def set_width(self, width: float) -> None:
        """Update the stereo width factor."""
        self.width = float(width)

    @staticmethod
    def mid_side(stereo: np.ndarray) -> tuple:
        """Return mid and side components as separate arrays."""
        if stereo.ndim != 2 or stereo.shape[1] != 2:
            raise ValueError("Input must be a stereo array with shape (samples, 2)")
        mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
        side = (stereo[:, 0] - stereo[:, 1]) * 0.5
        return mid, side

    @staticmethod
    def from_mid_side(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
        """Reconstruct stereo signal from mid and side components."""
        left = mid + side
        right = mid - side
        return np.stack([left, right], axis=1)

    @staticmethod
    def apply_width(stereo: np.ndarray, width: float) -> np.ndarray:
        """Convenience static method to process a stereo signal with a width factor."""
        processor = MidSideProcessor(width=width)
        return processor.process(stereo)
