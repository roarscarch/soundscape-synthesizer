"""
Mid-side stereo processing utilities for expanding the stereo image.
"""

import numpy as np


def mid_side_encode(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert stereo signal to mid and side components.

    Args:
        stereo: (n_samples, 2) float array.

    Returns:
        (mid, side) arrays, each (n_samples,).
    """
    left = stereo[:, 0]
    right = stereo[:, 1]
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    return mid, side


def mid_side_decode(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Convert mid and side components back to stereo.

    Args:
        mid: (n_samples,) array.
        side: (n_samples,) array.

    Returns:
        Stereo array (n_samples, 2).
    """
    left = mid + side
    right = mid - side
    return np.stack([left, right], axis=1)


def apply_mid_side_widening(
    stereo: np.ndarray,
    width: float = 1.0,
) -> np.ndarray:
    """Apply mid-side widening to a stereo signal.

    The width parameter controls how much the side (difference) component
    is amplified relative to the mid (sum) component. A width of 1.0 leaves
    the signal unchanged, values >1.0 widen the stereo image, and values
    between 0.0 and 1.0 narrow it.

    Args:
        stereo: (n_samples, 2) float array.
        width: Stereo width factor. Must be >= 0.0.

    Returns:
        Widened stereo array (n_samples, 2).
    """
    if width < 0.0:
        raise ValueError("Width must be non-negative.")
    if stereo.ndim != 2 or stereo.shape[1] != 2:
        raise ValueError("Input must be a (n_samples, 2) stereo array.")

    mid, side = mid_side_encode(stereo)
    # Scale side component only; mid stays the same.
    side_scaled = side * width
    return mid_side_decode(mid, side_scaled)


def apply_pan_lfo(
    stereo: np.ndarray,
    lfo: np.ndarray,
    depth: float = 1.0,
) -> np.ndarray:
    """Apply a panning LFO to a stereo signal.

    The LFO values are expected to be in [-1, 1] (or normalized).
    The panning is applied by adjusting the balance between left and right
    channels based on the LFO. The depth controls the maximum amount of
    panning (0.0 = no panning, 1.0 = full pan).

    Args:
        stereo: (n_samples, 2) float array.
        lfo: (n_samples,) array of LFO values in [-1, 1].
        depth: Maximum pan amount (0.0 to 1.0).

    Returns:
        Panned stereo array (n_samples, 2).
    """
    if stereo.shape[0] != len(lfo):
        raise ValueError("Stereo and LFO must have the same number of samples.")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("Depth must be between 0.0 and 1.0.")

    # Normalize LFO to [-1, 1] if needed? Assume it's already in that range.
    pan = np.clip(lfo, -1.0, 1.0) * depth

    # Convert pan to balance factor. pan = -1 (full left), +1 (full right)
    # balance = 0.5 + pan * 0.5, so left gain = balance, right gain = 1 - balance
    # But we want to preserve total energy, so use equal-power panning:
    # angle = (pan + 1) * pi/4, left = cos(angle), right = sin(angle)
    angle = (pan + 1.0) * (np.pi / 4.0)
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    left = stereo[:, 0] * left_gain
    right = stereo[:, 1] * right_gain
    return np.stack([left, right], axis=1)
