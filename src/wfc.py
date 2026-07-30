"""
Wave Function Collapse (WFC) module for deterministic grain grid generation.
Uses a seed phrase to drive a PRNG and collapse a 2D grid of grain indices,
ensuring non-repeating patterns.
"""

import numpy as np
from hashlib import sha256
from typing import List, Tuple, Optional


def _seed_to_int(seed: str) -> int:
    """Convert a seed string to a deterministic integer."""
    return int(sha256(seed.encode()).hexdigest(), 16)


def _prng(state: int) -> Tuple[int, int]:
    """
    Simple xorshift64 PRNG returning (next_state, random_int).
    Returns a 32-bit integer.
    """
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    # Ensure state fits in 64 bits
    state &= 0xFFFFFFFFFFFFFFFF
    return state, state & 0xFFFFFFFF


def generate_grid(
    seed: str,
    width: int,
    height: int,
    num_grains: int,
) -> np.ndarray:
    """
    Generate a 2D grid of grain indices using WFC.
    
    The algorithm maintains a set of possible grain indices for each cell,
    then iteratively collapses the cell with the lowest entropy (fewest options),
    using the seed PRNG to choose among possibilities. Adjacent cells are constrained
    to avoid immediate repetition (simple adjacency rule).
    
    :param seed: seed phrase for deterministic output
    :param width: grid width in cells
    :param height: grid height in cells
    :param num_grains: total number of available grain types (0..num_grains-1)
    :return: 2D numpy array of shape (height, width) with grain indices
    """
    if num_grains < 1:
        raise ValueError("num_grains must be at least 1")
    if width < 1 or height < 1:
        raise ValueError("grid dimensions must be positive")

    # Initialize grid: each cell can be any grain index (0..num_grains-1)
    # We store a set of possible indices as a boolean mask for efficiency
    possibilities = np.ones((height, width, num_grains), dtype=bool)
    collapsed = np.zeros((height, width), dtype=np.int32) - 1  # -1 = not collapsed

    # Deterministic PRNG state
    state = _seed_to_int(seed)

    def get_entropy(y: int, x: int) -> int:
        """Count remaining options for a cell."""
        return int(np.sum(possibilities[y, x]))

    def collapse_cell(y: int, x: int) -> None:
        """Collapse a cell to a single grain index."""
        nonlocal state
        options = np.where(possibilities[y, x])[0]
        if len(options) == 0:
            # Fallback: should not happen with proper constraint, but handle gracefully
            collapsed[y, x] = 0
            return
        state, rand = _prng(state)
        chosen = options[rand % len(options)]
        collapsed[y, x] = chosen
        # Remove all other possibilities
        possibilities[y, x, :] = False
        possibilities[y, x, chosen] = True

    def propagate(y: int, x: int) -> None:
        """
        Propagate constraints to neighbors: forbid the chosen grain index
        in adjacent cells to avoid immediate repetition.
        """
        chosen = collapsed[y, x]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and collapsed[ny, nx] == -1:
                # Remove the chosen grain from neighbor's possibilities
                possibilities[ny, nx, chosen] = False

    # Main WFC loop
    while True:
        # Find cell with minimum entropy > 0
        min_entropy = num_grains + 1
        target_y, target_x = -1, -1
        for y in range(height):
            for x in range(width):
                if collapsed[y, x] == -1:
                    ent = get_entropy(y, x)
                    if ent > 0 and ent < min_entropy:
                        min_entropy = ent
                        target_y, target_x = y, x

        if target_y == -1:
            # All cells collapsed or no valid options remain
            break

        collapse_cell(target_y, target_x)
        propagate(target_y, target_x)

    # If any cell remains uncollapsed (shouldn't happen with proper constraints),
    # fill with random using PRNG
    for y in range(height):
        for x in range(width):
            if collapsed[y, x] == -1:
                state, rand = _prng(state)
                collapsed[y, x] = rand % num_grains

    return collapsed


def get_grain_sequence(
    grid: np.ndarray,
    traversal: str = "hilbert",
    seed: Optional[str] = None,
) -> List[int]:
    """
    Convert a 2D grid of grain indices into a 1D sequence for playback.
    
    Supported traversal orders:
    - "hilbert": space-filling Hilbert curve (default)
    - "raster": row-major order
    - "spiral": outward spiral from center
    
    :param grid: 2D numpy array of grain indices
    :param traversal: traversal method
    :param seed: optional seed for randomization within traversal
    :return: list of grain indices in playback order
    """
    height, width = grid.shape
    
    if traversal == "raster":
        return grid.flatten().tolist()
    elif traversal == "spiral":
        return _spiral_traversal(grid)
    elif traversal == "hilbert":
        return _hilbert_traversal(grid)
    else:
        raise ValueError(f"Unknown traversal method: {traversal}