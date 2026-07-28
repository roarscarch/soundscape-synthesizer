"""
Core engine: deterministic wave function collapse over a 2D grid of audio grains.
Each grain is a short waveform segment. The collapse propagates constraints based on
adjacent grains to produce non-repeating, evolving soundscapes.
"""

import numpy as np
from hashlib import sha256
from typing import List, Tuple, Optional, Dict
from .grain_bank import GrainBank, Grain


def seed_to_int(seed: str) -> int:
    """Convert a seed string to a deterministic integer using SHA-256."""
    return int(sha256(seed.encode()).hexdigest(), 16)


class SoundscapeEngine:
    """
    Deterministic wave function collapse engine for ambient soundscape generation.
    
    Maintains a 2D grid of grains that evolves over time, with each cell's grain
    determined by a seeded PRNG and constrained by its neighbors.
    """

    def __init__(
        self,
        seed: str,
        grain_bank: GrainBank,
        grid_width: int = 8,
        grid_height: int = 8,
        sample_rate: int = 44100,
    ):
        """
        :param seed: seed phrase for deterministic generation
        :param grain_bank: GrainBank instance providing waveform grains
        :param grid_width: number of columns in the grid
        :param grid_height: number of rows in the grid
        :param sample_rate: audio sample rate in Hz
        """
        self.seed = seed
        self.grain_bank = grain_bank
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.sample_rate = sample_rate

        # Internal PRNG state derived from seed
        self._rng_state = seed_to_int(seed)

        # Current grid: 2D array of Grain objects or None
        self.grid: List[List[Optional[Grain]]] = [
            [None for _ in range(grid_width)] for _ in range(grid_height)
        ]

        # Collapse history for deterministic evolution
        self._collapse_step = 0

        # Precompute all possible grain types from the bank
        self._grain_types: List[Grain] = list(grain_bank.grains)
        if not self._grain_types:
            raise ValueError("Grain bank is empty; cannot generate soundscape.")

        # Initialize grid with first collapse
        self._initialize_grid()

    def _next_random(self) -> float:
        """Generate next pseudo-random float in [0, 1) using internal state."""
        # Simple LCG-like update (deterministic from seed)
        self._rng_state = (self._rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        return self._rng_state / 0x7FFFFFFF

    def _choose_grain(self, row: int, col: int) -> Grain:
        """
        Deterministically choose a grain for grid position (row, col)
        based on position and current collapse step.
        """
        # Combine position, step, and seed into a deterministic index
        pos_hash = int(sha256(
            f"{self.seed}:{self._collapse_step}:{row}:{col}