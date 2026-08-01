import numpy as np
from typing import List, Dict, Optional, Tuple
from .biomes import Biome
from .grain_bank import GrainBank
from .wfc import WaveFunctionCollapse
from .lfo import LFO
from .stereo import MidSideStereo
from .state import SoundscapeState


class SoundscapeEngine:
    """
    Generates an infinite, evolving ambient soundscape from a seed phrase
    using a deterministic wave function collapse algorithm.
    """

    def __init__(
        self,
        biome: Biome,
        seed_phrase: str,
        sample_rate: int = 44100,
        chunk_size: int = 1024,
        mid_side_enabled: bool = False,
        stereo_width: float = 1.0,
    ):
        self.biome = biome
        self.seed_phrase = seed_phrase
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.mid_side_enabled = mid_side_enabled
        self.stereo_width = stereo_width

        # Deterministic seed from phrase
        import hashlib
        self.seed = int(hashlib.sha256(seed_phrase.encode()).hexdigest(), 16) % (2**32)
        self.rng = np.random.default_rng(self.seed)

        # Core components
        self.grain_bank = GrainBank(biome, sample_rate=sample_rate)
        self.wfc = WaveFunctionCollapse(
            grid_size=(8, 8),
            seed=self.seed,
            num_grains=len(self.grain_bank.grains),
        )

        # LFOs for organic movement (volume and pan)
        self.volume_lfo = LFO(
            rate=biome.volume_lfo_rate,
            amplitude=0.3,
            offset=0.7,
            sample_rate=sample_rate,
        )
        self.pan_lfo = LFO(
            rate=biome.pan_lfo_rate,
            amplitude=biome.pan_spread,
            offset=0.0,
            sample_rate=sample_rate,
        )

        # Stereo processor for mid-side widening
        self.stereo_processor = MidSideStereo(width=stereo_width)

        # State tracking
        self.state = SoundscapeState(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
        )
        self._current_grid = self.wfc.initialize()

    def generate_chunk(self) -> np.ndarray:
        """
        Generate one chunk of audio (stereo float32, shape = (chunk_size, 2)).
        Evolves the WFC grid and applies LFO modulation per grain.
        """
        # Update LFOs for this chunk
        self.volume_lfo.update(self.chunk_size)
        self.pan_lfo.update(self.chunk_size)

        # Get LFO modulation envelopes for the chunk
        vol_env = self.volume_lfo.get_envelope()  # shape (chunk_size,)
        pan_env = self.pan_lfo.get_envelope()    # shape (chunk_size,)

        # Prepare output buffer
        output = np.zeros((self.chunk_size, 2), dtype=np.float32)

        # Generate audio from current WFC grid
        # Each cell in the grid maps to a grain index; we play a slice of each grain
        # and then advance the grid over time.
        num_cells = self._current_grid.size
        grains_per_cell = 1
        total_grains = num_cells * grains_per_cell

        # For now, we play the first grain of each cell, but we'll evolve the grid
        # after each chunk.
        for cell_idx, grain_idx in enumerate(self._current_grid.flatten()):
            grain = self.grain_bank.grains[grain_idx % len(self.grain_bank.grains)]
            # Get a slice of the grain of chunk_size (with wraparound for infinite)
            start = (self.state.global_time + cell_idx * self.chunk_size) % len(grain)
            slice_data = grain[start : start + self.chunk_size]
            if len(slice_data) < self.chunk_size:
                # wrap around
                slice_data = np.concatenate([slice_data, grain[: self.chunk_size - len(slice_data)]])

            # Apply per-grain volume and pan modulation
            # Volume modulation: global LFO plus slight per-grain variation
            vol = vol_env * (0.6 + 0.4 * np.sin(cell_idx * 0.1))
            # Pan: global LFO plus per-grain stereo offset
            pan = pan_env + 0.1 * np.sin(cell_idx * 0.3)
            left_gain = np.cos((pan + 1) * np.pi / 4)  # pan from -1 to 1
            right_gain = np.sin((pan + 1) * np.pi / 4)

            # Add to output
            output[:, 0] += slice_data * vol * left_gain
            output[:, 1] += slice_data * vol * right_gain

        # Normalize to prevent clipping (soft limit)
        max_val = np.max(np.abs(output)) + 1e-8
        if max_val > 0.99:
            output *= 0.99 / max_val

        # Apply mid-side stereo widening if enabled
        if self.mid_side_enabled:
            output = self.stereo_processor.process(output)

        # Update state
        self.state.advance(self.chunk_size)

        # Evolve the WFC grid for next chunk
        self._current_grid = self.wfc.evolve(self._current_grid)

        return output

    def run(self, duration_seconds: Optional[float] = None) -> None:
        """Run the engine for a specified duration (or indefinitely)."""
        import sounddevice as sd

        total_frames = int(duration_seconds * self.sample_rate) if duration_seconds else None
        frames_generated = 0

        stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            dtype="float32",
        )
        stream.start()
        try:
            while True:
                chunk = self.generate_chunk()
                stream.write(chunk)
                frames_generated += self.chunk_size
                if total_frames and frames_generated >= total_frames:
                    break
        finally:
            stream.stop()
            stream.close()

    def get_state(self) -> SoundscapeState:
        """Return current engine state."""
        return self.state
