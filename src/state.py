"""
Soundscape state persistence and resume capability.

Allows saving the current state of a soundscape (seed, biome, parameters,
and internal PRNG state) to a JSON file, and resuming from that state to
continue the same non-repeating audio stream.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class SoundscapeState:
    """Represents the full state of a soundscape for persistence."""

    def __init__(
        self,
        seed: str,
        biome_name: str,
        sample_rate: int = 44100,
        grid_width: int = 8,
        grid_height: int = 8,
        position: int = 0,
        prng_state: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.seed = seed
        self.biome_name = biome_name
        self.sample_rate = sample_rate
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.position = position
        self.prng_state = prng_state or {}
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary for serialization."""
        return {
            "seed": self.seed,
            "biome": self.biome_name,
            "sample_rate": self.sample_rate,
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "position": self.position,
            "prng_state": self.prng_state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SoundscapeState":
        """Create a SoundscapeState from a dictionary."""
        return cls(
            seed=data["seed"],
            biome_name=data["biome"],
            sample_rate=data.get("sample_rate", 44100),
            grid_width=data.get("grid_width", 8),
            grid_height=data.get("grid_height", 8),
            position=data.get("position", 0),
            prng_state=data.get("prng_state", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def save(self, filepath: str) -> None:
        """Save the state to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "SoundscapeState":
        """Load a state from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update_position(self, position: int) -> None:
        """Update the playhead position and timestamp."""
        self.position = position
        self.updated_at = datetime.utcnow().isoformat()


def save_state(
    state: SoundscapeState,
    directory: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    """
    Save the state to a file. If no directory is given, use a temporary directory.
    Returns the full path to the saved file.
    """
    if directory is None:
        directory = tempfile.gettempdir()
    if filename is None:
        filename = f"soundscape_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}