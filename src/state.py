"""
Soundscape state persistence: save and restore generation state
so a soundscape can be resumed from the same seed and biome.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from .biomes import BIOME_REGISTRY


def _state_file_path() -> Path:
    """Return the default state file path."""
    return Path(".soundscape_state.json")


def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a loaded state dict has all required keys.
    Fills missing keys with defaults based on the biome.
    """
    biome_name = state.get("biome", "forest")
    biome = BIOME_REGISTRY.get(biome_name)
    defaults: Dict[str, Any] = {
        "seed": "soundscape",
        "biome": "forest",
        "grid_width": 8,
        "grid_height": 8,
        "sample_rate": 44100,
        "step": 0,
        "timestamp": time.time(),
        "biome_params": {},
    }
    merged = {**defaults, **state}
    if biome is not None:
        merged["biome_params"] = {
            "base_frequencies": list(biome.base_frequencies),
            "harmonics": list(biome.harmonics),
            "envelope_attack": biome.envelope_attack,
            "envelope_decay": biome.envelope_decay,
            "grain_duration": biome.grain_duration,
        }