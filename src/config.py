"""
Configuration loader for Soundscape Synthesizer.
Supports TOML-based config files with command-line overrides.
"""

import tomllib
from pathlib import Path
from typing import Optional, Dict, Any


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file. If no path is given, look for
    soundscape.toml in the current directory. Returns a dict with defaults.
    """
    defaults: Dict[str, Any] = {
        "seed": "soundscape",
        "biome": "forest",
        "duration": 0,
        "export": None,
        "sample_rate": 44100,
        "grid_width": 8,
        "grid_height": 8,
        "sleep_timer": 0,
        "fade_duration": 5.0,
    }

    if config_path is None:
        config_path = "soundscape.toml"

    path = Path(config_path)
    if not path.exists():
        return defaults

    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML file {config_path}: {e}")

    # Merge loaded values into defaults (only known keys)
    for key in defaults:
        if key in data:
            defaults[key] = data[key]

    return defaults


def merge_config(cli_args: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge command-line arguments over file-based config.
    CLI args that are not None override file config.
    """
    merged = file_config.copy()
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged
