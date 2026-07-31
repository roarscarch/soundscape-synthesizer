import tomllib
from pathlib import Path
from typing import Optional, Dict, Any, List
import tomli_w


DEFAULTS: Dict[str, Any] = {
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

VALID_BIOMES = ["forest", "ocean", "space"]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a TOML file. If no path is given, look for
    soundscape.toml in the current directory. Returns a dict with defaults.
    """
    if config_path is None:
        config_path = "soundscape.toml"

    path = Path(config_path)
    if not path.exists():
        return DEFAULTS.copy()

    with open(path, "rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML file {config_path}: {e}")

    merged = DEFAULTS.copy()
    merged.update(data)
    validate_config(merged, path)
    return merged


def save_config(config: Dict[str, Any], config_path: str = "soundscape.toml") -> None:
    """
    Save a configuration dict to a TOML file. Creates parent directories if needed.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def validate_config(config: Dict[str, Any], path: Optional[Path] = None) -> None:
    """
    Validate configuration values, raising ValueError for invalid entries.
    """
    invalid = []
    if config.get("biome") not in VALID_BIOMES:
        invalid.append(f"biome must be one of {VALID_BIOMES}, got {config.get('biome')!r}")
    if not isinstance(config.get("sample_rate"), int) or config.get("sample_rate") <= 0:
        invalid.append(f"sample_rate must be a positive integer, got {config.get('sample_rate')!r}")
    if not isinstance(config.get("grid_width"), int) or config.get("grid_width") <= 0:
        invalid.append(f"grid_width must be a positive integer, got {config.get('grid_width')!r}")
    if not isinstance(config.get("grid_height"), int) or config.get("grid_height") <= 0:
        invalid.append(f"grid_height must be a positive integer, got {config.get('grid_height')!r}")
    if not isinstance(config.get("duration"), (int, float)) or config.get("duration") < 0:
        invalid.append(f"duration must be a non-negative number, got {config.get('duration')!r}")
    if not isinstance(config.get("sleep_timer"), (int, float)) or config.get("sleep_timer") < 0:
        invalid.append(f"sleep_timer must be a non-negative number, got {config.get('sleep_timer')!r}")
    if not isinstance(config.get("fade_duration"), (int, float)) or config.get("fade_duration") < 0:
        invalid.append(f"fade_duration must be a non-negative number, got {config.get('fade_duration')!r}")
    if config.get("seed") is not None and not isinstance(config.get("seed"), str):
        invalid.append(f"seed must be a string or null, got {config.get('seed')!r}")
    if config.get("export") is not None and not isinstance(config.get("export"), str):
        invalid.append(f"export must be a string or null, got {config.get('export')!r}")

    if invalid:
        loc = f" in {path}" if path else ""
        raise ValueError(f"Invalid configuration{loc}: " + "; ".join(invalid))
