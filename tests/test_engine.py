import numpy as np
from src.engine import SoundscapeEngine, seed_to_int
from src.grain_bank import GrainBank, Grain
from src.biomes import Biome, BIOME_REGISTRY


def test_seed_to_int_deterministic():
    """Same seed string should always produce the same integer."""
    seed = "forest morning"
    a = seed_to_int(seed)
    b = seed_to_int(seed)
    assert a == b, "seed_to_int should be deterministic"


def test_seed_to_int_different_seeds():
    """Different seed strings should produce different integers."""
    a = seed_to_int("forest morning")
    b = seed_to_int("deep space")
    assert a != b, "Different seeds should produce different integers"


def test_engine_initialization():
    """Engine should initialize with a valid grain bank and biome."""
    biome = BIOME_REGISTRY["forest"]
    grain_bank = GrainBank(biome, seed="test_seed", sample_rate=44100)
    engine = SoundscapeEngine(
        seed="test_seed",
        grain_bank=grain_bank,
        grid_width=4,
        grid_height=4,
        sample_rate=44100,
    )
    assert engine.grid_width == 4
    assert engine.grid_height == 4
    assert engine.sample_rate == 44100
    assert engine.prng is not None


def test_engine_generates_audio():
    """Engine should produce audio samples for a given time range."""
    biome = BIOME_REGISTRY["forest"]
    grain_bank = GrainBank(biome, seed="test_seed", sample_rate=44100)
    engine = SoundscapeEngine(
        seed="test_seed",
        grain_bank=grain_bank,
        grid_width=4,
        grid_height=4,
        sample_rate=44100,
    )
    duration = 1.0  # seconds
    audio = engine.generate(duration)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    expected_samples = int(44100 * duration)
    assert len(audio) == expected_samples, f"Expected {expected_samples} samples, got {len(audio)}