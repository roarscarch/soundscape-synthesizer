import unittest
import hashlib
import numpy as np
from src.seed_phrase import seed_to_hash, seed_to_biome_index


class TestSeedPhrase(unittest.TestCase):
    """Tests for seed phrase hashing and biome mapping."""

    def test_seed_to_hash_deterministic(self):
        """Same seed produces same hash."""
        seed = "forest morning"
        h1 = seed_to_hash(seed)
        h2 = seed_to_hash(seed)
        self.assertEqual(h1, h2)

    def test_seed_to_hash_different(self):
        """Different seeds produce different hashes."""
        seed1 = "forest morning"
        seed2 = "ocean waves"
        self.assertNotEqual(seed_to_hash(seed1), seed_to_hash(seed2))

    def test_seed_to_hash_empty(self):
        """Empty seed is allowed and produces a hash."""
        h = seed_to_hash("")
        self.assertIsInstance(h, bytes)
        self.assertEqual(len(h), 32)

    def test_seed_to_biome_index_in_range(self):
        """Biome index is within the number of biomes."""
        from src.biomes import BIOME_REGISTRY
        num_biomes = len(BIOME_REGISTRY)
        for seed in ["a", "b", "c", "forest", "space", "ocean"]:
            idx = seed_to_biome_index(seed, num_biomes)
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, num_biomes)

    def test_seed_to_biome_index_deterministic(self):
        """Same seed and count gives same index."""
        seed = "my phrase"
        count = 5
        self.assertEqual(seed_to_biome_index(seed, count), seed_to_biome_index(seed, count))

    def test_seed_to_biome_index_different_seeds(self):
        """Different seeds give different indices (probabilistically)."""
        count = 100
        seeds = [f"seed{i}" for i in range(10)]
        indices = {seed_to_biome_index(s, count) for s in seeds}
        self.assertGreater(len(indices), 1)


if __name__ == "__main__":
    unittest.main()
