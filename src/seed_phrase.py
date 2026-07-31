"""
Seed phrase generation: convert arbitrary text into a deterministic, compact phrase.
"""

import hashlib
import random
from typing import List, Optional


# Word lists for generating phrases. Keep them simple and evocative.
ADJECTIVES = [
    "amber", "azure", "calm", "deep", "distant", "dreamy", "emerald",
    "frozen", "gentle", "golden", "hollow", "ivory", "lush", "misty",
    "moonlit", "neon", "oceanic", "quiet", "radiant", "silent", "silver",
    "soft", "stellar", "tranquil", "velvet", "wandering", "whispering",
    "wild", "windy", "winter"
]

NOUNS = [
    "aurora", "breeze", "canyon", "cave", "cloud", "coast", "crystal",
    "dawn", "desert", "dream", "echo", "fjord", "forest", "garden",
    "glacier", "grove", "harbor", "horizon", "island", "lagoon", "lake",
    "meadow", "mountain", "night", "ocean", "peak", "rain", "river",
    "sea", "sky", "star", "stream", "sunset", "temple", "valley",
    "waterfall", "wood"
]


def generate_phrase(seed: Optional[str] = None, length: int = 3) -> str:
    """
    Generate a deterministic phrase from a seed string.

    The phrase is derived by hashing the seed and using the hash to select
    words from the word lists. The same seed always produces the same phrase.

    :param seed: optional seed string; if None, a random phrase is generated
    :param length: number of words in the phrase (must be at least 2)
    :return: a phrase like "misty forest" or "silent ocean"
    """
    if length < 2:
        raise ValueError("Phrase length must be at least 2")

    if seed is None:
        # Random phrase: use system randomness
        rng = random.Random()
        words = [rng.choice(ADJECTIVES), rng.choice(NOUNS)]
        for _ in range(length - 2):
            words.append(rng.choice(NOUNS))
        return " ".join(words)

    # Deterministic: hash the seed to get a stable integer
    hash_digest = hashlib.sha256(seed.encode("utf-8")).digest()
    # Use the first 8 bytes as a 64-bit integer
    seed_int = int.from_bytes(hash_digest[:8], byteorder="big")
    rng = random.Random(seed_int)

    # Build the phrase: adjective, then nouns
    words = [rng.choice(ADJECTIVES), rng.choice(NOUNS)]
    for _ in range(length - 2):
        words.append(rng.choice(NOUNS))
    return " ".join(words)


def phrase_to_seed(phrase: str) -> int:
    """
    Convert a phrase to a deterministic integer seed for the WFC algorithm.

    :param phrase: any string
    :return: a 64-bit integer derived from the phrase
    """
    digest = hashlib.sha256(phrase.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def random_phrase() -> str:
    """Generate a random three-word phrase using system randomness."""
    return generate_phrase(None, length=3)
