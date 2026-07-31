"""Generate random seed phrases for soundscape synthesis."""

import random
from typing import List, Optional

ADJECTIVES = [
    "misty", "golden", "silent", "deep", "ancient", "crimson", "azure",
    "gentle", "wild", "luminous", "hollow", "shimmering", "vast", "quiet",
    "frozen", "molten", "verdant", "stellar", "lunar", "solar"
]

NOUNS = [
    "forest", "ocean", "space", "meadow", "canyon", "glacier", "desert",
    "jungle", "tundra", "reef", "nebula", "valley", "peak", "river",
    "dune", "grove", "cavern", "lagoon", "prairie", "horizon"
]


def generate_seed_phrase(
    num_words: int = 2,
    separator: str = " ",
    rng: Optional[random.Random] = None
) -> str:
    """
    Generate a random seed phrase from curated word lists.

    Args:
        num_words: Number of words in the phrase (must be >= 1).
        separator: String to join words with.
        rng: Optional random.Random instance for deterministic generation.

    Returns:
        A seed phrase like "misty forest" or "deep space".

    Raises:
        ValueError: If num_words is less than 1.
    """
    if num_words < 1:
        raise ValueError("num_words must be at least 1")

    if rng is None:
        rng = random

    words: List[str] = []
    for i in range(num_words):
        if i % 2 == 0:
            # Even index (0, 2, ...) -> adjective
            words.append(rng.choice(ADJECTIVES))
        else:
            # Odd index (1, 3, ...) -> noun
            words.append(rng.choice(NOUNS))

    return separator.join(words)


def generate_biome_specific_seed_phrase(
    biome: str,
    rng: Optional[random.Random] = None
) -> str:
    """
    Generate a seed phrase that fits a given biome's theme.

    Args:
        biome: Biome name (e.g., 'forest', 'ocean', 'space').
        rng: Optional random.Random instance.

    Returns:
        A seed phrase with at least one word related to the biome.
    """
    if rng is None:
        rng = random

    biome_nouns = {
        "forest": ["forest", "grove", "meadow"],
        "ocean": ["ocean", "reef", "lagoon"],
        "space": ["space", "nebula", "stellar"],
    }

    nouns = biome_nouns.get(biome, NOUNS)
    adjective = rng.choice(ADJECTIVES)
    noun = rng.choice(nouns)
    return f"{adjective} {noun}