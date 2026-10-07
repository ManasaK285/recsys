from __future__ import annotations

import itertools
import random
from typing import List, Dict

from .models import Taxonomy


def sample_factor_combinations(
    taxonomy: Taxonomy,
    n: int,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """
    Controlled breadth-first-ish sampling.

    We first enumerate all combinations when the Cartesian product is small,
    then shuffle deterministically and cycle if more samples are requested.
    """
    rng = random.Random(seed)
    factors = taxonomy.factors

    names = [f.name for f in factors]
    values = [f.values for f in factors]

    combinations = [
        dict(zip(names, combo))
        for combo in itertools.product(*values)
    ]

    rng.shuffle(combinations)

    result = []
    for i in range(n):
        result.append(combinations[i % len(combinations)].copy())
    return result
