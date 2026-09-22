"""Seed derivation. Never touches the global NumPy RNG."""

from __future__ import annotations

import numpy as np


def spawn_seeds(root: int, count: int) -> list[int]:
    """Return ``count`` independent child seeds spawned from ``root``."""

    if count <= 0:
        return []
    parent = np.random.SeedSequence(root)
    children = parent.spawn(count)
    return [int(child.generate_state(1)[0]) for child in children]


def generator(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))
