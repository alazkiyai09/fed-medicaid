"""Random noise attack — replaces update with Gaussian noise.

Simplest attack: represents a compromised system sending garbage.
Tests the L2 norm check in SignGuard's statistical validation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from src.attacks.base import AttackBase


class RandomPoisonAttack(AttackBase):
    """Replace model update with random Gaussian noise.

    Args:
        intensity: Standard deviation of the noise. Higher = more obvious.
        seed: Random seed.
    """

    def poison_update(
        self,
        honest_update: OrderedDict[str, np.ndarray],
        global_weights: OrderedDict[str, np.ndarray],
        round_num: int,
    ) -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            (k, self.rng.normal(0, self.intensity, v.shape).astype(v.dtype))
            for k, v in honest_update.items()
        )
