"""Free rider attack — sends near-zero updates to extract model without contributing.

Represents a state that wants to benefit from the global model
without contributing meaningful training. Tests the reputation
system's zero-contribution detection.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from src.attacks.base import AttackBase


class FreeRiderAttack(AttackBase):
    """Send near-zero updates — extracts global model without contributing.

    Args:
        intensity: Scale of the noise (very small, ~0.001).
        seed: Random seed.
    """

    def __init__(self, intensity: float = 0.001, seed: int = 42):
        super().__init__(intensity=intensity, seed=seed)

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
