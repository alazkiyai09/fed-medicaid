"""Model poisoning attack — gradient ascent to maximize loss.

Actively degrades the fraud detector by sending the negated update.
Tests the cosine similarity check in SignGuard.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from src.attacks.base import AttackBase


class ModelPoisonAttack(AttackBase):
    """Send negated update (gradient ascent) to maximize loss.

    Args:
        intensity: Scaling factor for the negated update.
        seed: Random seed.
    """

    def poison_update(
        self,
        honest_update: OrderedDict[str, np.ndarray],
        global_weights: OrderedDict[str, np.ndarray],
        round_num: int,
    ) -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            (k, (-self.intensity * v).astype(v.dtype))
            for k, v in honest_update.items()
        )
