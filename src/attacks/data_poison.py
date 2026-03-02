"""Label flipping attack — flips fraud labels before local training.

The most subtle attack: represents corrupt officials protecting
fraudulent providers. Tests the reputation system's ability to
detect gradual performance degradation.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from src.attacks.base import AttackBase


class LabelFlipAttack(AttackBase):
    """Flip fraud labels in local data before honest training.

    This is a data-poisoning attack (pre-training), not a
    model-poisoning attack (post-training).

    Args:
        intensity: Fraction of labels to flip (0.0 to 1.0).
        flip_ratio: Additional control over flip proportion.
        seed: Random seed.
    """

    def __init__(self, intensity: float = 1.0, flip_ratio: float = 0.5, seed: int = 42):
        super().__init__(intensity=intensity, seed=seed)
        self.flip_ratio = flip_ratio

    def poison_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flip a fraction of fraud labels.

        Args:
            X: Feature matrix (unchanged).
            y: Binary label vector (0 = clean, 1 = fraud).

        Returns:
            Tuple of (X, poisoned_y).
        """
        n_flip = int(len(y) * self.flip_ratio * self.intensity)
        n_flip = min(n_flip, len(y))

        flip_idx = self.rng.choice(len(y), n_flip, replace=False)
        y_poisoned = y.copy()
        y_poisoned[flip_idx] = 1 - y_poisoned[flip_idx]

        return X, y_poisoned
