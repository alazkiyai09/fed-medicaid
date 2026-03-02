"""Base class for Byzantine attack simulations.

Provides abstract interface for poisoning model updates and training data,
used to test the robustness of different aggregation strategies.
"""

from __future__ import annotations

import abc
from collections import OrderedDict

import numpy as np


class AttackBase(abc.ABC):
    """Abstract base class for Byzantine attacks.

    Args:
        intensity: Attack intensity factor (0.0 to 1.0+).
            Interpretation varies by attack type.
        seed: Random seed for reproducibility.
    """

    def __init__(self, intensity: float = 1.0, seed: int = 42):
        self.intensity = intensity
        self.rng = np.random.RandomState(seed)

    def poison_update(
        self,
        honest_update: OrderedDict[str, np.ndarray],
        global_weights: OrderedDict[str, np.ndarray],
        round_num: int,
    ) -> OrderedDict[str, np.ndarray]:
        """Poison a model update (post-training attack).

        Default: returns honest update unchanged. Override in subclasses.

        Args:
            honest_update: The genuine update from local training.
            global_weights: Current global model parameters.
            round_num: Current federation round.

        Returns:
            Poisoned update delta.
        """
        return honest_update

    def poison_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Poison training data (pre-training attack).

        Default: returns data unchanged. Override in subclasses.

        Args:
            X: Feature matrix.
            y: Label vector.

        Returns:
            Tuple of (poisoned_X, poisoned_y).
        """
        return X, y
