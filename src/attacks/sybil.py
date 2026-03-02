"""Sybil attack — multiple fake identities with coordinated malicious updates.

Represents an organized fraud ring compromising multiple states.
Tests reputation + FoolsGold-like similarity detection.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from src.attacks.base import AttackBase
from src.attacks.model_poison import ModelPoisonAttack


class SybilAttack(AttackBase):
    """Multiple fake clients sending coordinated malicious updates.

    Generates N Sybil identities, each sending a slightly perturbed
    version of a base malicious update (to evade simple deduplication).

    Args:
        num_sybils: Number of fake identities to create.
        base_attack: Underlying attack to use for poisoning (default: ModelPoisonAttack).
        intensity: Intensity for the base attack.
        seed: Random seed.
    """

    def __init__(
        self,
        num_sybils: int = 5,
        base_attack: AttackBase | None = None,
        intensity: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(intensity=intensity, seed=seed)
        self.num_sybils = num_sybils
        self.base_attack = base_attack or ModelPoisonAttack(intensity=intensity, seed=seed)

    def generate_sybil_updates(
        self,
        honest_update: OrderedDict[str, np.ndarray],
        global_weights: OrderedDict[str, np.ndarray],
        round_num: int,
    ) -> list[OrderedDict[str, np.ndarray]]:
        """Generate coordinated updates from multiple fake identities.

        Each Sybil sends a slightly different version of the same
        poisoned update (with tiny noise to evade exact-match detection).

        Args:
            honest_update: A genuine update to base the poison on.
            global_weights: Current global model parameters.
            round_num: Current federation round.

        Returns:
            List of N Sybil updates.
        """
        base_poison = self.base_attack.poison_update(
            honest_update, global_weights, round_num
        )

        sybil_updates = []
        for _ in range(self.num_sybils):
            perturbed = OrderedDict(
                (k, v + self.rng.normal(0, 1e-6, v.shape).astype(v.dtype))
                for k, v in base_poison.items()
            )
            sybil_updates.append(perturbed)

        return sybil_updates

    def poison_update(
        self,
        honest_update: OrderedDict[str, np.ndarray],
        global_weights: OrderedDict[str, np.ndarray],
        round_num: int,
    ) -> OrderedDict[str, np.ndarray]:
        """Return a single Sybil update (use generate_sybil_updates for multiple)."""
        return self.base_attack.poison_update(honest_update, global_weights, round_num)
