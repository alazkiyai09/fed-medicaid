"""Gaussian mechanism for local differential privacy.

Implements per-update gradient clipping and calibrated Gaussian noise
addition to provide (ε, δ)-differential privacy guarantees.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class GaussianMechanism:
    """Gaussian DP mechanism for clipping and noising model updates.

    Provides (ε, δ)-DP by:
    1. Clipping the L2 norm of the update to max_grad_norm.
    2. Adding Gaussian noise calibrated to the sensitivity and privacy budget.

    Noise multiplier: σ = max_grad_norm × √(2 ln(1.25/δ)) / ε

    Args:
        epsilon: Privacy budget per round.
        delta: Failure probability (default: 1e-5).
        max_grad_norm: Maximum L2 norm for gradient clipping (default: 1.0).
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm

        if math.isinf(epsilon):
            self.noise_multiplier = 0.0
        else:
            self.noise_multiplier = (
                max_grad_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            )

        logger.info(
            "GaussianMechanism: ε=%.2f, δ=%.1e, clip=%.2f, σ=%.4f",
            epsilon, delta, max_grad_norm, self.noise_multiplier,
        )

    def clip_and_noise(
        self,
        update: OrderedDict[str, np.ndarray],
    ) -> OrderedDict[str, np.ndarray]:
        """Clip update norm and add calibrated Gaussian noise.

        1. Flatten the update and compute L2 norm.
        2. Clip if norm exceeds max_grad_norm.
        3. Add Gaussian noise with σ = noise_multiplier × max_grad_norm.

        Args:
            update: Model update delta.

        Returns:
            Clipped and noised update delta.
        """
        if self.noise_multiplier == 0.0:
            return update  # No DP (ε = ∞)

        # Compute global L2 norm
        flat = np.concatenate([v.flatten() for v in update.values()])
        global_norm = np.linalg.norm(flat)

        # Clip factor
        clip_factor = min(1.0, self.max_grad_norm / max(global_norm, 1e-10))

        # Apply clipping + noise
        noise_std = self.noise_multiplier * self.max_grad_norm
        noised_update = OrderedDict()

        for k, v in update.items():
            clipped = v * clip_factor
            noise = np.random.normal(0, noise_std, v.shape).astype(v.dtype)
            noised_update[k] = clipped + noise

        logger.debug(
            "DP: norm=%.4f → clipped=%.4f, noise_std=%.4f",
            global_norm, global_norm * clip_factor, noise_std,
        )

        return noised_update

    @property
    def per_round_epsilon(self) -> float:
        """Per-round epsilon (same as configured epsilon for this simple mechanism)."""
        return self.epsilon
