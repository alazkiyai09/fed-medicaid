"""Privacy accountant — tracks cumulative privacy budget across rounds.

Implements advanced composition theorem for tracking total (ε, δ)-DP
budget consumed across multiple rounds of federated learning.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PrivacySpent:
    """Cumulative privacy budget consumed."""
    epsilon: float = 0.0
    delta: float = 0.0
    rounds: int = 0


class PrivacyAccountant:
    """Tracks cumulative privacy budget across federation rounds.

    Supports two composition modes:
    - Basic composition: ε_total = k × ε_per_round
    - Advanced composition: ε_total = √(2k ln(1/δ')) × ε + k × ε²

    Args:
        total_epsilon: Maximum total privacy budget.
        total_delta: Maximum total failure probability.
        per_round_epsilon: Per-round privacy budget.
        per_round_delta: Per-round failure probability.
        composition: "basic" or "advanced".
    """

    def __init__(
        self,
        total_epsilon: float = 10.0,
        total_delta: float = 1e-5,
        per_round_epsilon: float = 0.1,
        per_round_delta: float = 1e-7,
        composition: str = "advanced",
    ):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.per_round_epsilon = per_round_epsilon
        self.per_round_delta = per_round_delta
        self.composition = composition
        self._rounds_consumed: int = 0

    def step(self) -> PrivacySpent:
        """Record one round of privacy spending.

        Returns:
            Current cumulative privacy spent.

        Raises:
            RuntimeError: If privacy budget is exhausted.
        """
        self._rounds_consumed += 1
        spent = self.get_privacy_spent()

        if spent.epsilon > self.total_epsilon:
            raise RuntimeError(
                f"Privacy budget exhausted: ε={spent.epsilon:.4f} > {self.total_epsilon}"
            )

        if self._rounds_consumed % 10 == 0:
            logger.info(
                "Privacy accountant: %d rounds, ε=%.4f/%.1f",
                self._rounds_consumed, spent.epsilon, self.total_epsilon,
            )

        return spent

    def get_privacy_spent(self) -> PrivacySpent:
        """Compute cumulative privacy spent using chosen composition theorem.

        Returns:
            PrivacySpent with total epsilon and delta.
        """
        k = self._rounds_consumed
        eps = self.per_round_epsilon
        delta = self.per_round_delta

        if self.composition == "basic":
            total_eps = k * eps
            total_delta = k * delta
        elif self.composition == "advanced":
            # Advanced composition theorem (Dwork et al. 2010)
            # ε_total ≤ √(2k ln(1/δ')) × ε + k × ε × (e^ε - 1)
            delta_prime = self.total_delta / 2
            if delta_prime > 0:
                total_eps = (
                    math.sqrt(2 * k * math.log(1 / delta_prime)) * eps
                    + k * eps * (math.exp(eps) - 1)
                )
            else:
                total_eps = k * eps
            total_delta = k * delta + self.total_delta / 2
        else:
            raise ValueError(f"Unknown composition: {self.composition}")

        return PrivacySpent(
            epsilon=total_eps,
            delta=total_delta,
            rounds=k,
        )

    def remaining_rounds(self) -> int | None:
        """Estimate how many rounds remain before budget exhaustion.

        Returns:
            Estimated rounds remaining, or None if cannot estimate.
        """
        if self.per_round_epsilon <= 0:
            return None

        # Binary search for max rounds
        lo, hi = self._rounds_consumed, 10000
        saved = self._rounds_consumed

        while lo < hi:
            mid = (lo + hi + 1) // 2
            self._rounds_consumed = mid
            spent = self.get_privacy_spent()
            if spent.epsilon <= self.total_epsilon:
                lo = mid
            else:
                hi = mid - 1

        self._rounds_consumed = saved
        return max(0, lo - self._rounds_consumed)

    @property
    def is_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.get_privacy_spent().epsilon > self.total_epsilon
