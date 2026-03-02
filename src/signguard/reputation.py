"""EMA-based reputation system for SignGuard protocol.

Tracks per-client trustworthiness over federation rounds using
Exponential Moving Average (EMA) of round behavior scores.
Controls aggregation weights to reduce malicious influence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReputationRecord:
    """History record of a client's reputation change."""
    round_num: int
    old_reputation: float
    new_reputation: float
    signature_ok: bool
    validation_ok: bool
    performance_impact: float


class ReputationManager:
    """EMA-based reputation tracking per client.

    Reputation update rule:
        reputation_{t+1} = α × round_score_t + (1-α) × reputation_t

    Aggregation weight rules (spec §6.6):
    - rep > 0.8  → full weight (trustworthy)
    - 0.5 < rep < 0.8 → linearly reduced weight
    - rep < 0.5  → weight = 0 (excluded from aggregation)
    - rep < 0.2  → flagged for investigation

    Args:
        alpha: EMA smoothing factor (default: 0.3).
        initial_reputation: Starting reputation for new clients (default: 1.0).
        exclusion_threshold: Reputation below this → excluded (default: 0.5).
        full_weight_threshold: Reputation above this → full weight (default: 0.8).
        investigation_threshold: Reputation below this → flagged (default: 0.2).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        initial_reputation: float = 1.0,
        exclusion_threshold: float = 0.5,
        full_weight_threshold: float = 0.8,
        investigation_threshold: float = 0.2,
    ):
        self.alpha = alpha
        self.initial_reputation = initial_reputation
        self.exclusion_threshold = exclusion_threshold
        self.full_weight_threshold = full_weight_threshold
        self.investigation_threshold = investigation_threshold

        self.reputation: dict[str, float] = {}
        self.history: dict[str, list[ReputationRecord]] = {}
        self.flagged_clients: set[str] = set()

    def get_reputation(self, client_id: str) -> float:
        """Get current reputation for a client.

        Returns initial_reputation for unknown clients.
        """
        return self.reputation.get(client_id, self.initial_reputation)

    def update_reputation(
        self,
        client_id: str,
        round_num: int,
        signature_ok: bool,
        validation_ok: bool,
        performance_impact: float,
    ) -> float:
        """Update a client's reputation based on round behavior.

        Args:
            client_id: Client identifier.
            round_num: Current federation round.
            signature_ok: Whether ECDSA signature was valid.
            validation_ok: Whether statistical validation passed.
            performance_impact: Change in global AUPRC attributed to this client.

        Returns:
            New reputation value.
        """
        old_rep = self.get_reputation(client_id)

        if not signature_ok:
            # Immediate zero — crypto failure is definitive
            new_rep = 0.0
        else:
            round_score = old_rep

            if not validation_ok:
                round_score -= 0.3

            if performance_impact < -0.01:
                round_score -= 0.15
            elif performance_impact > 0.005:
                round_score += 0.1

            round_score = max(0.0, min(1.0, round_score))
            new_rep = self.alpha * round_score + (1 - self.alpha) * old_rep

        self.reputation[client_id] = new_rep

        # Track history
        record = ReputationRecord(
            round_num=round_num,
            old_reputation=old_rep,
            new_reputation=new_rep,
            signature_ok=signature_ok,
            validation_ok=validation_ok,
            performance_impact=performance_impact,
        )
        self.history.setdefault(client_id, []).append(record)

        # Flag for investigation
        if new_rep < self.investigation_threshold:
            if client_id not in self.flagged_clients:
                logger.warning(
                    "Client %s flagged for investigation (rep=%.3f)",
                    client_id, new_rep,
                )
                self.flagged_clients.add(client_id)

        return new_rep

    def get_aggregation_weight(
        self,
        client_id: str,
        data_size: int,
        total_data: int,
    ) -> float:
        """Compute the aggregation weight for a client.

        Weight combines reputation and data size:
        - rep > full_weight_threshold → full data-proportional weight
        - exclusion < rep < full_weight_threshold → linearly reduced
        - rep < exclusion_threshold → 0

        Args:
            client_id: Client identifier.
            data_size: Number of samples from this client.
            total_data: Total samples across all clients.

        Returns:
            Aggregation weight (unnormalized — caller should normalize).
        """
        rep = self.get_reputation(client_id)
        data_weight = data_size / max(total_data, 1)

        if rep < self.exclusion_threshold:
            return 0.0
        elif rep >= self.full_weight_threshold:
            reputation_factor = 1.0
        else:
            # Linear interpolation between exclusion and full weight
            reputation_factor = (rep - self.exclusion_threshold) / (
                self.full_weight_threshold - self.exclusion_threshold
            )

        return data_weight * reputation_factor

    def is_excluded(self, client_id: str) -> bool:
        """Check if a client is excluded from aggregation."""
        return self.get_reputation(client_id) < self.exclusion_threshold

    def is_flagged(self, client_id: str) -> bool:
        """Check if a client is flagged for investigation."""
        return client_id in self.flagged_clients

    def get_all_reputations(self) -> dict[str, float]:
        """Return all current client reputations."""
        return dict(self.reputation)

    def get_reputation_history(self, client_id: str) -> list[ReputationRecord]:
        """Return reputation history for a specific client."""
        return self.history.get(client_id, [])

    def reset(self) -> None:
        """Reset all reputation data."""
        self.reputation.clear()
        self.history.clear()
        self.flagged_clients.clear()
