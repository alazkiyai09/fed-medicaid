"""Aggregation strategies for federated learning.

Implements 6 strategies:
- FedAvg: Standard weighted average by data size
- FedProx: Same aggregation as FedAvg (proximal term is client-side)
- Krum: Byzantine-resilient — selects update closest to neighbors
- TrimmedMean: Coordinate-wise trimmed mean
- FoolsGold: Penalizes similar updates (anti-Sybil)
- SignGuard: ECDSA verification + statistical validation + reputation-weighted aggregation
"""

from __future__ import annotations

import abc
import collections
import logging
from typing import OrderedDict

import numpy as np

from src.federation.client import ClientUpdate

logger = logging.getLogger(__name__)


class AggregationStrategy(abc.ABC):
    """Base class for federated aggregation strategies."""

    def __init__(self):
        self.last_accepted_count: int = 0

    @abc.abstractmethod
    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
        client_data_sizes: dict[str, int],
    ) -> OrderedDict[str, np.ndarray]:
        """Aggregate client updates into a single global delta.

        Args:
            updates: List of client updates.
            global_params: Current global model parameters.
            round_num: Current federation round.
            client_data_sizes: Mapping of client_id → local dataset size.

        Returns:
            Aggregated delta to apply to global model.
        """
        ...


# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

class FedAvgStrategy(AggregationStrategy):
    """Standard Federated Averaging — weighted by data size."""

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
        client_data_sizes: dict[str, int],
    ) -> OrderedDict[str, np.ndarray]:
        if not updates:
            raise ValueError("No updates to aggregate")

        total_samples = sum(u.num_samples for u in updates)
        self.last_accepted_count = len(updates)

        aggregated: OrderedDict[str, np.ndarray] = collections.OrderedDict()
        for name in updates[0].delta:
            weighted_sum = np.zeros_like(updates[0].delta[name], dtype=np.float64)
            for u in updates:
                weight = u.num_samples / total_samples
                weighted_sum += weight * u.delta[name].astype(np.float64)
            aggregated[name] = weighted_sum.astype(np.float32)

        return aggregated


class FedProxStrategy(FedAvgStrategy):
    """FedProx — same aggregation as FedAvg (proximal term is client-side)."""
    pass


# ---------------------------------------------------------------------------
# Krum
# ---------------------------------------------------------------------------

class KrumStrategy(AggregationStrategy):
    """Multi-Krum: selects the update(s) closest to their neighbors.

    Byzantine-resilient: tolerates up to f Byzantine clients, selects
    the update with smallest sum of distances to its n-f-2 closest peers.

    Args:
        num_byzantine: Expected number of Byzantine clients.
        multi_k: Number of updates to select (1 = Krum, >1 = Multi-Krum).
    """

    def __init__(self, num_byzantine: int = 0, multi_k: int = 1):
        super().__init__()
        self.num_byzantine = num_byzantine
        self.multi_k = multi_k

    def _flatten(self, delta: OrderedDict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([v.flatten() for v in delta.values()])

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
        client_data_sizes: dict[str, int],
    ) -> OrderedDict[str, np.ndarray]:
        n = len(updates)
        if n == 0:
            raise ValueError("No updates to aggregate")

        f = self.num_byzantine
        k = max(1, n - f - 2)  # Number of closest neighbors to consider

        # Flatten all deltas
        flat_updates = [self._flatten(u.delta) for u in updates]

        # Pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(flat_updates[i] - flat_updates[j])
                distances[i, j] = d
                distances[j, i] = d

        # Score: sum of k closest distances
        scores = np.zeros(n)
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            scores[i] = np.sum(sorted_dists[1:k + 1])  # Exclude self (0)

        # Select multi_k updates with lowest scores
        selected_indices = np.argsort(scores)[:self.multi_k]
        self.last_accepted_count = len(selected_indices)

        # Average selected updates
        aggregated: OrderedDict[str, np.ndarray] = collections.OrderedDict()
        for name in updates[0].delta:
            avg = np.mean(
                [updates[i].delta[name].astype(np.float64) for i in selected_indices],
                axis=0,
            )
            aggregated[name] = avg.astype(np.float32)

        selected_ids = [updates[i].client_id for i in selected_indices]
        logger.debug("Krum selected: %s (scores: %s)", selected_ids, scores[selected_indices])
        return aggregated


# ---------------------------------------------------------------------------
# Trimmed Mean
# ---------------------------------------------------------------------------

class TrimmedMeanStrategy(AggregationStrategy):
    """Coordinate-wise trimmed mean — trims extreme values per coordinate.

    For each parameter coordinate, sorts values across clients and removes
    the top and bottom trim_ratio fraction before averaging.

    Args:
        trim_ratio: Fraction to trim from each end (e.g., 0.1 = 10%).
    """

    def __init__(self, trim_ratio: float = 0.1):
        super().__init__()
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
        client_data_sizes: dict[str, int],
    ) -> OrderedDict[str, np.ndarray]:
        n = len(updates)
        if n == 0:
            raise ValueError("No updates to aggregate")

        trim_count = max(1, int(n * self.trim_ratio))
        self.last_accepted_count = n - 2 * trim_count

        aggregated: OrderedDict[str, np.ndarray] = collections.OrderedDict()
        for name in updates[0].delta:
            shape = updates[0].delta[name].shape
            # Stack all clients' values for this parameter
            stacked = np.stack(
                [u.delta[name].astype(np.float64).flatten() for u in updates]
            )  # (n, d)

            # Sort along client axis and trim
            sorted_vals = np.sort(stacked, axis=0)
            trimmed = sorted_vals[trim_count:n - trim_count]  # (n-2*trim, d)

            # Average the remaining
            mean = trimmed.mean(axis=0).reshape(shape)
            aggregated[name] = mean.astype(np.float32)

        return aggregated


# ---------------------------------------------------------------------------
# FoolsGold
# ---------------------------------------------------------------------------

class FoolsGoldStrategy(AggregationStrategy):
    """FoolsGold: penalizes clients with similar updates (anti-Sybil).

    Computes pairwise cosine similarity between client updates. Clients
    with highly correlated updates receive lower aggregation weights,
    defending against Sybil attacks.
    """

    def __init__(self):
        super().__init__()
        self._history: dict[str, np.ndarray] = {}

    def _flatten(self, delta: OrderedDict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([v.flatten() for v in delta.values()])

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
        client_data_sizes: dict[str, int],
    ) -> OrderedDict[str, np.ndarray]:
        n = len(updates)
        if n == 0:
            raise ValueError("No updates to aggregate")

        self.last_accepted_count = n

        # Track cumulative gradients per client
        for u in updates:
            flat = self._flatten(u.delta)
            if u.client_id in self._history:
                self._history[u.client_id] += flat
            else:
                self._history[u.client_id] = flat.copy()

        # Build contribution matrix from history
        client_ids = [u.client_id for u in updates]
        contributions = np.stack([self._history[cid] for cid in client_ids])

        # Pairwise cosine similarity
        norms = np.linalg.norm(contributions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = contributions / norms
        similarity = normalized @ normalized.T

        # FoolsGold weight computation
        # Each client's weight is inversely related to its max similarity with others
        weights = np.ones(n)
        for i in range(n):
            # Max similarity with any other client (excluding self)
            similarities_i = similarity[i].copy()
            similarities_i[i] = -1.0  # Exclude self
            max_sim = np.max(similarities_i)
            if max_sim > 0:
                weights[i] = 1.0 - max_sim
            else:
                weights[i] = 1.0

        # Normalize weights
        weights = np.maximum(weights, 0.0)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weight_sum
        else:
            weights = np.ones(n) / n

        # Weighted average
        aggregated: OrderedDict[str, np.ndarray] = collections.OrderedDict()
        for name in updates[0].delta:
            weighted_sum = np.zeros_like(updates[0].delta[name], dtype=np.float64)
            for i, u in enumerate(updates):
                weighted_sum += weights[i] * u.delta[name].astype(np.float64)
            aggregated[name] = weighted_sum.astype(np.float32)

        logger.debug(
            "FoolsGold weights: %s",
            {cid: f"{w:.3f}" for cid, w in zip(client_ids, weights)},
        )
        return aggregated


# ---------------------------------------------------------------------------
# SignGuard
# ---------------------------------------------------------------------------

class SignGuardStrategy(AggregationStrategy):
    """SignGuard: ECDSA verification + statistical validation + reputation-weighted aggregation.

    Four-phase pipeline:
    1. Cryptographic verification (ECDSA signature check)
    2. Statistical validation (L2 norm + cosine similarity)
    3. Reputation-weighted aggregation
    4. Reputation update

    Args:
        verifier: SignGuard UpdateVerifier instance.
        validator: SignGuard StatisticalValidator instance.
        reputation: SignGuard ReputationManager instance.
    """

    def __init__(self, verifier=None, validator=None, reputation=None):
        super().__init__()
        self.verifier = verifier
        self.validator = validator
        self.reputation = reputation

    def _flatten(self, delta: OrderedDict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([v.flatten() for v in delta.values()])

    def aggregate(
        self,
        updates: list[ClientUpdate],
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
        client_data_sizes: dict[str, int],
    ) -> OrderedDict[str, np.ndarray]:
        if not updates:
            raise ValueError("No updates to aggregate")

        # --- Phase 1: Cryptographic verification ---
        if self.verifier is not None:
            crypto_valid = []
            for u in updates:
                sig = u.metadata.get("signature")
                update_hash = u.metadata.get("update_hash")
                if sig is not None and update_hash is not None:
                    if self.verifier.verify_signature(u.client_id, u.delta, sig, update_hash):
                        crypto_valid.append(u)
                    else:
                        logger.warning("Round %d: ECDSA failed for %s", round_num, u.client_id)
                        if self.reputation:
                            self.reputation.update_reputation(
                                u.client_id, round_num,
                                signature_ok=False, validation_ok=False, performance_impact=0.0,
                            )
                else:
                    # No signature — skip crypto check (testing mode)
                    crypto_valid.append(u)
        else:
            crypto_valid = list(updates)

        # --- Phase 2: Statistical validation ---
        if self.validator is not None:
            all_norms = [
                np.linalg.norm(self._flatten(u.delta)) for u in crypto_valid
            ]
            global_flat = self._flatten(global_params)

            stat_valid = []
            stat_failed = []
            for u in crypto_valid:
                result = self.validator.validate(u.delta, global_params, all_norms)
                if result.is_valid:
                    stat_valid.append(u)
                else:
                    stat_failed.append(u)
                    logger.info(
                        "Round %d: Statistical validation failed for %s: %s",
                        round_num, u.client_id, result.failures,
                    )
        else:
            stat_valid = crypto_valid
            stat_failed = []

        # --- Phase 3: Reputation-weighted aggregation ---
        if not stat_valid:
            logger.warning("Round %d: No valid updates after filtering!", round_num)
            self.last_accepted_count = 0
            return collections.OrderedDict(
                (name, np.zeros_like(val)) for name, val in global_params.items()
            )

        self.last_accepted_count = len(stat_valid)

        total_data = sum(
            client_data_sizes.get(u.client_id, 1) for u in stat_valid
        )

        aggregated: OrderedDict[str, np.ndarray] = collections.OrderedDict()
        total_weight = 0.0

        for name in stat_valid[0].delta:
            weighted_sum = np.zeros_like(stat_valid[0].delta[name], dtype=np.float64)

            for u in stat_valid:
                if self.reputation is not None:
                    w = self.reputation.get_aggregation_weight(
                        u.client_id,
                        client_data_sizes.get(u.client_id, 1),
                        total_data,
                    )
                else:
                    w = client_data_sizes.get(u.client_id, 1) / total_data

                weighted_sum += w * u.delta[name].astype(np.float64)
                if name == list(stat_valid[0].delta.keys())[0]:
                    total_weight += w

            aggregated[name] = weighted_sum.astype(np.float32)

        # Normalize
        if total_weight > 0:
            aggregated = collections.OrderedDict(
                (name, val / total_weight) for name, val in aggregated.items()
            )

        # --- Phase 4: Update reputations ---
        if self.reputation is not None:
            for u in stat_valid:
                self.reputation.update_reputation(
                    u.client_id, round_num,
                    signature_ok=True, validation_ok=True,
                    performance_impact=0.0,
                )
            for u in stat_failed:
                self.reputation.update_reputation(
                    u.client_id, round_num,
                    signature_ok=True, validation_ok=False,
                    performance_impact=-0.01,
                )

        logger.debug(
            "Round %d SignGuard: %d/%d/%d (crypto/stat/total) accepted",
            round_num, len(crypto_valid), len(stat_valid), len(updates),
        )
        return aggregated
