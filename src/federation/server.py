"""Federation server — global model management and round orchestration.

Manages the global model, broadcasts weights to clients, collects updates,
delegates aggregation to the chosen strategy, and tracks per-round metrics.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import OrderedDict

import numpy as np

from src.federation.client import ClientUpdate, FederatedClient
from src.federation.models import FraudMLP, apply_update_delta, get_model_params

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result of a single federation round."""
    round_num: int
    global_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    num_participating: int = 0
    num_accepted: int = 0
    duration_seconds: float = 0.0
    client_losses: dict[str, float] = field(default_factory=dict)


class FederationServer:
    """Coordinates federated learning across state clients.

    Args:
        model: Global model instance.
        strategy: Aggregation strategy (FedAvg, Krum, SignGuard, etc.).
        num_rounds: Total number of federation rounds.
        participation_rate: Fraction of clients participating per round.
        seed: Random seed for client sampling.
    """

    def __init__(
        self,
        model: FraudMLP,
        strategy,
        num_rounds: int = 100,
        participation_rate: float = 1.0,
        seed: int = 42,
    ):
        self.model = model
        self.strategy = strategy
        self.num_rounds = num_rounds
        self.participation_rate = participation_rate
        self.rng = np.random.RandomState(seed)

        self.global_params = get_model_params(model)
        self.history: list[RoundResult] = []

    def select_clients(
        self, clients: dict[str, FederatedClient],
    ) -> list[str]:
        """Select a random subset of clients for this round.

        Args:
            clients: All available clients.

        Returns:
            List of selected client IDs.
        """
        all_ids = sorted(clients.keys())
        n_select = max(1, int(len(all_ids) * self.participation_rate))
        selected = self.rng.choice(all_ids, size=n_select, replace=False).tolist()
        return selected

    def run_round(
        self,
        round_num: int,
        clients: dict[str, FederatedClient],
        attack_updates: list[ClientUpdate] | None = None,
    ) -> RoundResult:
        """Execute a single federation round.

        1. Select participating clients.
        2. Broadcast global weights.
        3. Collect local updates (+ any attack updates).
        4. Aggregate using the strategy.
        5. Update global model.

        Args:
            round_num: Current round number.
            clients: All available federated clients.
            attack_updates: Optional list of malicious updates to inject.

        Returns:
            RoundResult with metrics for this round.
        """
        start_time = time.time()

        # 1. Select clients
        selected_ids = self.select_clients(clients)
        logger.info("Round %d: %d / %d clients selected", round_num, len(selected_ids), len(clients))

        # 2-3. Collect updates from selected honest clients
        updates: list[ClientUpdate] = []
        client_data_sizes: dict[str, int] = {}

        for cid in selected_ids:
            client = clients[cid]
            update = client.train_local(self.global_params, round_num)
            updates.append(update)
            client_data_sizes[cid] = update.num_samples

        # Inject attack updates if any
        if attack_updates:
            updates.extend(attack_updates)
            for au in attack_updates:
                client_data_sizes[au.client_id] = au.num_samples

        # 4. Aggregate
        aggregated_delta = self.strategy.aggregate(
            updates=updates,
            global_params=self.global_params,
            round_num=round_num,
            client_data_sizes=client_data_sizes,
        )

        # 5. Update global model
        self.global_params = apply_update_delta(self.global_params, aggregated_delta)

        duration = time.time() - start_time

        result = RoundResult(
            round_num=round_num,
            num_participating=len(updates),
            num_accepted=self.strategy.last_accepted_count
            if hasattr(self.strategy, "last_accepted_count") else len(updates),
            duration_seconds=duration,
            client_losses={u.client_id: u.local_loss for u in updates},
        )

        self.history.append(result)
        logger.info(
            "Round %d complete: %d accepted, %.2fs",
            round_num, result.num_accepted, duration,
        )
        return result

    def run_training(
        self,
        clients: dict[str, FederatedClient],
        attack_updates_fn=None,
        eval_fn=None,
        log_every: int = 5,
    ) -> list[RoundResult]:
        """Run the full federation training loop.

        Args:
            clients: All federated clients.
            attack_updates_fn: Optional callable(round_num, global_params) → list[ClientUpdate].
            eval_fn: Optional callable(round_num, global_params) → dict[str, float].
            log_every: Log metrics every N rounds.

        Returns:
            List of RoundResults for all rounds.
        """
        logger.info("Starting federation: %d rounds, %d clients", self.num_rounds, len(clients))

        for r in range(1, self.num_rounds + 1):
            attack_updates = None
            if attack_updates_fn is not None:
                attack_updates = attack_updates_fn(r, self.global_params)

            result = self.run_round(r, clients, attack_updates)

            # Evaluate if callback provided
            if eval_fn is not None and r % log_every == 0:
                metrics = eval_fn(r, self.global_params)
                result.metrics.update(metrics)
                logger.info("Round %d metrics: %s", r, metrics)

        logger.info("Training complete: %d rounds", self.num_rounds)
        return self.history

    def get_global_params(self) -> OrderedDict[str, np.ndarray]:
        """Return current global model parameters."""
        return collections.OrderedDict(
            (k, v.copy()) for k, v in self.global_params.items()
        )
