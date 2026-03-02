"""Federated learning client — local training and update computation.

Each client represents a US state with its own local data partition.
Handles local model training, update delta computation, optional
SignGuard signing, and optional differential privacy noise.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.federation.models import (
    FraudMLP,
    compute_update_delta,
    get_model_params,
    set_model_params,
)

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for a federated client."""
    client_id: str
    local_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 256
    fedprox_mu: float = 0.0  # 0 = FedAvg, >0 = FedProx
    device: str = "cpu"


@dataclass
class ClientUpdate:
    """Result of local training sent to the server."""
    client_id: str
    round_num: int
    delta: OrderedDict[str, np.ndarray]
    num_samples: int
    local_loss: float
    metadata: dict = field(default_factory=dict)


class FederatedClient:
    """Handles local model training for a single federated client (state).

    Args:
        config: Client configuration.
        train_loader: DataLoader for local training data.
        val_loader: Optional DataLoader for local validation.
        signer: Optional SignGuard UpdateSigner for cryptographic signing.
        dp_mechanism: Optional differential privacy mechanism.
    """

    def __init__(
        self,
        config: ClientConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        signer=None,
        dp_mechanism=None,
        model_kwargs: dict | None = None,
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.signer = signer
        self.dp_mechanism = dp_mechanism
        self.device = torch.device(config.device)

        # Infer input dimension from data
        sample_x, _ = next(iter(train_loader))
        input_dim = sample_x.shape[1]
        kwargs = {"input_dim": input_dim}
        if model_kwargs:
            kwargs.update(model_kwargs)
        self.model = FraudMLP(**kwargs).to(self.device)
        self.num_samples = len(train_loader.dataset)

    def train_local(
        self,
        global_params: OrderedDict[str, np.ndarray],
        round_num: int,
    ) -> ClientUpdate:
        """Perform local training starting from global model weights.

        1. Load global weights.
        2. Train for E local epochs.
        3. Compute update delta (Δ = W_local - W_global).
        4. Optionally apply DP noise.
        5. Optionally sign with ECDSA.

        Args:
            global_params: Current global model parameters.
            round_num: Current federation round number.

        Returns:
            ClientUpdate with the delta and metadata.
        """
        # Set model to global weights
        set_model_params(self.model, global_params)
        global_params_copy = collections.OrderedDict(
            (k, v.copy()) for k, v in global_params.items()
        )

        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Binary cross-entropy with class imbalance weighting
        pos_weight = torch.tensor(
            [self.train_loader.dataset.pos_weight], device=self.device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)

                # FedProx proximal term
                if self.config.fedprox_mu > 0:
                    proximal_loss = 0.0
                    for name, param in self.model.named_parameters():
                        target = torch.from_numpy(global_params_copy[name]).to(self.device)
                        proximal_loss += ((param - target) ** 2).sum()
                    loss += (self.config.fedprox_mu / 2) * proximal_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(num_batches, 1)

        # Compute delta
        new_params = get_model_params(self.model)
        delta = compute_update_delta(new_params, global_params_copy)

        # Apply DP if configured
        if self.dp_mechanism is not None:
            delta = self.dp_mechanism.clip_and_noise(delta)

        logger.debug(
            "Client %s round %d: loss=%.4f, samples=%d",
            self.config.client_id, round_num, avg_loss, self.num_samples,
        )

        update = ClientUpdate(
            client_id=self.config.client_id,
            round_num=round_num,
            delta=delta,
            num_samples=self.num_samples,
            local_loss=avg_loss,
        )

        # Sign if signer is available
        if self.signer is not None:
            signed = self.signer.sign_update(delta, round_num, {"loss": avg_loss})
            update.metadata["signature"] = signed.signature
            update.metadata["update_hash"] = signed.update_hash

        return update

    def evaluate_local(
        self,
        global_params: OrderedDict[str, np.ndarray],
    ) -> dict[str, float]:
        """Evaluate global model on local validation data.

        Args:
            global_params: Global model parameters to evaluate.

        Returns:
            Dictionary with loss and accuracy metrics.
        """
        if self.val_loader is None:
            return {}

        set_model_params(self.model, global_params)
        self.model.eval()

        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

        return {
            "val_loss": total_loss / max(total, 1),
            "val_accuracy": correct / max(total, 1),
        }
