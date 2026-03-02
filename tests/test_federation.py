"""Tests for federation server and client."""

from __future__ import annotations

import collections

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.federation.client import ClientConfig, FederatedClient
from src.federation.models import FraudMLP, get_model_params
from src.federation.server import FederationServer
from src.federation.strategies import FedAvgStrategy


def _make_loader(n_samples: int = 100, n_features: int = 10, seed: int = 42):
    """Create a synthetic DataLoader for testing."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (rng.random(n_samples) > 0.9).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    # Add pos_weight attribute to match FraudDataset interface
    dataset.pos_weight = float((1 - y).sum() / max(y.sum(), 1))
    return DataLoader(dataset, batch_size=32, shuffle=True)


class TestFederatedClient:
    def test_local_training_produces_update(self):
        loader = _make_loader(n_features=10)
        config = ClientConfig(client_id="CA", local_epochs=1)
        client = FederatedClient(config, loader, model_kwargs={"hidden_dims": [16, 8]})

        global_params = get_model_params(client.model)
        update = client.train_local(global_params, round_num=1)

        assert update.client_id == "CA"
        assert update.round_num == 1
        assert update.num_samples == 100
        assert update.local_loss > 0
        # Delta should be non-zero
        total_delta = sum(np.abs(v).sum() for v in update.delta.values())
        assert total_delta > 0

    def test_fedprox_proximal_term(self):
        loader = _make_loader(n_features=10)
        config = ClientConfig(client_id="NY", local_epochs=2, fedprox_mu=0.1)
        client = FederatedClient(config, loader, model_kwargs={"hidden_dims": [16, 8]})

        global_params = get_model_params(client.model)
        update = client.train_local(global_params, round_num=1)
        # Should still produce a valid update
        assert update is not None
        assert update.local_loss > 0


class TestFederationServer:
    def test_run_round_aggregates(self):
        model = FraudMLP(input_dim=10, hidden_dims=[16, 8])
        strategy = FedAvgStrategy()
        server = FederationServer(model, strategy, num_rounds=5)

        clients = {}
        for state in ["CA", "NY", "TX"]:
            loader = _make_loader(n_features=10, seed=hash(state) % 1000)
            config = ClientConfig(client_id=state, local_epochs=1)
            clients[state] = FederatedClient(config, loader, model_kwargs={"hidden_dims": [16, 8]})

        result = server.run_round(1, clients)
        assert result.num_participating == 3
        assert result.round_num == 1

    def test_e2e_small_federation(self):
        """End-to-end smoke test: 3 clients, 5 rounds of FedAvg."""
        model = FraudMLP(input_dim=10, hidden_dims=[16, 8])
        strategy = FedAvgStrategy()
        server = FederationServer(model, strategy, num_rounds=5)

        clients = {}
        for state in ["CA", "NY", "TX"]:
            loader = _make_loader(n_features=10, seed=hash(state) % 1000)
            config = ClientConfig(client_id=state, local_epochs=2)
            clients[state] = FederatedClient(config, loader, model_kwargs={"hidden_dims": [16, 8]})

        history = server.run_training(clients)
        assert len(history) == 5

        # Global params should have changed
        final_params = server.get_global_params()
        initial_params = get_model_params(model)
        total_change = sum(
            np.abs(final_params[k] - initial_params[k]).sum()
            for k in initial_params
        )
        assert total_change > 0

    def test_client_selection(self):
        model = FraudMLP(input_dim=10, hidden_dims=[16, 8])
        strategy = FedAvgStrategy()
        server = FederationServer(
            model, strategy, participation_rate=0.5, seed=42
        )

        clients = {f"S{i}": None for i in range(10)}  # Dummy clients
        selected = server.select_clients(clients)
        assert len(selected) == 5
