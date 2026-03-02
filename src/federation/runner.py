"""Experiment runner — orchestrates end-to-end federation experiments.

Loads configuration, sets up clients, server, and strategy, then
runs the federation training loop with optional attacks and DP.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import OrderedDict

import numpy as np
import yaml

from src.federation.models import FraudMLP, get_model_params, set_model_params

logger = logging.getLogger(__name__)


def load_experiment_config(
    base_path: str | Path = "configs/base.yaml",
    experiment_path: str | Path | None = None,
) -> dict:
    """Load and merge base + experiment-specific configuration.

    Args:
        base_path: Path to base config.
        experiment_path: Optional experiment-specific config to overlay.

    Returns:
        Merged configuration dictionary.
    """
    with open(base_path) as f:
        config = yaml.safe_load(f)

    if experiment_path is not None:
        with open(experiment_path) as f:
            exp_config = yaml.safe_load(f)
        config = _deep_merge(config, exp_config)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def create_strategy(config: dict):
    """Create an aggregation strategy from config.

    Args:
        config: Full configuration dictionary.

    Returns:
        An AggregationStrategy instance.
    """
    from src.federation.strategies import (
        FedAvgStrategy,
        FedProxStrategy,
        FoolsGoldStrategy,
        KrumStrategy,
        SignGuardStrategy,
        TrimmedMeanStrategy,
    )

    strategy_name = config.get("strategy", "fedavg")

    if strategy_name == "fedavg":
        return FedAvgStrategy()
    elif strategy_name == "fedprox":
        return FedProxStrategy()
    elif strategy_name == "krum":
        krum_cfg = config.get("krum", {})
        return KrumStrategy(
            num_byzantine=krum_cfg.get("num_byzantine", 0),
            multi_k=krum_cfg.get("multi_k", 1),
        )
    elif strategy_name == "trimmed_mean":
        tm_cfg = config.get("trimmed_mean", {})
        return TrimmedMeanStrategy(trim_ratio=tm_cfg.get("trim_ratio", 0.1))
    elif strategy_name == "foolsgold":
        return FoolsGoldStrategy()
    elif strategy_name == "signguard":
        sg_cfg = config.get("signguard", {})
        verifier = None
        validator = None
        reputation = None

        if sg_cfg.get("enabled", True):
            from src.signguard.reputation import ReputationManager
            from src.signguard.verification import StatisticalValidator, UpdateVerifier

            reputation = ReputationManager(
                alpha=sg_cfg.get("reputation_alpha", 0.3),
                initial_reputation=sg_cfg.get("initial_reputation", 1.0),
                exclusion_threshold=sg_cfg.get("exclusion_threshold", 0.5),
            )
            validator = StatisticalValidator(
                l2_norm_threshold=sg_cfg.get("l2_norm_threshold", 100.0),
                relative_norm_threshold=sg_cfg.get("relative_norm_threshold", 5.0),
                cosine_threshold=sg_cfg.get("cosine_threshold", -0.5),
            )
            # Verifier needs public keys — set up during key registration
            verifier = UpdateVerifier()

        return SignGuardStrategy(
            verifier=verifier,
            validator=validator,
            reputation=reputation,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def create_model(config: dict) -> FraudMLP:
    """Create a model from config.

    Args:
        config: Full configuration dictionary.

    Returns:
        FraudMLP instance.
    """
    model_cfg = config.get("model", {})
    return FraudMLP(
        input_dim=model_cfg.get("input_dim", 38),
        hidden_dims=model_cfg.get("hidden_dims", [128, 64, 32]),
        dropout=model_cfg.get("dropout", 0.3),
        use_batch_norm=model_cfg.get("use_batch_norm", True),
    )


def run_centralized(
    config: dict,
    train_loader,
    val_loader=None,
    num_epochs: int = 50,
) -> dict:
    """Run centralized (non-federated) training baseline.

    Args:
        config: Configuration dictionary.
        train_loader: Combined training DataLoader (all states).
        val_loader: Optional validation DataLoader.
        num_epochs: Number of training epochs.

    Returns:
        Dictionary with training history and final metrics.
    """
    import torch
    import torch.nn as nn

    model = create_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("federation", {}).get("learning_rate", 0.001),
    )
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        if val_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    loss = criterion(model(X_batch), y_batch)
                    val_loss += loss.item() * len(y_batch)
                    n_val += len(y_batch)
            history["val_loss"].append(val_loss / max(n_val, 1))
            logger.info("Epoch %d: train_loss=%.4f, val_loss=%.4f",
                        epoch + 1, avg_loss, history["val_loss"][-1])

    return {
        "history": history,
        "params": get_model_params(model),
    }
