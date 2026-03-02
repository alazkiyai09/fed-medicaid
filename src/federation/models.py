"""FraudMLP model and weight utilities for federated learning.

Architecture from spec §5:
  [38] → [128, BN, ReLU, Drop(0.3)] → [64, BN, ReLU, Drop(0.3)] → [32, ReLU] → [1, Sigmoid]
"""

from __future__ import annotations

import collections
import io
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """Multi-layer perceptron for provider fraud detection.

    A 4-layer MLP with batch normalization and dropout, designed for
    federated weight averaging (FedAvg-compatible).

    Args:
        input_dim: Number of input features (default: 38 from P1).
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout probability.
        use_batch_norm: Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int = 38,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim

        layers: list[nn.Module] = []
        in_dim = input_dim

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm and i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits (pre-sigmoid)."""
        h = self.encoder(x)
        return self.classifier(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probability (post-sigmoid)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def get_model_params(model: nn.Module) -> OrderedDict[str, np.ndarray]:
    """Extract model parameters as numpy arrays.

    Args:
        model: PyTorch model.

    Returns:
        OrderedDict mapping parameter names to numpy arrays.
    """
    return collections.OrderedDict(
        (name, param.detach().cpu().numpy().copy())
        for name, param in model.state_dict().items()
    )


def set_model_params(model: nn.Module, params: OrderedDict[str, np.ndarray]) -> None:
    """Set model parameters from numpy arrays.

    Args:
        model: PyTorch model.
        params: OrderedDict mapping parameter names to numpy arrays.
    """
    state_dict = collections.OrderedDict(
        (name, torch.from_numpy(np.asarray(arr))) for name, arr in params.items()
    )
    model.load_state_dict(state_dict)


def serialize_weights(params: OrderedDict[str, np.ndarray]) -> bytes:
    """Deterministically serialize model weights to bytes.

    Uses numpy's save format for reproducible byte representation,
    which is critical for ECDSA signature verification.

    Args:
        params: OrderedDict of parameter name → numpy array.

    Returns:
        Deterministic byte representation of the weights.
    """
    buffer = io.BytesIO()
    # Sort keys for determinism
    sorted_params = collections.OrderedDict(sorted(params.items()))
    np.savez(buffer, **sorted_params)
    return buffer.getvalue()


def deserialize_weights(data: bytes) -> OrderedDict[str, np.ndarray]:
    """Deserialize weights from bytes.

    Args:
        data: Byte representation from serialize_weights.

    Returns:
        OrderedDict of parameter name → numpy array.
    """
    buffer = io.BytesIO(data)
    npz = np.load(buffer)
    return collections.OrderedDict((k, npz[k]) for k in sorted(npz.files))


def compute_update_delta(
    new_params: OrderedDict[str, np.ndarray],
    old_params: OrderedDict[str, np.ndarray],
) -> OrderedDict[str, np.ndarray]:
    """Compute the difference between new and old model parameters.

    Args:
        new_params: Parameters after local training.
        old_params: Parameters before local training (global weights).

    Returns:
        Delta: new_params - old_params.
    """
    return collections.OrderedDict(
        (name, new_params[name] - old_params[name]) for name in old_params
    )


def apply_update_delta(
    params: OrderedDict[str, np.ndarray],
    delta: OrderedDict[str, np.ndarray],
) -> OrderedDict[str, np.ndarray]:
    """Apply an update delta to parameters.

    Args:
        params: Current parameters.
        delta: Update to add.

    Returns:
        Updated parameters.
    """
    return collections.OrderedDict(
        (name, params[name] + delta[name]) for name in params
    )
