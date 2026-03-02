"""Evaluation metrics for federated fraud detection.

Implements AUPRC (primary), AUC-ROC, Recall@k, convergence rate,
communication cost, and computational overhead metrics.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (primary metric).

    Handles extreme class imbalance (fraud < 1%) better than AUC-ROC.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted fraud probabilities.

    Returns:
        AUPRC score (0.0 to 1.0).
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true — AUPRC undefined")
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (secondary metric).

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted fraud probabilities.

    Returns:
        AUC-ROC score (0.0 to 1.0).
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true — AUROC undefined")
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    """Compute Recall@k: of the top k flagged providers, how many are truly excluded?

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted fraud probabilities.
        k: Number of top-scored providers to consider.

    Returns:
        Fraction of top-k that are true positives.
    """
    k = min(k, len(y_true))
    top_k_indices = np.argsort(y_score)[-k:]
    true_positives = y_true[top_k_indices].sum()
    total_positives = y_true.sum()

    if total_positives == 0:
        return 0.0
    return float(true_positives / total_positives)


def convergence_rate(
    auprc_history: list[float],
    target_fraction: float = 0.95,
) -> int | None:
    """Compute the round at which AUPRC reaches target_fraction of final value.

    Args:
        auprc_history: List of AUPRC values per round.
        target_fraction: Fraction of final AUPRC to reach (default: 95%).

    Returns:
        Round number at which target was first reached, or None.
    """
    if not auprc_history:
        return None

    final_auprc = auprc_history[-1]
    target = target_fraction * final_auprc

    for i, val in enumerate(auprc_history):
        if val >= target:
            return i + 1  # 1-indexed round

    return None


def compute_communication_cost(
    params: OrderedDict[str, np.ndarray],
    num_clients: int,
    num_rounds: int,
) -> dict[str, float]:
    """Compute total communication cost in bytes.

    Each round: server broadcasts global model + clients send updates.
    Total per round: (1 + num_clients) × model_size.

    Args:
        params: Model parameters (to calculate size).
        num_clients: Number of participating clients.
        num_rounds: Total rounds.

    Returns:
        Dictionary with per-round and total MB.
    """
    total_params = sum(v.nbytes for v in params.values())
    bytes_per_round = total_params * (1 + num_clients)  # Broadcast + uploads
    total_bytes = bytes_per_round * num_rounds

    return {
        "model_size_mb": total_params / (1024 * 1024),
        "per_round_mb": bytes_per_round / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "total_params": sum(v.size for v in params.values()),
    }


def compute_attack_degradation(
    clean_auprc: float,
    attack_auprc: float,
) -> float:
    """Compute AUPRC degradation under attack relative to clean baseline.

    Args:
        clean_auprc: AUPRC under clean (no attack) conditions.
        attack_auprc: AUPRC under attack conditions.

    Returns:
        Fraction of AUPRC lost (0.0 = no degradation, 1.0 = total loss).
    """
    if clean_auprc == 0:
        return 0.0
    return max(0.0, (clean_auprc - attack_auprc) / clean_auprc)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics at once.

    Args:
        y_true: Binary ground truth labels.
        y_score: Predicted fraud probabilities.
        ks: List of k values for Recall@k.

    Returns:
        Dictionary of all metric values.
    """
    if ks is None:
        ks = [50, 100, 500]

    metrics = {
        "auprc": compute_auprc(y_true, y_score),
        "auroc": compute_auroc(y_true, y_score),
    }
    for k in ks:
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k)

    return metrics
