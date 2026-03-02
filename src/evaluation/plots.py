"""Visualization plots for experiment results.

Generates publication-ready figures for:
- Convergence curves, attack robustness heatmaps, DP trade-off curves,
  reputation evolution, per-state performance, communication costs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Default styling
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def plot_convergence_curves(
    histories: dict[str, list[float]],
    metric_name: str = "AUPRC",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot convergence curves for multiple strategies.

    Args:
        histories: Mapping strategy_name → list of metric values per round.
        metric_name: Y-axis label.
        output_path: Optional path to save figure.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots()

    for name, values in histories.items():
        ax.plot(range(1, len(values) + 1), values, label=name, linewidth=2)

    ax.set_xlabel("Federation Round")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Convergence")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved convergence plot to %s", output_path)

    return fig


def plot_attack_heatmap(
    results: dict[tuple[str, str], float],
    attacks: list[str],
    defenses: list[str],
    metric_name: str = "AUPRC",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot attack robustness heatmap (5 attacks × 6 defenses).

    Args:
        results: Mapping (attack, defense) → metric value.
        attacks: List of attack names (rows).
        defenses: List of defense names (columns).
        metric_name: Title metric.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    matrix = np.zeros((len(attacks), len(defenses)))
    for i, attack in enumerate(attacks):
        for j, defense in enumerate(defenses):
            matrix[i, j] = results.get((attack, defense), 0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=defenses,
        yticklabels=attacks,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"Attack Robustness ({metric_name})")
    ax.set_xlabel("Defense Strategy")
    ax.set_ylabel("Attack Type")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved heatmap to %s", output_path)

    return fig


def plot_dp_tradeoff(
    epsilon_values: list[float],
    auprc_values: dict[str, list[float]],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot DP privacy-accuracy trade-off curves.

    Args:
        epsilon_values: List of epsilon values (log scale).
        auprc_values: Mapping strategy_name → list of AUPRC at each epsilon.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots()

    for name, values in auprc_values.items():
        ax.plot(epsilon_values, values, marker="o", label=name, linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("ε (Privacy Budget)")
    ax.set_ylabel("AUPRC")
    ax.set_title("Privacy-Accuracy Trade-off")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved DP trade-off plot to %s", output_path)

    return fig


def plot_reputation_evolution(
    reputations: dict[str, list[float]],
    malicious_clients: list[str] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-client reputation evolution over rounds.

    Args:
        reputations: Mapping client_id → list of reputation values per round.
        malicious_clients: List of known malicious client IDs (highlighted).
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if malicious_clients is None:
        malicious_clients = []

    fig, ax = plt.subplots(figsize=(12, 6))

    for cid, values in reputations.items():
        is_malicious = cid in malicious_clients
        ax.plot(
            range(1, len(values) + 1),
            values,
            label=f"{cid} {'(malicious)' if is_malicious else ''}",
            linewidth=2 if is_malicious else 1,
            linestyle="--" if is_malicious else "-",
            alpha=0.9 if is_malicious else 0.4,
            color="red" if is_malicious else None,
        )

    ax.axhline(y=0.5, color="orange", linestyle=":", label="Exclusion threshold")
    ax.axhline(y=0.2, color="red", linestyle=":", label="Investigation threshold")

    ax.set_xlabel("Federation Round")
    ax.set_ylabel("Reputation Score")
    ax.set_title("Client Reputation Evolution")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved reputation plot to %s", output_path)

    return fig


def plot_per_state_performance(
    state_auprc: dict[str, float],
    state_sizes: dict[str, int] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-state local AUPRC bar chart sorted by data size.

    Args:
        state_auprc: Mapping state → AUPRC.
        state_sizes: Optional mapping state → dataset size (for sorting).
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    if state_sizes:
        sorted_states = sorted(state_auprc.keys(), key=lambda s: state_sizes.get(s, 0), reverse=True)
    else:
        sorted_states = sorted(state_auprc.keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    values = [state_auprc[s] for s in sorted_states]
    ax.bar(range(len(sorted_states)), values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(sorted_states)))
    ax.set_xticklabels(sorted_states, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUPRC")
    ax.set_title("Per-State AUPRC (sorted by data size)")
    ax.grid(True, alpha=0.3, axis="y")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved per-state plot to %s", output_path)

    return fig


def plot_communication_cost(
    strategy_costs: dict[str, float],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot total communication cost per strategy.

    Args:
        strategy_costs: Mapping strategy_name → total MB.
        output_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots()

    names = list(strategy_costs.keys())
    values = list(strategy_costs.values())

    ax.barh(names, values, color="teal", alpha=0.8)
    ax.set_xlabel("Total Communication (MB)")
    ax.set_title("Communication Cost by Strategy")
    ax.grid(True, alpha=0.3, axis="x")

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        logger.info("Saved communication cost plot to %s", output_path)

    return fig
