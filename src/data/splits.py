"""Per-state train/val/test splitting with stratification.

Creates reproducible splits for each state partition, ensuring balanced
representation of fraud vs. non-fraud labels in each split.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


def create_state_splits(
    state_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    label_col: str = "is_excluded",
    seed: int = 42,
) -> dict[str, int]:
    """Create stratified train/val/test splits for a single state.

    Args:
        state_path: Path to the state's Parquet file.
        output_dir: Base output directory (splits saved under state subdirectory).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation. Test = 1 - train - val.
        label_col: Column name for stratification label.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with split counts: {train: N, val: N, test: N}.
    """
    state_path = Path(state_path)
    state_code = state_path.stem
    output_dir = Path(output_dir) / state_code
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(state_path)
    n = len(df)
    rng = np.random.RandomState(seed)

    # Stratified splitting
    if label_col in df.columns:
        indices_pos = np.where(df[label_col].to_numpy() == 1)[0]
        indices_neg = np.where(df[label_col].to_numpy() == 0)[0]

        rng.shuffle(indices_pos)
        rng.shuffle(indices_neg)

        def _split_indices(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            n_train = int(len(idx) * train_ratio)
            n_val = int(len(idx) * val_ratio)
            return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

        train_pos, val_pos, test_pos = _split_indices(indices_pos)
        train_neg, val_neg, test_neg = _split_indices(indices_neg)

        train_idx = np.concatenate([train_pos, train_neg])
        val_idx = np.concatenate([val_pos, val_neg])
        test_idx = np.concatenate([test_pos, test_neg])
    else:
        # No label column — random split
        indices = np.arange(n)
        rng.shuffle(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

    # Write splits
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    counts = {}
    for split_name, idx in splits.items():
        split_df = df[idx.tolist()]
        split_df.write_parquet(output_dir / f"{split_name}.parquet")
        counts[split_name] = len(split_df)

    logger.info(
        "  %s: train=%d, val=%d, test=%d",
        state_code, counts["train"], counts["val"], counts["test"],
    )
    return counts


def create_all_splits(
    partitioned_dir: str | Path = "data/partitioned",
    output_dir: str | Path = "data/splits",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    label_col: str = "is_excluded",
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Create splits for all state partitions.

    Args:
        partitioned_dir: Directory with per-state Parquet files.
        output_dir: Base output directory for splits.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        label_col: Label column for stratification.
        seed: Random seed.

    Returns:
        Nested dict: {state: {train: N, val: N, test: N}}.
    """
    partitioned_dir = Path(partitioned_dir)
    all_counts = {}

    logger.info("Creating train/val/test splits (%.0f/%.0f/%.0f) …",
                train_ratio * 100, val_ratio * 100, (1 - train_ratio - val_ratio) * 100)

    for parquet_file in sorted(partitioned_dir.glob("*.parquet")):
        counts = create_state_splits(
            parquet_file, output_dir, train_ratio, val_ratio, label_col, seed,
        )
        all_counts[parquet_file.stem] = counts

    total_train = sum(c["train"] for c in all_counts.values())
    total_val = sum(c["val"] for c in all_counts.values())
    total_test = sum(c["test"] for c in all_counts.values())
    logger.info(
        "Total: train=%d, val=%d, test=%d across %d states",
        total_train, total_val, total_test, len(all_counts),
    )
    return all_counts
