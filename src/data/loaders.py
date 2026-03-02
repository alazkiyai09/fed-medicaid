"""PyTorch data loaders for federated client training.

Wraps per-state Parquet splits into torch Datasets and DataLoaders,
handling feature/label separation and tensor conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Columns that are identifiers / metadata, not features
_META_COLS = {"BILLING_PROVIDER_NPI_NUM", "state", "is_excluded", "npi"}


class FraudDataset(Dataset):
    """PyTorch Dataset wrapping a Polars DataFrame of provider features.

    Attributes:
        features: Tensor of shape (N, D) with feature values.
        labels: Tensor of shape (N,) with binary fraud labels.
        npi_ids: Optional list of NPI identifiers for traceability.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        label_col: str = "is_excluded",
        feature_cols: list[str] | None = None,
    ):
        if feature_cols is None:
            # Only include numeric columns (excluding metadata)
            numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8)
            feature_cols = [
                c for c in df.columns 
                if c not in _META_COLS and isinstance(df.schema[c], numeric_types)
            ]

        # Extract features as float32 numpy array
        feature_data = df.select(feature_cols).fill_null(0.0).to_numpy().astype(np.float32)
        self.features = torch.from_numpy(feature_data)

        # Extract labels
        if label_col in df.columns:
            labels = df[label_col].fill_null(0).to_numpy().astype(np.float32)
            self.labels = torch.from_numpy(labels)
        else:
            self.labels = torch.zeros(len(df), dtype=torch.float32)

        # Keep NPI IDs for traceability
        if "BILLING_PROVIDER_NPI_NUM" in df.columns:
            self.npi_ids = df["BILLING_PROVIDER_NPI_NUM"].to_list()
        else:
            self.npi_ids = list(range(len(df)))

        self.feature_cols = feature_cols
        logger.debug(
            "FraudDataset: %d samples, %d features, %d positive",
            len(self), self.features.shape[1], int(self.labels.sum()),
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def input_dim(self) -> int:
        """Number of input features."""
        return self.features.shape[1]

    @property
    def pos_weight(self) -> float:
        """Class imbalance weight for BCEWithLogitsLoss."""
        n_pos = self.labels.sum().item()
        n_neg = len(self.labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos


def get_client_dataloader(
    state_code: str,
    split: str = "train",
    splits_dir: str | Path = "data/splits",
    batch_size: int = 256,
    shuffle: bool | None = None,
    label_col: str = "is_excluded",
) -> DataLoader:
    """Get a DataLoader for a specific state and split.

    Args:
        state_code: US state code (e.g., "CA", "NY").
        split: One of "train", "val", "test".
        splits_dir: Base directory containing per-state split directories.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle. Defaults to True for train, False otherwise.
        label_col: Column name for labels.

    Returns:
        PyTorch DataLoader.
    """
    path = Path(splits_dir) / state_code / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")

    df = pl.read_parquet(path)
    dataset = FraudDataset(df, label_col=label_col)

    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,  # Polars doesn't need multiprocessing
    )


def get_all_client_loaders(
    splits_dir: str | Path = "data/splits",
    split: str = "train",
    batch_size: int = 256,
    label_col: str = "is_excluded",
) -> dict[str, DataLoader]:
    """Get DataLoaders for all state clients.

    Args:
        splits_dir: Directory containing per-state split subdirectories.
        split: Which split to load.
        batch_size: Batch size.
        label_col: Label column name.

    Returns:
        Dictionary mapping state code → DataLoader.
    """
    splits_dir = Path(splits_dir)
    loaders = {}

    for state_dir in sorted(splits_dir.iterdir()):
        if state_dir.is_dir():
            split_path = state_dir / f"{split}.parquet"
            if split_path.exists():
                loaders[state_dir.name] = get_client_dataloader(
                    state_dir.name, split, splits_dir, batch_size, label_col=label_col,
                )

    logger.info("Loaded %d client %s loaders", len(loaders), split)
    return loaders
