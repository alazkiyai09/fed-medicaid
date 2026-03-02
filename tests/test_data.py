"""Tests for the data module — partitioning, splitting, and loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.partition import create_federation_config, partition_by_state
from src.data.splits import create_state_splits


@pytest.fixture
def synthetic_features():
    """Create a synthetic feature DataFrame mimicking P1 output."""
    rng = np.random.RandomState(42)
    n = 500
    return pl.DataFrame({
        "BILLING_PROVIDER_NPI_NUM": [f"NPI{i:06d}" for i in range(n)],
        "total_paid_all_time": rng.exponential(100000, n),
        "total_claims_all_time": rng.poisson(50, n).astype(float),
        "unique_hcpcs_codes": rng.poisson(10, n).astype(float),
        "avg_cost_per_claim": rng.exponential(500, n),
        "is_excluded": (rng.random(n) < 0.02).astype(int),
    })


@pytest.fixture
def synthetic_npi_registry():
    """Create a synthetic NPI registry with state assignments."""
    states = ["CA", "NY", "TX", "FL", "OH", "WY"]
    rng = np.random.RandomState(42)
    n = 500
    return pl.DataFrame({
        "npi": [f"NPI{i:06d}" for i in range(n)],
        "state": [states[i % len(states)] for i in range(n)],
    })


class TestPartitionByState:
    def test_partitions_created(self, synthetic_features, synthetic_npi_registry, tmp_path):
        counts = partition_by_state(synthetic_features, synthetic_npi_registry, tmp_path)
        assert len(counts) > 0
        assert sum(counts.values()) <= len(synthetic_features)

    def test_parquet_files_written(self, synthetic_features, synthetic_npi_registry, tmp_path):
        partition_by_state(synthetic_features, synthetic_npi_registry, tmp_path)
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) > 0

    def test_all_states_represented(self, synthetic_features, synthetic_npi_registry, tmp_path):
        counts = partition_by_state(synthetic_features, synthetic_npi_registry, tmp_path)
        expected_states = {"CA", "NY", "TX", "FL", "OH", "WY"}
        assert set(counts.keys()) == expected_states


class TestCreateFederationConfig:
    def test_filters_small_states(self):
        counts = {"CA": 500, "NY": 300, "WY": 10, "VT": 5}
        config = create_federation_config(counts, min_providers=100)
        assert config["num_clients"] == 2
        assert "WY" in config["excluded_states"]
        assert "VT" in config["excluded_states"]

    def test_weights_sum_to_one(self):
        counts = {"CA": 600, "NY": 400}
        config = create_federation_config(counts, min_providers=100)
        total_weight = sum(c["weight"] for c in config["clients"].values())
        assert abs(total_weight - 1.0) < 1e-6

    def test_empty_counts(self):
        config = create_federation_config({}, min_providers=100)
        assert config["num_clients"] == 0


class TestCreateStateSplits:
    def test_split_sizes(self, synthetic_features, tmp_path):
        # Save features as a state parquet
        state_path = tmp_path / "CA.parquet"
        synthetic_features.write_parquet(state_path)
        output_dir = tmp_path / "splits"

        counts = create_state_splits(state_path, output_dir)
        total = sum(counts.values())
        assert total == len(synthetic_features)
        assert counts["train"] > counts["val"]
        assert counts["train"] > counts["test"]

    def test_split_files_exist(self, synthetic_features, tmp_path):
        state_path = tmp_path / "NY.parquet"
        synthetic_features.write_parquet(state_path)
        output_dir = tmp_path / "splits"

        create_state_splits(state_path, output_dir)
        assert (output_dir / "NY" / "train.parquet").exists()
        assert (output_dir / "NY" / "val.parquet").exists()
        assert (output_dir / "NY" / "test.parquet").exists()
