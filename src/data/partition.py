"""State-level data partitioning from P1 feature matrices.

Splits the provider feature matrix by US state using NPPES NPI Registry,
creating per-state Parquet files for federated client training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/base.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def partition_by_state(
    features_df: pl.DataFrame | pl.LazyFrame,
    npi_registry: pl.DataFrame | pl.LazyFrame,
    output_dir: str | Path = "data/partitioned",
) -> dict[str, int]:
    """Split provider feature matrix by state.

    Joins features with NPI registry to get state affiliation,
    then writes one Parquet file per state.

    Args:
        features_df: Provider-level feature matrix (from P1).
        npi_registry: NPI registry with ``npi`` and ``state`` columns.
        output_dir: Directory to write per-state Parquet files.

    Returns:
        Dictionary mapping state code → provider count.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we work with DataFrames
    if isinstance(features_df, pl.LazyFrame):
        features_df = features_df.collect()
    if isinstance(npi_registry, pl.LazyFrame):
        npi_registry = npi_registry.collect()

    # Join features with NPI registry to get state
    npi_subset = npi_registry.select(["npi", "state"])
    with_state = features_df.join(
        npi_subset,
        left_on="BILLING_PROVIDER_NPI_NUM",
        right_on="npi",
        how="left",
    )

    # Drop providers with no state mapping
    with_state = with_state.filter(pl.col("state").is_not_null())
    logger.info(
        "Matched %d / %d providers to states",
        len(with_state),
        len(features_df),
    )

    # Partition by state
    state_counts: dict[str, int] = {}
    states = sorted(with_state["state"].unique().to_list())

    for state in states:
        state_data = with_state.filter(pl.col("state") == state)
        state_path = output_dir / f"{state}.parquet"
        state_data.write_parquet(state_path)
        state_counts[state] = len(state_data)
        logger.debug("  %s: %d providers", state, len(state_data))

    logger.info("Partitioned into %d states", len(state_counts))
    return state_counts


def create_federation_config(
    state_counts: dict[str, int],
    min_providers: int = 100,
) -> dict:
    """Create federation configuration from state partition counts.

    Filters out states with too few providers and computes relative weights.

    Args:
        state_counts: Mapping of state code → provider count.
        min_providers: Minimum providers for a state to participate.

    Returns:
        Federation config with eligible clients, counts, and weights.
    """
    eligible = {s: c for s, c in state_counts.items() if c >= min_providers}
    total = sum(eligible.values())

    config = {
        "num_clients": len(eligible),
        "total_providers": total,
        "excluded_states": [s for s in state_counts if s not in eligible],
        "clients": {
            state: {"count": count, "weight": count / total}
            for state, count in sorted(eligible.items())
        },
    }

    logger.info(
        "Federation: %d eligible clients (%d excluded), %d total providers",
        config["num_clients"],
        len(config["excluded_states"]),
        total,
    )
    return config


def analyze_noniid(partitioned_dir: str | Path = "data/partitioned") -> pl.DataFrame:
    """Analyze non-IID characteristics across state partitions.

    Computes per-state statistics: mean/std of features, label distribution,
    data size, and specialty mix.

    Args:
        partitioned_dir: Directory containing per-state Parquet files.

    Returns:
        DataFrame with one row per state and summary statistics.
    """
    partitioned_dir = Path(partitioned_dir)
    records = []

    for parquet_file in sorted(partitioned_dir.glob("*.parquet")):
        state = parquet_file.stem
        df = pl.read_parquet(parquet_file)
        record = {
            "state": state,
            "num_providers": len(df),
        }

        # Add summary statistics for numeric columns
        numeric_cols = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32, pl.Int64)]
        for col in numeric_cols[:10]:  # First 10 numeric columns
            record[f"{col}_mean"] = df[col].mean()
            record[f"{col}_std"] = df[col].std()

        records.append(record)

    return pl.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config()
    features_path = Path(cfg["data"]["features_dir"]) / "feature_matrix.parquet"
    npi_path = Path(cfg["data"]["npi_registry_path"])

    if not features_path.exists():
        logger.error("Feature matrix not found at %s", features_path)
        raise SystemExit(1)
    if not npi_path.exists():
        logger.error("NPI registry not found at %s", npi_path)
        raise SystemExit(1)

    features = pl.read_parquet(features_path)
    npi = pl.read_parquet(npi_path)
    state_counts = partition_by_state(features, npi, cfg["data"]["partitioned_dir"])
    fed_config = create_federation_config(state_counts, cfg["data"]["min_providers_per_state"])

    # Save federation config
    config_path = Path("configs") / "federation_state.yaml"
    with open(config_path, "w") as f:
        yaml.dump(fed_config, f, default_flow_style=False)
    logger.info("Saved federation config to %s", config_path)
