"""Script to prepare data for FedMedicaid.

1. Extracts NPI -> State mapping from the 11GB NPPES CSV.
2. Runs the partition_by_state to create client datasets.
3. Creates train/val/test splits for each state.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.data.partition import create_federation_config, partition_by_state
from src.data.splits import create_state_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_npi_registry(csv_path: Path, out_path: Path):
    """Extract NPI and State from the massive NPPES CSV."""
    logger.info("Extracting NPI registry from %s...", csv_path)
    
    # Use scan_csv for memory efficiency on 11GB file
    df = pl.scan_csv(
        csv_path,
        infer_schema_length=0,  # Read all as string to avoid type errors
        ignore_errors=True
    ).select([
        pl.col("NPI").alias("npi"),
        pl.col("Provider Business Practice Location Address State Name").alias("state")
    ]).filter(
        pl.col("npi").is_not_null() & 
        pl.col("state").is_not_null() &
        (pl.col("state").str.len_chars() == 2)
    ).collect()
    
    logger.info("Saving %d NPI records to %s...", len(df), out_path)
    df.write_parquet(out_path)
    return df


def main():
    # Paths
    p1_npi_csv = Path(r"F:\Projects\medicaid-guard\data\external\nppes_npi.csv")
    p1_features = Path(r"F:\Projects\medicaid-guard\data\processed\feature_matrix.parquet")
    
    p2_data_dir = Path("data")
    npi_registry_path = p2_data_dir / "external" / "npi_registry.parquet"
    features_dir = p2_data_dir / "features"
    partitioned_dir = p2_data_dir / "partitioned"
    splits_dir = p2_data_dir / "splits"
    
    # 1. Prepare NPI registry
    if not npi_registry_path.exists():
        npi_registry_df = prepare_npi_registry(p1_npi_csv, npi_registry_path)
    else:
        logger.info("Loading existing NPI registry from %s", npi_registry_path)
        npi_registry_df = pl.read_parquet(npi_registry_path)

    # 2. Partition features by state
    logger.info("Loading feature matrix from %s...", p1_features)
    features_df = pl.read_parquet(p1_features)
    
    # Symlink/copy feature matrix for reference
    dest_features = features_dir / "feature_matrix.parquet"
    if not dest_features.exists():
        import shutil
        shutil.copy2(p1_features, dest_features)
    
    logger.info("Partitioning features by state...")
    state_counts = partition_by_state(features_df, npi_registry_df, partitioned_dir)
    
    # 3. Create federation config
    logger.info("Creating federation configuration...")
    fed_config = create_federation_config(state_counts, min_providers=100)
    
    config_path = partitioned_dir / "federation_state.yaml"
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(fed_config, f, default_flow_style=False)
    logger.info("Saved federation config to %s", config_path)
    
    # 4. Create splits for eligible states
    logger.info("Creating train/val/test splits...")
    for state_id in fed_config["clients"].keys():
        state_file = partitioned_dir / f"{state_id}.parquet"
        if state_file.exists():
            create_state_splits(state_file, splits_dir)
            
    logger.info("Data preparation complete! Generated %d federated clients.", fed_config["num_clients"])

if __name__ == "__main__":
    main()
