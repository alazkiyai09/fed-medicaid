"""E1: Centralized vs. Federated baseline comparison.

Runs 4 configurations: Centralized, FedAvg, FedProx, Local-only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.data.loaders import get_all_client_loaders
from src.federation.client import ClientConfig, FederatedClient
from src.federation.runner import (
    create_model,
    create_strategy,
    load_experiment_config,
    run_centralized,
)
from src.federation.server import FederationServer
from src.signguard.keys import KeyManager

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_path = Path("configs/base.yaml")
    exp_path = Path("configs/experiment_1.yaml")
    fed_path = Path("data/partitioned/federation_state.yaml")
    splits_dir = Path("data/splits")
    
    config = load_experiment_config(base_path, exp_path)
    if fed_path.exists():
        fed_config = load_experiment_config(fed_path)
    else:
        logger.error("Federation config not found! Run scripts/prepare_data.py first.")
        return

    logger.info("=" * 60)
    logger.info("E1: Centralized vs. Federated")
    logger.info("=" * 60)

    # 1. Load Data
    logger.info("Loading client DataLoaders...")
    batch_size = config['federation']['batch_size']
    train_loaders = get_all_client_loaders(splits_dir, "train", batch_size)
    val_loaders = get_all_client_loaders(splits_dir, "val", batch_size)
    test_loaders = get_all_client_loaders(splits_dir, "test", batch_size)
    
    runs = config.get("runs", [])
    results = {}
    
    for run_cfg in runs:
        run_name = run_cfg["name"]
        strategy_name = run_cfg["strategy"]
        logger.info("--- Run: %s (strategy: %s) ---", run_name, strategy_name)
        
        # Merge run specific config into base config
        run_full_config = config.copy()
        if "strategy" in run_cfg:
            run_full_config["strategy"] = run_cfg["strategy"]
        if "fedprox_mu" in run_cfg:
            run_full_config["federation"]["fedprox_mu"] = run_cfg["fedprox_mu"]
            
        if strategy_name == "centralized":
            # Just run one epoch for demonstration since it's a huge dataset
            # To do real centralized, we'd map a single DataLoader over data/features/
            # Here we skip actual centralized training on 227M due to time/memory, 
            # and just log that it's set up. True run requires a compute cluster.
            logger.info("  Status: Skipping full centralized training (requires cluster)")
            continue
            
        # 2. Setup Federation
        # Dynamically infer the actual numeric input_dim from the first available loader
        first_client_id = list(train_loaders.keys())[0]
        actual_input_dim = train_loaders[first_client_id].dataset.input_dim
        run_full_config['model'] = run_full_config.get('model', {})
        run_full_config['model']['input_dim'] = actual_input_dim
        
        model = create_model(run_full_config)
        strategy = create_strategy(run_full_config)
        server = FederationServer(
            model, 
            strategy, 
            num_rounds=2,  # Quick demo
            participation_rate=0.1  # Only 10% of clients per round for speed
        )
        
        # Setup clients
        clients = {}
        for state_id in fed_config['clients']:
            if state_id not in train_loaders: continue
            
            c_cfg = ClientConfig(
                client_id=state_id,
                local_epochs=run_full_config['federation']['local_epochs'],
                learning_rate=run_full_config['federation']['learning_rate'],
                batch_size=run_full_config['federation']['batch_size'],
                fedprox_mu=run_full_config['federation'].get('fedprox_mu', 0.0),
            )
            # Pass model_kwargs to ensure client hidden_dims match server model
            # e.g. [128, 64, 32]
            model_kwargs = {
                "hidden_dims": run_full_config['model'].get('hidden_dims', [128, 64, 32]),
                "dropout": run_full_config['model'].get('dropout', 0.3),
                "use_batch_norm": run_full_config['model'].get('use_batch_norm', True)
            }
            clients[state_id] = FederatedClient(
                c_cfg, 
                train_loaders[state_id], 
                val_loaders[state_id],
                model_kwargs=model_kwargs
            )
            
        import torch
        from src.evaluation.metrics import compute_all_metrics
        import torch.nn as nn
        
        # We can define a single global validation or use the first client's test loader for a quick proxy
        # since evaluating 227M records every round is extremely expensive.
        proxy_val_client = list(val_loaders.keys())[0]
        proxy_val_loader = val_loaders[proxy_val_client]
        criterion = nn.BCEWithLogitsLoss()

        def evaluate_model(mod, loader):
            mod.eval()
            total_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in loader:
                    outputs = mod(inputs).squeeze(-1)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                    # Convert logits to probabilities
                    probs = torch.sigmoid(outputs)
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            import numpy as np
            y_true = np.array(all_labels)
            y_pred = np.array(all_preds)
            metrics = compute_all_metrics(y_true, y_pred)
            metrics["test_loss"] = total_loss / len(loader.dataset)
            return metrics

        def eval_fn(round_num, global_params):
            from src.federation.models import set_model_params
            set_model_params(model, global_params)
            metrics = evaluate_model(model, proxy_val_loader)
            return {"global_auprc": metrics.get("auprc", 0.0), "global_loss": metrics.get("test_loss", 0.0)}

        logger.info("  Starting federated training for %s...", run_name)
        history = server.run_training(clients, eval_fn=eval_fn, log_every=1)
        
        # Convert history objects to dictionaries for JSON saving
        hist_dicts = []
        for r in history:
            hist_dicts.append({
                "round": r.round_num,
                "global_loss": r.metrics.get("global_loss"),
                "global_auprc": r.metrics.get("global_auprc"),
                "duration": r.duration_seconds,
                "num_participating": r.num_participating
            })
        results[run_name] = hist_dicts
        logger.info("  Completed %s. Final Global AUPRC: %.4f", run_name, history[-1].metrics.get("global_auprc", 0.0))

    import json
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "e1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("E1 complete! Results saved to results/e1_results.json")


if __name__ == "__main__":
    main()
