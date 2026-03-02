"""E2: Defense comparison under clean conditions.

Compares: No defense, Krum, TrimmedMean, FoolsGold, SignGuard (5 runs).
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.data.loaders import get_all_client_loaders
from src.federation.client import ClientConfig, FederatedClient
from src.federation.runner import (
    create_model,
    create_strategy,
    load_experiment_config,
)
from src.federation.server import FederationServer
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_path = Path("configs/base.yaml")
    exp_path = Path("configs/experiment_2.yaml")
    fed_path = Path("data/partitioned/federation_state.yaml")
    splits_dir = Path("data/splits")
    
    config = load_experiment_config(base_path, exp_path)
    if fed_path.exists():
        fed_config = load_experiment_config(fed_path)
    else:
        logger.error("Federation config not found! Run scripts/prepare_data.py first.")
        return

    logger.info("=" * 60)
    logger.info("E2: Defense Comparison (Clean Conditions)")
    logger.info("=" * 60)

    logger.info("Loading client DataLoaders...")
    batch_size = config['federation']['batch_size']
    train_loaders = get_all_client_loaders(splits_dir, "train", batch_size)
    val_loaders = get_all_client_loaders(splits_dir, "val", batch_size)
    
    runs = config.get("runs", [])
    results = {}
    
    # We use the first client's test loader for a quick proxy
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
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        metrics = compute_all_metrics(y_true, y_pred)
        metrics["test_loss"] = total_loss / len(loader.dataset)
        return metrics

    for run_cfg in runs:
        run_name = run_cfg["name"]
        strategy_name = run_cfg["strategy"]
        logger.info("--- Run: %s (strategy: %s) ---", run_name, strategy_name)
        
        run_full_config = config.copy()
        if "strategy" in run_cfg:
            run_full_config["strategy"] = run_cfg["strategy"]
        
        first_client_id = list(train_loaders.keys())[0]
        actual_input_dim = train_loaders[first_client_id].dataset.input_dim
        run_full_config['model'] = run_full_config.get('model', {})
        run_full_config['model']['input_dim'] = actual_input_dim
        
        model = create_model(run_full_config)
        strategy = create_strategy(run_full_config)
        
        # Override params for specific strategies if needed
        if strategy_name == "trimmed_mean" and "trimmed_mean" in run_cfg:
            run_full_config["federation"]["trimmed_mean_ratio"] = run_cfg["trimmed_mean"]["trim_ratio"]
            strategy = create_strategy(run_full_config)

        # Ensure SignGuard is properly configured
        if strategy_name == "signguard" and "signguard" in run_cfg:
            run_full_config["federation"]["signguard"] = run_cfg["signguard"]
            strategy = create_strategy(run_full_config)
        
        server = FederationServer(
            model, 
            strategy, 
            num_rounds=2,  # Quick demo for E2
            participation_rate=0.1  # Only 10%
        )
        
        clients = {}
        for state_id in fed_config['clients']:
            if state_id not in train_loaders: continue
            
            c_cfg = ClientConfig(
                client_id=state_id,
                local_epochs=run_full_config['federation']['local_epochs'],
                learning_rate=run_full_config['federation']['learning_rate'],
                batch_size=run_full_config['federation']['batch_size'],
            )
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
            
        def eval_fn(round_num, global_params):
            from src.federation.models import set_model_params
            set_model_params(model, global_params)
            metrics = evaluate_model(model, proxy_val_loader)
            return {"global_auprc": metrics.get("auprc", 0.0), "global_loss": metrics.get("test_loss", 0.0)}

        logger.info("  Starting federated training for %s...", run_name)
        history = server.run_training(clients, eval_fn=eval_fn, log_every=1)
        
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

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "e2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("E2 complete! Results saved to results/e2_results.json")


if __name__ == "__main__":
    main()
