"""E4: Differential privacy trade-off sweep.

Sweeps epsilon values to measure the privacy-accuracy trade-off.
"""

from __future__ import annotations

import logging
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.data.loaders import get_all_client_loaders
from src.federation.client import ClientConfig, FederatedClient
from src.federation.runner import create_model, create_strategy, load_experiment_config
from src.federation.server import FederationServer
from src.evaluation.metrics import compute_all_metrics
from src.privacy.dp_mechanism import GaussianMechanism

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_path = Path("configs/base.yaml")
    exp_path = Path("configs/experiment_4.yaml")
    fed_path = Path("data/partitioned/federation_state.yaml")
    splits_dir = Path("data/splits")
    
    config = load_experiment_config(base_path, exp_path)
    if fed_path.exists():
        fed_config = load_experiment_config(fed_path)
    else:
        logger.error("Federation config not found!")
        return

    logger.info("=" * 60)
    logger.info("E4: DP Trade-off Sweep")
    logger.info("=" * 60)

    batch_size = config['federation']['batch_size']
    train_loaders = get_all_client_loaders(splits_dir, "train", batch_size)
    val_loaders = get_all_client_loaders(splits_dir, "val", batch_size)
    
    epsilons = config.get("epsilon_values", [0.1, 1.0, 10.0, float('inf')])
    dp_config = config.get("privacy", {"delta": 1e-5, "max_grad_norm": 1.0})
    
    results = {}
    
    proxy_val_client = list(val_loaders.keys())[0]
    proxy_val_loader = val_loaders[proxy_val_client]
    criterion = nn.BCEWithLogitsLoss()

    def evaluate_model(mod, loader):
        mod.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []
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

    first_client_id = list(train_loaders.keys())[0]
    actual_input_dim = train_loaders[first_client_id].dataset.input_dim

    for eps in epsilons:
        eps_val = float(eps)
        if hasattr(math, 'isinf') and math.isinf(eps_val):
            run_name = "dp_eps_inf"
        else:
            run_name = f"dp_eps_{eps_val}"
            
        logger.info("--- Run: %s ---", run_name)
        
        run_full_config = config.copy()
        run_full_config['model'] = run_full_config.get('model', {})
        run_full_config['model']['input_dim'] = actual_input_dim
        
        model = create_model(run_full_config)
        strategy = create_strategy(run_full_config)
        
        server = FederationServer(
            model, strategy, num_rounds=2, participation_rate=0.1
        )
        
        dp_mech = GaussianMechanism(
            epsilon=eps_val, 
            delta=dp_config.get("delta", 1e-5), 
            max_grad_norm=dp_config.get("max_grad_norm", 1.0)
        )
        
        clients = {}
        for cid in fed_config['clients']:
            if cid not in train_loaders: continue
            c_cfg = ClientConfig(client_id=cid, local_epochs=run_full_config['federation']['local_epochs'],
                                 learning_rate=run_full_config['federation']['learning_rate'], batch_size=batch_size)
            
            clients[cid] = FederatedClient(
                c_cfg, train_loaders[cid], val_loaders[cid],
                dp_mechanism=dp_mech,
                model_kwargs={"hidden_dims": run_full_config['model'].get('hidden_dims', [128, 64, 32]),
                              "dropout": run_full_config['model'].get('dropout', 0.3),
                              "use_batch_norm": run_full_config['model'].get('use_batch_norm', True)}
            )

        def eval_fn(round_num, global_params):
            from src.federation.models import set_model_params
            set_model_params(model, global_params)
            metrics = evaluate_model(model, proxy_val_loader)
            return {"global_auprc": metrics.get("auprc", 0.0), "global_loss": metrics.get("test_loss", 0.0)}

        history = server.run_training(clients, eval_fn=eval_fn, log_every=1)
        
        results[run_name] = [
            {"round": h.round_num, "global_auprc": h.metrics.get("global_auprc", 0.0), 
             "num_participating": h.num_participating, "num_accepted": h.num_accepted} 
             for h in history
        ]
        logger.info("  Completed %s. AUPRC: %.4f", run_name, history[-1].metrics.get("global_auprc", 0.0))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "e4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("E4 complete! Results saved to results/e4_results.json")


if __name__ == "__main__":
    main()
