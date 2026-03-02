"""E5: Scalability analysis (participation rates: 20%, 50%, 80%, 100%)."""

from __future__ import annotations

import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from src.data.loaders import get_all_client_loaders
from src.federation.client import ClientConfig, FederatedClient
from src.federation.runner import create_model, create_strategy, load_experiment_config
from src.federation.server import FederationServer
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_path = Path("configs/base.yaml")
    exp_path = Path("configs/experiment_5.yaml")
    fed_path = Path("data/partitioned/federation_state.yaml")
    splits_dir = Path("data/splits")
    
    config = load_experiment_config(base_path, exp_path)
    if fed_path.exists():
        fed_config = load_experiment_config(fed_path)
    else:
        logger.error("Federation config not found!")
        return

    logger.info("=" * 60)
    logger.info("E5: Scalability Analysis")
    logger.info("=" * 60)

    batch_size = config['federation']['batch_size']
    train_loaders = get_all_client_loaders(splits_dir, "train", batch_size)
    val_loaders = get_all_client_loaders(splits_dir, "val", batch_size)
    
    rates = config.get("participation_rates", [0.2, 0.5, 0.8, 1.0])
    
    # Due to local execution memory/time limits for 227M records,
    # we simulate the scalability loop specifically logging the num_participating and duration.
    # In a real environment, 100% participation (54 clients) will process all clients in parallel.
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

    for rate in rates:
        run_name = f"participation_{int(rate * 100)}pct"
        logger.info("--- Run: %s ---", run_name)
        
        run_full_config = config.copy()
        # Ensure we use signguard for the scalability stress test
        run_full_config["strategy"] = "signguard"
        run_full_config["federation"]["signguard"] = {"enabled": True}
        run_full_config['model'] = run_full_config.get('model', {})
        run_full_config['model']['input_dim'] = actual_input_dim
        
        model = create_model(run_full_config)
        strategy = create_strategy(run_full_config)
        
        server = FederationServer(
            model, strategy, num_rounds=2, participation_rate=rate
        )
        
        clients = {}
        for cid in fed_config['clients']:
            if cid not in train_loaders: continue
            c_cfg = ClientConfig(client_id=cid, local_epochs=run_full_config['federation']['local_epochs'],
                                 learning_rate=run_full_config['federation']['learning_rate'], batch_size=batch_size)
            
            clients[cid] = FederatedClient(
                c_cfg, train_loaders[cid], val_loaders[cid],
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
             "num_participating": h.num_participating, "num_accepted": h.num_accepted, "duration": h.duration_seconds} 
             for h in history
        ]
        logger.info("  Completed %s. Duration (R[-1]): %.2fs", run_name, history[-1].duration_seconds)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "e5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("E5 complete! Results saved to results/e5_results.json")


if __name__ == "__main__":
    main()
