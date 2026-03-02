"""E3: Attack robustness matrix.

Evaluates combinations of Attacks, Defenses, and Intensities.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import time
from collections import OrderedDict

from src.data.loaders import get_all_client_loaders
from src.federation.client import ClientConfig, FederatedClient, ClientUpdate
from src.federation.runner import create_model, create_strategy, load_experiment_config
from src.federation.server import FederationServer
from src.evaluation.metrics import compute_all_metrics

# Attacks
from src.attacks.random_poison import RandomPoisonAttack
from src.attacks.model_poison import ModelPoisonAttack
from src.attacks.free_rider import FreeRiderAttack
from src.attacks.sybil import SybilAttack
from src.signguard.signing import UpdateSigner

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_path = Path("configs/base.yaml")
    exp_path = Path("configs/experiment_3.yaml")
    fed_path = Path("data/partitioned/federation_state.yaml")
    splits_dir = Path("data/splits")
    
    config = load_experiment_config(base_path, exp_path)
    if fed_path.exists():
        fed_config = load_experiment_config(fed_path)
    else:
        logger.error("Federation config not found!")
        return

    logger.info("=" * 60)
    logger.info("E3: Attack Robustness Matrix")
    logger.info("=" * 60)

    batch_size = config['federation']['batch_size']
    train_loaders = get_all_client_loaders(splits_dir, "train", batch_size)
    val_loaders = get_all_client_loaders(splits_dir, "val", batch_size)
    
    attacks = config.get("attacks", [])
    defenses = config.get("defenses", [])
    intensities = config.get("intensities", [0.10])
    
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

    # For demo speed: pick 1 intensity, 2 attacks, 2 defenses, 1 round
    demo_attacks = attacks
    demo_defenses = defenses
    demo_intensities = intensities
    
    logger.info("Running full matrix (%d attacks x %d defenses x %d intensity)", 
                len(demo_attacks), len(demo_defenses), len(demo_intensities))

    for attack_cfg in demo_attacks:
        attack_type = attack_cfg["type"]
        for defense_cfg in demo_defenses:
            defense_strategy = defense_cfg["strategy"]
            for intensity in demo_intensities:
                run_name = f"{attack_type}_vs_{defense_strategy}_int{intensity}"
                logger.info("--- Run: %s ---", run_name)
                
                run_full_config = config.copy()
                run_full_config["strategy"] = defense_strategy
                run_full_config['model'] = run_full_config.get('model', {})
                run_full_config['model']['input_dim'] = actual_input_dim
                
                if defense_strategy == "signguard" and "signguard" in defense_cfg:
                    run_full_config["federation"]["signguard"] = defense_cfg["signguard"]
                    
                model = create_model(run_full_config)
                strategy = create_strategy(run_full_config)
                
                server = FederationServer(
                    model, strategy, num_rounds=1, participation_rate=0.05
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

                # Setup attack
                n_compromised = max(1, int(len(clients) * intensity))
                if attack_type == "random_poison":
                    attack_obj = RandomPoisonAttack(intensity=5.0) # High noise
                elif attack_type == "sybil":
                    attack_obj = SybilAttack(num_sybils=n_compromised, intensity=2.0)
                else:
                    attack_obj = ModelPoisonAttack(intensity=2.0)

                def eval_fn(round_num, global_params):
                    from src.federation.models import set_model_params
                    set_model_params(model, global_params)
                    metrics = evaluate_model(model, proxy_val_loader)
                    return {"global_auprc": metrics.get("auprc", 0.0), "global_loss": metrics.get("test_loss", 0.0)}

                def attack_updates_fn(r, global_params):
                    # Generate a baseline honest update to poison
                    base_cid = list(clients.keys())[0]
                    honest_update = clients[base_cid].train_local(global_params, r)
                    
                    malicious_updates = []
                    if attack_type == "sybil":
                        poi_dicts = attack_obj.generate_sybil_updates(honest_update.delta, global_params, r)
                        for i, p_dict in enumerate(poi_dicts):
                            malicious_updates.append(ClientUpdate(
                                client_id=f"sybil_{i}", round_num=r,
                                delta=p_dict, num_samples=honest_update.num_samples,
                                local_loss=honest_update.local_loss
                            ))
                    else:
                        for i in range(n_compromised):
                            p_dict = attack_obj.poison_update(honest_update.delta, global_params, r)
                            malicious_updates.append(ClientUpdate(
                                client_id=f"attacker_{i}", round_num=r,
                                delta=p_dict, num_samples=honest_update.num_samples,
                                local_loss=honest_update.local_loss
                            ))
                    return malicious_updates

                history = server.run_training(clients, attack_updates_fn=attack_updates_fn, eval_fn=eval_fn, log_every=1)
                
                results[run_name] = [
                    {"round": h.round_num, "global_auprc": h.metrics.get("global_auprc", 0.0), 
                     "num_participating": h.num_participating, "num_accepted": h.num_accepted} 
                     for h in history
                ]
                logger.info("  Completed %s. AUPRC: %.4f", run_name, history[-1].metrics.get("global_auprc", 0.0))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "e3_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("E3 complete! Results saved to results/e3_results.json")


if __name__ == "__main__":
    main()
