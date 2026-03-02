"""Generate all evaluation plots from JSON results."""

import json
from pathlib import Path
from src.evaluation.plots import (
    plot_convergence_curves,
    plot_attack_heatmap,
    plot_dp_tradeoff,
    plot_communication_cost
)

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        print("Results directory not found. Please run experiments first.")
        return

    # 1. E2 Convergence Curves
    e2_path = results_dir / "e2_results.json"
    if e2_path.exists():
        with open(e2_path, "r") as f:
            e2_data = json.load(f)
        
        histories = {}
        for strategy, runs in e2_data.items():
            histories[strategy] = [r.get("global_auprc", 0.0) for r in runs]
            
        plot_convergence_curves(
            histories, 
            output_path=results_dir / "e2_convergence.png"
        )
        print("Generated E2 Convergence Curves.")

    # 3. E4 DP Trade-off
    e4_path = results_dir / "e4_results.json"
    if e4_path.exists():
        with open(e4_path, "r") as f:
            e4_data = json.load(f)
            
        eps_values = []
        auprc_values = []
        for run_name, runs in e4_data.items():
            # Extract eps value from string e.g., "dp_eps_0.1"
            eps_str = run_name.replace("dp_eps_", "")
            eps = float(eps_str) if eps_str != "inf" else 100.0 # Cap inf to 100 on graph
            eps_values.append(eps)
            auprc_values.append(runs[-1].get("global_auprc", 0.0))
            
        # Sort by eps
        eps_values, auprc_values = zip(*sorted(zip(eps_values, auprc_values)))
        plot_dp_tradeoff(
            list(eps_values), 
            {"SignGuard": list(auprc_values)}, 
            output_path=results_dir / "e4_dp_tradeoff.png"
        )
        print("Generated E4 DP Trade-off Curve.")

    # 4. E5 Scalability Communication Cost (Approximated via runtime)
    e5_path = results_dir / "e5_results.json"
    if e5_path.exists():
        with open(e5_path, "r") as f:
            e5_data = json.load(f)
            
        costs = {}
        for title, runs in e5_data.items():
            # Sum up durations directly representing computation/communication overhead
            total_duration = sum(r.get("duration", 0.0) for r in runs)
            costs[title] = total_duration
            
        plot_communication_cost(
            costs, 
            output_path=results_dir / "e5_scalability.png"
        )
        print("Generated E5 Scalability Cost Chart.")

if __name__ == "__main__":
    main()
