"""
src/ablation_observation_radius.py

Ablation Study (report Section VII): isolates the effect of traffic
observability on GNN-DQN's performance, by re-evaluating the SAME trained
checkpoint (no retraining) under several observation_radius values.

The checkpoint's weights don't depend on observation_radius, only the
environment's traffic_obs/traffic_mask fields change.

Usage:
    python -m src.ablation_observation_radius
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.env import load_config, DynamicCVRPEnv
from src.eval import DQNGreedyPolicy, evaluate, write_csv, find_latest_checkpoint

# Sweep: well below training value, the trained value itself (sanity check --
# should reproduce Section VI's numbers almost exactly), moderately above,
# and near-full observability (max possible Manhattan distance on a [0,1]^2
# map is 2.0, so 1.5 reveals nearly everything, everywhere).
RADII = [0, 0.15, 0.30, 0.50, 1.50]
N_SEEDS = 100  # match Section VI/IV for directly comparable numbers


def main():
    base_cfg = load_config("configs/default.yaml")
    trained_radius = base_cfg["traffic"]["observation_radius"]
    print(f"Checkpoint was trained with observation_radius = {trained_radius}")

    ckpt_path = find_latest_checkpoint("gnn_dqn_seed42")
    if ckpt_path is None:
        raise FileNotFoundError(
            "No GNN-DQN checkpoint found in checkpoints/. "
            "Expected e.g. checkpoints/gnn_dqn_seed42_step20000.pt"
        )
    print(f"Using checkpoint: {ckpt_path}\n")

    seeds = list(range(base_cfg["seed"], base_cfg["seed"] + N_SEEDS))
    all_rows = []
    summary = []

    for radius in RADII:
        cfg = copy.deepcopy(base_cfg)
        cfg["traffic"]["observation_radius"] = radius
        env = DynamicCVRPEnv(cfg)
        n_nodes = env.N

        policy = DQNGreedyPolicy(str(ckpt_path), n_nodes)
        tag = f"gnn_dqn_r{radius}"
        print(f"Evaluating at observation_radius = {radius} ...")
        rows = evaluate(env, policy, tag, seeds, step=0)
        all_rows.extend(rows)

        returns = np.array([r["episode_return"] for r in rows])
        costs = np.array([r["total_cost"] for r in rows])
        lates = np.array([r["n_late"] for r in rows])
        served = np.array([r["n_served"] for r in rows])
        late_rate = 100.0 * lates.sum() / max(served.sum(), 1)

        summary.append((radius, returns.mean(), returns.std(ddof=1),
                        costs.mean(), costs.std(ddof=1), late_rate))

    out_path = "results/ablation_observation_radius.csv"
    write_csv(all_rows, out_path)

    print(f"\nSaved {len(all_rows)} rows to {out_path}\n")
    print(f"{'radius':>8} {'return (mean+-std)':>22} {'cost (mean+-std)':>20} {'late %':>8}")
    for radius, rm, rs, cm, cs, lr in summary:
        tag = "  <- trained value" if radius == trained_radius else ""
        print(f"{radius:8.2f} {rm:9.2f} +/- {rs:6.2f} {cm:8.2f} +/- {cs:5.2f} {lr:7.1f}%{tag}")


if __name__ == "__main__":
    main()
