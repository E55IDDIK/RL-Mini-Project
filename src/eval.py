"""Unified evaluation script for all methods on the Dynamic CVRP environment.

Evaluates Random, Greedy Nearest-Neighbor, GNN-DQN, and GNN-PPO on the exact
same matched test seeds, computing the 4 key performance metrics from the
cahier des charges.

Usage:
    python src/eval.py
    python src/eval.py --n-episodes 100
    python src/eval.py --n-episodes 5   # quick 5-seed check matching training logs (seeds 42-46)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# make sure src is importable regardless where we run it from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.baselines import Policy, RandomPolicy, GreedyNearestNeighbor
from src.env import DynamicCVRPEnv, load_config
from src.gnn_encoder import GNNQNetwork
from src.ppo import GNNActorCritic

RESULTS_SCHEMA = [
    "method", "seed", "step", "episode", "episode_return",
    "total_cost", "n_served", "n_customers",
    "terminated", "truncated",
    "n_on_time", "n_waited", "n_late",
]


class DQNGreedyPolicy(Policy):
    """Evaluates a trained GNN-DQN checkpoint greedily (argmax over Q-values)."""

    def __init__(self, ckpt_path: str, n_nodes: int, device: str = "cpu"):
        self.device = device
        self.net = GNNQNetwork(n_nodes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.net.load_state_dict(ckpt["online_state_dict"])
        self.net.eval()

    def act(self, obs: dict) -> int:
        with torch.no_grad():
            obs_t = {k: torch.as_tensor(v, device=self.device).unsqueeze(0)
                     for k, v in obs.items()}
            q = self.net(obs_t)
            mask = obs_t["action_mask"]
            masked_q = q.masked_fill(mask == 0, float("-inf"))
            return int(masked_q.argmax(dim=1).item())


class PPOGreedyPolicy(Policy):
    """Evaluates a trained GNN-PPO checkpoint greedily (argmax over logits)."""

    def __init__(self, ckpt_path: str, n_nodes: int, device: str = "cpu"):
        self.device = device
        self.ac = GNNActorCritic(n_nodes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        self.ac.load_state_dict(ckpt["ac_state_dict"])
        self.ac.eval()

    def act(self, obs: dict) -> int:
        with torch.no_grad():
            obs_t = {k: torch.as_tensor(v, device=self.device).unsqueeze(0)
                     for k, v in obs.items()}
            logits, _, mask = self.ac._logits_and_value(obs_t)
            masked_logits = logits.masked_fill(mask == 0, float("-inf"))
            return int(masked_logits.argmax(dim=1).item())


def run_episode(env: DynamicCVRPEnv, policy: Policy, seed: int) -> dict:
    """Run one evaluation episode with matched seed across environment and policy."""
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)
    terminated = truncated = False
    episode_return = 0.0

    while not (terminated or truncated):
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

    return {
        "episode_return": episode_return,
        "total_cost": info["total_cost"],
        "n_served": info["n_served"],
        "n_customers": info["n_customers"],
        "terminated": int(terminated),
        "truncated": int(truncated),
        "n_on_time": env.n_on_time,
        "n_waited": env.n_waited,
        "n_late": env.n_late,
    }


def evaluate(env: DynamicCVRPEnv, policy: Policy, method_name: str,
             seeds: list[int], step: int = 0) -> list[dict]:
    rows = []
    for i, seed in enumerate(seeds):
        result = run_episode(env, policy, seed)
        rows.append({
            "method": method_name,
            "seed": seed,
            "step": step,
            "episode": i,
            **result,
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_SCHEMA)
        writer.writeheader()
        writer.writerows(rows)


def find_latest_checkpoint(prefix: str) -> str | None:
    """Find the highest-step checkpoint for a given prefix in checkpoints/."""
    ckpts = list(Path("checkpoints").glob(f"{prefix}_step*.pt"))
    if not ckpts:
        return None
    # Sort by step number extracted from filename
    def extract_step(p: Path) -> int:
        try:
            return int(p.stem.split("step")[-1])
        except ValueError:
            return 0
    ckpts.sort(key=extract_step)
    return str(ckpts[-1])


def print_summary_table(all_rows: list[dict], seeds: list[int]) -> None:
    methods = []
    seen = set()
    for r in all_rows:
        m = r["method"]
        if m not in seen:
            seen.add(m)
            methods.append(m)

    print("\n" + "=" * 92)
    print(f"EVALUATION SUMMARY ({len(seeds)} episodes, seeds {seeds[0]}..{seeds[-1]})")
    print("=" * 92)
    print(f"{'Method':<16} | {'Return (mean±std)':<18} | {'Cost (mean±std)':<16} | {'On-Time':<8} | {'Waited':<8} | {'Late':<8} | {'Served'}")
    print("-" * 92)

    for m in methods:
        rows = [r for r in all_rows if r["method"] == m]
        returns = [r["episode_return"] for r in rows]
        costs = [r["total_cost"] for r in rows]
        on_time = np.mean([r["n_on_time"] for r in rows])
        waited = np.mean([r["n_waited"] for r in rows])
        late = np.mean([r["n_late"] for r in rows])
        served = np.mean([r["n_served"] for r in rows])
        total_cust = rows[0]["n_customers"]

        ret_str = f"{np.mean(returns):.2f} ± {np.std(returns):.2f}"
        cost_str = f"{np.mean(costs):.2f} ± {np.std(costs):.2f}"

        print(f"{m:<16} | {ret_str:<18} | {cost_str:<16} | {on_time:<8.1f} | {waited:<8.1f} | {late:<8.1f} | {served:.1f}/{total_cust}")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="number of test seeds (defaults to cfg['evaluation']['n_episodes'])")
    parser.add_argument("--dqn-ckpt", type=str, default=None,
                        help="path to GNN-DQN checkpoint (auto-detects if None)")
    parser.add_argument("--ppo-ckpt", type=str, default=None,
                        help="path to GNN-PPO checkpoint (auto-detects if None)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="results/eval_comparison.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = DynamicCVRPEnv(cfg)
    n_nodes = env.N

    n_eval = args.n_episodes if args.n_episodes is not None else cfg["evaluation"]["n_episodes"]
    seeds = list(range(cfg["seed"], cfg["seed"] + n_eval))

    # Setup policies
    policies: dict[str, Policy] = {
        "random": RandomPolicy(seed=cfg["seed"]),
        "greedy": GreedyNearestNeighbor(),
    }

    # Auto-detect or use specified DQN checkpoint
    dqn_path = args.dqn_ckpt or find_latest_checkpoint("gnn_dqn_seed42")
    if dqn_path and os.path.exists(dqn_path):
        print(f"Loading GNN-DQN checkpoint: {dqn_path}")
        policies["gnn_dqn"] = DQNGreedyPolicy(dqn_path, n_nodes, device=args.device)
    else:
        print("Note: GNN-DQN checkpoint not found, skipping DQN evaluation.")

    # Auto-detect or use specified PPO checkpoint
    ppo_path = args.ppo_ckpt or find_latest_checkpoint("gnn_ppo_seed42")
    if ppo_path and os.path.exists(ppo_path):
        print(f"Loading GNN-PPO checkpoint: {ppo_path}")
        policies["gnn_ppo"] = PPOGreedyPolicy(ppo_path, n_nodes, device=args.device)
    else:
        print("Note: GNN-PPO checkpoint not found, skipping PPO evaluation.")

    print(f"Running evaluation on {len(seeds)} matched seeds (seeds {seeds[0]}..{seeds[-1]})...\n")

    all_rows = []
    baseline_rows = []

    for method_name, policy in policies.items():
        rows = evaluate(env, policy, method_name, seeds)
        all_rows.extend(rows)
        if method_name in ("random", "greedy"):
            baseline_rows.extend(rows)

    print_summary_table(all_rows, seeds)

    # Save outputs
    write_csv(all_rows, args.out)
    print(f"Wrote all results to {args.out}")

    # Also update results/baselines.csv with the properly seeded baselines
    if baseline_rows:
        write_csv(baseline_rows, "results/baselines.csv")
        print(f"Updated results/baselines.csv with seeded baseline runs ({len(baseline_rows)} rows)")