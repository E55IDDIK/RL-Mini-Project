import sys
import csv
from pathlib import Path


#make sure src is importable regardless where we run it from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.env import load_config, DynamicCVRPEnv
from src.baselines import RandomPolicy, GreedyNearestNeighbor

RESULTS_SCHEMA = [
    "method", "seed", "step", "episode", "episode_return",
    "total_cost","n_served","n_customers",
    "terminated","truncated",
    "n_on_time","n_waited","n_late"
]

def run_episode(env:DynamicCVRPEnv, policy, seed:int)->dict:
    obs,info = env.reset(seed=seed)
    policy.reset()
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

def evaluate(env, policy, method_name:str, seeds, step:int=0)->list:
    rows = []
    for i, seed in enumerate(seeds):
        result = run_episode(env, policy, seed)
        rows.append({"method": method_name, "seed": seed, "step": step, "episode": i, **result})
    return rows

def write_csv(rows:list, path:str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_SCHEMA)
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    env = DynamicCVRPEnv(cfg)
    n_eval = cfg["evaluation"]["n_episodes"]
    seeds = list(range(cfg["seed"], cfg["seed"] + n_eval))
    policies = {
        "random": RandomPolicy(seed=cfg["seed"]),
        "greedy": GreedyNearestNeighbor(),
    }

    all_rows = []
    for method_name, policy in policies.items():
        rows = evaluate(env, policy, method_name, seeds)
        all_rows.extend(rows)
        

        returns = [r["episode_return"] for r in rows]
        costs = [r["total_cost"] for r in rows]
        print(f"{method_name:10s}  episode_return: mean={np.mean(returns):2.2f}  std={np.std(returns):2.2f}"
              f"   |   total_cost: mean={np.mean(costs):2.2f}  std={np.std(costs):2.2f}")

    out_path = "results/baselines.csv"
    write_csv(all_rows, out_path)
    print(f"\nwrote {len(all_rows)} rows to {out_path}")