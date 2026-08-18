import argparse
import os
import re
from pathlib import Path
import numpy as np

from src.env import DynamicCVRPEnv, load_config
from src.eval import DQNGreedyPolicy, PPOGreedyPolicy, evaluate, write_csv

def get_checkpoints(prefix):
    ckpts = list(Path("checkpoints").glob(f"{prefix}_step*.pt"))
    def extract_step(p):
        try:
            return int(p.stem.split("step")[-1])
        except ValueError:
            return 0
    ckpts.sort(key=extract_step)
    return [(ckpt, extract_step(ckpt)) for ckpt in ckpts]

def main():
    cfg = load_config("configs/default.yaml")
    env = DynamicCVRPEnv(cfg)
    n_nodes = env.N
    seeds = [42, 43, 44, 45, 46]
    
    all_rows = []
    
    dqn_ckpts = get_checkpoints("gnn_dqn_seed42")
    for ckpt_path, step in dqn_ckpts:
        print(f"Evaluating DQN step {step}...")
        policy = DQNGreedyPolicy(str(ckpt_path), n_nodes)
        rows = evaluate(env, policy, "gnn_dqn", seeds, step=step)
        all_rows.extend(rows)
        
    ppo_ckpts = get_checkpoints("gnn_ppo_seed42")
    for ckpt_path, step in ppo_ckpts:
        print(f"Evaluating PPO step {step}...")
        policy = PPOGreedyPolicy(str(ckpt_path), n_nodes)
        rows = evaluate(env, policy, "gnn_ppo", seeds, step=step)
        all_rows.extend(rows)
        
    out_path = "results/learning_curve_data.csv"
    write_csv(all_rows, out_path)
    print(f"Saved true evaluated data to {out_path}")

if __name__ == "__main__":
    main()
