"""GNN-PPO training loop for the Dynamic CVRP environment.

Usage:
    python src/train_ppo.py --config configs/dev.yaml
    python src/train_ppo.py --config configs/default.yaml --steps 20000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# make sure src is importable regardless where we run it from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.env import DynamicCVRPEnv, load_config
from src.ppo import GNNActorCritic, PPORolloutBuffer, compute_ppo_loss, OBS_KEYS

# columns for results/.csv -- identical schema to train.py (GNN-DQN)
CSV_FIELDS = [
    "method", "seed", "step", "episode", "episode_return", "total_cost",
    "n_served", "n_customers", "terminated", "truncated",
    "n_on_time", "n_waited", "n_late",
]


# ---------------------------------------------------------------- evaluation
def run_eval_episode(env: DynamicCVRPEnv, ac: GNNActorCritic, device: str,
                     seed: int | None = None) -> dict:
    """Run one full episode greedily (argmax over logits) to evaluate the
    current policy.  Returns a result dict compatible with CSV_FIELDS."""
    obs, info = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    ac.eval()
    while not done:
        with torch.no_grad():
            obs_t = {k: torch.as_tensor(obs[k], device=device).unsqueeze(0)
                     for k in OBS_KEYS}
            # get action greedily (deterministic evaluation)
            action, _, _, _ = ac.get_action_and_value(obs_t)
            # override with greedy argmax: re-compute logits and take argmax
            logits, _, mask = ac._logits_and_value(obs_t)
            masked_logits = logits.masked_fill(mask == 0, float("-inf"))
            action = int(masked_logits.argmax(dim=1).item())
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        done = terminated or truncated
    ac.train()
    return {
        "episode_return": ep_return,
        "total_cost": info["total_cost"],
        "n_served": info["n_served"],
        "n_customers": info["n_customers"],
        "terminated": int(terminated),
        "truncated": int(truncated),
        "n_on_time": env.n_on_time,
        "n_waited": env.n_waited,
        "n_late": env.n_late,
    }


# ----------------------------------------------------------------- CSV utils
def init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_row(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


# ----------------------------------------------------------------- training
def train(cfg: dict, device: str = "cpu") -> None:
    pcfg = cfg["ppo"]
    seed = int(pcfg.get("seed", cfg.get("seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = DynamicCVRPEnv(cfg)
    eval_env = DynamicCVRPEnv(cfg)  # separate instance so eval doesn't disturb training
    n_nodes = env.N

    ac = GNNActorCritic(n_nodes).to(device)
    optimizer = torch.optim.Adam(ac.parameters(), lr=float(pcfg["learning_rate"]))

    # read config
    total_steps     = int(pcfg["total_steps"])
    rollout_length  = int(pcfg["rollout_length"])
    batch_size      = int(pcfg["batch_size"])
    n_epochs        = int(pcfg["n_epochs"])
    gamma           = float(pcfg["gamma"])
    gae_lambda      = float(pcfg["gae_lambda"])
    clip_eps        = float(pcfg["clip_eps"])
    vf_coef         = float(pcfg["vf_coef"])
    ent_coef        = float(pcfg["ent_coef"])
    max_grad_norm   = float(pcfg["max_grad_norm"])
    eval_every      = int(pcfg["eval_every"])
    checkpoint_every = int(pcfg["checkpoint_every"])
    n_eval_seeds    = 5

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "gnn_ppo.csv")
    init_csv(results_path)

    # TensorBoard logging
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("logs", "gnn_ppo")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to {log_dir}")
    except ImportError:
        writer = None
        print("TensorBoard not available, skipping TB logging.")

    obs, info = env.reset()
    global_step = 0
    episode = 0
    ep_return = 0.0

    next_eval_step = eval_every
    next_ckpt_step = checkpoint_every

    print(f"Starting GNN-PPO training: {total_steps} total steps, "
          f"rollout_length={rollout_length}, batch_size={batch_size}, "
          f"n_epochs={n_epochs}")
    t_start = time.time()

    while global_step < total_steps:
        # ======== Phase 1: Collect a rollout ========
        buffer = PPORolloutBuffer(rollout_length, n_nodes)
        ac.eval()  # no dropout etc., but we still need gradients=off for collection

        for t in range(rollout_length):
            with torch.no_grad():
                obs_t = {k: torch.as_tensor(obs[k], device=device).unsqueeze(0)
                         for k in OBS_KEYS}
                action, log_prob, _, value = ac.get_action_and_value(obs_t)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.add(obs, action, reward, terminated, truncated, log_prob, value)

            ep_return += reward
            obs = next_obs
            global_step += 1
            done = terminated or truncated

            if done:
                episode += 1
                ep_return = 0.0
                obs, info = env.reset()

            if global_step >= total_steps:
                break

        # ======== Phase 2: Compute GAE advantages ========
        # bootstrap value for the state after the last collected transition
        with torch.no_grad():
            obs_t = {k: torch.as_tensor(obs[k], device=device).unsqueeze(0)
                     for k in OBS_KEYS}
            last_value = ac.get_value(obs_t).item()

        buffer.compute_gae(last_value, gamma, gae_lambda)

        # ======== Phase 3: PPO update (multiple epochs over the rollout) ========
        ac.train()
        for epoch in range(n_epochs):
            for batch in buffer.get_batches(batch_size, device=device):
                loss, loss_info = compute_ppo_loss(ac, batch, clip_eps, vf_coef, ent_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()

        # TensorBoard: log training losses from the last mini-batch
        if writer is not None:
            writer.add_scalar("train/policy_loss", loss_info["policy_loss"], global_step)
            writer.add_scalar("train/value_loss", loss_info["value_loss"], global_step)
            writer.add_scalar("train/entropy", loss_info["entropy"], global_step)
            writer.add_scalar("train/approx_kl", loss_info["approx_kl"], global_step)

        # ======== Phase 4: Periodic evaluation (after network update) ========
        if global_step >= next_eval_step:
            eval_seeds = [seed + i for i in range(n_eval_seeds)]
            results = [run_eval_episode(eval_env, ac, device, seed=s)
                       for s in eval_seeds]

            mean_return    = float(np.mean([r["episode_return"] for r in results]))
            mean_cost      = float(np.mean([r["total_cost"] for r in results]))
            mean_served    = float(np.mean([r["n_served"] for r in results]))
            mean_terminated = float(np.mean([r["terminated"] for r in results]))
            mean_truncated = float(np.mean([r["truncated"] for r in results]))
            mean_on_time   = float(np.mean([r["n_on_time"] for r in results]))
            mean_waited    = float(np.mean([r["n_waited"] for r in results]))
            mean_late      = float(np.mean([r["n_late"] for r in results]))

            # TensorBoard
            if writer is not None:
                writer.add_scalar("eval/episode_return", mean_return, global_step)
                writer.add_scalar("eval/total_cost", mean_cost, global_step)
                writer.add_scalar("eval/n_served", mean_served, global_step)
                writer.add_scalar("eval/n_late", mean_late, global_step)

            seed_label = f"{eval_seeds[0]}-{eval_seeds[-1]}"
            row = {
                "method": "gnn_ppo",
                "seed": seed_label,
                "step": global_step,
                "episode": episode,
                "episode_return": mean_return,
                "total_cost": mean_cost,
                "n_served": mean_served,
                "n_customers": results[0]["n_customers"],
                "terminated": mean_terminated,
                "truncated": mean_truncated,
                "n_on_time": mean_on_time,
                "n_waited": mean_waited,
                "n_late": mean_late,
            }
            append_csv_row(results_path, row)
            elapsed = time.time() - t_start
            print(f"[step {global_step}/{total_steps}] eval_return={mean_return:.2f} "
                  f"cost={mean_cost:.2f} n_served={mean_served:.1f}/{results[0]['n_customers']} "
                  f"n_late={mean_late:.1f} ({elapsed:.0f}s)")

            while next_eval_step <= global_step:
                next_eval_step += eval_every

        # ======== Phase 5: Periodic checkpoints ========
        if global_step >= next_ckpt_step:
            ckpt_path = os.path.join(
                "checkpoints", f"gnn_ppo_seed{seed}_step{global_step}.pt")
            torch.save({
                "step": global_step,
                "ac_state_dict": ac.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"[step {global_step}] checkpoint saved -> {ckpt_path}")

            while next_ckpt_step <= global_step:
                next_ckpt_step += checkpoint_every

    if writer is not None:
        writer.close()
    print("training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dev.yaml")
    parser.add_argument("--steps", type=int, default=None,
                        help="override ppo.total_steps")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["ppo"]["total_steps"] = args.steps

    # import nn here for clip_grad_norm_ used in the training loop
    import torch.nn as nn
    train(cfg, device=args.device)
