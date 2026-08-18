"""GNN-DQN training loop for the Dynamic CVRP environment.

Usage:
    python src/train_dqn.py --config configs/dev.yaml
    python src/train_dqn.py --config configs/default.yaml --steps 20000
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

from src.dqn import (
    double_dqn_loss,
    make_target_network,
    select_action,
    sync_target_network,
)
from src.env import DynamicCVRPEnv, load_config
from src.gnn_encoder import GNNQNetwork
from src.replay_buffer import ReplayBuffer

# columns for results/.csv to help compare
CSV_FIELDS = [
    "method", "seed", "step", "episode", "episode_return", "total_cost",
    "n_served", "n_customers", "terminated", "truncated",
    "n_on_time", "n_waited", "n_late",
]

# exploration vs exploitaion 
def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    frac = min(1.0, step / decay_steps)
    return start + frac * (end - start)

# creation the network 
def build_network(n_nodes: int) -> torch.nn.Module:
    return GNNQNetwork(n_nodes)

#evaluation 
def run_eval_episode(env: DynamicCVRPEnv, network: torch.nn.Module, device: str,
                     seed: int | None = None) -> dict:
    """Run one full episode greedily (epsilon=0) to evaluate the agent's
    performance. Returns a result dict compatible with CSV_FIELDS."""
    obs, info = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    network.eval()
    while not done:
        with torch.no_grad(): #no gradients needed in evaluation
            obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
            q = network(obs_t) #network producing q values
            mask_t = obs_t["action_mask"]
            #invalid actions receive -inf to never be selected by argmax()
            masked_q = q.masked_fill(mask_t == 0, float("-inf"))
            # choosing best action using argmax 
            action = int(masked_q.argmax(dim=1).item())
        # next step
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        done = terminated or truncated
    network.train()
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


def init_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()


def append_csv_row(path: str, row: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)


def train(cfg: dict, device: str = "cpu") -> None:
    #read training config
    tcfg = cfg["training"]
    seed = int(tcfg.get("seed", cfg.get("seed", 42)))
    #generate randomness for DQN exploration
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    env = DynamicCVRPEnv(cfg)
    eval_env = DynamicCVRPEnv(cfg)  # separate instance so eval rollouts don't disturb training state
    n_nodes = env.N
    # main gets updated every training step 
    online = build_network(n_nodes).to(device)
    target = make_target_network(online).to(device)
    # optimizer modifies online net weights
    optimizer = torch.optim.Adam(online.parameters(), lr=float(tcfg["learning_rate"]))
    # experience replay memory
    buffer = ReplayBuffer(capacity=int(tcfg["buffer_size"]), n_nodes=n_nodes, seed=seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join("logs", "gnn_dqn")
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logging to {log_dir}")
    except ImportError:
        writer = None
        print("TensorBoard not available, skipping TB logging.")
    results_path = os.path.join("results", "gnn_dqn.csv")
    init_csv(results_path)  # start fresh for this training run

    # loding training configs
    total_steps = int(tcfg["total_steps"])
    warmup_steps = int(tcfg["warmup_steps"])
    batch_size = int(tcfg["batch_size"])
    gamma = float(tcfg["gamma"])
    eps_start = float(tcfg["epsilon_start"])
    eps_end = float(tcfg["epsilon_end"])
    eps_decay = int(tcfg["epsilon_decay_steps"])
    target_update_freq = int(tcfg["target_update_freq"])
    eval_every = int(tcfg["eval_every"])
    checkpoint_every = int(tcfg["checkpoint_every"])
    n_eval_seeds = 5

    obs, info = env.reset()
    episode = 0
    ep_return = 0.0
    ep_step = 0
    # training loop 
    for step in range(1, total_steps + 1):
        epsilon = linear_epsilon(step, eps_start, eps_end, eps_decay)
        action = select_action(online, obs, epsilon, rng, n_nodes)

        next_obs, reward, terminated, truncated, info = env.step(action)
        # store experience
        buffer.add(obs, action, reward, next_obs, terminated, truncated)
        # update the current state
        ep_return += reward
        ep_step += 1
        obs = next_obs
        done = terminated or truncated

        if done:
            # log the spisode and start another one
            writer.add_scalar("rollout/episode_return", ep_return, step)
            writer.add_scalar("rollout/episode_length", ep_step, step)
            obs, info = env.reset()
            episode += 1
            ep_return = 0.0
            ep_step = 0

        # learning after warming up the buffer 
        if step >= warmup_steps and len(buffer) >= batch_size:
            batch = buffer.sample(batch_size, device=device)
            loss = double_dqn_loss(online, target, batch, gamma)
            # backpropagation 
            optimizer.zero_grad() #clears old grad
            loss.backward() #calculate grad
            optimizer.step() #updates online weights

            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/epsilon", epsilon, step)
            # synching the target net periodically
            if step % target_update_freq == 0:
                sync_target_network(online, target)

        # periodic evaluation every eval_every (average over 5 fixed seeds)
        if step % eval_every == 0:
            eval_seeds = [seed + i for i in range(n_eval_seeds)]
            results = [run_eval_episode(eval_env, online, device, seed=s) for s in eval_seeds]

            mean_return = float(np.mean([r["episode_return"] for r in results]))
            mean_cost = float(np.mean([r["total_cost"] for r in results]))
            mean_served = float(np.mean([r["n_served"] for r in results]))
            mean_terminated = float(np.mean([r["terminated"] for r in results]))
            mean_truncated = float(np.mean([r["truncated"] for r in results]))
            mean_on_time = float(np.mean([r["n_on_time"] for r in results]))
            mean_waited = float(np.mean([r["n_waited"] for r in results]))
            mean_late = float(np.mean([r["n_late"] for r in results]))

            # logging
            writer.add_scalar("eval/episode_return", mean_return, step)
            writer.add_scalar("eval/total_cost", mean_cost, step)
            writer.add_scalar("eval/n_served", mean_served, step)
            writer.add_scalar("eval/n_late", mean_late, step)

            seed_label = f"{eval_seeds[0]}-{eval_seeds[-1]}"
            row = {
                "method": "gnn_dqn",
                "seed": seed_label,
                "step": step,
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
            print(f"[step {step}/{total_steps}] eval_return={mean_return:.2f} "
                  f"n_served={mean_served:.1f}/{results[0]['n_customers']} "
                  f"n_late={mean_late:.1f} eps={epsilon:.3f}")

        # periodic checkpoints to save the model 
        if step % checkpoint_every == 0:
            ckpt_path = os.path.join("checkpoints", f"gnn_dqn_seed{seed}_step{step}.pt")
            torch.save({
                "step": step,
                "online_state_dict": online.state_dict(),
                "target_state_dict": target.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)
            print(f"[step {step}] checkpoint saved -> {ckpt_path}")

    if writer is not None:
        writer.close()
    print("training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dev.yaml")
    parser.add_argument("--steps", type=int, default=None, help="override training.total_steps")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps

    train(cfg, device=args.device)