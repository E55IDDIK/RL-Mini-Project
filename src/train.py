"""GNN-DQN training loop for the Dynamic CVRP environment.

Runs against the StubQNetwork (dqn.py) until the real GNN encoder + GRU +
DQN head is ready -- swap it in via `--network real` (see `build_network`)
once it exists; nothing else in this file needs to change, since both
networks share the exact same (obs) -> Q-values contract.

Usage:
    python src/train.py --config configs/dev.yaml
    python src/train.py --config configs/default.yaml --steps 20000
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from baselines import GreedyNearestNeighbor, RandomPolicy
from dqn import (
    StubQNetwork,
    double_dqn_loss,
    make_target_network,
    select_action,
    sync_target_network,
)
from env import DynamicCVRPEnv, load_config
from replay_buffer import ReplayBuffer

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
def build_network(n_nodes: int, kind: str = "stub") -> torch.nn.Module:
    if kind == "stub":
        return StubQNetwork(n_nodes)
    raise ValueError(
        f"unknown network kind '{kind}' -- the real GNN-GRU-DQN network isn't "
        "wired in yet; pass kind='stub' or add the real network here at "
        "integration time (same (obs) -> Q-values contract, no other "
        "changes needed in this file)."
    )

#evaluation 
def run_eval_episode(env: DynamicCVRPEnv, network: torch.nn.Module, device: str) -> dict:
    """function that runs one full episode, greedy (epsilon=0) to evaluate 
     the agent's performace Returns a result dict."""
    obs, info = env.reset()
    done = False
    ep_return = 0.0
    while not done:
        with torch.no_grad(): #no gradients needed in evaluation
            obs_t = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in obs.items()}
            q = network(obs_t) #nerwork producing q values
            mask_t = obs_t["action_mask"]
            #invalid actions receive -inf to never be selected by argmax()
            masked_q = q.masked_fill(mask_t == 0, float("-inf"))
            # choosing best action using argmax 
            action = int(masked_q.argmax(dim=1).item())
        # next step
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        done = terminated or truncated
    return {
        "episode_return": ep_return,
        "total_cost": info["total_cost"],
        "n_served": info["n_served"],
        "n_customers": info["n_customers"],
        "terminated": terminated, # type: ignore
        "truncated": truncated, # type: ignore
        "n_on_time": env.n_on_time,
        "n_waited": env.n_waited,
        "n_late": env.n_late,
    }


def append_csv_row(path: str, row: dict) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(cfg: dict, network_kind: str = "stub", device: str = "cpu") -> None:
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
    online = build_network(n_nodes, network_kind).to(device)
    target = make_target_network(online).to(device)
    # optimizer modifies online net weights
    optimizer = torch.optim.Adam(online.parameters(), lr=float(tcfg["learning_rate"]))
    # experience replay memory
    buffer = ReplayBuffer(capacity=int(tcfg["buffer_size"]), n_nodes=n_nodes, seed=seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", f"gnn_dqn_seed{seed}_{int(time.time())}"))
    results_path = os.path.join("results", "gnn_dqn.csv")
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
        # periodic evaluation every eval_every 
        # no-exploration eval + CSV logging 
        if step % eval_every == 0:
            result = run_eval_episode(eval_env, online, device)
            #logging
            writer.add_scalar("eval/episode_return", result["episode_return"], step)
            writer.add_scalar("eval/n_served", result["n_served"], step)

            row = {
                "method": "gnn_dqn",
                "seed": seed,
                "step": step,
                "episode": episode,
                "episode_return": result["episode_return"],
                "total_cost": result["total_cost"],
                "n_served": result["n_served"],
                "n_customers": result["n_customers"],
                "terminated": result["terminated"],
                "truncated": result["truncated"],
                "n_on_time": result["n_on_time"],
                "n_waited": result["n_waited"],
                "n_late": result["n_late"],
            }
            append_csv_row(results_path, row)
            print(f"[step {step}/{total_steps}] eval_return={result['episode_return']:.2f} "
                  f"n_served={result['n_served']}/{result['n_customers']} "
                  f"eps={epsilon:.3f} terminated={result['terminated']}")

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

    writer.close()
    print("training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None, help="override training.total_steps")
    parser.add_argument("--network", type=str, default="stub", choices=["stub"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps

    train(cfg, network_kind=args.network, device=args.device)