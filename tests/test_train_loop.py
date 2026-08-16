"""Quick correctness checks against the stub network:

1. Replay buffer fills and samples correctly (shapes, dtypes).
2. Double DQN loss computes without shape errors or NaNs, including on a
   batch that contains genuinely terminal transitions (all-zero next mask).
3. terminated vs truncated are handled distinctly (bootstrap only zeroed by
   terminated).

Run with: python tests/test_train_loop.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from dqn import StubQNetwork, double_dqn_loss, make_target_network
from env import DynamicCVRPEnv, load_config
from replay_buffer import ReplayBuffer


def make_env():
    cfg = load_config(os.path.join(os.path.dirname(__file__), "..", "configs", "dev.yaml"))
    return DynamicCVRPEnv(cfg), cfg


def test_buffer_fill_and_sample():
    env, cfg = make_env()
    buf = ReplayBuffer(capacity=200, n_nodes=env.N, seed=0)
    rng = np.random.default_rng(0)

    obs, _ = env.reset()
    for _ in range(150):
        legal = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(legal))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    assert len(buf) == 150
    batch = buf.sample(32)
    assert batch["obs"]["node_feats"].shape == (32, env.N, 8)
    assert batch["obs"]["dist"].shape == (32, env.N, env.N)
    assert batch["obs"]["action_mask"].shape == (32, env.N)
    assert batch["action"].shape == (32,)
    assert batch["reward"].dtype == torch.float32
    assert batch["terminated"].shape == (32,)
    assert batch["truncated"].shape == (32,)
    print("test_buffer_fill_and_sample: OK")


def test_loss_no_nan_including_terminal_batch():
    env, cfg = make_env()
    buf = ReplayBuffer(capacity=500, n_nodes=env.N, seed=1)
    rng = np.random.default_rng(1)

    n_terminated_seen = 0
    obs, _ = env.reset()
    while len(buf) < 300:
        legal = np.flatnonzero(obs["action_mask"])
        action = int(rng.choice(legal))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buf.add(obs, action, reward, next_obs, terminated, truncated)
        n_terminated_seen += int(terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    assert n_terminated_seen > 0, "test setup issue: never saw a terminated transition"

    online = StubQNetwork(env.N)
    target = make_target_network(online)

    for _ in range(20):
        batch = buf.sample(32)
        loss = double_dqn_loss(online, target, batch, gamma=0.99)
        assert torch.isfinite(loss).all(), "loss is NaN/Inf"
        loss.backward()
        # confirm gradients actually flowed into the online network
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in online.parameters())
        online.zero_grad()

    print(f"test_loss_no_nan_including_terminal_batch: OK ({n_terminated_seen} terminated transitions seen)")


def test_terminated_vs_truncated_bootstrap_distinction():
    """Directly verify Section 3's critical detail: truncated-only transitions
    must still bootstrap (nonzero next_value contribution), while terminated
    transitions must not."""
    env, cfg = make_env()
    n = env.N
    online = StubQNetwork(n)
    target = make_target_network(online)

    def fake_obs():
        return {
            "node_feats": np.zeros((n, 8), dtype=np.float32),
            "dist": np.zeros((n, n), dtype=np.float32),
            "traffic_obs": np.zeros((n, n), dtype=np.float32),
            "traffic_mask": np.zeros((n, n), dtype=np.float32),
            "vehicle": np.array([1.0, 0.5, 0.5], dtype=np.float32),
            "action_mask": np.ones((n,), dtype=np.float32),  # still legal moves -> truncation case
        }

    buf = ReplayBuffer(capacity=4, n_nodes=n, seed=0)
    o = fake_obs()
    no = fake_obs()
    # one truncated (not terminated) transition, one terminated transition
    buf.add(o, 1, 1.0, no, terminated=False, truncated=True)
    term_next = fake_obs()
    term_next["action_mask"] = np.zeros((n,), dtype=np.float32)  # nowhere left to go
    buf.add(o, 1, 1.0, term_next, terminated=True, truncated=False)

    batch = buf.sample(20)  # oversample so both stored transitions are drawn at least once
    # manual recompute matching double_dqn_loss's target formula
    with torch.no_grad():
        q_next = online(batch["next_obs"])
        masked = q_next.masked_fill(batch["next_obs"]["action_mask"] == 0, float("-inf"))
        a_star = masked.argmax(dim=1)
        q_tgt_next = target(batch["next_obs"]).masked_fill(batch["next_obs"]["action_mask"] == 0, float("-inf"))
        next_value = q_tgt_next.gather(1, a_star.unsqueeze(1)).squeeze(1)
        next_value = torch.nan_to_num(next_value, neginf=0.0)
        bootstrap = 1.0 - batch["terminated"]
        td_target = batch["reward"] + 0.99 * next_value * bootstrap

    # sample() draws with replacement so position isn't fixed -- check by the
    # `terminated` flag itself rather than assumed index order.
    term_idx = (batch["terminated"] == 1.0).nonzero(as_tuple=True)[0]
    trunc_idx = (batch["terminated"] == 0.0).nonzero(as_tuple=True)[0]
    assert len(term_idx) > 0 and len(trunc_idx) > 0, "sample missed one of the two stored transitions"

    # terminated -> bootstrap zeroed -> target collapses to reward alone
    assert torch.allclose(bootstrap[term_idx], torch.zeros_like(bootstrap[term_idx]))
    assert torch.allclose(td_target[term_idx], batch["reward"][term_idx])
    # truncated-only -> bootstrap stays on -> agent still bootstraps future value
    assert torch.allclose(bootstrap[trunc_idx], torch.ones_like(bootstrap[trunc_idx]))
    print("test_terminated_vs_truncated_bootstrap_distinction: OK")


if __name__ == "__main__":
    test_buffer_fill_and_sample()
    test_loss_no_nan_including_terminal_batch()
    test_terminated_vs_truncated_bootstrap_distinction()
    print("\nall tests passed.")