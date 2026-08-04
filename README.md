# Dynamic Vehicle Routing under Uncertain Traffic — Deep Reinforcement Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-3C2179?logo=pytorch&logoColor=white" alt="PyG">
  <img src="https://img.shields.io/badge/Gymnasium-0081A5" alt="Gymnasium">
  <img src="https://img.shields.io/badge/status-in%20development-yellow" alt="Status">
</div>

> A Graph Neural Network–based Deep Reinforcement Learning framework for the
> **Capacitated Vehicle Routing Problem (CVRP)** under **dynamic, partially
> observable traffic** — formulated as a **POMDP** and benchmarked against
> classical heuristics.

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Approach](#approach)
- [Installation](#installation)
- [Roadmap](#roadmap)
- [Tech Stack](#tech-stack)
- [Team](#team)

## Overview

Classical vehicle-routing solutions assume travel times are known in advance and
fixed. In reality, congestion, incidents, and changing conditions can make a
route that was optimal at planning time quickly become inefficient. This
project asks:

> How can Reinforcement Learning dynamically adapt routing decisions to
> uncertain traffic while minimizing travel cost and delays?

We build a controllable simulator and train two **GNN-based Deep RL agents** to
route a single capacitated vehicle over a graph whose edge costs vary with
traffic the agent can only observe **locally**. Their performance is compared
against two non-learning baselines on a common, held-out test set.

## Problem Formulation

The environment is a graph over `{depot} ∪ {customers}`:

- **Nodes** — the depot (fixed start/end) and customers, each with a delivery demand.
- **Edges** — base cost is the distance between two nodes.
- **Dynamics & uncertainty** — traffic is drawn from a spatially-correlated
  field once per episode and is **only locally observable**, making this a
  **Partially Observable MDP (POMDP)**.
- **Objective** — serve every customer exactly once and return to the depot at
  minimum total travel cost, without exceeding vehicle capacity (the vehicle
  may return to the depot to reload).

| Component | Definition |
|---|---|
| **State / Observation** | Node features (position, remaining demand, served/current flags), the distance matrix, locally-revealed traffic with an observation mask, and vehicle state (remaining capacity, elapsed time). |
| **Action** | The next node to visit, among currently reachable candidates (invalid moves are masked). |
| **Reward** | Negative effective travel cost per step, a completion bonus for finishing at the depot, and a penalty for any customer left unserved. |

## Approach

Four methods are implemented and compared under identical conditions:

| Method | Type | Learns? | Role |
|---|---|:---:|---|
| **Random Policy** | Heuristic | No | Lower-bound reference |
| **Greedy Nearest-Neighbor** | Heuristic | No | Common-sense reference |
| **GNN-DQN** | Value-based DRL (Double DQN) | Yes | Learned agent |
| **GNN-PPO** | Policy-based DRL (PPO) | Yes | Learned agent |

Both learning agents share the same perception backbone:

- **GNN encoder** (PyTorch Geometric, `NNConv` edge-conditioned convolution) to
  reason over the graph and its edge features.
- **GRU belief state** that summarizes everything observed so far in the
  episode, compensating for partial observability of traffic.
- **Action masking** so the policy only ever selects legal moves, during both
  training and evaluation.

They differ only in the decision head and the learning rule — value estimation
for DQN vs. a stochastic policy for PPO.

## Installation

Requires **Python 3.10+**.

```bash
# Clone the repository
git clone https://github.com/E55IDDIK/RL-Mini-Project.git
cd RL-Mini-Project

# Create and activate a virtual environment (recommended)
python3 -m venv venv
venv\Scripts\activate     # On Windows
# OR
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Roadmap

Aligned with the 4-week plan in the project specification.

- **Week 1 — Environment & Baselines**
  - [ ] Graph instance generator with spatially-correlated stochastic traffic
  - [ ] Gymnasium POMDP environment (state / action / reward, action masking, capacity & reload)
  - [ ] Random policy (used as the environment smoke test)
  - [ ] Greedy Nearest-Neighbor baseline + evaluation harness
- **Week 2 — GNN Encoder & GNN-DQN**
  - [ ] Shared GNN encoder (`NNConv`) + GRU belief state
  - [ ] GNN-DQN (Double DQN) with action masking + reproducible checkpoints
- **Week 3 — GNN-PPO**
  - [ ] GNN-PPO (actor-critic) with action masking + checkpoints
- **Week 4 — Evaluation & Report**
  - [ ] Common evaluation protocol across all four methods
  - [ ] Comparison tables and figures + final report

## Tech Stack

- **Language** — Python 3.10+
- **RL environment** — Gymnasium
- **Deep learning** — PyTorch, PyTorch Geometric (`NNConv`), GRU
- **Training / logging** — TensorBoard
- **Visualization** — Matplotlib, Plotly

## Team

**Sultan Moulay Slimane University** — Multidisciplinary Faculty of Beni Mellal
Master's Program of Excellence — *Data Science and Information Systems Security* (2025–2026)

**Group**
- KHARBOUCH Essiddik
- ER-RACHEDY Sana
- BOUCHFIRA Hind
- KHARBOUCH Yahya

**Supervisors**
- Pr. Hamza ALLAGA
- Pr. Meriem EL HARKAOUI
