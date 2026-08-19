# Dynamic Vehicle Routing under Uncertain Traffic — Deep Reinforcement Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-3C2179?logo=pytorch&logoColor=white" alt="PyG">
  <img src="https://img.shields.io/badge/Gymnasium-0081A5" alt="Gymnasium">
  <img src="https://img.shields.io/badge/status-completed-brightgreen" alt="Status">
</div>

> A Graph Neural Network–based Deep Reinforcement Learning framework for the
> **Capacitated Vehicle Routing Problem (CVRP)** under **dynamic, partially
> observable traffic** — formulated as a **POMDP** and benchmarked against
> classical heuristics.

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Approach](#approach)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Viewing the Results](#viewing-the-results)
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

- **Nodes** : the depot (fixed start/end) and customers, each with a delivery demand.
- **Edges** : base cost is the distance between two nodes.
- **Dynamics & uncertainty** : traffic is drawn from a spatially-correlated
  field once per episode and is **only locally observable**, making this a
  **Partially Observable MDP (POMDP)**.
- **Objective** : serve every customer exactly once and return to the depot at
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
- *Note on GRU belief state: Initially planned, the GRU was scoped out because the replay buffer shuffles transitions (breaking the temporal sequence) and the `traffic_mask` already sufficiently accumulates relevant temporal information.*
- **Action masking** so the policy only ever selects legal moves, during both
  training and evaluation.

They differ only in the decision head and the learning rule — value estimation
for DQN vs. a stochastic policy for PPO.

## Results

Both learned agents demonstrate a strong ability to adapt to dynamic traffic conditions. Comparing the best-performing learned policy (GNN-DQN) against the Greedy Nearest-Neighbor baseline, on the same 100 held-out seeds:

| Metric | Improvement | Details |
|---|---|---|
| **Return** | **+97%** (-57.71 → -1.69) | GNN-DQN vs. Greedy Nearest-Neighbor, mean episode return. |
| **Late Deliveries** | **43.1% → 2.7%** | Massive reduction in the rate of late deliveries, proving effective dynamic routing. |

GNN-PPO shows the same qualitative pattern (return -19.91, late rate 12.5%) but a smaller effect, consistent with it still being undertrained relative to GNN-DQN at the shared training-step cutoff — see `report.pdf` (Section 7, Discussion) for the full analysis. These results indicate the framework successfully generalizes to hold-out maps and correctly adapts to locally observed traffic in real-time. The full write-up — problem formulation, architecture, training setup, and this comparison — is assembled automatically into [`report.pdf`](report.pdf) (see [Viewing the Results](#viewing-the-results)).

## Repository Structure

```
RL-Mini-Project/
├── report.pdf              # <- the final written report (generated, see below)
├── run_all.sh / run_all.bat  # regenerates every result, figure, and report.pdf in one command
├── schema                  # the required submission folder layout (reference)
├── notebook.md              # weekly lab log: every design decision, dated, with evidence
├── configs/
│   └── default.yaml         # every hyperparameter (env, reward, DQN, PPO, evaluation) — no hardcoded values
├── src/
│   ├── env.py                # DynamicCVRPEnv — the Gymnasium POMDP environment
│   ├── baselines.py          # Random Policy, Greedy Nearest-Neighbor
│   ├── gnn_encoder.py        # shared NNConv perception backbone
│   ├── dqn.py                 # GNN-DQN network + Double DQN loss
│   ├── replay_buffer.py       # DQN replay buffer
│   ├── train_dqn.py           # trains GNN-DQN, writes checkpoints + logs/gnn_dqn/
│   ├── ppo.py                  # GNN-PPO actor-critic + PPO loss
│   ├── train_ppo.py            # trains GNN-PPO, writes checkpoints + logs/gnn_ppo/
│   ├── eval_all_checkpoints.py # evaluates every saved checkpoint -> results/learning_curve_data.csv
│   ├── eval.py                  # final 4-method comparison -> results/eval_comparison.csv
│   ├── plot.py / plot.ipynb     # renders figures/*.pdf from the results/ CSVs
│   ├── visualize.py             # renders videos/*.mp4 and the interactive dashboard
│   └── generate_report.py       # assembles report.pdf from configs/ + results/ + figures/
├── checkpoints/    # trained model weights, one file per checkpoint step — generated
├── logs/           # TensorBoard event logs — generated
├── results/        # *.csv tables (schema: seed, step, episode_return, ...) — generated
├── figures/        # *.pdf vector figures (mean + shaded std) — generated
├── videos/         # *.mp4 before/after routing videos + interactive_dashboard.html — generated
└── requirements.txt
```

Everything marked "generated" is reproducible output — safe to delete and
regenerate at any time with `run_all.sh` / `run_all.bat`, as long as
`checkpoints/*.pt` exist (train first if starting from a truly empty clone).

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

## Running the Project

The trained checkpoints shipped with this repo (`checkpoints/*_step*.pt`) mean
you do **not** need to train anything to reproduce the results and the report
— `run_all` regenerates everything downstream of those checkpoints from
scratch. Training from zero is only needed if you want new checkpoints
(different hyperparameters, a different seed, more steps, etc.).

### Reproduce everything (results, figures, report.pdf) from the shipped checkpoints

```bash
# Linux / macOS
./run_all.sh

# Windows
run_all.bat
```

This runs, in order: (1) evaluate every saved checkpoint to build the
learning-curve data, (2) evaluate the final checkpoints against the two
baselines on 100 held-out seeds, (3) render the figures, (4) assemble
`report.pdf`. It then verifies every expected output file actually exists and
is non-empty, and exits non-zero if anything is missing — so a green run is a
real guarantee, not just "the script didn't crash".

### Train new checkpoints from scratch

```bash
python -m src.train_dqn      # writes checkpoints/gnn_dqn_seed42_step*.pt, logs/gnn_dqn/
python -m src.train_ppo      # writes checkpoints/gnn_ppo_seed42_step*.pt, logs/gnn_ppo/
./run_all.sh                 # or run_all.bat — regenerate results/figures/report.pdf against the new checkpoints
```

To try a different graph size, reward weights, or training budget, edit
[`configs/default.yaml`](configs/default.yaml) — every parameter used by the
environment and both agents lives there, nothing is hardcoded in `src/`.

### Regenerate only the report (after results/ and figures/ already exist)

```bash
python -m src.generate_report
```

## Viewing the Results

**1. The written report — start here.**
Open [`report.pdf`](report.pdf) in any PDF viewer. It covers the problem
formulation (POMDP, Manhattan distance, traffic observation radius, time
windows), the architecture (including *why* the GRU belief state was dropped),
the training setup for both agents, the training-convergence figures, the
four-method comparison table and figure, and a discussion of the results —
regenerate it any time with `python -m src.generate_report` and its numbers
will always match the current contents of `results/eval_comparison.csv`.

**2. The lab notebook — the "why" behind every decision.**
[`notebook.md`](notebook.md) is a dated log (claim / evidence / decision) of
every design choice made across the four weeks — useful if you want the
reasoning behind a specific line of code, not just what it does.

**3. Raw numbers and vector figures.**
- `results/eval_comparison.csv` — one row per (method, seed) for the final
  100-episode comparison; `results/learning_curve_data.csv` — one row per
  (method, checkpoint step) used to draw the convergence figures.
- `figures/*.pdf` — the same three figures embedded in the report, as
  standalone vector PDFs.

**4. Live/interactive extras.**
- `tensorboard --logdir logs` then open `http://localhost:6006` for live
  training curves (separate `gnn_dqn` / `gnn_ppo` runs, toggle to overlay).
- `videos/*.mp4` — before-vs-after routing videos per method/seed, and
  `videos/interactive_dashboard.html` — open it directly in a browser, no
  server needed.

## Roadmap

Aligned with the 4-week plan in the project specification.

- **Week 1 - Environment & Baselines**
  - [x] Graph instance generator with spatially-correlated stochastic traffic
  - [x] Gymnasium POMDP environment (state / action / reward, action masking, capacity & reload)
  - [x] Random policy (used as the environment smoke test)
  - [x] Greedy Nearest-Neighbor baseline + evaluation harness
- **Week 2 - GNN Encoder & GNN-DQN**
  - [x] Shared GNN encoder (`NNConv`)
  - [x] GNN-DQN (Double DQN) with action masking + reproducible checkpoints
- **Week 3 - GNN-PPO**
  - [x] GNN-PPO (actor-critic) with action masking + checkpoints
- **Week 4 - Evaluation & Report**
  - [x] Common evaluation protocol across all four methods
  - [x] Comparison tables and figures + final report

## Tech Stack

- **Language** : Python 3.10+
- **RL environment** : Gymnasium
- **Deep learning** : PyTorch, PyTorch Geometric (`NNConv`)
- **Training / logging** : TensorBoard
- **Data analysis & Visualization** : Pandas, Matplotlib, Seaborn, Plotly, imageio
- **Report generation** : ReportLab, pypdfium2 (`report.pdf`)

## Team

**Sultan Moulay Slimane University** - Multidisciplinary Faculty of Beni Mellal
Master's Program of Excellence - *Data Science and Information Systems Security* (2025–2026)

**Group**
- KHARBOUCH Essiddik
- ER-RACHEDY Sana
- BOUCHFIRA Hind
- KHARBOUCH Yahya

**Supervisors**
- Pr. Hamza ALLAGA
- Pr. Meriem EL HARKAOUI
