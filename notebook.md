# Notebook : Weekly Lab Log

## 2026-08-02 : MDP/POMDP formalization (Week 1)

**Claim :** The project requirements (Section 4.3) describes the problem as an MDP with a state, action and reward. We wanted to clarify the exact definition of each before starting the environment code. We also wanted to determine whether the action space should be considered static or dynamic during an episode.

**Evidence :** In reality, a delivery driver does not know the traffic on every road in the city, they only find out how congested a road actually is once they get close to it, every other information (customers position, orders, vehicle capacity …) is known from the start. That's why this problem is **POMDP** (partially observed MDP).

**Decision :**

**Real State vs Observation :** real state is everything that exists in the environment, whether the agent can see it or not (e.g. traffic everywhere). The observation is only the part the agent actually gets to see.

```
ENVIRONMENT
knows everything
        |
   -----|-----
   |         |
known      hidden
traffic    traffic
   |         |
   v         X
AGENT OBSERVATION
```

**Observation :** differs from state by one thing, the **traffic**, here it's restricted to the current vehicle position.

**State :** as mentioned in the project requirements
- Customers position
- Customers demands
- Traffic
- Vehicle position
- Vehicle capacity
- Visited customers
- Current time

**Action :** We defined a fixed discrete action space consisting of selecting the next customer to visit. Although the action space is fixed, the set of valid actions changes dynamically during an episode through action masking, which filters out infeasible actions (e.g., already served …).

**Reward :** Defined to give feedback at every step; each time the vehicle moves, it receives a negative penalty.

Travel cost = distance × current traffic multiplier

| Component | Sign | When it applies |
|---|---|---|
| Travel cost | Negative, every step | Every move the vehicle makes. |
| Time window violation | Negative | Per occurrence. |
| Successful delivery | Small positive bonus | Each time a customer is served. |
| Unserved customer at episode end | Negative, per customer | Only if the episode is truncated (ran out of time) with people still unserved. |
| Completing the full route | Positive bonus | Only if all customers served and vehicle back at depot. |

## 2026-08-06 : Repository structure & remaining open items (Week 1)

**Claim :** Before writing any code, we needed to align the project with the professor's exact submission requirements, and resolve the open items left from the cahier des charges.

**Evidence :** The submission guidelines require a specific structure (`configs/*.yaml`, `src/env.py`, `results/*.csv`...). The team confirmed synthetic data only for this edition (real data stays a possible Stage 2). The team also resolved the previously open time-windows question : time windows are included in this first version.

**Decision :** Built the project from scratch inside the required folder structure. Every environment parameter now lives in `configs/default.yaml` (problem size, traffic, time windows, episode limits, reward weights). Time windows are added to the state/observation/reward as a soft constraint (penalty-based, per cahier des charges §4.3.3).
and since we used yaml as a config file we add it to the `requirements.txt` as well `pyyaml`

## 2026-08-08 : Distance metric, switching to Manhattan distance

**Claim :** Straight-line (Euclidean) distance between customers doesn't reflect how a vehicle actually travels through a city.

**Evidence :** Real streets are grid-like, a vehicle can't cut diagonally through a block. Manhattan distance (`|dx| + |dy|`) is the standard way to model grid-constrained travel. Switching the node-to-node distance also changes what `observation_radius` reveals : a Manhattan region of radius `r` covers less area (`2r²`) than a Euclidean one (`πr²`) for the same `r`. Recalibrated `observation_radius` from `0.25` to `0.313` (`= 0.25 × √(π/2)`) to keep the same revealed-area fraction, confirmed with a Monte Carlo check and by running real episodes (both land close to the original ~20-24%).

**Decision :** Node-to-node distance, `observation_radius`, and the congestion-hotspot field all use Manhattan distance consistently.

## 2026-08-09 : Time-window mechanic, early arrival & lateness

**Claim :** Needed to decide what happens if the vehicle reaches a customer before their time window opens.

**Evidence :** Checked the VRPTW (Vehicle Routing Problem with Time Windows) literature. Standard convention (Solomon 1987 benchmark, PyVRP solver) : early arrival triggers a free wait, not a penalty.

```
service_start = max(arrival_time, window_start)
lateness      = max(0, service_start - window_end)
elapsed_time  = service_start
```

The wait carries forward, so later customers' time checks stay correct. Only lateness (arriving after the window closes) is penalized.

**Decision :** Adopted the force-to-wait convention, matching standard VRPTW practice rather than an ad hoc simplification.

## 2026-08-10 : `src/env.py` built and validated (Week 1)

**Claim :** Translate the finalized POMDP (state, observation, action, reward) into a working Gymnasium environment.

**Evidence :** Built `generate_instance()` (positions, demands, Manhattan distances, spatially-correlated traffic via Gaussian hotspots, time windows) and `DynamicCVRPEnv` (`reset`/`step`, action masking, observation), all reading from `configs/default.yaml`. Each step, the vehicle picks the next node to visit; travel cost = distance × traffic, and the action mask blocks illegal moves (already served, over capacity, …).

**Decision :** Environment considered functionally complete for Week 1. Random Policy and Greedy Nearest-Neighbor baselines are next.

## 2026-08-12 : Independent code review & bug fixes (Week 1)

**Claim :** Wanted a second opinion on `env.py` before moving on.

**Evidence :** Had the code reviewed externally, then checked every point against our actual design and the cahier des charges. Adopted : normalizing time-window features in the observation, making `terminated`(served all customers)/`truncated`(force to cut the episode) mutually exclusive means just one of them can be true ,never both at the same time

**Decision :** Bug fixed, environment re-verified. `env.py` considered stable.

## 2026-08-13 : Baselines, eval.py, and Week 1 wrap-up

**Claim :** Finish Week 1, implement the Random and Greedy Nearest-Neighbor baselines and log results in the required CSV schema.

**Evidence :** Built `src/baselines.py` with a shared `Policy` interface (`act(obs)` only, no `info`, matching exactly what a real trained agent will have access to) : `RandomPolicy` and `GreedyNearestNeighbor` (nearest by raw distance, matching the cahier des charges' literal definition, not traffic-adjusted cost). Built `src/eval.py`, logging one row per episode to `results/*.csv` with the required schema (`seed, step, episode_return, ...`); `step` stays `0` for baselines since they never train. Ran 100 matched-seed episodes per method : Greedy cuts average travel cost by ~41% vs Random (22.21 → 13.01), but only marginally improves the late-delivery rate (47.5% → 43.1%). Decomposing `episode_return` into its components showed the lateness penalty is the single largest term for both methods, distance-minimization alone doesn't solve this problem. Also noticed `n_served`/`terminated` are saturated at 100% for both baselines under the current config, so those two metrics won't discriminate between methods until the GNN agents are in the comparison too.

**Decision :** Week 1 (MDP formalization, environment, baselines) considered complete. Real, reproducible baseline numbers are now available for the report.

## 2026-08-14 : GNN encoder built (Week 2)

**Claim :** Translate the shared perception module (cahier des charges §4.4.2) into working code : dense observation arrays to per-node embeddings.

**Evidence :** Built `src/gnn_encoder.py`. `build_complete_edge_index` converts the environment's dense `(N,N)` distance/traffic matrices into the `edge_index`/`edge_attr` format PyTorch Geometric's `NNConv` expects (240 directed edges for `N=16`). `GNNLayer` wraps one `NNConv` with a small edge-network mapping `[distance, traffic, observed]` to a per-edge weight matrix, so a message from a nearby calm road is weighted differently than one from a distant, congested road. `GNNEncoder` stacks two `GNNLayer`s (information propagates two hops) and pools the result into a fixed-size vector : the current node's embedding, a global mean-pooled embedding across all nodes, and the raw vehicle state (capacity, elapsed time), concatenated.

**Decision :** Verified against real data from `env.py` at each step (`edge_attr` values checked by hand against direct matrix lookups; a full 3-node forward pass recomputed by hand matched the actual `NNConv` output to four decimal places). Encoder considered correct and ready to feed a decision head.

## 2026-08-15 : GRU belief module dropped (Week 2, divergence from cahier des charges)

**Claim :** The original design (cahier des charges §4.4.2) specifies a GRU belief state on top of the GNN encoder. Needed to decide whether to build it before the DQN/PPO heads.

**Evidence :** A recurrent hidden state is only trained correctly on ordered, sequential transitions. DQN's replay buffer samples transitions in random, shuffled order, which breaks that sequential dependency without extra machinery (sequence-chunked replay, burn-in) that was outside this project's timeline. Separately, `env.py`'s `traffic_mask` already accumulates monotonically within an episode : once an edge is revealed it stays revealed, so the observation itself already functions as a running summary of everything seen so far, reducing the marginal value a separate learned memory would add on top.

**Decision :** Dropped the GRU. Final architecture is the GNN encoder feeding a per-node decision head directly, no recurrent component.

## 2026-08-16 : GNN-DQN implemented and trained (Week 2)

**Claim :** Build the first learning agent : Double DQN with action masking on top of the GNN encoder.

**Evidence :** Built `src/dqn.py` (per-node Q-head, `GNNQNetwork`), `src/replay_buffer.py`, and `src/train_dqn.py`. Double DQN : the online network selects the next-state action, the target network evaluates it, avoiding the overestimation bias of vanilla DQN. The buffer stores `terminated` and `truncated` separately, and only `terminated` zeroes the Bellman bootstrap target, a truncated episode (hit `max_steps`) is not a true terminal state, so the agent should still bootstrap through it. Hard target-network sync every 500 steps. Trained for 20,000 environment steps, 1,000 warm-up, buffer size 50,000, batch size 64, epsilon 1.0 → 0.05.

**Decision :** Training converges by roughly step 15,000 and plateaus (checkpoint eval : return −53.19 at step 5,000 → −2.72 at step 15,000, on 5 held-out seeds). Checkpoints saved every 5,000 steps to `checkpoints/`.

## 2026-08-17 : GNN-PPO implemented and trained (Week 3)

**Claim :** Build the second learning agent, sharing the same GNN encoder, per cahier des charges §4.4.2's shared-backbone requirement.

**Evidence :** Built `src/ppo.py` (actor-critic heads on the shared encoder) and `src/train_ppo.py`. Clipped surrogate objective, Generalized Advantage Estimation, entropy bonus for exploration, on-policy updates every 512 steps. Same 20,000-step training budget as GNN-DQN for a matched comparison.

**Decision :** Training still visibly improving at the 20,000-step cutoff (checkpoint eval return : −66.33 at step 5,120 → −11.52 at step 20,000, still trending down in cost/lateness, not yet plateaued like GNN-DQN). Noted as a limitation for the results comparison, not treated as a finished, converged result.

## 2026-08-18 : Four-method evaluation and results analysis (Week 3/4)

**Claim :** Evaluate Random, Greedy Nearest-Neighbor, GNN-DQN, and GNN-PPO under one identical protocol, per cahier des charges §5.2.

**Evidence :** Extended `src/eval.py` to run all four methods on the same 100 held-out seeds. GNN-DQN reaches a mean episode return of −1.69 (95% CI [−3.09, −0.30]) versus Greedy's −57.71 and Random's −72.27, a 97% improvement over the best baseline. Late-delivery rate falls from 43.1% (Greedy) to 2.7% (GNN-DQN). Not achieved via lower cost, Greedy stays cheapest (13.01), but via waiting more (7.27/episode vs Greedy's 3.14), exploiting the reward's free-wait/penalized-lateness asymmetry. GNN-PPO shows the same pattern, smaller effect (return −19.91, late rate 12.5%), consistent with being undertrained rather than weaker.

**Decision :** Core finding : distance-minimizing heuristics can't fix time-window compliance ; a learned agent can. Open item : GNN-DQN vs. GNN-PPO isn't yet a fair comparison, confounded by GNN-PPO's earlier training cutoff.

## 2026-08-19 : Reproducibility hardening, run_all.sh, requirements.txt (Week 4)

**Claim :** Professor's Rule 1 : "if `run_all.sh` fails to run, the project is not validated." Needed to verify this end to end, not just write it once.

**Evidence :** Found two real gaps : `requirements.txt` was missing `pandas` and `seaborn` (both imported by `src/plot.py`), which would crash the figure-generation step on a fresh install ; and `run_all.sh` had no guard against missing checkpoints, meaning a submission without `checkpoints/*.pt` would run to completion, silently write header-only CSVs and blank figures, and still exit `0`. Added a checkpoint-count guard and a post-run check that every output CSV has data rows and every figure PDF is non-empty. Tested both : confirmed the checkpoint guard fires correctly, and confirmed the output check catches a deliberately-emptied CSV. Mirrored the same guards into `run_all.bat` for Windows, with CRLF line endings verified.

**Decision :** `run_all.sh`/`run_all.bat` now fail loudly instead of silently on the two most likely submission-day failure modes. `requirements.txt` corrected.

## 2026-08-19 : Missing `report.pdf` deliverable found and fixed (Week 4)

**Claim :** The required submission `schema` lists a top-level `report.pdf` (6-20 pages) alongside `results/`, `figures/`, `logs/`, `notebook.md`. Checked whether this branch actually produces one.

**Evidence :** It didn't. `notebook.md` (this file) and the README's Results section cover the narrative, and `figures/*.pdf` + `results/*.csv` cover the numbers, but nothing assembled them into the single PDF report the schema requires — `run_all.sh`/`run_all.bat` regenerated everything except that file. Wrote `src/generate_report.py` (reportlab + pypdfium2, both pure-Python/prebuilt-wheel so no system Poppler/LaTeX dependency is required beyond `requirements.txt`) to read `configs/default.yaml`, `results/eval_comparison.csv`, and `figures/*.pdf` and assemble a 7-page report: problem formulation, the environment's actual design choices (Manhattan distance, the observation-radius POMDP, the documented GRU drop), architecture, training setup tables pulled directly from the config, both learning-curve figures, the comparative-evaluation table and figure, and a discussion section with numbers computed live from `results/eval_comparison.csv` (not hand-copied, so it can't drift out of sync with a re-run). Caught and fixed two rendering bugs before treating it as done: ReportLab's base Helvetica font has no glyphs for several Unicode characters used in the first draft (union symbol, pi, arrows, Greek letters, minus vs. hyphen), which rendered as blank boxes — replaced with plain-ASCII equivalents throughout; and the first table-column-width pass overflowed the page's right margin — fixed by wrapping cell text in `Paragraph` flowables instead of raw strings so long cells wrap instead of clipping. Verified by rasterizing every page of the output and inspecting it, not just checking the script exited `0`.

**Decision :** Wired `src/generate_report.py` in as step 4 of `run_all.sh`/`run_all.bat`, and added `report.pdf` to the post-run verification check (Section 2026-08-19 above's non-empty-output guard now covers it too). `reportlab` and `pypdfium2` added to `requirements.txt`. Submission schema now fully satisfied.
