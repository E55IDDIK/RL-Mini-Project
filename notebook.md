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
 