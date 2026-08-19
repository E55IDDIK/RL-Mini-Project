"""Generates the top-level report.pdf required by the submission schema.

Reads the already-produced evaluation data (results/eval_comparison.csv) and
figures (figures/*.pdf, rasterized here for embedding) and assembles them,
together with the architecture/config described in configs/default.yaml and
src/{env,dqn,ppo,gnn_encoder}.py, into a single PDF report.

Usage (from the repository root, after `run_all.sh` / `run_all.bat` has been
run at least once so results/ and figures/ are populated):

    python -m src.generate_report
"""

from __future__ import annotations

import os

import pandas as pd
import pypdfium2 as pdfium
import yaml
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(ROOT, "configs", "default.yaml")) as f:
        return yaml.safe_load(f)


def rasterize_figures(out_dir: str):
    """Rasterize the vector figures/*.pdf to PNG so reportlab can embed them
    (Platypus' Image flowable needs a raster format, not a PDF page).
    Uses pypdfium2 (a pure-Python/prebuilt-binary PyPI package) instead of
    shelling out to poppler's pdftoppm, so this doesn't require any system
    package beyond what's in requirements.txt."""
    os.makedirs(out_dir, exist_ok=True)
    names = ["benchmark_comparison", "learning_curve_return", "learning_curve_metrics"]
    paths = {}
    for name in names:
        src = os.path.join(ROOT, "figures", f"{name}.pdf")
        pdf = pdfium.PdfDocument(src)
        bitmap = pdf[0].render(scale=200 / 72)
        image = bitmap.to_pil()
        out_path = os.path.join(out_dir, f"{name}.png")
        image.save(out_path)
        paths[name] = out_path
    return paths


def summarize_eval(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    g = df.groupby("method").agg(
        episode_return_mean=("episode_return", "mean"),
        episode_return_std=("episode_return", "std"),
        total_cost_mean=("total_cost", "mean"),
        total_cost_std=("total_cost", "std"),
        n_on_time_mean=("n_on_time", "mean"),
        n_waited_mean=("n_waited", "mean"),
        n_late_mean=("n_late", "mean"),
        n_served_mean=("n_served", "mean"),
        n_customers_mean=("n_customers", "mean"),
    )
    order = ["random", "greedy", "gnn_dqn", "gnn_ppo"]
    g = g.reindex([m for m in order if m in g.index])
    return g


def build_report(out_path: str):
    cfg = load_config()
    fig_dir = os.path.join(ROOT, ".report_assets")
    figs = rasterize_figures(fig_dir)
    summary = summarize_eval(os.path.join(ROOT, "results", "eval_comparison.csv"))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Justify", parent=styles["BodyText"], alignment=4, spaceAfter=10))
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="CenterTitle", parent=styles["Title"], alignment=1, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CenterSub", parent=styles["Normal"], alignment=1, fontSize=11, spaceAfter=4,
        textColor=colors.HexColor("#333333"),
    ))
    caption_style = ParagraphStyle(
        name="Caption", parent=styles["Normal"], alignment=1, fontSize=9,
        textColor=colors.HexColor("#555555"), spaceAfter=14,
    )

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        title="Reinforcement Learning for Dynamic Vehicle Routing under Uncertain Traffic Conditions",
        author="KHARBOUCH Essiddik, ER-RACHEDY Sana, BOUCHFIRA Hind, KHARBOUCH Yahya",
    )
    story = []

    # ---- Title page -------------------------------------------------
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(
        "Reinforcement Learning for Dynamic Vehicle Routing<br/>under Uncertain Traffic Conditions",
        styles["CenterTitle"],
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Mini-Project Report", styles["CenterSub"]))
    story.append(Paragraph(
        "Master's Program of Excellence  -  Data Science and Information Systems Security",
        styles["CenterSub"],
    ))
    story.append(Paragraph(
        "Sultan Moulay Slimane University  -  Multidisciplinary Faculty of Beni Mellal",
        styles["CenterSub"],
    ))
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph(
        "<b>Group:</b> KHARBOUCH Essiddik, ER-RACHEDY Sana, BOUCHFIRA Hind, KHARBOUCH Yahya",
        styles["CenterSub"],
    ))
    story.append(Paragraph(
        "<b>Supervisors:</b> Pr. Hamza ALLAGA, Pr. Meriem EL HARKAOUI", styles["CenterSub"],
    ))
    story.append(Paragraph("Annee universitaire : 2025-2026", styles["CenterSub"]))
    story.append(PageBreak())

    # ---- 1. Introduction ---------------------------------------------
    story.append(Paragraph("1. Introduction &amp; Problem Statement", styles["H1"]))
    story.append(Paragraph(
        "Vehicle routing is a core problem in transportation and logistics. Classical solvers assume "
        "travel times, demands, and the road network are known and fixed in advance, which breaks down "
        "under real traffic conditions: congestion can turn a route that was optimal at planning time "
        "into a poor one during execution. This project applies Reinforcement Learning to the Dynamic "
        "Vehicle Routing Problem (DVRP): a single capacitated vehicle must serve every customer on a "
        "graph at minimum cost, subject to per-customer delivery time windows, while traffic on each "
        "road segment is drawn from a spatially-correlated congestion field once per episode and is "
        "only observable within a limited radius of the vehicle's current position. The guiding "
        "question, taken directly from the project specification, is: how can Reinforcement Learning "
        "dynamically adapt vehicle routing decisions to uncertain traffic conditions while minimizing "
        "travel time, distance, and delays?", styles["Justify"],
    ))

    # ---- 2. MDP / POMDP formulation -----------------------------------
    story.append(Paragraph("2. Problem Formulation (POMDP)", styles["H1"]))
    story.append(Paragraph(
        "The environment (<font face='Courier'>src/env.py</font>, <font face='Courier'>DynamicCVRPEnv</font>) "
        "is a Gymnasium-compatible environment built around a complete graph over {depot} + {customers}. "
        "The real state  -  everything that exists in the environment, whether observed or not  -  includes "
        "every customer's position, demand, and time window, the vehicle's position and remaining "
        "capacity, and the true traffic multiplier on every road segment. The agent's observation differs "
        "from that real state in exactly one respect: traffic. Congestion is only revealed on edges "
        "within a fixed <font face='Courier'>observation_radius</font> of the vehicle's current position; "
        "everywhere else the observation reports zero (unobserved) with an explicit observed/unobserved "
        "mask. Because the agent must act on an incomplete view of the true state, this is formally a "
        "Partially Observable MDP (POMDP), not a plain MDP.", styles["Justify"],
    ))
    prob = cfg["problem"]
    traf = cfg["traffic"]
    tw = cfg["time_windows"]
    story.append(Paragraph(
        f"<b>Nodes / graph size:</b> {prob['n_customers']} customers plus one fixed depot node "
        f"(N = {prob['n_customers'] + 1}), positioned uniformly at random on a "
        f"{cfg['map']['size']}x{cfg['map']['size']} square (the depot is fixed at its center). "
        f"<b>Distance metric:</b> Manhattan (L1) distance is used instead of Euclidean distance, since a "
        "vehicle on a grid-like street network cannot cut diagonally through a block; this also changes "
        "how much area a fixed observation radius reveals (a Manhattan disk of radius r covers 2 x r^2 versus "
        "pi x r^2 for a Euclidean disk of the same r), which was explicitly recalibrated and validated with a "
        "Monte Carlo check (see <font face='Courier'>notebook.md</font>, 2026-08-08).",
        styles["Justify"],
    ))
    story.append(Paragraph(
        f"<b>Traffic / uncertainty:</b> a spatially-correlated congestion field is generated from "
        f"2-4 random Gaussian \"hotspots\" per episode and mapped to a multiplier range "
        f"[{traf['low']}, {traf['high']}] (free-flow to worst-case congestion). The multiplier for edge "
        f"(i, j) is fixed for the whole episode but only revealed to the agent once the vehicle comes "
        f"within <font face='Courier'>observation_radius = {traf['observation_radius']}</font> of it  -  "
        "modelling the fact that a real driver only learns actual road congestion by approaching it, not "
        "by consulting a global traffic map. <b>Time windows:</b> "
        f"{'enabled' if tw['enabled'] else 'disabled'}, each customer's delivery window is "
        f"[{tw['earliest_start']}, {tw['latest_start']}] wide "
        f"{tw['window_width_low']}-{tw['window_width_high']} time units, following the standard VRPTW "
        "convention (Solomon 1987): arriving early triggers a free wait until the window opens "
        "(<font face='Courier'>service_start = max(arrival_time, window_start)</font>), only arriving "
        "after the window closes is penalized.", styles["Justify"],
    ))

    story.append(Paragraph("Action space", styles["H2"]))
    story.append(Paragraph(
        "A fixed discrete action space of size N: choose the next node to visit. The set of actions that "
        "are actually legal changes at every step through action masking  -  a customer is masked out once "
        "served or if its demand exceeds the vehicle's remaining capacity, and the depot is masked out "
        "while the vehicle is already there (except to end the episode). Invalid actions are never sampled, "
        "in training or at evaluation.", styles["Justify"],
    ))

    story.append(Paragraph("Reward function", styles["H2"]))
    rew = cfg["reward"]
    cell_style = ParagraphStyle(name="Cell", parent=styles["Normal"], fontSize=8.5, leading=10.5)
    hdr_style = ParagraphStyle(name="CellHdr", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold")

    def cell(text, header=False):
        return Paragraph(text, hdr_style if header else cell_style)

    reward_rows = [
        [cell("Component", True), cell("Weight", True), cell("When it applies", True)],
        [cell("Travel cost penalty"), cell(f"-{rew['w_distance']} x (distance x traffic)"), cell("Every move")],
        [cell("Time-window violation"), cell(f"-{rew['w_time_window_penalty']}"), cell("Only if genuinely late (early arrival waits for free)")],
        [cell("Delivery bonus"), cell(f"+{rew['w_delivery_bonus']}"), cell("Each successful delivery")],
        [cell("Route completion bonus"), cell(f"+{rew['w_complete_bonus']}"), cell("All customers served and vehicle back at depot")],
        [cell("Unserved customer penalty"), cell(f"-{rew['w_unserved_penalty']} per customer"), cell("Only if the episode is truncated (ran out of steps) with customers still unserved")],
    ]
    t = Table(reward_rows, colWidths=[4.0 * cm, 3.6 * cm, 6.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # ---- 3. Architecture -----------------------------------------------
    story.append(Paragraph("3. Agent Architecture", styles["H1"]))
    story.append(Paragraph(
        "Both learning agents share a graph-neural-network perception backbone "
        "(<font face='Courier'>src/gnn_encoder.py</font>): the dense per-step observation "
        "(node features, distance matrix, observed-traffic matrix, observation mask) is converted into a "
        "complete directed graph (N x (N-1) edges), and two stacked edge-conditioned graph convolution "
        "layers (PyTorch Geometric <font face='Courier'>NNConv</font>) propagate information two hops "
        "across it. For every edge, a small neural network maps its [distance, observed traffic, "
        "observed flag] features to a full weight matrix, so a message travelling along a calm, "
        "well-observed road is treated differently from one along a distant, congested, or still-unknown "
        "one. The encoder outputs per-node embeddings, which a linear head turns into one Q-value "
        "(GNN-DQN) or one policy logit plus a shared state value (GNN-PPO) per candidate next node; "
        "invalid actions are masked to a very large negative value before the arg-max / softmax in both agents.",
        styles["Justify"],
    ))
    story.append(Paragraph(
        "<b>Note on the GRU belief state.</b> The original design considered a GRU carrying a recurrent "
        "\"belief\" across steps within an episode, to compensate for the traffic field only being "
        "partially observed. This was deliberately dropped after Week 2 (documented in "
        "<font face='Courier'>notebook.md</font>, 2026-08-15): a recurrent hidden state is only trained "
        "correctly on ordered, sequential transitions, but the DQN replay buffer samples randomly-ordered "
        "transitions from many different episodes, which breaks that sequential dependency without extra "
        "machinery (sequence-chunked replay, burn-in) that was out of scope for this timeline. Separately, "
        "the environment's observed-traffic mask never resets within an episode  -  it only grows as the "
        "vehicle moves  -  so every observation already carries a running summary of everything revealed so "
        "far. In effect, the belief the GRU was meant to build is already externalized into the "
        "environment's persistent state, which removes most of the marginal benefit a separate learned "
        "memory would add.", styles["Justify"],
    ))
    story.append(Paragraph(
        "<b>GNN-DQN</b> (value-based, <font face='Courier'>src/dqn.py</font>): Double DQN  -  the online "
        "network selects the next state's action, the target network evaluates it, reducing the "
        "overestimation bias of vanilla DQN. A replay buffer stores transitions with <font "
        "face='Courier'>terminated</font> and <font face='Courier'>truncated</font> tracked separately, "
        "since only a true termination should zero the bootstrap target  -  a truncated episode (hit "
        "<font face='Courier'>max_steps</font>) is not a genuine terminal state.<br/>"
        "<b>GNN-PPO</b> (policy-based, <font face='Courier'>src/ppo.py</font>): actor-critic PPO sharing "
        "the same GNN encoder, with the clipped surrogate objective, Generalized Advantage Estimation, "
        "and an entropy bonus for exploration; on-policy updates run every "
        f"{cfg['ppo']['rollout_length']} collected steps.<br/>"
        "<b>Baselines</b> (<font face='Courier'>src/baselines.py</font>): a Random Policy (uniform over "
        "valid actions) and a Greedy Nearest-Neighbor heuristic (always moves to the nearest unserved "
        "customer by raw distance, matching the cahier des charges' literal definition rather than a "
        "traffic-adjusted cost).", styles["Justify"],
    ))

    # ---- 4. Training setup ---------------------------------------------
    story.append(Paragraph("4. Training Setup", styles["H1"]))
    dqn_cfg, ppo_cfg = cfg["training"], cfg["ppo"]
    setup_rows = [
        ["", "GNN-DQN", "GNN-PPO"],
        ["Total environment steps", f"{dqn_cfg['total_steps']:,}", f"{ppo_cfg['total_steps']:,}"],
        ["Learning rate", str(dqn_cfg["learning_rate"]), str(ppo_cfg["learning_rate"])],
        ["Discount (gamma)", str(dqn_cfg["gamma"]), str(ppo_cfg["gamma"])],
        ["Batch size", str(dqn_cfg["batch_size"]), str(ppo_cfg["batch_size"])],
        ["Replay buffer size", f"{dqn_cfg['buffer_size']:,}", "-"],
        ["Epsilon-greedy schedule", f"{dqn_cfg['epsilon_start']} -> {dqn_cfg['epsilon_end']} over {dqn_cfg['epsilon_decay_steps']:,} steps", "-"],
        ["Target net sync every", f"{dqn_cfg['target_update_freq']} steps", "-"],
        ["Rollout length", "-", f"{ppo_cfg['rollout_length']} steps"],
        ["Epochs per rollout", "-", str(ppo_cfg["n_epochs"])],
        ["GAE lambda / clip epsilon", "-", f"{ppo_cfg['gae_lambda']} / {ppo_cfg['clip_eps']}"],
        ["Entropy coefficient", " - ", str(ppo_cfg["ent_coef"])],
        ["Checkpoint every", f"{dqn_cfg['checkpoint_every']:,} steps", f"{ppo_cfg['checkpoint_every']:,} steps"],
    ]
    t2 = Table(setup_rows, colWidths=[5.4 * cm, 5.4 * cm, 5.4 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Both agents train on the same fixed seed (42) and the same environment configuration, for a "
        "matched total-step budget, so any gap between them reflects the algorithms themselves rather "
        "than unequal training resources.", styles["Justify"],
    ))
    story.append(PageBreak())

    # ---- 5. Training convergence ----------------------------------------
    story.append(Paragraph("5. Training Convergence", styles["H1"]))
    story.append(Paragraph(
        "Both agents are periodically checkpointed and evaluated on 5 held-out seeds during training. "
        "GNN-DQN's return rises steadily and plateaus by roughly step 15,000; GNN-PPO improves more "
        "slowly and was still trending downward in cost/lateness at the 20,000-step cutoff, i.e. not yet "
        "fully converged within this training budget (a limitation noted for the comparison below, not a "
        "finished result).", styles["Justify"],
    ))
    story.append(Image(figs["learning_curve_return"], width=15.5 * cm, height=15.5 * cm * 969 / 1478))
    story.append(Paragraph("Figure 1  -  Mean episode return vs. training steps, 5 held-out evaluation seeds, shaded band = +/-1 std.", caption_style))
    story.append(Image(figs["learning_curve_metrics"], width=15.5 * cm, height=15.5 * cm * 818 / 2000))
    story.append(Paragraph("Figure 2  -  Travel cost and late-delivery count vs. training steps for both learned agents.", caption_style))
    story.append(PageBreak())

    # ---- 6. Comparative evaluation ----------------------------------------
    story.append(Paragraph("6. Comparative Evaluation", styles["H1"]))
    n_eval = cfg["evaluation"]["n_episodes"]
    story.append(Paragraph(
        f"All four routing methods (Random, Greedy Nearest-Neighbor, GNN-DQN, GNN-PPO) are evaluated on "
        f"the same {n_eval} held-out seeds (seeds 42-{42 + n_eval - 1}), unseen during training. Final "
        "checkpoints (step 20,000) are used for both learned agents, and both act greedily (no "
        "exploration) at evaluation time.", styles["Justify"],
    ))

    header = ["Method", "Return (mean+/-std)", "Cost (mean+/-std)", "On-time", "Waited", "Late"]
    rows = [header]
    label_map = {"random": "Random Policy", "greedy": "Greedy Nearest-Neighbor", "gnn_dqn": "GNN-DQN", "gnn_ppo": "GNN-PPO"}
    for method, row in summary.iterrows():
        rows.append([
            label_map.get(method, method),
            f"{row['episode_return_mean']:.2f} +/- {row['episode_return_std']:.2f}",
            f"{row['total_cost_mean']:.2f} +/- {row['total_cost_std']:.2f}",
            f"{row['n_on_time_mean']:.1f}",
            f"{row['n_waited_mean']:.1f}",
            f"{row['n_late_mean']:.1f}",
        ])
    t3 = Table(rows, colWidths=[3.7 * cm, 3.2 * cm, 3.2 * cm, 1.9 * cm, 1.9 * cm, 1.9 * cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ]))
    story.append(t3)
    story.append(Spacer(1, 10))
    story.append(Image(figs["benchmark_comparison"], width=16 * cm, height=16 * cm * 421 / 1490))
    story.append(Paragraph("Figure 3  -  Final episode return, total travel cost, and on-time delivery count, all four methods, 100 held-out episodes.", caption_style))

    # ---- 7. Discussion -----------------------------------------------------
    story.append(Paragraph("7. Discussion", styles["H1"]))
    r = summary
    dqn_ret, greedy_ret = r.loc["gnn_dqn", "episode_return_mean"], r.loc["greedy", "episode_return_mean"]
    improvement = (dqn_ret - greedy_ret) / abs(greedy_ret) * 100
    story.append(Paragraph(
        f"GNN-DQN reaches a mean return of {dqn_ret:.2f} versus Greedy Nearest-Neighbor's "
        f"{greedy_ret:.2f}  -  a {improvement:.0f}% improvement  -  and Random's "
        f"{r.loc['random', 'episode_return_mean']:.2f}. The late-delivery rate falls from "
        f"{r.loc['greedy', 'n_late_mean'] / r.loc['greedy', 'n_customers_mean'] * 100:.1f}% for Greedy to "
        f"{r.loc['gnn_dqn', 'n_late_mean'] / r.loc['gnn_dqn', 'n_customers_mean'] * 100:.1f}% for GNN-DQN. "
        "Notably, this improvement is not achieved through a lower travel cost  -  Greedy actually stays "
        f"cheaper on raw distance ({r.loc['greedy', 'total_cost_mean']:.2f} vs. "
        f"{r.loc['gnn_dqn', 'total_cost_mean']:.2f})  -  but through deliberately waiting more "
        f"({r.loc['gnn_dqn', 'n_waited_mean']:.1f} waits/episode vs. Greedy's "
        f"{r.loc['greedy', 'n_waited_mean']:.1f}), exploiting the reward's asymmetry between a free early "
        "wait and a heavily penalized late arrival. This is the project's core empirical finding: a "
        "distance-minimizing heuristic cannot, by construction, fix time-window compliance  -  it has no "
        "notion of when to wait  -  while a learned agent conditioned on the traffic and time-window "
        "features can trade a small distance cost for a much larger compliance gain.", styles["Justify"],
    ))
    story.append(Paragraph(
        f"GNN-PPO shows the same qualitative pattern (return {r.loc['gnn_ppo', 'episode_return_mean']:.2f}, "
        f"late rate {r.loc['gnn_ppo', 'n_late_mean'] / r.loc['gnn_ppo', 'n_customers_mean'] * 100:.1f}%) "
        "but a smaller effect than GNN-DQN. Given the learning curves in Section 5 show GNN-PPO still "
        "improving, not yet plateaued, at the shared 20,000-step training cutoff, this gap is best read as "
        "GNN-PPO being undertrained relative to GNN-DQN rather than a weaker method in principle  -  Double "
        "DQN's replay buffer reuses every transition many times over, giving it more effective gradient "
        "steps per environment step than PPO's strictly on-policy updates, which is a plausible mechanism "
        "for GNN-DQN converging faster within an equal environment-step budget. A fair, decisive comparison "
        "between the two algorithms would need either a larger step budget for GNN-PPO or a training-time "
        "budget (wall clock) matched instead of an environment-step budget.", styles["Justify"],
    ))

    # ---- 8. Conclusion ---------------------------------------------------
    story.append(Paragraph("8. Conclusion and Future Work", styles["H1"]))
    story.append(Paragraph(
        "This project delivers a Gymnasium-based DVRP simulator with Manhattan-distance travel costs, a "
        "spatially-correlated and only locally-observable traffic field, and per-customer time windows; "
        "two graph-neural-network Deep RL agents (GNN-DQN and GNN-PPO) sharing an NNConv perception "
        "backbone with action masking; two heuristic baselines; and a common evaluation protocol across "
        "return, cost, and time-window compliance. The central finding  -  a learned agent trades a small "
        "increase in travel distance for a large reduction in time-window violations by learning when to "
        "wait, something a distance-only heuristic cannot do  -  directly answers the problem statement in "
        "Section 1. Future work, out of scope for this first edition, includes multi-vehicle coordination, "
        "real-time traffic feeds (OSM/GPS), and training GNN-PPO for a larger or wall-clock-matched budget "
        "to settle the DQN-vs-PPO comparison decisively.", styles["Justify"],
    ))

    doc.build(story)
    print(f"Saved report to {out_path}")


if __name__ == "__main__":
    build_report(os.path.join(ROOT, "report.pdf"))
