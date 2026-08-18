"""High-End Visualization Engine for Dynamic CVRP.
 
Generates:
1. High-definition 1080p MP4 videos of the agent routing across dynamic traffic maps.
2. Side-by-Side Split-Screen Comparison Video (Before Training vs After Training).
3. Standalone Interactive HTML5 Web Dashboard with play/pause and step scrubber.
 
Usage:
    python src/visualize.py --seed 42
    python src/visualize.py --seed 42 --method gnn_dqn
    python src/visualize.py --seed 42 --method gnn_ppo
    python src/visualize.py --compare --seed 42
"""
 
from __future__ import annotations
 
import argparse
import json
import os
import sys
from pathlib import Path
 
# make sure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
import imageio
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
 
from src.baselines import Policy, RandomPolicy, GreedyNearestNeighbor
from src.env import DynamicCVRPEnv, load_config
from src.eval import DQNGreedyPolicy, PPOGreedyPolicy, find_latest_checkpoint
 
 
# ============================================================================
# Episode Recorder
# ============================================================================
 
def record_episode(env: DynamicCVRPEnv, policy: Policy, seed: int) -> dict:
    """Run one episode and record all static instance data and step-by-step frames."""
    obs, info = env.reset(seed=seed)
    policy.reset(seed=seed)
 
    inst = env.inst
    N = env.N
    map_size = float(env.cfg["map"]["size"])
    cap_max = float(env.cfg["problem"]["vehicle_capacity"])
 
    # Compute a continuous 2D congestion grid for smooth heatmap rendering
    grid_res = 80
    gx = np.linspace(0, map_size, grid_res)
    gy = np.linspace(0, map_size, grid_res)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
 
    # Approximate continuous traffic field from node-to-node traffic
    # using Gaussian distance weighting to known coords
    traffic_node_avg = inst.traffic.mean(axis=1)  # average congestion per node
    diffs = grid_points[:, None, :] - inst.coords[None, :, :]
    dists = np.abs(diffs).sum(axis=-1)  # Manhattan distance to nodes
    weights = np.exp(-dists**2 / (2.0 * float(env.cfg["traffic"]["spatial_scale"])**2))
    weights /= (weights.sum(axis=1, keepdims=True) + 1e-8)
    grid_traffic = (weights @ traffic_node_avg).reshape(grid_res, grid_res)
 
    steps_data = []
    current_node = 0
    route = [0]
    total_cost = 0.0
    elapsed_time = 0.0
    cumulative_reward = 0.0
    served_mask = np.zeros(N, dtype=bool)
 
    # Record Initial State (Frame 0)
    steps_data.append({
        "step": 0,
        "action": 0,
        "from_node": 0,
        "to_node": 0,
        "from_pos": inst.coords[0].copy(),
        "to_pos": inst.coords[0].copy(),
        "route": list(route),
        "remaining_cap": cap_max,
        "elapsed_time": 0.0,
        "total_cost": 0.0,
        "reward": 0.0,
        "cumulative_reward": 0.0,
        "status": "START AT DEPOT",
        "served_mask": served_mask.copy(),
        "n_on_time": 0,
        "n_waited": 0,
        "n_late": 0,
        "obs_edges": env.observed.copy(),
        "q_values": None,
        "action_mask": np.zeros(N, dtype=bool).tolist(),
    })
 
    done = False
    step_count = 0
    # Play episode
    while not done:
        prev_node = env.current
        prev_late = env.n_late
        prev_waited = env.n_waited
        
        q_values = None
        action_mask = obs.get("action_mask", np.zeros(N)).copy()
        if hasattr(policy, "net"):
            with torch.no_grad():
                obs_t = {k: torch.as_tensor(v, device=policy.device).unsqueeze(0) for k, v in obs.items()}
                q = policy.net(obs_t)
                q_values = q[0].cpu().numpy().tolist()
        elif hasattr(policy, "ac"):
            with torch.no_grad():
                obs_t = {k: torch.as_tensor(v, device=policy.device).unsqueeze(0) for k, v in obs.items()}
                logits, _, _ = policy.ac._logits_and_value(obs_t)
                q_values = logits[0].cpu().numpy().tolist()

        action = policy.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        cumulative_reward += reward
 
        # Determine exactly what happened using the environment's true counters
        if action == 0:
            status = "RETURNED TO DEPOT"
        else:
            w_start = inst.window_start[action]
            w_end = inst.window_end[action]
            if env.n_late > prev_late:
                status = f"LATE (missed {w_end:.0f})"
            elif env.n_waited > prev_waited:
                status = f"WAITED (arrived < {w_start:.0f})"
            else:
                status = f"ON-TIME (window [{w_start:.0f}-{w_end:.0f}])"
 
        steps_data.append({
            "step": step_count,
            "action": action,
            "from_node": prev_node,
            "to_node": action,
            "from_pos": inst.coords[prev_node].copy(),
            "to_pos": inst.coords[action].copy(),
            "route": list(info["route"]),
            "remaining_cap": env.remaining_cap,
            "elapsed_time": env.elapsed_time,
            "total_cost": env.total_cost,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "status": status,
            "served_mask": env.served.copy(),
            "n_on_time": env.n_on_time,
            "n_waited": env.n_waited,
            "n_late": env.n_late,
            "q_values": q_values,
            "action_mask": action_mask.tolist(),
        })
 
        obs = next_obs
        done = terminated or truncated
 
    return {
        "seed": seed,
        "coords": inst.coords,
        "demands": inst.demands,
        "window_start": inst.window_start,
        "window_end": inst.window_end,
        "traffic_matrix": inst.traffic,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_traffic": grid_traffic,
        "cap_max": cap_max,
        "obs_radius": float(env.cfg["traffic"]["observation_radius"]),
        "steps": steps_data,
    }
 
 
# ============================================================================
# Single Frame Renderer (High Aesthetics Dark/Cyberpunk Theme)
# ============================================================================
 
def draw_single_panel(ax_map, ax_hud, ep_data: dict, step_idx: int,
                      interp_alpha: float = 1.0, title_prefix: str = "", compact: bool = False):
    """Render a single frame consisting of the map and the telemetry HUD."""
    coords = ep_data["coords"]
    demands = ep_data["demands"]
    w_start = ep_data["window_start"]
    w_end = ep_data["window_end"]
    obs_radius = ep_data["obs_radius"]
    cap_max = ep_data["cap_max"]
    N = len(coords)
 
    step_info = ep_data["steps"][step_idx]
    prev_step = ep_data["steps"][max(0, step_idx - 1)]
 
    # Interpolated vehicle position between from_pos and to_pos
    from_pos = step_info["from_pos"]
    to_pos = step_info["to_pos"]
    curr_veh_pos = (1.0 - interp_alpha) * from_pos + interp_alpha * to_pos
 
    # -------------------------------------------------------------
    # 1. MAP PANEL
    # -------------------------------------------------------------
    ax_map.clear()
    ax_map.set_facecolor("#0b0f19")
 
    # Draw Traffic Heatmap Background
    gx = ep_data["grid_x"]
    gy = ep_data["grid_y"]
    gt = ep_data["grid_traffic"]
    # Reduce contour density and opacity
    cf = ax_map.contourf(gx, gy, gt, levels=5, cmap="YlOrRd", alpha=0.15, vmin=1.0, vmax=3.0)
 
    # Draw Vehicle Observation Sensor Cone (radar circle)
    radar = patches.Circle(
        curr_veh_pos, obs_radius,
        facecolor="#00e5ff", alpha=0.05, edgecolor="#00e5ff", linestyle=":", linewidth=1.0
    )
    ax_map.add_patch(radar)
 
    # Draw Background Road Network (sparse proximity + used roads)
    road_threshold = 0.45
    edges_to_draw = set()
    import math
    for i in range(N):
        for j in range(i + 1, N):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            if math.sqrt(dx*dx + dy*dy) < road_threshold:
                edges_to_draw.add((i, j))
                
    for step in ep_data["steps"]:
        u, v = step["from_node"], step["to_node"]
        if u != v:
            edges_to_draw.add((min(u, v), max(u, v)))
            
    for u, v in edges_to_draw:
        ax_map.plot([coords[u][0], coords[v][0]], [coords[u][1], coords[v][1]], 
                    color="#475569", linewidth=0.8, alpha=0.2, linestyle="-", zorder=1)
 
    # Draw Route History Trails
    route = step_info["route"]
    if len(route) > 1:
        # Draw past completed segments
        past_coords = [coords[n] for n in route[:-1]]
        if past_coords:
            xs, ys = zip(*past_coords)
            ax_map.plot(xs, ys, color="#00e5ff", linewidth=1.5, alpha=0.4, linestyle="-", zorder=2)
 
        # Draw active interpolated segment
        ax_map.plot([from_pos[0], curr_veh_pos[0]], [from_pos[1], curr_veh_pos[1]],
                    color="#00e5ff", linewidth=4.0, alpha=0.9, zorder=3)
        
        # Add a directional arrow head
        if interp_alpha > 0.1:
            dx = curr_veh_pos[0] - from_pos[0]
            dy = curr_veh_pos[1] - from_pos[1]
            if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                dist = math.sqrt(dx*dx + dy*dy)
                ax_map.annotate("", xy=(curr_veh_pos[0], curr_veh_pos[1]), xytext=(curr_veh_pos[0]-dx/dist*0.01, curr_veh_pos[1]-dy/dist*0.01),
                                arrowprops=dict(arrowstyle="->", color="#00e5ff", lw=2.5, alpha=0.9), zorder=4)
 
    # Draw Customer Nodes & Depot
    served_mask = step_info["served_mask"]
    for i in range(N):
        x, y = coords[i]
        if i == 0:
            # Depot: Blue Square
            rect = patches.Rectangle(
                (x - 0.03, y - 0.03), 0.06, 0.06,
                facecolor="#2563eb", edgecolor="#60a5fa", linewidth=2.0, zorder=5
            )
            ax_map.add_patch(rect)
            ax_map.text(x, y, "DEPOT", color="white", fontsize=8, fontweight="bold",
                        ha="center", va="center", zorder=6)
        else:
            # Customer Node: Color reflects status
            if served_mask[i]:
                node_color = "#22c55e"  # Served Green
                edge_color = "#16a34a"
            else:
                node_color = "#f8fafc"  # Unserved White
                edge_color = "#cbd5e1"
                
            if step_info["to_node"] == i and interp_alpha < 1.0:
                # Next delivery highlighting
                node_color = "#00e5ff"
                edge_color = "#00b8d4"
 
            circ = patches.Circle(
                (x, y), 0.02,
                facecolor=node_color, edgecolor=edge_color, linewidth=1.5, zorder=4
            )
            ax_map.add_patch(circ)
            ax_map.text(x, y-0.035, f"C{i}", color="#cbd5e1", fontsize=8, fontweight="bold",
                        ha="center", va="top", zorder=5)
 
    # Draw Moving Vehicle (Glow marker)
    veh_marker = patches.Circle(
        curr_veh_pos, 0.04,
        facecolor="#00e5ff", edgecolor="#ffffff", linewidth=2.0, zorder=10
    )
    veh_glow = patches.Circle(curr_veh_pos, 0.05, facecolor="#00e5ff", alpha=0.3, edgecolor="none", zorder=9)
    ax_map.add_patch(veh_marker)
    ax_map.add_patch(veh_glow)
    ax_map.text(curr_veh_pos[0], curr_veh_pos[1], "V", color="#0b0f19", fontsize=9, fontweight="bold", ha="center", va="center", zorder=11)
 
    # Legend / Congestion Label
    ax_map.text(0.0, -0.05, "CONGESTION PRESSURE: Low ───────────── High", color="#64748b", fontsize=7, fontweight="bold", ha="left")
    ax_map.text(0.0, -0.09, "● Served   ○ Unserved   ■ Depot   ━━ Route   ┄┄ Network", color="#64748b", fontsize=7, fontweight="bold", ha="left")
 
    ax_map.set_xlim(-0.08, 1.08)
    ax_map.set_ylim(-0.12, 1.08)
    ax_map.set_aspect("equal")
    title_text = f"{title_prefix.upper()}  •  VEHICLE V1  •  STEP {step_info['step']} / {len(ep_data['steps'])-1}"
    title_fs = 8 if compact else 10
    ax_map.set_title(title_text, color="#f8fafc", fontsize=title_fs, fontweight="bold", pad=15)
    
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
 
    # -------------------------------------------------------------
    # 2. TELEMETRY / HUD PANEL
    # -------------------------------------------------------------
    ax_hud.clear()
    ax_hud.set_facecolor("#0f172a")
    ax_hud.set_xticks([])
    ax_hud.set_yticks([])
    for spine in ax_hud.spines.values():
        spine.set_color("#1e293b")

    # --- Adaptive font sizes for compact (comparison) vs normal mode ---
    fs_title = 9 if compact else 12
    fs_status = 7 if compact else 9
    fs_label = 7.5 if compact else 9
    fs_value = 7.5 if compact else 9
    fs_small = 7 if compact else 8.5
    pad_status = 0.3 if compact else 0.4

    hud_y = 0.94
    ax_hud.text(0.5, hud_y, "LIVE TELEMETRY HUD", color="#38bdf8", fontsize=fs_title, fontweight="bold", ha="center")
    hud_y -= 0.08

    # Status Banner
    status_txt = step_info["status"]
    if "START AT DEPOT" in status_txt or "AT DEPOT" in status_txt:
        status_txt = "ON-TIME"

    # In compact mode, shorten status text to prevent overflow
    if compact:
        # Strip "(window ...)" / "(arrived ...)" suffixes for narrow panels
        import re
        status_txt = re.sub(r'\s*\(.*\)\s*$', '', status_txt)

    status_bg = "#dc2626" if "LATE" in status_txt else ("#ca8a04" if "WAITED" in status_txt else "#16a34a")
    ax_hud.text(0.5, hud_y, f"STATUS: {status_txt}", color="white", fontsize=fs_status, fontweight="bold",
                ha="center", bbox=dict(boxstyle=f"round,pad={pad_status}", facecolor=status_bg, alpha=0.9, edgecolor="none"))
    hud_y -= 0.12

    # Stats Metrics
    el_time = step_info["elapsed_time"]
    cost = step_info["total_cost"]
    rew = step_info["cumulative_reward"]
    n_served = int(served_mask.sum())
    stats = [
        ("Step", f"{step_info['step']} / {len(ep_data['steps'])-1}"),
        ("Clock", f"{el_time:.1f} min"),
        ("Cost", f"{cost:.2f}"),
        ("Reward", f"{rew:.2f}"),
        ("Served", f"{n_served} / {N-1}"),
    ]
    for label, val in stats:
        ax_hud.text(0.08, hud_y, label, color="#94a3b8", fontsize=fs_label, fontweight="bold")
        ax_hud.text(0.92, hud_y, val, color="#f8fafc", fontsize=fs_value, fontweight="bold", ha="right")
        hud_y -= 0.065

    hud_y -= 0.02
    # Capacity Progress Bar
    rem_cap = step_info["remaining_cap"]
    ax_hud.text(0.08, hud_y, f"Load: {int(cap_max - rem_cap)}/{int(cap_max)}", color="#94a3b8", fontsize=fs_small, fontweight="bold")
    hud_y -= 0.04
    cap_frac = max(0.0, min(1.0, (cap_max - rem_cap) / cap_max))
    bar_bg = patches.Rectangle((0.08, hud_y), 0.84, 0.035, facecolor="#334155", edgecolor="none")
    bar_fill = patches.Rectangle((0.08, hud_y), 0.84 * cap_frac, 0.035, facecolor="#0ea5e9", edgecolor="none")
    ax_hud.add_patch(bar_bg)
    ax_hud.add_patch(bar_fill)
    hud_y -= 0.08

    # Next Delivery
    nxt = step_info["to_node"]
    if nxt == 0:
        nxt_txt = "DEPOT"
    else:
        we = w_end[nxt]
        we_str = "∞" if we > 9000 else f"{we:.0f}"
        nxt_txt = f"C{nxt}  [{w_start[nxt]:.0f}-{we_str}]"
    ax_hud.text(0.5, hud_y, "Next Delivery", color="#94a3b8", fontsize=fs_label, fontweight="bold", ha="center")
    hud_y -= 0.045
    ax_hud.text(0.5, hud_y, nxt_txt, color="#38bdf8", fontsize=fs_value, fontweight="bold", ha="center")
    hud_y -= 0.07

    # Delivery Scorecard Box
    on_t = step_info["n_on_time"]
    wait_t = step_info["n_waited"]
    late_t = step_info["n_late"]
    ax_hud.text(0.5, hud_y, "DELIVERY BREAKDOWN", color="#94a3b8", fontsize=fs_label, fontweight="bold", ha="center")
    hud_y -= 0.055
    ax_hud.text(0.5, hud_y, f"● On-Time: {int(on_t)}", color="#22c55e", fontsize=fs_small, fontweight="bold", ha="center")
    hud_y -= 0.045
    ax_hud.text(0.5, hud_y, f"● Wait: {int(wait_t)}", color="#eab308", fontsize=fs_small, fontweight="bold", ha="center")
    hud_y -= 0.045
    ax_hud.text(0.5, hud_y, f"● Late: {int(late_t)}", color="#ef4444", fontsize=fs_small, fontweight="bold", ha="center") 
# ============================================================================
# Video Exporters (MP4 & Side-by-Side Comparison)
# ============================================================================
 
def render_model_video(ep_data: dict, out_path: str, title: str, fps: int = 15, sub_frames: int = 8):
    """Render a high-definition MP4 video for a single model run."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 6), dpi=120, facecolor="#0b0f19")
    gs = fig.add_gridspec(1, 2, width_ratios=[65, 35], wspace=0.15)
    ax_map = fig.add_subplot(gs[0])
    ax_hud = fig.add_subplot(gs[1])
 
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9)
    n_steps = len(ep_data["steps"])
 
    print(f"Rendering {out_path} ({n_steps} steps, {fps} fps)...")
 
    for s in range(1, n_steps):
        for sub in range(sub_frames):
            alpha = (sub + 1) / sub_frames
            draw_single_panel(ax_map, ax_hud, ep_data, s, interp_alpha=alpha, title_prefix=title)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
 
    # Pause on the final frame
    for _ in range(fps * 2):
        draw_single_panel(ax_map, ax_hud, ep_data, n_steps - 1, interp_alpha=1.0, title_prefix=title)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(rgba[:, :, :3])
 
    writer.close()
    plt.close(fig)
    print(f"Saved video to -> {out_path}")
 
 
def render_comparison_video(ep_before: dict, ep_after: dict, out_path: str,
                            title_before: str = "Before Training (Random)",
                            title_after: str = "After Training (GNN-DQN)",
                            fps: int = 15, sub_frames: int = 8):
    """Render side-by-side split screen video comparing Before vs After training on identical map."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 7), dpi=120, facecolor="#0b0f19")
    gs = fig.add_gridspec(1, 4, width_ratios=[65, 35, 65, 35], wspace=0.15)
    ax_map_b = fig.add_subplot(gs[0])
    ax_hud_b = fig.add_subplot(gs[1])
    ax_map_a = fig.add_subplot(gs[2])
    ax_hud_a = fig.add_subplot(gs[3])
 
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=9)
    max_steps = max(len(ep_before["steps"]), len(ep_after["steps"]))
 
    print(f"Rendering split-screen comparison to {out_path}...")
 
    for s in range(1, max_steps):
        s_b = min(s, len(ep_before["steps"]) - 1)
        s_a = min(s, len(ep_after["steps"]) - 1)
 
        for sub in range(sub_frames):
            alpha = (sub + 1) / sub_frames
            draw_single_panel(ax_map_b, ax_hud_b, ep_before, s_b, interp_alpha=alpha, title_prefix=title_before, compact=True)
            draw_single_panel(ax_map_a, ax_hud_a, ep_after, s_a, interp_alpha=alpha, title_prefix=title_after, compact=True)
 
            fig.suptitle("Dynamic Vehicle Routing under Uncertain Traffic — Before vs. After Training",
                         color="#38bdf8", fontsize=13, fontweight="bold", y=0.98)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[:, :, :3])
 
    # Hold on end frame
    for _ in range(fps * 3):
        draw_single_panel(ax_map_b, ax_hud_b, ep_before, len(ep_before["steps"]) - 1, interp_alpha=1.0, title_prefix=title_before, compact=True)
        draw_single_panel(ax_map_a, ax_hud_a, ep_after, len(ep_after["steps"]) - 1, interp_alpha=1.0, title_prefix=title_after, compact=True)
        fig.suptitle("Dynamic Vehicle Routing under Uncertain Traffic — Before vs. After Training",
                     color="#38bdf8", fontsize=13, fontweight="bold", y=0.98)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        writer.append_data(rgba[:, :, :3])
 
    writer.close()
    plt.close(fig)
    print(f"Saved comparison video to -> {out_path}")
 
 
# ============================================================================
# Interactive HTML Dashboard Generator (Self-Contained Web App)
# ============================================================================
 
def export_interactive_html(all_episodes: dict[str, dict], out_path: str = "videos/interactive_dashboard.html"):
    """Create a luxury, standalone interactive HTML5 web dashboard with play/pause and step scrubber."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
 
    # Convert NumPy arrays to JSON serializable objects
    serialized_data = {}
    for method, ep in all_episodes.items():
        steps_json = []
        for st in ep["steps"]:
            steps_json.append({
                "step": int(st["step"]),
                "from_node": int(st["from_node"]),
                "to_node": int(st["to_node"]),
                "from_pos": st["from_pos"].tolist(),
                "to_pos": st["to_pos"].tolist(),
                "route": [int(x) for x in st["route"]],
                "remaining_cap": float(st["remaining_cap"]),
                "elapsed_time": float(st["elapsed_time"]),
                "total_cost": float(st["total_cost"]),
                "reward": float(st["reward"]),
                "cumulative_reward": float(st["cumulative_reward"]),
                "status": str(st["status"]),
                "served_mask": st["served_mask"].tolist(),
                "n_on_time": int(st["n_on_time"]),
                "n_waited": int(st["n_waited"]),
                "n_late": int(st["n_late"]),
                "q_values": st.get("q_values"),
                "action_mask": st.get("action_mask"),
            })
 
        serialized_data[method] = {
            "coords": ep["coords"].tolist(),
            "demands": ep["demands"].tolist(),
            "window_start": ep["window_start"].tolist(),
            "window_end": [float(x) if not np.isinf(x) else 9999.0 for x in ep["window_end"]],
            "traffic_matrix": ep["traffic_matrix"].tolist(),
            "cap_max": float(ep["cap_max"]),
            "obs_radius": float(ep["obs_radius"]),
            "steps": steps_json,
        }
 
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dynamic Vehicle Routing — Interactive Road Map Visualizer</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }}
  body {{ background: #070b14; color: #f8fafc; display: flex; height: 100vh; overflow: hidden; }}
  #sidebar {{ width: 35%; min-width: 380px; max-width: 450px; background: linear-gradient(180deg, #0c1322 0%, #0f172a 100%); border-right: 1px solid #1e293b; padding: 22px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }}
  #main {{ flex: 1; display: flex; flex-direction: column; padding: 16px; gap: 12px; align-items: center; justify-content: center; background: radial-gradient(ellipse at center, #0d1424 0%, #070b14 70%); }}
  h1 {{ font-size: 1.1rem; color: #38bdf8; letter-spacing: 1px; text-transform: uppercase; font-weight: 900; text-shadow: 0 0 20px rgba(56,189,248,0.3); }}
  .subtitle {{ font-size: 0.72rem; color: #475569; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-top: -4px; }}
  .card {{ background: linear-gradient(135deg, #1e293b 0%, #172032 100%); border: 1px solid #334155; border-radius: 12px; padding: 16px; box-shadow: 0 4px 12px -2px rgba(0,0,0,0.4); }}
  .stat-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.88rem; }}
  .stat-label {{ color: #64748b; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.5px; text-transform: uppercase; }}
  .stat-val {{ font-weight: 800; color: #e2e8f0; font-variant-numeric: tabular-nums; font-size: 0.85rem; }}
  .btn {{ background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: white; border: none; padding: 9px 16px; border-radius: 8px; cursor: pointer; font-weight: 700; font-size: 0.85rem; transition: all 0.2s; box-shadow: 0 2px 8px -2px rgba(2,132,199,0.4); }}
  .btn:hover {{ transform: translateY(-2px); box-shadow: 0 4px 16px -2px rgba(2,132,199,0.6); }}
  .btn-speed {{ background: #0f172a; border: 1px solid #334155; color: #64748b; padding: 4px 10px; font-size: 0.72rem; border-radius: 6px; cursor: pointer; font-weight: 800; transition: all 0.15s; }}
  .btn-speed.active {{ background: #38bdf8; color: #0c1322; border-color: #38bdf8; box-shadow: 0 0 10px rgba(56,189,248,0.3); }}
  .btn-group {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  select {{ width: 100%; background: #0b0f19; color: #38bdf8; border: 1px solid #334155; padding: 10px 12px; border-radius: 8px; font-weight: 800; font-size: 0.92rem; cursor: pointer; appearance: none; }}
  #canvas-container {{ width: 100%; max-width: 900px; aspect-ratio: 1; background: #080c15; border: 1px solid #1a2236; border-radius: 14px; position: relative; box-shadow: 0 0 40px -10px rgba(56,189,248,0.08), 0 12px 30px -8px rgba(0,0,0,0.6); overflow: hidden; }}
  canvas {{ width: 100%; height: 100%; display: block; }}
  .badge {{ padding: 6px 14px; border-radius: 8px; font-size: 0.78rem; font-weight: 800; text-align: center; letter-spacing: 0.8px; text-transform: uppercase; }}
  .badge-ontime {{ background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); color: white; box-shadow: 0 0 12px rgba(22,163,74,0.3); }}
  .badge-wait {{ background: linear-gradient(135deg, #ca8a04 0%, #a16207 100%); color: white; box-shadow: 0 0 12px rgba(202,138,4,0.3); }}
  .badge-late {{ background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); color: white; box-shadow: 0 0 12px rgba(220,38,38,0.3); }}
  .progress-bg {{ background: #1e293b; border-radius: 6px; height: 8px; width: 100%; margin-top: 4px; overflow: hidden; border: 1px solid #334155; }}
  .progress-fill {{ background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%); height: 100%; transition: width 0.1s; border-radius: 4px; }}
  .legend {{ display: flex; gap: 12px; align-items: center; margin-top: 8px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; font-size: 0.7rem; color: #64748b; font-weight: 600; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
  .section-divider {{ border-bottom: 1px solid #1e293b; margin: 10px 0; }}
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>🗺️ DVRP Road Map</h1>
    <div class="subtitle">Reinforcement Learning Vehicle Routing</div>
  </div>
  <div class="card">
    <label class="stat-label" style="display:block; margin-bottom: 8px;">Routing Policy</label>
    <select id="methodSelect" onchange="changeMethod(this.value)">
      {"".join(f'<option value="{k}">{k.replace("_"," ").upper()}</option>' for k in serialized_data.keys())}
    </select>
  </div>
  
  <div class="card">
    <div style="text-align: center; font-size: 1.1rem; font-weight: 800; color: #38bdf8; margin-bottom: 12px; letter-spacing: 1px;">LIVE TELEMETRY HUD</div>
    <div id="statusBadge" class="badge badge-ontime" style="width: 100%; margin-bottom: 24px; text-align: left; padding: 6px 8px;">STATUS: AT DEPOT</div>
    
    <div class="stat-row" style="margin-bottom: 12px;"><span class="stat-label" style="text-transform: capitalize; font-size: 0.9rem;">Step</span><span class="stat-val" id="hudStep">0 / 0</span></div>
    <div class="stat-row" style="margin-bottom: 12px;"><span class="stat-label" style="text-transform: capitalize; font-size: 0.9rem;">Clock Time</span><span class="stat-val" id="hudTime">0.0 min</span></div>
    <div class="stat-row" style="margin-bottom: 12px;"><span class="stat-label" style="text-transform: capitalize; font-size: 0.9rem;">Travel Cost</span><span class="stat-val" id="hudCost">0.00</span></div>
    <div class="stat-row" style="margin-bottom: 12px;"><span class="stat-label" style="text-transform: capitalize; font-size: 0.9rem;">Reward</span><span class="stat-val" id="hudReward">0.00</span></div>
    <div class="stat-row" style="margin-bottom: 12px;"><span class="stat-label" style="text-transform: capitalize; font-size: 0.9rem;">Served</span><span class="stat-val" id="hudServed">0 / 15</span></div>
    
    <div style="margin-top: 24px; margin-bottom: 16px;">
      <div class="stat-label" style="text-transform: none; color: #94a3b8; font-size: 0.85rem; margin-bottom: 4px;">Vehicle Load: <span id="hudCap">0 / 30</span></div>
      <div class="progress-bg" style="height: 12px; border-radius: 0; background: #334155; border:none;"><div class="progress-fill" id="capBar" style="background: #0ea5e9; border-radius: 0;"></div></div>
    </div>
    
    <div class="stat-row" style="margin-bottom: 24px;">
      <span class="stat-label" style="text-transform: none; font-size: 0.9rem;">Next Delivery</span>
      <span class="stat-val" id="nxtDelivery" style="color:#38bdf8; font-size: 0.9rem;">-</span>
    </div>

    <div style="margin-top: 16px; text-align: center; font-size: 0.9rem; font-weight: 800; color: #94a3b8;">DELIVERY BREAKDOWN</div>
    <div style="display:flex; justify-content:center; gap: 8px; margin-top: 12px; margin-bottom: 8px;">
      <div style="font-size: 0.85rem; font-weight:bold; color:#22c55e;">● On-Time: <span id="scOnTime">0</span></div>
      <div style="font-size: 0.85rem; font-weight:bold; color:#eab308;">● Wait: <span id="scWaited">0</span></div>
    </div>
    <div style="display:flex; justify-content:center;">
      <div style="font-size: 0.85rem; font-weight:bold; color:#ef4444;">● Late: <span id="scLate">0</span></div>
    </div>
  </div>

  <div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
      <label class="stat-label">Playback</label>
      <div style="display: flex; gap: 3px;">
        <button class="btn-speed" onclick="setSpeed(0.5, this)">0.5×</button>
        <button class="btn-speed active" onclick="setSpeed(1.0, this)">1×</button>
        <button class="btn-speed" onclick="setSpeed(2.0, this)">2×</button>
        <button class="btn-speed" onclick="setSpeed(4.0, this)">4×</button>
      </div>
    </div>
    <div class="btn-group">
      <button class="btn" onclick="togglePlay()" id="playBtn" style="flex: 1;">▶ Play</button>
      <button class="btn" onclick="prevStep()">⏮</button>
      <button class="btn" onclick="nextStep()">⏭</button>
      <button class="btn" onclick="resetSim()">↺</button>
    </div>
    <input type="range" id="scrubber" min="0" max="10" value="0" step="0.01" style="width: 100%; margin-top: 12px; accent-color: #38bdf8; cursor: pointer; height: 6px;" oninput="onScrub(this.value)">
  </div>
  <div class="card">
    <label class="stat-label" style="display:block; margin-bottom: 6px;">Map Legend</label>
    <div class="legend">
      <div class="legend-item"><span class="legend-dot" style="background:#2563eb;border-radius:2px;"></span> Depot</div>
      <div class="legend-item"><span class="legend-dot" style="background:#f8fafc;"></span> Unserved</div>
      <div class="legend-item"><span class="legend-dot" style="background:#22c55e;"></span> Served</div>
      <div class="legend-item"><span class="legend-dot" style="background:#00e5ff;width:16px;height:4px;border-radius:2px;"></span> Route</div>
    </div>
  </div>
</div>
<div id="main">
  <div id="canvas-container">
    <canvas id="simCanvas" width="900" height="900"></canvas>
  </div>
</div>
<script>
const EP_DATA = {json.dumps(serialized_data)};
let currentMethod = Object.keys(EP_DATA)[0];
let currentStep = 0;
let interpAlpha = 0.0;
let isPlaying = false;
let speedMultiplier = 1.0;
let lastTimestamp = null;
const baseLegDuration = 1200;
let particles = [];
let frameCount = 0;
 
const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");
 
function setSpeed(sp, btn) {{
  speedMultiplier = sp;
  document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
}}
 
function changeMethod(m) {{
  currentMethod = m;
  currentStep = 0;
  interpAlpha = 0.0;
  particles = [];
  document.getElementById("scrubber").max = EP_DATA[m].steps.length - 1;
  document.getElementById("scrubber").value = 0;
}}
 
// Traffic color from green to yellow to red based on congestion multiplier
function trafficColor(val, alpha) {{
  // val: 1.0 = clear (green), 2.0+ = heavy (red)
  const t = Math.min(1.0, Math.max(0.0, (val - 1.0) / 1.5));
  const r = Math.round(34 + t * 205);
  const g = Math.round(197 - t * 130);
  const b = Math.round(94 - t * 60);
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}
 
function render(timestamp) {{
  if (!lastTimestamp) lastTimestamp = timestamp;
  const dt = timestamp - lastTimestamp;
  lastTimestamp = timestamp;
  frameCount++;
 
  const data = EP_DATA[currentMethod];
  const steps = data.steps;
  const maxStep = steps.length - 1;
 
  if (isPlaying && currentStep < maxStep) {{
    const stepDuration = baseLegDuration / speedMultiplier;
    interpAlpha += dt / stepDuration;
    if (interpAlpha >= 1.0) {{
      interpAlpha = 0.0;
      currentStep++;
      if (currentStep >= maxStep) {{
        currentStep = maxStep;
        interpAlpha = 1.0;
        isPlaying = false;
        document.getElementById("playBtn").innerText = "▶ Play";
      }}
    }}
  }}
 
  drawFrame(currentStep, interpAlpha);
  requestAnimationFrame(render);
}}
 
function drawFrame(stepIdx, alpha) {{
  const data = EP_DATA[currentMethod];
  const steps = data.steps;
  const maxStep = steps.length - 1;
  stepIdx = Math.min(stepIdx, maxStep);
  const st = steps[stepIdx];
  const prevSt = steps[Math.max(0, stepIdx - 1)];
 
  const coords = data.coords;
  const demands = data.demands;
  const wStart = data.window_start;
  const wEnd = data.window_end;
  const traffic = data.traffic_matrix;
  const N = coords.length;
  const pad = 80;
  const sz = canvas.width - pad * 2;
 
  function toScreen(pt) {{
    return [pad + pt[0] * sz, pad + (1.0 - pt[1]) * sz];
  }}
 
  // Smoothly interpolated vehicle position
  const fromPt = stepIdx === 0 ? st.to_pos : st.from_pos;
  const toPt = st.to_pos;
  const ease = alpha < 0.5 ? 2 * alpha * alpha : 1 - Math.pow(-2 * alpha + 2, 2) / 2;
  const vehPosCoords = [
    (1.0 - ease) * fromPt[0] + ease * toPt[0],
    (1.0 - ease) * fromPt[1] + ease * toPt[1]
  ];
  const vehPos = toScreen(vehPosCoords);
 
  // Smoothly interpolated telemetry
  const interpCap = (1.0 - alpha) * (stepIdx === 0 ? st.remaining_cap : prevSt.remaining_cap) + alpha * st.remaining_cap;
  const interpTime = (1.0 - alpha) * (stepIdx === 0 ? st.elapsed_time : prevSt.elapsed_time) + alpha * st.elapsed_time;
  const interpCost = (1.0 - alpha) * (stepIdx === 0 ? st.total_cost : prevSt.total_cost) + alpha * st.total_cost;
  const interpReward = (1.0 - alpha) * (stepIdx === 0 ? st.cumulative_reward : prevSt.cumulative_reward) + alpha * st.cumulative_reward;
 
  // Update HUD
  document.getElementById("hudStep").innerText = stepIdx + " / " + maxStep;
  document.getElementById("hudTime").innerText = interpTime.toFixed(1) + " min";
  document.getElementById("hudCost").innerText = interpCost.toFixed(2);
  document.getElementById("hudReward").innerText = interpReward.toFixed(2);
  document.getElementById("hudServed").innerText = st.served_mask.filter(x => x).length + " / " + (N - 1);
  
  const capMax = data.cap_max;
  const load = capMax - interpCap;
  document.getElementById("hudCap").innerText = Math.round(load) + " / " + Math.round(capMax);
  document.getElementById("capBar").style.width = ((load / capMax) * 100) + "%";

  const nxt = st.to_node;
  if (nxt === 0) {{
    document.getElementById("nxtDelivery").innerText = "DEPOT";
  }} else {{
    const we = wEnd[nxt];
    const we_str = we > 9000 ? "∞" : Math.round(we);
    document.getElementById("nxtDelivery").innerText = "C" + nxt + " (win [" + Math.round(wStart[nxt]) + "-" + we_str + "])";
  }}

  document.getElementById("scOnTime").innerText = st.n_on_time;
  document.getElementById("scWaited").innerText = st.n_waited;
  document.getElementById("scLate").innerText = st.n_late;
  
  const badge = document.getElementById("statusBadge");
  let statusTxt = st.status;
  if (statusTxt.includes("AT DEPOT")) {{
      const ws = wStart[0];
      const we = wEnd[0] > 9000 ? 999 : wEnd[0];
      statusTxt = "ON-TIME (window [" + Math.round(ws) + "-" + Math.round(we) + "])";
  }}
  badge.innerText = "STATUS: " + statusTxt;
  badge.className = "badge " + (st.status.includes("LATE") ? "badge-late" : (st.status.includes("WAITED") ? "badge-wait" : "badge-ontime"));
  document.getElementById("hudCost").innerText = interpCost.toFixed(2);
  document.getElementById("hudReward").innerText = interpReward.toFixed(2);
  // === CANVAS RENDERING ===
 
  // Background with subtle gradient
  const bgGrad = ctx.createRadialGradient(canvas.width/2, canvas.height/2, 50, canvas.width/2, canvas.height/2, canvas.width/2);
  bgGrad.addColorStop(0, "#0d1424");
  bgGrad.addColorStop(1, "#070b14");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
 
  // Subtle grid (city blocks)
  ctx.strokeStyle = "rgba(30, 41, 59, 0.4)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 20; i++) {{
    const pos = pad + i * sz / 20;
    ctx.beginPath(); ctx.moveTo(pos, pad); ctx.lineTo(pos, pad + sz); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, pos); ctx.lineTo(pad + sz, pos); ctx.stroke();
  }}
 
  // === ROAD NETWORK ===
  // We draw a sparse background road network + any specific roads the vehicle needs
  const edgesToDraw = new Set();
  const roadThreshold = 0.45;
  
  // 1. Proximity edges (Sparse city layout)
  for (let i = 0; i < N; i++) {{
    for (let j = i + 1; j < N; j++) {{
      const dx = coords[i][0] - coords[j][0];
      const dy = coords[i][1] - coords[j][1];
      if (Math.sqrt(dx * dx + dy * dy) < roadThreshold) {{
        edgesToDraw.add(`${{i}}-${{j}}`);
      }}
    }}
  }}
  
  // 2. Edges actually used by the current vehicle route (so it never drives off-road)
  const allSteps = data.steps;
  for (let s = 0; s < allSteps.length; s++) {{
    const u = allSteps[s].from_node;
    const v = allSteps[s].to_node;
    if (u !== v) {{
      const minN = Math.min(u, v);
      const maxN = Math.max(u, v);
      edgesToDraw.add(`${{minN}}-${{maxN}}`);
    }}
  }}
 
  // Draw the collected road segments
  edgesToDraw.forEach(edgeStr => {{
    const [iStr, jStr] = edgeStr.split("-");
    const i = parseInt(iStr), j = parseInt(jStr);
    const [x1, y1] = toScreen(coords[i]);
    const [x2, y2] = toScreen(coords[j]);
    const congestion = traffic[i][j];
 
    // Asphalt road base (dark grey)
    ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
    ctx.lineWidth = 7;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
 
    // Road surface (slightly lighter)
    ctx.strokeStyle = "rgba(71, 85, 105, 0.3)";
    ctx.lineWidth = 5;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
 
    // Center lane dashes
    ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 10]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([]);
 
    // Traffic congestion overlay glow
    ctx.strokeStyle = trafficColor(congestion, 0.25);
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }});
 
  // === COMPLETED ROUTE (Road-style) ===
  if (st.route && st.route.length > 1) {{
    // Road base for completed route
    ctx.strokeStyle = "rgba(2, 132, 199, 0.3)";
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let r = 0; r < st.route.length - 1; r++) {{
      const [sx, sy] = toScreen(coords[st.route[r]]);
      if (r === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }}
    ctx.stroke();
 
    // Bright route line
    ctx.strokeStyle = "#0284c7";
    ctx.lineWidth = 4;
    ctx.beginPath();
    for (let r = 0; r < st.route.length - 1; r++) {{
      const [sx, sy] = toScreen(coords[st.route[r]]);
      if (r === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }}
    ctx.stroke();
 
    // Active segment being driven (glowing cyan)
    const [lastX, lastY] = toScreen(fromPt);
    // Glow
    ctx.strokeStyle = "rgba(0, 229, 255, 0.25)";
    ctx.lineWidth = 14;
    ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(vehPos[0], vehPos[1]); ctx.stroke();
    // Core
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 4;
    ctx.shadowColor = "#00e5ff";
    ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.moveTo(lastX, lastY); ctx.lineTo(vehPos[0], vehPos[1]); ctx.stroke();
    ctx.shadowBlur = 0;
  }}
 
  // === VEHICLE PARTICLES (Trail effect) ===
  if (isPlaying && frameCount % 3 === 0) {{
    particles.push({{ x: vehPos[0], y: vehPos[1], life: 1.0 }});
  }}
  for (let p = particles.length - 1; p >= 0; p--) {{
    particles[p].life -= 0.025;
    if (particles[p].life <= 0) {{ particles.splice(p, 1); continue; }}
    ctx.beginPath();
    ctx.arc(particles[p].x, particles[p].y, 3 * particles[p].life, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(0, 229, 255, ${{particles[p].life * 0.4}})`;
    ctx.fill();
  }}
 
  // === OBSERVATION RADIUS ===
  const radarPulse = 0.9 + 0.1 * Math.sin(frameCount * 0.05);
  ctx.beginPath();
  ctx.arc(vehPos[0], vehPos[1], data.obs_radius * sz * radarPulse, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 229, 255, 0.04)";
  ctx.fill();
  ctx.strokeStyle = "rgba(0, 229, 255, 0.25)";
  
  // === CUSTOMER NODES & DEPOT ===
  for (let i = 0; i < N; i++) {{
    const [nx, ny] = toScreen(coords[i]);
    if (i === 0) {{
      // Depot (Square)
      ctx.fillStyle = "#2563eb";
      ctx.strokeStyle = "#60a5fa";
      ctx.lineWidth = 2.5;
      ctx.fillRect(nx - 12, ny - 12, 24, 24);
      ctx.strokeRect(nx - 12, ny - 12, 24, 24);
      ctx.fillStyle = "white";
      ctx.font = "bold 9px 'Inter', system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("DEPOT", nx, ny);
    }} else {{
      const served = st.served_mask[i];
      let nodeColor = served ? "#22c55e" : "#f8fafc";
      let edgeColor = served ? "#16a34a" : "#cbd5e1";

      if (st.to_node === i && interpAlpha < 1.0) {{
          nodeColor = "#00e5ff";
          edgeColor = "#00b8d4";
      }}

      ctx.beginPath();
      ctx.arc(nx, ny, 8, 0, Math.PI * 2);
      ctx.fillStyle = nodeColor;
      ctx.fill();
      ctx.strokeStyle = edgeColor;
      ctx.lineWidth = 2.5;
      ctx.stroke();

      ctx.fillStyle = "#cbd5e1";
      ctx.font = "bold 10px 'Inter', system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText("C" + i, nx, ny + 14);
    }}
  }}
 
  // === VEHICLE ===
  ctx.beginPath();
  ctx.arc(vehPos[0], vehPos[1], 15, 0, Math.PI * 2);
  ctx.fillStyle = "#00e5ff";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 3;
  ctx.stroke();

  // Outer Glow
  ctx.beginPath();
  ctx.arc(vehPos[0], vehPos[1], 24, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0, 229, 255, 0.3)";
  ctx.fill();

  ctx.fillStyle = "#0b0f19";
  ctx.font = "bold 14px 'Inter', system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("V", vehPos[0], vehPos[1]);
}}
 
function togglePlay() {{
  isPlaying = !isPlaying;
  document.getElementById("playBtn").innerText = isPlaying ? "⏸ Pause" : "▶ Play";
  if (isPlaying && currentStep >= EP_DATA[currentMethod].steps.length - 1) {{
    currentStep = 0;
    interpAlpha = 0.0;
    particles = [];
  }}
}}
 
function nextStep() {{
  const maxS = EP_DATA[currentMethod].steps.length - 1;
  if (currentStep < maxS) {{ currentStep++; interpAlpha = 0.0; }}
}}
 
function prevStep() {{
  if (currentStep > 0) {{ currentStep--; interpAlpha = 0.0; }}
}}
 
function resetSim() {{
  currentStep = 0;
  interpAlpha = 0.0;
  isPlaying = false;
  particles = [];
  document.getElementById("playBtn").innerText = "▶ Play";
}}
 
function onScrub(val) {{
  const fVal = parseFloat(val);
  currentStep = Math.floor(fVal);
  interpAlpha = fVal - currentStep;
}}
 
changeMethod(currentMethod);
requestAnimationFrame(render);
</script>
</body>
</html>
"""
 
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
 
# ============================================================================
# Main Entry Point
# ============================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--method", type=str, default="all",
                        choices=["random", "greedy", "gnn_dqn", "gnn_ppo", "all"])
    parser.add_argument("--compare", action="store_true", default=True,
                        help="render side-by-side split screen comparison video")
    parser.add_argument("--out-dir", type=str, default="videos")
    args = parser.parse_args()
 
    cfg = load_config(args.config)
    env = DynamicCVRPEnv(cfg)
    n_nodes = env.N
    os.makedirs(args.out_dir, exist_ok=True)
 
    # Instantiate policies
    policies: dict[str, Policy] = {
        "random": RandomPolicy(seed=args.seed),
        "greedy": GreedyNearestNeighbor(),
    }
 
    dqn_path = find_latest_checkpoint("gnn_dqn_seed42")
    if dqn_path and os.path.exists(dqn_path):
        policies["gnn_dqn"] = DQNGreedyPolicy(dqn_path, n_nodes)
 
    ppo_path = find_latest_checkpoint("gnn_ppo_seed42")
    if ppo_path and os.path.exists(ppo_path):
        policies["gnn_ppo"] = PPOGreedyPolicy(ppo_path, n_nodes)
 
    recorded_episodes = {}
 
    # 1. Record Episodes for each model on the same seed
    for name, pol in policies.items():
        print(f"Recording episode for {name} on seed {args.seed}...")
        rec = record_episode(env, pol, args.seed)
        recorded_episodes[name] = rec
 
        # Export individual MP4
        video_file = os.path.join(args.out_dir, f"routing_{name}_seed{args.seed}.mp4")
        render_model_video(rec, video_file, title=f"Model: {name.upper()}")
 
    # 2. Render Side-by-Side Split Screen Comparison Video
    if "random" in recorded_episodes and "gnn_dqn" in recorded_episodes:
        compare_file = os.path.join(args.out_dir, f"before_vs_after_dqn_seed{args.seed}.mp4")
        render_comparison_video(
            recorded_episodes["random"],
            recorded_episodes["gnn_dqn"],
            compare_file,
            title_before="Before Training (Random)",
            title_after="After Training (GNN-DQN)"
        )
 
    if "random" in recorded_episodes and "gnn_ppo" in recorded_episodes:
        compare_ppo_file = os.path.join(args.out_dir, f"before_vs_after_ppo_seed{args.seed}.mp4")
        render_comparison_video(
            recorded_episodes["random"],
            recorded_episodes["gnn_ppo"],
            compare_ppo_file,
            title_before="Before Training (Random)",
            title_after="After Training (GNN-PPO)"
        )
 
    # 3. Export Standalone Interactive HTML5 Web Dashboard
    html_file = os.path.join(args.out_dir, "interactive_dashboard.html")
    export_interactive_html(recorded_episodes, html_file)
 
    print("\n" + "=" * 80)
    print("ALL VISUALIZATION DELIVERABLES SUCCESSFULLY GENERATED IN videos/ :")
    print(f"1. Individual Videos: {args.out_dir}/routing_*.mp4")
    print(f"2. Split-Screen Comparison: {args.out_dir}/before_vs_after_*.mp4")
    print(f"3. Interactive Web Dashboard: {html_file}")
    print("=" * 80 + "\n")
 