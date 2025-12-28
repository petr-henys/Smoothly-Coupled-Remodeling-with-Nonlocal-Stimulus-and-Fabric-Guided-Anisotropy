"""Anderson acceleration diagnostic plot.

Multi-panel figure showing Anderson accelerator behavior:
- Convergence curves (residual vs subiteration)
- Event timeline (restarts: stall/cond, step limiting)
- Residual timeline with step boundaries
- Gram matrix conditioning
- History size evolution

Input:
    results/<run_dir>/ containing subiterations.csv, steps.csv and config.json

Output:
    manuscript/images/anderson_diagnostic.png

Usage:
    python3 analysis/anderson_diagnostic_plot.py [run_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from analysis.plot_utils import (
    PLOT_LINEWIDTH,
    PUBLICATION_DPI,
    apply_style,
    save_manuscript_figure,
    setup_axis_style,
)
from postprocessor import SimulationLoader

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RUN_DIR = Path(".results_box/")
OUTPUT_FILE = Path("manuscript/images/anderson_diagnostic.png")

# Colors for events
COLOR_STALL = "#e74c3c"   # Red - stall restart
COLOR_COND = "#3498db"    # Blue - condition restart / conditioning line
COLOR_LIMITED = "#f39c12" # Orange - step limiting
COLOR_RESTART = "#9b59b6" # Purple - general restart (for cumulative plot)


# =============================================================================
# Analysis functions
# =============================================================================


def print_statistics(subiters: pd.DataFrame, config: dict, run_dir: Path) -> None:
    """Print summary statistics to console."""
    solver_cfg = config.get("solver", {})
    m = solver_cfg.get("m", "?")
    beta = solver_cfg.get("beta", "?")
    lam = solver_cfg.get("lam", "?")

    print("\n=== ANDERSON DIAGNOSTIC ANALYSIS ===")
    print(f"Run directory: {run_dir}")
    print(f"Anderson parameters: m={m}, β={beta}, λ={lam}")
    print(f"Total subiterations: {len(subiters)}")

    steps = subiters["step"].unique()
    print(f"Number of timesteps: {len(steps)}")

    subiters_per_step = subiters.groupby("step")["iter"].max()
    print(
        f"Subiterations per step: min={subiters_per_step.min()}, "
        f"max={subiters_per_step.max()}, mean={subiters_per_step.mean():.1f}"
    )

    # Restart events
    if "restart" in subiters.columns:
        restart_count = subiters["restart"].astype(bool).sum()
        print(f"AA restarts: {restart_count}")
        
        # Show restart reasons if available
        if "restart_reason" in subiters.columns and restart_count > 0:
            rst_reasons = subiters.loc[subiters["restart"].astype(bool), "restart_reason"]
            rst_reasons = rst_reasons[rst_reasons.astype(str).str.len() > 0]
            if len(rst_reasons) > 0:
                # Group by prefix (stall, cond)
                stall_count = sum(1 for r in rst_reasons if str(r).startswith("stall"))
                cond_count = sum(1 for r in rst_reasons if str(r).startswith("cond"))
                if stall_count:
                    print(f"  - stall: {stall_count}")
                if cond_count:
                    print(f"  - cond: {cond_count}")
    
    # Step limiting events
    if "limited" in subiters.columns:
        lim_count = subiters["limited"].astype(bool).sum()
        lim_rate = 100 * lim_count / max(len(subiters), 1)
        print(f"Step limiting events: {lim_count} ({lim_rate:.1f}%)")

    if "condH" in subiters.columns:
        cond_vals = subiters["condH"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(cond_vals) > 0:
            print(
                f"Gram condition: min={cond_vals.min():.1e}, "
                f"max={cond_vals.max():.1e}, median={cond_vals.median():.1e}"
            )


def create_global_index(subiters: pd.DataFrame) -> np.ndarray:
    """Create a global iteration index for plotting across all timesteps."""
    return np.arange(len(subiters))


def plot_convergence(ax: plt.Axes, subiters: pd.DataFrame, config: dict, steps_df: pd.DataFrame | None = None) -> None:
    """Plot convergence curves colored by timestep.
    
    Each (step, attempt) pair is treated as a unique curve.
    """
    # Create unique key for each step attempt
    if "attempt" in subiters.columns:
        subiters = subiters.copy()
        subiters["step_attempt"] = subiters["step"].astype(str) + "_" + subiters["attempt"].astype(str)
        unique_keys = subiters.groupby(["step", "attempt"]).ngroups
        step_attempts = subiters.groupby(["step", "attempt"], sort=True).first().index.tolist()
    else:
        # Fallback for old data without attempt column
        step_attempts = [(s, 1) for s in sorted(subiters["step"].unique())]
        unique_keys = len(step_attempts)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=max(unique_keys - 1, 1))

    for i, (step, attempt) in enumerate(step_attempts):
        if "attempt" in subiters.columns:
            step_data = subiters[(subiters["step"] == step) & (subiters["attempt"] == attempt)].sort_values("iter")
        else:
            step_data = subiters[subiters["step"] == step].sort_values("iter")
        
        if step_data.empty or "proj_res" not in step_data.columns:
            continue

        ax.semilogy(
            step_data["iter"].values,
            step_data["proj_res"].values,
            color=cmap(norm(i)),
            linestyle="-",
            linewidth=PLOT_LINEWIDTH,
            alpha=0.7,
        )

    solver_cfg = config.get("solver", {})
    coupling_tol = solver_cfg.get("coupling_tol", 1e-6)
    ax.axhline(coupling_tol, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add colorbar for step progression
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("Step", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    setup_axis_style(ax, xlabel="Subiteration", ylabel="Residual", title="", loglog=False)
    ax.set_title("Convergence", fontsize=8)


# Markers for restart reasons
RESTART_MARKERS = {
    "stall": ("v", COLOR_STALL),   # Red triangle down - stall
    "cond": ("d", COLOR_COND),     # Blue diamond - condition
}

# Color for rejected steps
COLOR_REJECTED = "#c0392b"  # Dark red


def plot_dt_evolution(ax: plt.Axes, steps_df: pd.DataFrame) -> None:
    """Plot timestep size evolution over simulation steps."""
    if steps_df is None or steps_df.empty or "dt_days" not in steps_df.columns:
        ax.text(0.5, 0.5, "No dt data", ha="center", va="center", transform=ax.transAxes)
        return

    step_nums = steps_df["step"].values
    dt_vals = steps_df["dt_days"].values
    
    # Identify rejected steps (accepted=0)
    rejected_mask = np.zeros(len(steps_df), dtype=bool)
    if "accepted" in steps_df.columns:
        rejected_mask = steps_df["accepted"].values == 0
    elif "converged" in steps_df.columns:
        # Fallback for old data without 'accepted' column
        rejected_mask = steps_df["converged"].values == 0

    # Plot accepted steps
    accepted_mask = ~rejected_mask
    ax.semilogy(step_nums[accepted_mask], dt_vals[accepted_mask], 
                color="#2ecc71", linewidth=PLOT_LINEWIDTH, alpha=0.8, label="accepted")
    
    # Highlight rejected steps
    if rejected_mask.any():
        ax.scatter(step_nums[rejected_mask], dt_vals[rejected_mask],
                   c=COLOR_REJECTED, s=40, marker="x", zorder=5, 
                   linewidths=1.5, label="rejected")

    ax.set_xlim(step_nums.min() - 0.5, step_nums.max() + 0.5)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
    setup_axis_style(ax, xlabel="Step", ylabel="dt [days]", title="", loglog=False)
    ax.set_title("Timestep evolution", fontsize=8)


def plot_events_unified(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray, 
                        steps_df: pd.DataFrame | None = None) -> None:
    """Unified plot for all events: AA restarts (stall/cond), step limiting, and rejected steps.
    
    Shows cumulative event count with markers for different event types.
    """
    n_total = len(subiters)
    
    # Background: residual magnitude (semi-log)
    if "proj_res" in subiters.columns:
        res = subiters["proj_res"].values
        res_log = np.log10(np.clip(res, 1e-12, None))
        res_norm = (res_log - res_log.min()) / max(res_log.max() - res_log.min(), 1e-10)
        ax.fill_between(global_idx, 0, res_norm * 0.3, alpha=0.1, color="#7f8c8d", label=None)
    
    # Collect all events: (global_idx, event_type, color, marker)
    events = []
    
    # 1. AA Restart events (stall, cond)
    if "restart" in subiters.columns:
        restart_mask = subiters["restart"].astype(bool).values
        if restart_mask.any() and "restart_reason" in subiters.columns:
            reasons = subiters["restart_reason"].fillna("").astype(str).values
            for prefix, (marker, color) in RESTART_MARKERS.items():
                mask = restart_mask & np.array([str(r).startswith(prefix) for r in reasons])
                for idx in np.where(mask)[0]:
                    events.append((idx, prefix, color, marker))
    
    # 2. Step limiting events
    if "limited" in subiters.columns:
        limited_mask = subiters["limited"].astype(bool).values
        for idx in np.where(limited_mask)[0]:
            events.append((idx, "limited", COLOR_LIMITED, "o"))
    
    # 3. Rejected steps - distinguish between error (rej:err) and divergence (rej:div)
    if steps_df is not None and "accepted" in steps_df.columns:
        rejected_df = steps_df[steps_df["accepted"] == 0]
        for _, row in rejected_df.iterrows():
            step = row["step"]
            step_mask = subiters["step"].values == step
            if not step_mask.any():
                continue
            last_idx = np.where(step_mask)[0][-1]
            
            # Distinguish rejection reason
            if "converged" in row and row["converged"] == 0:
                # Rejected due to solver divergence
                events.append((last_idx, "rej:div", COLOR_REJECTED, "X"))
            else:
                # Rejected due to error tolerance
                events.append((last_idx, "rej:err", COLOR_LIMITED, "x"))
    elif steps_df is not None and "converged" in steps_df.columns:
        # Fallback for old data - only divergence info available
        rejected_steps = set(steps_df.loc[steps_df["converged"] == 0, "step"].values)
        for step in rejected_steps:
            step_mask = subiters["step"].values == step
            if step_mask.any():
                last_idx = np.where(step_mask)[0][-1]
                events.append((last_idx, "rej:div", COLOR_REJECTED, "X"))
    
    # Sort events by index
    events.sort(key=lambda x: x[0])
    
    # Build cumulative count
    cum_events = np.zeros(n_total, dtype=int)
    for i, (idx, _, _, _) in enumerate(events):
        cum_events[idx:] = i + 1
    
    # Plot cumulative line
    if events:
        ax.plot(global_idx, cum_events, color="#2c3e50", linewidth=PLOT_LINEWIDTH * 0.8,
                alpha=0.6)
    
    # Plot markers by type
    plotted_types = set()
    for idx, etype, color, marker in events:
        label = etype if etype not in plotted_types else None
        ax.scatter([global_idx[idx]], [cum_events[idx]], c=color, s=35, 
                   marker=marker, zorder=5, edgecolors="white", linewidths=0.5, 
                   label=label)
        plotted_types.add(etype)
    
    # Count summary
    n_stall = sum(1 for e in events if e[1] == "stall")
    n_cond = sum(1 for e in events if e[1] == "cond")
    n_lim = sum(1 for e in events if e[1] == "limited")
    n_rej_err = sum(1 for e in events if e[1] == "rej:err")
    n_rej_div = sum(1 for e in events if e[1] == "rej:div")
    
    ax.set_xlim(global_idx.min(), global_idx.max())
    ax.set_ylim(0, max(len(events) + 1, 1))
    if events:
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8, ncol=2)
    
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Cumulative events", title="", loglog=False)
    
    # Build title with counts
    parts = []
    if n_stall: parts.append(f"stall={n_stall}")
    if n_cond: parts.append(f"cond={n_cond}")
    if n_lim: parts.append(f"lim={n_lim}")
    if n_rej_err: parts.append(f"rej:err={n_rej_err}")
    if n_rej_div: parts.append(f"rej:div={n_rej_div}")
    title = "Events" + (f" ({', '.join(parts)})" if parts else "")
    ax.set_title(title, fontsize=8)


def plot_events(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray) -> None:
    """Plot AA events timeline: restarts (stall/cond) and step limiting.
    
    Shows:
    - Background: residual magnitude as gray fill
    - Markers: restart events (stall=red, cond=blue) and limiting (orange)
    - Right axis: cumulative event count
    """
    n_total = len(subiters)
    
    # Background: residual magnitude (semi-log)
    if "proj_res" in subiters.columns:
        res = subiters["proj_res"].values
        res_log = np.log10(np.clip(res, 1e-12, None))
        res_norm = (res_log - res_log.min()) / max(res_log.max() - res_log.min(), 1e-10)
        ax.fill_between(global_idx, 0, res_norm, alpha=0.15, color="#7f8c8d", label=None)
    
    # Collect events
    events = []  # (idx, event_type, color, marker)
    
    # Restart events
    if "restart" in subiters.columns:
        restart_mask = subiters["restart"].astype(bool).values
        if restart_mask.any() and "restart_reason" in subiters.columns:
            reasons = subiters["restart_reason"].fillna("").astype(str).values
            for prefix, (marker, color) in RESTART_MARKERS.items():
                mask = restart_mask & np.array([str(r).startswith(prefix) for r in reasons])
                for idx in np.where(mask)[0]:
                    events.append((idx, f"RST:{prefix}", color, marker))
    
    # Step limiting events
    if "limited" in subiters.columns:
        limited_mask = subiters["limited"].astype(bool).values
        for idx in np.where(limited_mask)[0]:
            events.append((idx, "LIM", COLOR_LIMITED, "o"))
    
    # Sort events by index
    events.sort(key=lambda x: x[0])
    
    # Plot cumulative event counts and markers
    if events:
        event_indices = [e[0] for e in events]
        event_colors = [e[2] for e in events]
        event_markers = [e[3] for e in events]
        
        # Cumulative count line
        cum_events = np.zeros(n_total, dtype=int)
        for i, idx in enumerate(event_indices):
            cum_events[idx:] = i + 1
        
        ax.plot(global_idx, cum_events, color="#2c3e50", linewidth=PLOT_LINEWIDTH * 0.8,
                alpha=0.6)
        
        # Mark individual events with type-specific markers
        plotted_types = set()
        for idx, etype, color, marker in events:
            label = etype if etype not in plotted_types else None
            ax.scatter([global_idx[idx]], [cum_events[idx]], c=color, s=30, 
                       marker=marker, zorder=5, edgecolors="white", linewidths=0.5, 
                       label=label)
            plotted_types.add(etype)
    
    # Count summary
    n_stall = sum(1 for e in events if e[1] == "RST:stall")
    n_cond = sum(1 for e in events if e[1] == "RST:cond")
    n_lim = sum(1 for e in events if e[1] == "LIM")
    
    ax.set_xlim(global_idx.min(), global_idx.max())
    ax.set_ylim(0, max(len(events) + 1, 1))
    if events:
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8)
    
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Cumulative events", title="", loglog=False)
    ax.set_title(f"Events (stall={n_stall}, cond={n_cond}, lim={n_lim})", fontsize=8)


def plot_restarts(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray) -> None:
    """Plot cumulative restart count over iterations."""
    if "restart" not in subiters.columns:
        ax.text(0.5, 0.5, "No restart data", ha="center", va="center", transform=ax.transAxes)
        return

    # restart column is 0/1 (bool)
    restarts = subiters["restart"].astype(bool).astype(int).values
    cumulative = np.cumsum(restarts)
    total = cumulative[-1]

    ax.plot(global_idx, cumulative, color=COLOR_RESTART, linewidth=PLOT_LINEWIDTH, alpha=0.8)
    ax.fill_between(global_idx, 0, cumulative, alpha=0.3, color=COLOR_RESTART)

    # Mark individual restart events with different markers per reason
    restart_mask = restarts.astype(bool)
    if restart_mask.any() and "restart_reason" in subiters.columns:
        reasons = subiters["restart_reason"].fillna("").astype(str).values
        plotted_reasons = set()
        for reason_prefix, (marker, color) in RESTART_MARKERS.items():
            if reason_prefix == "":
                continue  # handle default separately
            # Match prefix (e.g., "stall" matches "stall>(x1.50)")
            mask = restart_mask & np.array([r.startswith(reason_prefix) for r in reasons])
            if mask.any():
                label = reason_prefix if reason_prefix not in plotted_reasons else None
                ax.scatter(global_idx[mask], cumulative[mask],
                           c=color, s=25, marker=marker, zorder=5, 
                           edgecolors="white", linewidths=0.5, label=label)
                plotted_reasons.add(reason_prefix)
        # Handle unmatched reasons
        matched = np.zeros(len(reasons), dtype=bool)
        for reason_prefix in RESTART_MARKERS.keys():
            if reason_prefix:
                matched |= np.array([r.startswith(reason_prefix) for r in reasons])
        unmatched = restart_mask & ~matched
        if unmatched.any():
            ax.scatter(global_idx[unmatched], cumulative[unmatched],
                       c="#9b59b6", s=20, marker="o", zorder=5, 
                       edgecolors="white", linewidths=0.5, label="other")
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8)
    elif restart_mask.any():
        ax.scatter(global_idx[restart_mask], cumulative[restart_mask],
                   c=COLOR_RESTART, s=20, zorder=5, edgecolors="white", linewidths=0.5)

    ax.set_xlim(global_idx.min(), global_idx.max())
    ax.set_ylim(0, max(total + 1, 1))
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Cumulative", title="", loglog=False)
    ax.set_title(f"Restarts (total={total})", fontsize=8)


def plot_conditioning(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray, config: dict) -> None:
    """Plot Gram matrix condition number evolution."""
    if "condH" not in subiters.columns:
        ax.text(0.5, 0.5, "No condH data", ha="center", va="center", transform=ax.transAxes)
        return

    cond = subiters["condH"].values.copy()
    cond = np.where(np.isfinite(cond), cond, np.nan)

    ax.semilogy(global_idx, cond, color=COLOR_COND, linewidth=PLOT_LINEWIDTH, alpha=0.7)

    # Mark restart_on_cond threshold from config
    solver_cfg = config.get("solver", {})
    cond_threshold = solver_cfg.get("restart_on_cond", 1e5)
    ax.axhline(cond_threshold, color="r", linestyle="--", linewidth=1.0, alpha=0.7, 
               label=f"restart κ={cond_threshold:.0e}")

    ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
    setup_axis_style(ax, xlabel="Global iteration", ylabel="cond(H)", title="", loglog=False)
    ax.set_title("Gram conditioning", fontsize=8)


def plot_history_size(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray, config: dict) -> None:
    """Plot Anderson history size evolution."""
    if "aa_hist" not in subiters.columns:
        ax.text(0.5, 0.5, "No aa_hist data", ha="center", va="center", transform=ax.transAxes)
        return

    aa_hist = subiters["aa_hist"].values

    ax.plot(global_idx, aa_hist, color="#2c3e50", linewidth=PLOT_LINEWIDTH, alpha=0.7)
    ax.fill_between(global_idx, 0, aa_hist, alpha=0.3, color="#2c3e50")

    # Max history line
    solver_cfg = config.get("solver", {})
    m = solver_cfg.get("m", None)
    if m is not None:
        ax.axhline(m, color="k", linestyle="--", linewidth=0.8, alpha=0.5, label=f"m={m}")
        ax.legend(loc="upper right", fontsize=6, framealpha=0.8)

    ax.set_ylim(0, max(aa_hist.max() + 1, (m or 0) + 1))
    setup_axis_style(ax, xlabel="Global iteration", ylabel="History", title="", loglog=False)
    ax.set_title("AA history size", fontsize=8)


def plot_residual_timeline(
    ax: plt.Axes, 
    subiters: pd.DataFrame, 
    global_idx: np.ndarray, 
    config: dict,
    steps: pd.DataFrame | None = None,
) -> None:
    """Plot residual evolution with step boundaries and wRMS error.
    
    Shows:
    - Primary: Picard residual per iteration (semi-log)
    - Background: wRMS time-integration error at step boundaries (shaded)
    - Vertical lines at timestep boundaries
    """
    if "proj_res" not in subiters.columns:
        return

    res = subiters["proj_res"].values

    ax.semilogy(global_idx, res, color="#34495e", linewidth=PLOT_LINEWIDTH * 0.8, 
                alpha=0.8, label="Picard res")

    # Mark step boundaries and optionally show wRMS
    step_arr = subiters["step"].values
    step_changes = np.where(np.diff(step_arr) != 0)[0] + 1
    step_changes = np.concatenate([[0], step_changes])  # Include first
    
    # Plot wRMS error from steps data as background bars
    if steps is not None and "error_norm" in steps.columns and len(step_changes) > 0:
        # Map step number to global_idx range and error_norm
        for i, sc_start in enumerate(step_changes):
            sc_end = step_changes[i + 1] if i + 1 < len(step_changes) else len(global_idx)
            step_num = step_arr[sc_start]
            
            # Find matching step in steps DataFrame
            step_row = steps[steps["step"] == step_num]
            if len(step_row) > 0:
                err = step_row["error_norm"].values[0]
                if err > 0 and np.isfinite(err):
                    # Draw horizontal bar at error level
                    ax.axhspan(err * 0.8, err * 1.2, 
                               xmin=(global_idx[sc_start] - global_idx[0]) / max(global_idx[-1] - global_idx[0], 1),
                               xmax=(global_idx[sc_end - 1] - global_idx[0]) / max(global_idx[-1] - global_idx[0], 1),
                               alpha=0.15, color="#e67e22", zorder=0)
    
    # Vertical lines at step boundaries (skip first)
    for sc in step_changes[1:]:
        ax.axvline(global_idx[sc], color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    # Convergence threshold from config
    solver_cfg = config.get("solver", {})
    coupling_tol = solver_cfg.get("coupling_tol", 1e-6)
    ax.axhline(coupling_tol, color="g", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"tol={coupling_tol:.0e}")
    
    ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Residual", title="", loglog=False)
    ax.set_title("Residual timeline", fontsize=8)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    from mpi4py import MPI

    loader = SimulationLoader(run_dir, MPI.COMM_SELF)
    config = loader.get_config()
    subiters = loader.get_subiterations_metrics()
    steps = loader.get_steps_metrics()

    if subiters.empty:
        raise RuntimeError("No subiterations data found.")

    print_statistics(subiters, config, run_dir)

    # Create global iteration index
    global_idx = create_global_index(subiters)

    # Create figure with 6 subplots (2x3 grid)
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))

    # Row 1: Convergence + Residual timeline + dt evolution
    plot_convergence(axes[0, 0], subiters, config, steps)
    plot_residual_timeline(axes[0, 1], subiters, global_idx, config, steps)
    plot_dt_evolution(axes[0, 2], steps)

    # Row 2: History size + Conditioning + Events (unified: restarts + limiting + rejected)
    plot_history_size(axes[1, 0], subiters, global_idx, config)
    plot_conditioning(axes[1, 1], subiters, global_idx, config)
    plot_events_unified(axes[1, 2], subiters, global_idx, steps)

    fig.tight_layout()
    save_manuscript_figure(fig, OUTPUT_FILE.name, dpi=PUBLICATION_DPI)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
