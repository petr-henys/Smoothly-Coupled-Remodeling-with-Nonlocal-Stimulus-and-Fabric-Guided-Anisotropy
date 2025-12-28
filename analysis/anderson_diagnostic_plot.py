"""Anderson acceleration diagnostic plot.

Multi-panel figure showing Anderson accelerator behavior:
- Convergence curves (residual vs subiteration)
- Acceptance/rejection events
- Restart events and reasons
- Gram matrix conditioning
- History size evolution

Input:
    results/<run_dir>/ containing subiterations.csv and config.json

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
COLOR_ACCEPT = "#2ecc71"  # Green
COLOR_REJECT = "#e74c3c"  # Red
COLOR_RESTART = "#9b59b6"  # Purple
COLOR_COND = "#3498db"  # Blue


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

    if "accepted" in subiters.columns:
        acc_rate = subiters["accepted"].mean() * 100
        rej_count = (~subiters["accepted"].astype(bool)).sum()
        print(f"AA acceptance rate: {acc_rate:.1f}% ({rej_count} rejections)")
        
        # Show rejection reasons if available
        if "reject_reason" in subiters.columns and rej_count > 0:
            rej_reasons = subiters.loc[~subiters["accepted"].astype(bool), "reject_reason"]
            rej_reasons = rej_reasons[rej_reasons.astype(str).str.len() > 0]
            if len(rej_reasons) > 0:
                for reason, count in rej_reasons.value_counts().items():
                    print(f"  - {reason}: {count}")

    if "restart" in subiters.columns:
        # restart column contains 0/1 (bool)
        restart_count = subiters["restart"].astype(bool).sum()
        print(f"AA restarts: {restart_count}")
        
        # Show restart reasons if available
        if "restart_reason" in subiters.columns and restart_count > 0:
            rst_reasons = subiters.loc[subiters["restart"].astype(bool), "restart_reason"]
            rst_reasons = rst_reasons[rst_reasons.astype(str).str.len() > 0]
            if len(rst_reasons) > 0:
                for reason, count in rst_reasons.value_counts().items():
                    print(f"  - {reason}: {count}")

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


def plot_convergence(ax: plt.Axes, subiters: pd.DataFrame, config: dict) -> None:
    """Plot convergence curves colored by timestep."""
    steps = sorted(subiters["step"].unique())
    n_steps = len(steps)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=max(n_steps - 1, 1))

    for i, step in enumerate(steps):
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

    setup_axis_style(ax, xlabel="Subiteration", ylabel="Residual", title="", loglog=False)
    ax.set_title("Convergence", fontsize=8)


# Markers and colors for reject reasons
REJECT_MARKERS = {
    "res_increase": ("^", "#e74c3c"),   # red triangle up
    "bt_fail": ("v", "#c0392b"),         # dark red triangle down
    "": ("x", "#e74c3c"),                 # default red x
}

# Window size for running acceptance rate
ACCEPTANCE_WINDOW = 20


def plot_acceptance(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray) -> None:
    """Plot cumulative accept/reject counts with running acceptance rate.
    
    Shows:
    - Left axis: Cumulative accept (green) and reject (red) counts as lines
    - Right axis: Running acceptance rate (%) as gray dashed line
    - Markers on reject curve showing reason types
    """
    if "accepted" not in subiters.columns:
        ax.text(0.5, 0.5, "No acceptance data", ha="center", va="center", transform=ax.transAxes)
        return

    accepted = subiters["accepted"].astype(bool).values
    rejected = ~accepted
    
    n_acc = accepted.sum()
    n_rej = rejected.sum()
    
    # Cumulative counts
    cum_accept = np.cumsum(accepted.astype(int))
    cum_reject = np.cumsum(rejected.astype(int))
    
    # Plot cumulative accept (green area)
    ax.fill_between(global_idx, 0, cum_accept, alpha=0.3, color=COLOR_ACCEPT)
    ax.plot(global_idx, cum_accept, color=COLOR_ACCEPT, linewidth=PLOT_LINEWIDTH,
            label=f"Accept: {n_acc}")
    
    # Plot cumulative reject (red area, stacked on top if any)
    if n_rej > 0:
        ax.fill_between(global_idx, 0, cum_reject, alpha=0.3, color=COLOR_REJECT)
        ax.plot(global_idx, cum_reject, color=COLOR_REJECT, linewidth=PLOT_LINEWIDTH,
                label=f"Reject: {n_rej}")
        
        # Mark reject events with reason-specific markers
        if "reject_reason" in subiters.columns:
            reasons = subiters["reject_reason"].fillna("").astype(str).values
            plotted_reasons = set()
            for reason, (marker, color) in REJECT_MARKERS.items():
                mask = rejected & (reasons == reason)
                if mask.any():
                    label = reason if reason and reason not in plotted_reasons else None
                    ax.scatter(global_idx[mask], cum_reject[mask],
                               c=color, s=20, marker=marker, zorder=5,
                               edgecolors="white", linewidths=0.5, label=label)
                    plotted_reasons.add(reason)
        else:
            # Just mark reject points
            ax.scatter(global_idx[rejected], cum_reject[rejected],
                       c=COLOR_REJECT, s=15, marker="x", zorder=5)
    
    # Secondary axis: Running acceptance rate
    ax2 = ax.twinx()
    window = min(ACCEPTANCE_WINDOW, len(accepted))
    if window > 0:
        # Use pandas rolling for efficiency
        acc_series = pd.Series(accepted.astype(float))
        running_rate = acc_series.rolling(window=window, min_periods=1).mean() * 100
        ax2.plot(global_idx, running_rate, color="#7f8c8d", linewidth=0.8,
                 linestyle="--", alpha=0.7, label=f"Rate (w={window})")
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Acc. rate %", fontsize=7, color="#7f8c8d")
        ax2.tick_params(axis='y', labelsize=6, colors="#7f8c8d")
        ax2.axhline(100, color="#7f8c8d", linestyle=":", linewidth=0.5, alpha=0.5)
    
    # Final acceptance rate annotation
    final_rate = 100 * n_acc / max(n_acc + n_rej, 1)
    ax.annotate(f"{final_rate:.0f}%", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="#7f8c8d",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#7f8c8d", alpha=0.8))
    
    ax.set_xlim(global_idx.min(), global_idx.max())
    ax.set_ylim(0, max(cum_accept.max(), cum_reject.max() if n_rej > 0 else 1) * 1.1)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.8)
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Cumulative count", title="", loglog=False)
    ax.set_title("Accept/Reject", fontsize=8)


# Markers and colors for restart reasons
RESTART_MARKERS = {
    "reject_streak": ("o", "#9b59b6"),    # purple circle
    "stall": ("s", "#8e44ad"),             # dark purple square  
    "cond": ("d", "#6c3483"),              # diamond
    "": ("o", "#9b59b6"),                   # default
}


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


def plot_residual_timeline(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray, config: dict) -> None:
    """Plot residual evolution as a timeline."""
    if "proj_res" not in subiters.columns:
        return

    res = subiters["proj_res"].values

    ax.semilogy(global_idx, res, color="#34495e", linewidth=PLOT_LINEWIDTH * 0.8, alpha=0.7)

    # Mark step boundaries
    step_changes = np.where(np.diff(subiters["step"].values) != 0)[0] + 1
    for sc in step_changes:
        ax.axvline(global_idx[sc], color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

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

    if subiters.empty:
        raise RuntimeError("No subiterations data found.")

    print_statistics(subiters, config, run_dir)

    # Create global iteration index
    global_idx = create_global_index(subiters)

    # Create figure with 6 subplots (2x3 grid)
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))

    # Row 1: Convergence + Residual timeline + Accept/Reject
    plot_convergence(axes[0, 0], subiters, config)
    plot_residual_timeline(axes[0, 1], subiters, global_idx, config)
    plot_acceptance(axes[0, 2], subiters, global_idx)

    # Row 2: History size + Conditioning + Restarts
    plot_history_size(axes[1, 0], subiters, global_idx, config)
    plot_conditioning(axes[1, 1], subiters, global_idx, config)
    plot_restarts(axes[1, 2], subiters, global_idx)

    fig.tight_layout()
    save_manuscript_figure(fig, OUTPUT_FILE.name, dpi=PUBLICATION_DPI)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
