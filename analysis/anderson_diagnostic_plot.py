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

import json
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from analysis.plot_utils import (
    PLOT_LINEWIDTH,
    PUBLICATION_DPI,
    apply_style,
    save_manuscript_figure,
    setup_axis_style,
)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RUN_DIR = Path(".stiff_results_femur/")
OUTPUT_FILE = Path("manuscript/images/anderson_diagnostic.png")

# Colors for events
COLOR_STALL = "#e74c3c"   # Red - stall restart
COLOR_COND = "#3498db"    # Blue - condition restart / conditioning line
COLOR_LIMITED = "#f39c12" # Orange - step limiting
COLOR_PICARD = "#8e44ad"  # Purple - Picard mode (AA off)
COLOR_RESTART = "#9b59b6" # Purple - general restart (for cumulative plot)
COLOR_OUTER_STALL = "#d35400"  # Dark orange - outer coupling stall (no_progress)
COLOR_MAX_SUBITERS = "#7f8c8d" # Gray - max subiters reached

# Color for rejected steps
COLOR_REJECTED = "#c0392b"  # Dark red

# Rejection reason styling (from steps.csv `reject_reason`)
REJECT_REASON_STYLES: dict[str, tuple[str, str, str]] = {
    "reject:time_error": ("rej:time", COLOR_LIMITED, "x"),
    "reject:coupling_max_subiters": ("rej:maxit", COLOR_REJECTED, "X"),
    "reject:coupling_no_progress": ("rej:stall", COLOR_REJECTED, "P"),
    "reject:coupling_nonconverged": ("rej:coupling", COLOR_REJECTED, "D"),
}
REJECT_FALLBACK = ("rej:unknown", COLOR_REJECTED, "X")


# =============================================================================
# Analysis functions
# =============================================================================


def print_statistics(
    subiters: pd.DataFrame,
    config: dict,
    run_dir: Path,
    steps_df: pd.DataFrame | None = None,
) -> None:
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

    if steps_df is not None and not steps_df.empty:
        steps_df = steps_df.copy()
        if "attempt" not in steps_df.columns:
            steps_df["attempt"] = 1

        if "accepted" in steps_df.columns:
            rejected = steps_df[steps_df["accepted"].astype(int) == 0]
        elif "reject_reason" in steps_df.columns:
            rr = steps_df["reject_reason"].fillna("").astype(str)
            rejected = steps_df[rr.str.startswith("reject:")]
        elif "converged" in steps_df.columns:
            rejected = steps_df[steps_df["converged"].astype(int) == 0]
        else:
            rejected = steps_df.iloc[0:0]

        if not rejected.empty:
            if "reject_reason" in rejected.columns:
                counts = (
                    rejected["reject_reason"]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .to_dict()
                )
            else:
                counts = {"reject:coupling_nonconverged": int(len(rejected))}

            print("Rejected timestep attempts:")
            for reason, n in sorted(counts.items()):
                if reason:
                    print(f"  - {reason}: {n}")


def create_global_index(subiters: pd.DataFrame) -> np.ndarray:
    """Create a global iteration index for plotting across all timesteps."""
    return np.arange(len(subiters))


def plot_convergence(ax: plt.Axes, subiters: pd.DataFrame, config: dict, steps_df: pd.DataFrame | None = None) -> None:
    """Plot convergence curves colored by timestep.
    
    Each (step, attempt) pair is treated as a unique curve.
    Picard iterations (aa_off=1) are shown with dotted lines.
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
    
    # Check for Picard/Anderson distinction - ONLY use picard_mode (hysteresis), NOT aa_off (restart)
    # aa_off indicates restart used Picard, not hysteresis mode switch
    picard_col = "picard_mode" if "picard_mode" in subiters.columns else None
    has_picard_col = picard_col is not None

    for i, (step, attempt) in enumerate(step_attempts):
        if "attempt" in subiters.columns:
            step_data = subiters[(subiters["step"] == step) & (subiters["attempt"] == attempt)].sort_values("iter")
        else:
            step_data = subiters[subiters["step"] == step].sort_values("iter")
        
        if step_data.empty or "proj_res" not in step_data.columns:
            continue
        
        color = cmap(norm(i))
        iters = step_data["iter"].values
        res = step_data["proj_res"].values
        
        if has_picard_col:
            # Plot segments with different line styles for Picard vs Anderson
            picard_vals = step_data[picard_col].astype(bool).values
            
            # Find contiguous segments of same mode
            j = 0
            while j < len(iters):
                is_picard = picard_vals[j]
                seg_start = j
                while j < len(iters) and picard_vals[j] == is_picard:
                    j += 1
                seg_end = j
                
                # Include overlap point for continuity
                end_idx = min(seg_end + 1, len(iters))
                
                ax.semilogy(
                    iters[seg_start:end_idx],
                    res[seg_start:end_idx],
                    color=color,
                    linestyle=":" if is_picard else "-",
                    linewidth=PLOT_LINEWIDTH,
                    alpha=0.7,
                )
        else:
            # No picard info, plot as solid line
            ax.semilogy(
                iters,
                res,
                color=color,
                linestyle="-",
                linewidth=PLOT_LINEWIDTH,
                alpha=0.7,
            )

    solver_cfg = config.get("solver", {})
    coupling_tol = solver_cfg.get("coupling_tol", 1e-6)
    ax.axhline(coupling_tol, color="k", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add horizontal colorbar inside plot for step progression
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = inset_axes(ax, width="30%", height="4%", loc="upper right", borderpad=1.0)
    cbar = ax.figure.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Step", fontsize=6, labelpad=1)
    cbar.ax.tick_params(labelsize=5)

    setup_axis_style(ax, xlabel="Subiteration", ylabel="Residual", title="(a) Convergence", loglog=False)


# Markers for restart reasons
RESTART_MARKERS = {
    "stall": ("v", COLOR_STALL),   # Red triangle down - stall
    "cond": ("d", COLOR_COND),     # Blue diamond - condition
}


def plot_dt_evolution(ax: plt.Axes, steps_df: pd.DataFrame, subiters_df: pd.DataFrame = None) -> None:
    """Plot timestep size evolution over simulation steps with subiteration count on secondary axis."""
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
        rejected_mask = steps_df["converged"].values == 0

    # Plot accepted steps (linear scale for even ticks)
    accepted_mask = ~rejected_mask
    line_dt, = ax.plot(
        step_nums[accepted_mask],
        dt_vals[accepted_mask],
        color="#2ecc71",
        linewidth=PLOT_LINEWIDTH,
        alpha=0.8,
        label=r"$\Delta t$",
    )
    
    # Highlight rejected steps
    if rejected_mask.any():
        ax.scatter(
            step_nums[rejected_mask],
            dt_vals[rejected_mask],
            c=COLOR_REJECTED,
            s=40,
            marker="x",
            zorder=5,
            linewidths=1.5,
            label="rejected",
        )

    ax.set_xlim(step_nums.min() - 0.5, step_nums.max() + 0.5)
    ax.set_ylabel(r"$\Delta t$ [days]", color="#2ecc71")
    ax.tick_params(axis="y", labelcolor="#2ecc71")
    
    # Secondary axis: subiteration count per step
    if subiters_df is not None and not subiters_df.empty:
        ax2 = ax.twinx()
        # Count subiterations per (step, attempt)
        if "attempt" in subiters_df.columns:
            grouped = subiters_df.groupby(["step", "attempt"]).size().reset_index(name="n_iters")
            # For accepted steps, take last attempt
            accepted_steps = steps_df[accepted_mask]["step"].values
            iters_per_step = []
            step_nums_for_iters = []
            for step in accepted_steps:
                step_data = grouped[grouped["step"] == step]
                if len(step_data) > 0:
                    # Take last attempt's iteration count
                    iters_per_step.append(step_data["n_iters"].iloc[-1])
                    step_nums_for_iters.append(step)
        else:
            grouped = subiters_df.groupby("step").size()
            iters_per_step = [grouped.get(s, 0) for s in step_nums[accepted_mask]]
            step_nums_for_iters = step_nums[accepted_mask]
        
        if len(iters_per_step) > 0:
            line_iters, = ax2.plot(
                step_nums_for_iters,
                iters_per_step,
                color="#9b59b6",  # Purple
                linewidth=PLOT_LINEWIDTH * 0.8,
                alpha=0.7,
                linestyle="--",
                label="subiters",
            )
            ax2.set_ylabel("Subiterations", color="#9b59b6")
            ax2.tick_params(axis="y", labelcolor="#9b59b6")
            ax2.set_ylim(0, max(iters_per_step) * 1.1)
            
            # Combined legend
            lines = [line_dt, line_iters]
            labels = [r"$\Delta t$", "subiters"]
            if rejected_mask.any():
                # Add rejected marker to legend
                from matplotlib.lines import Line2D
                rej_handle = Line2D([0], [0], marker="x", color=COLOR_REJECTED, linestyle="None",
                                    markersize=6, label="rejected")
                lines.append(rej_handle)
                labels.append("rejected")
            ax.legend(lines, labels, loc="upper left", fontsize=6, framealpha=0.8, frameon=False)
    else:
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8, frameon=False)
    
    setup_axis_style(ax, xlabel="Step", ylabel="", title="(c) Timestep evolution", loglog=False)


def plot_events_unified(ax: plt.Axes, subiters: pd.DataFrame, global_idx: np.ndarray, 
                        steps_df: pd.DataFrame | None = None) -> None:
    """Unified plot for all events: AA restarts (stall/cond), step limiting, and rejected steps.
    
    Shows cumulative event count with markers for different event types.
    Includes secondary x-axis showing simulation time.
    """
    n_total = len(subiters)
    
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
    
    # 3. Fixed-point stop reason events (outer stall, max_subiters)
    # Try from fp_stop_reason column first (new format)
    if "fp_stop_reason" in subiters.columns:
        stop_reasons = subiters["fp_stop_reason"].fillna("").astype(str).values
        for idx, reason in enumerate(stop_reasons):
            if reason == "no_progress":
                events.append((idx, "outer_stall", COLOR_OUTER_STALL, "H"))  # Hexagon
            elif reason == "max_subiters":
                events.append((idx, "max_iters", COLOR_MAX_SUBITERS, "8"))  # Octagon
    elif steps_df is not None and not steps_df.empty:
        # Fallback: infer from steps.csv for old data format
        # Mark last iteration of each step that didn't converge
        steps_copy = steps_df.copy()
        if "attempt" not in steps_copy.columns:
            steps_copy["attempt"] = 1
        
        for _, row in steps_copy.iterrows():
            step = int(row.get("step", 0))
            attempt = int(row.get("attempt", 1))
            converged = int(row.get("converged", 1))
            num_subiters = int(row.get("num_subiters", 0))
            max_subiters_cfg = 25  # Default, could be read from config
            
            if converged == 0:  # Step didn't converge
                # Find last iteration of this step/attempt in subiters
                if "attempt" in subiters.columns:
                    step_mask = (subiters["step"].values == step) & (subiters["attempt"].values == attempt)
                else:
                    step_mask = subiters["step"].values == step
                if step_mask.any():
                    last_idx = np.where(step_mask)[0][-1]
                    # Determine reason: max_subiters if reached limit
                    if num_subiters >= max_subiters_cfg:
                        events.append((last_idx, "max_iters", COLOR_MAX_SUBITERS, "8"))
    
    # 4. Picard mode switch events (hysteresis transitions from Anderson to Picard)
    # Use picard_mode column (new format) or fall back to aa_off (old format)
    picard_col = "picard_mode" if "picard_mode" in subiters.columns else "aa_off"
    if picard_col in subiters.columns:
        picard_mode = subiters[picard_col].astype(bool).values
        # Detect transitions: picard_mode goes from False (0) to True (1)
        for idx in range(len(picard_mode)):
            if picard_mode[idx]:
                # Check if this is a transition (previous was Anderson or first in step)
                if idx == 0:
                    events.append((idx, "picard", COLOR_PICARD, "s"))
                else:
                    # Check if same step
                    same_step = subiters.iloc[idx]["step"] == subiters.iloc[idx-1]["step"]
                    if "attempt" in subiters.columns:
                        same_step = same_step and (subiters.iloc[idx]["attempt"] == subiters.iloc[idx-1]["attempt"])
                    if same_step and not picard_mode[idx - 1]:
                        # Transition from Anderson to Picard within same step
                        events.append((idx, "picard", COLOR_PICARD, "s"))
    
    # 5. Rejected attempts (all reasons, from steps.csv `reject_reason`)
    if steps_df is not None and not steps_df.empty:
        steps_df = steps_df.copy()
        if "attempt" not in steps_df.columns:
            steps_df["attempt"] = 1

        if "accepted" in steps_df.columns:
            rejected_df = steps_df[steps_df["accepted"].astype(int) == 0]
        elif "reject_reason" in steps_df.columns:
            rr = steps_df["reject_reason"].fillna("").astype(str)
            rejected_df = steps_df[rr.str.startswith("reject:")]
        elif "converged" in steps_df.columns:
            rejected_df = steps_df[steps_df["converged"].astype(int) == 0]
        else:
            rejected_df = steps_df.iloc[0:0]

        for _, row in rejected_df.iterrows():
            step = int(row.get("step", 0))
            attempt = int(row.get("attempt", 1))
            if "attempt" in subiters.columns:
                step_mask = (subiters["step"].values == step) & (subiters["attempt"].values == attempt)
            else:
                step_mask = subiters["step"].values == step
            if not step_mask.any():
                continue
            last_idx = np.where(step_mask)[0][-1]

            code = str(row.get("reject_reason", "")).strip()
            if not code.startswith("reject:"):
                if int(row.get("converged", 1)) == 0:
                    code = "reject:coupling_nonconverged"
                else:
                    code = "reject:time_error"

            label, color, marker = REJECT_REASON_STYLES.get(code, REJECT_FALLBACK)
            events.append((last_idx, label, color, marker))
    
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
    
    # Group events by index to handle overlaps
    from collections import defaultdict
    events_by_idx = defaultdict(list)
    for idx, etype, color, marker in events:
        events_by_idx[idx].append((etype, color, marker))
    
    # Calculate vertical offset for overlapping markers (stack above each other)
    # Each marker needs to be offset by roughly its size in data coordinates
    y_max = max(len(events), 1)
    marker_offset_y = y_max * 0.06  # ~6% of y-range per marker
    
    # Plot markers by type with vertical offset for overlaps
    plotted_types = set()
    for idx, event_list in events_by_idx.items():
        n_events = len(event_list)
        
        for i, (etype, color, marker) in enumerate(event_list):
            label = etype if etype not in plotted_types else None
            x_pos = global_idx[idx]
            # Stack markers vertically: first at base, others above
            y_pos = cum_events[idx] + i * marker_offset_y
            
            scatter_kwargs = {
                "c": color,
                "s": 40,  # Marker size
                "marker": marker,
                "zorder": 5 + i,  # Higher zorder for stacked markers
                "linewidths": 0.8,
                "label": label,
            }
            if marker not in {"x", "+", "|", "_"}:
                scatter_kwargs["edgecolors"] = "white"
            ax.scatter([x_pos], [y_pos], **scatter_kwargs)
            plotted_types.add(etype)
    
    # Count summary
    n_stall = sum(1 for e in events if e[1] == "stall")
    n_cond = sum(1 for e in events if e[1] == "cond")
    n_lim = sum(1 for e in events if e[1] == "limited")
    n_picard = sum(1 for e in events if e[1] == "picard")
    n_outer_stall = sum(1 for e in events if e[1] == "outer_stall")
    n_max_iters = sum(1 for e in events if e[1] == "max_iters")
    rej_counts = {}
    for _, etype, _, _ in events:
        if etype.startswith("rej:"):
            rej_counts[etype] = rej_counts.get(etype, 0) + 1
    
    # Calculate max y considering stacked markers
    max_stack = max((len(evs) for evs in events_by_idx.values()), default=1)
    y_top = len(events) + (max_stack - 1) * marker_offset_y + 1
    
    # Add padding to x-axis for markers at edges
    x_range = global_idx.max() - global_idx.min() if len(global_idx) > 1 else 1
    x_padding = x_range * 0.02  # 2% padding on each side
    ax.set_xlim(global_idx.min() - x_padding, global_idx.max() + x_padding)
    ax.set_ylim(0, y_top)
    if events:
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8, ncol=2, frameon=False)
    
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Cumulative events", title="", loglog=False)
    
    # Add secondary x-axis with simulation time
    if "time_days" in subiters.columns:
        ax2 = ax.twiny()
        time_vals = subiters["time_days"].values
        
        # Map global_idx to time for the secondary axis
        ax2.set_xlim(
            time_vals[0] if len(time_vals) > 0 else 0,
            time_vals[-1] if len(time_vals) > 0 else 1
        )
        ax2.set_xlabel("Time [days]", fontsize=8)
        ax2.tick_params(axis='x', labelsize=7)
        ax2.grid(False)  # Disable grid for secondary axis
        
        # Create tick positions at step boundaries for cleaner display
        if len(time_vals) > 0:
            # Get unique times (one per step change)
            step_times = subiters.groupby("step")["time_days"].first().values
            if len(step_times) > 10:
                # Too many steps, use ~5-7 evenly spaced ticks
                n_ticks = min(7, len(step_times))
                tick_indices = np.linspace(0, len(step_times) - 1, n_ticks, dtype=int)
                tick_times = step_times[tick_indices]
            else:
                tick_times = step_times
            ax2.set_xticks(tick_times)
            ax2.set_xticklabels([f"{t:.0f}" for t in tick_times])
    
    # Build title with counts
    parts = []
    if n_stall: parts.append(f"stall={n_stall}")
    if n_cond: parts.append(f"cond={n_cond}")
    if n_lim: parts.append(f"lim={n_lim}")
    if n_picard: parts.append(f"picard={n_picard}")
    if n_outer_stall: parts.append(f"outer_stall={n_outer_stall}")
    if n_max_iters: parts.append(f"max_iters={n_max_iters}")
    for k in sorted(rej_counts.keys()):
        parts.append(f"{k}={rej_counts[k]}")
    ax.set_title("(f) Cumulative Events", fontweight="bold")


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
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8, frameon=False)
    
    setup_axis_style(
        ax,
        xlabel="Global iteration",
        ylabel="Cumulative events",
        title=f"Events (stall={n_stall}, cond={n_cond}, lim={n_lim})",
        loglog=False,
    )


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
                       c=COLOR_RESTART, s=20, marker="o", zorder=5, 
                       edgecolors="white", linewidths=0.5, label="other")
        ax.legend(loc="upper left", fontsize=6, framealpha=0.8, frameon=False)
    elif restart_mask.any():
        ax.scatter(global_idx[restart_mask], cumulative[restart_mask],
                   c=COLOR_RESTART, s=20, zorder=5, edgecolors="white", linewidths=0.5)

    ax.set_xlim(global_idx.min(), global_idx.max())
    ax.set_ylim(0, max(total + 1, 1))
    setup_axis_style(
        ax,
        xlabel="Global iteration",
        ylabel="Cumulative",
        title=f"Restarts (total={total})",
        loglog=False,
    )


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

    setup_axis_style(ax, xlabel="Global iteration", ylabel="cond(H)", title="(e) Gram conditioning", loglog=False)
    ax.legend(loc="upper left", fontsize=6, framealpha=0.8, frameon=False)


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
        ax.legend(loc="upper right", fontsize=6, framealpha=0.8, frameon=False)

    ax.set_ylim(0, max(aa_hist.max() + 1, (m or 0) + 1))
    setup_axis_style(ax, xlabel="Global iteration", ylabel="History", title="(d) AA history size", loglog=False)


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
    if "attempt" in subiters.columns:
        attempt_arr = subiters["attempt"].values
        changes = (np.diff(step_arr) != 0) | (np.diff(attempt_arr) != 0)
    else:
        attempt_arr = None
        changes = np.diff(step_arr) != 0
    step_changes = np.where(changes)[0] + 1
    step_changes = np.concatenate([[0], step_changes])
    
    # Plot wRMS error from steps data as background bars
    if steps is not None and "error_norm" in steps.columns and len(step_changes) > 0:
        # Map step number to global_idx range and error_norm
        for i, sc_start in enumerate(step_changes):
            sc_end = step_changes[i + 1] if i + 1 < len(step_changes) else len(global_idx)
            step_num = step_arr[sc_start]
            
            # Find matching step in steps DataFrame
            if attempt_arr is not None and "attempt" in steps.columns:
                attempt_num = int(attempt_arr[sc_start])
                step_row = steps[(steps["step"] == step_num) & (steps["attempt"] == attempt_num)]
            else:
                step_row = steps[steps["step"] == step_num]
            if len(step_row) > 0:
                err = step_row["error_norm"].values[0]
                if err > 0 and np.isfinite(err):
                    # Draw horizontal bar at error level (label only first one for legend)
                    lbl = "wRMS err" if i == 0 else None
                    ax.axhspan(err * 0.8, err * 1.2, 
                               xmin=(global_idx[sc_start] - global_idx[0]) / max(global_idx[-1] - global_idx[0], 1),
                               xmax=(global_idx[sc_end - 1] - global_idx[0]) / max(global_idx[-1] - global_idx[0], 1),
                               alpha=0.15, color="#e67e22", zorder=0, label=lbl)
    
    # Vertical lines at step boundaries (skip first)
    for sc in step_changes[1:]:
        ax.axvline(global_idx[sc], color="gray", linestyle=":", linewidth=0.5, alpha=0.4)

    # Convergence threshold from config
    solver_cfg = config.get("solver", {})
    coupling_tol = solver_cfg.get("coupling_tol", 1e-6)
    ax.axhline(coupling_tol, color="g", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"tol={coupling_tol:.0e}")
    
    # Secondary axis: contraction ratio rho = r_k / r_{k-1} (per-step average)
    if "contraction" in subiters.columns:
        ax2 = ax.twinx()
        
        # Compute per-step average contraction
        step_arr = subiters["step"].values
        rho_arr = subiters["contraction"].values
        
        # Group by step and compute mean contraction
        step_rho_avg = []
        step_centers = []  # global_idx center of each step
        
        for sc_idx, sc_start in enumerate(step_changes):
            sc_end = step_changes[sc_idx + 1] if sc_idx + 1 < len(step_changes) else len(global_idx)
            step_rho = rho_arr[sc_start:sc_end]
            valid_rho = step_rho[np.isfinite(step_rho)]
            if len(valid_rho) > 0:
                step_rho_avg.append(np.mean(valid_rho))
                step_centers.append((global_idx[sc_start] + global_idx[sc_end - 1]) / 2)
        
        if len(step_rho_avg) > 0:
            ax2.plot(
                step_centers,
                step_rho_avg,
                color="#e74c3c",  # Red
                linewidth=PLOT_LINEWIDTH * 0.8,
                alpha=0.7,
                linestyle="-",
                marker="o",
                markersize=3,
                label=r"$\bar{\rho}$",
            )
            # Reference line at rho=1 (contraction threshold)
            ax2.axhline(1.0, color="#e74c3c", linestyle=":", linewidth=0.8, alpha=0.4)
            ax2.set_ylabel(r"Avg contraction $\bar{\rho}$", color="#e74c3c")
            ax2.tick_params(axis="y", labelcolor="#e74c3c")
            # Limit y-axis to reasonable range
            rho_max = min(np.percentile(step_rho_avg, 95) * 1.2, 2.0)
            ax2.set_ylim(0, max(rho_max, 1.1))
    
    setup_axis_style(ax, xlabel="Global iteration", ylabel="Residual", title="(b) Residual timeline", loglog=False)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.8, frameon=False)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    subiters_path = run_dir / "subiterations.csv"
    if not subiters_path.exists():
        raise FileNotFoundError(f"Subiterations metrics not found: {subiters_path}")
    subiters = pd.read_csv(subiters_path)

    steps_path = run_dir / "steps.csv"
    if not steps_path.exists():
        raise FileNotFoundError(f"Steps metrics not found: {steps_path}")
    steps = pd.read_csv(steps_path)

    if subiters.empty:
        raise RuntimeError("No subiterations data found.")

    print_statistics(subiters, config, run_dir, steps_df=steps)

    # Create global iteration index
    global_idx = create_global_index(subiters)

    # Create figure with 6 subplots (2x3 grid)
    # Use 3:2 aspect ratio for equal-sized square subplots
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.5))

    # Row 1: Convergence + Residual timeline + dt evolution
    plot_convergence(axes[0, 0], subiters, config, steps)
    plot_residual_timeline(axes[0, 1], subiters, global_idx, config, steps)
    plot_dt_evolution(axes[0, 2], steps, subiters)

    # Row 2: History size + Conditioning + Events (unified: restarts + limiting + rejected)
    plot_history_size(axes[1, 0], subiters, global_idx, config)
    plot_conditioning(axes[1, 1], subiters, global_idx, config)
    plot_events_unified(axes[1, 2], subiters, global_idx, steps)

    fig.tight_layout()
    save_manuscript_figure(fig, OUTPUT_FILE.name, dpi=PUBLICATION_DPI)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
