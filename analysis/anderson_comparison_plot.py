"""Anderson acceleration comparison plot: Physio vs Stiff configuration.

Demonstrates that Anderson acceleration is essential for the stiff (reaction-dominated)
configuration where Picard iteration fails to converge.

Three-panel figure:
    (a) Final residual per step: Shows stiff+Picard fails to converge (high residual)
    (b) Convergence rate: % of steps that actually converged vs hit max_subiters
    (c) Residual evolution: Typical convergence curves for each configuration

Input directories:
    .physio_results_box/         - Physio + Anderson
    .physio_results_box_picard/  - Physio + Picard
    .stiff_results_box/          - Stiff + Anderson
    .stiff_results_box_picard/   - Stiff + Picard

Output:
    manuscript/images/anderson_comparison.png

Usage:
    python3 analysis/anderson_comparison_plot.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.plot_utils import (
    PUBLICATION_DPI,
    apply_style,
    save_manuscript_figure,
    setup_axis_style,
    PLOT_LINEWIDTH,
    COLORS as TOL_COLORS,
)

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_FILE = Path("manuscript/images/anderson_comparison.png")

RESULT_DIRS = {
    "physio_aa": Path(".physio_results_box"),
    "physio_picard": Path(".physio_results_box_picard"),
    "stiff_aa": Path(".stiff_results_box"),
    "stiff_picard": Path(".stiff_results_box_picard"),
}

# Modern color-blind friendly palette (from plot_utils TOL palette)
COLORS = {
    "physio_aa": TOL_COLORS["blue"],      # Blue
    "physio_picard": TOL_COLORS["cyan"],  # Cyan
    "stiff_aa": TOL_COLORS["orange"],     # Orange
    "stiff_picard": TOL_COLORS["red"],    # Red (failure case)
}

LABELS = {
    "physio_aa": "Physio + AA",
    "physio_picard": "Physio + Picard",
    "stiff_aa": "Stiff + AA",
    "stiff_picard": "Stiff + Picard",
}


# =============================================================================
# Data loading
# =============================================================================

@dataclass
class RunMetrics:
    """Aggregated metrics from a simulation run."""
    name: str
    label: str
    color: str
    config: dict | None
    subiters_df: pd.DataFrame | None = None
    
    # Residual-based metrics (the real measure of convergence!)
    n_steps: int = 0
    n_converged: int = 0           # Steps that hit coupling_tol
    n_max_subiters: int = 0        # Steps that hit max_subiters limit
    n_no_progress: int = 0         # Steps that stalled
    convergence_rate: float = 0.0  # % truly converged
    
    # Time metrics (breakdown by stop reason - dynamic)
    total_time_sec: float = 0.0           # Total wall-clock time
    time_by_reason: dict = None           # {reason: time_sec}
    
    # Residual statistics
    median_final_residual: float = 0.0
    max_final_residual: float = 0.0
    final_residuals: np.ndarray = None  # All final residuals per step
    
    @classmethod
    def from_directory(cls, name: str, run_dir: Path) -> "RunMetrics":
        """Load metrics from a result directory."""
        label = LABELS.get(name, name)
        color = COLORS.get(name, "#888888")
        
        if not run_dir.exists():
            print(f"  [WARN] Directory not found: {run_dir}")
            return cls(name=name, label=label, color=color, config=None)
        
        config_path = run_dir / "config.json"
        config = None
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        subiters_df = pd.read_csv(run_dir / "subiterations.csv") if (run_dir / "subiterations.csv").exists() else None
        
        if subiters_df is None or subiters_df.empty:
            print(f"  [WARN] subiterations.csv not found or empty in {run_dir}")
            return cls(name=name, label=label, color=color, config=config)
        
        instance = cls(name=name, label=label, color=color, config=config, subiters_df=subiters_df)
        instance._compute_aggregates()
        return instance
    
    def _compute_aggregates(self) -> None:
        """Compute aggregate statistics from subiterations data."""
        df = self.subiters_df
        if df is None or df.empty:
            return
        
        # Group by (step, attempt) and get the LAST iteration of each
        if "attempt" in df.columns:
            grouped = df.groupby(["step", "attempt"])
        else:
            grouped = df.groupby("step")
        
        # Get final residual and stop reason for each step/attempt
        final_rows = grouped.last().reset_index()
        
        self.n_steps = len(final_rows)
        
        # Count by stop reason (fp_stop_reason column)
        if "fp_stop_reason" in final_rows.columns:
            stop_reasons = final_rows["fp_stop_reason"].fillna("unknown")
            self.n_converged = (stop_reasons == "converged").sum()
            self.n_max_subiters = (stop_reasons == "max_subiters").sum()
            self.n_no_progress = (stop_reasons == "no_progress").sum()
        else:
            # Fallback: check if final residual is below tolerance
            tol = self.config.get("solver", {}).get("coupling_tol", 1e-6) if self.config else 1e-6
            if "proj_res" in final_rows.columns:
                self.n_converged = (final_rows["proj_res"] <= tol).sum()
                self.n_max_subiters = self.n_steps - self.n_converged
        
        self.convergence_rate = 100.0 * self.n_converged / self.n_steps if self.n_steps > 0 else 0.0
        
        # Total computation time (sum of all solver times), broken down by stop reason
        time_cols = ["mech_time", "fab_time", "stim_time", "dens_time"]
        available_time_cols = [c for c in time_cols if c in df.columns]
        self.time_by_reason = {}
        if available_time_cols:
            # Add total time per iteration
            df = df.copy()
            df["iter_time"] = df[available_time_cols].sum(axis=1)
            self.total_time_sec = df["iter_time"].sum()
            
            # Group by step to get stop reason, then sum times by reason
            if "fp_stop_reason" in df.columns:
                step_times = df.groupby("step").agg({"iter_time": "sum", "fp_stop_reason": "last"}).reset_index()
                for reason in step_times["fp_stop_reason"].unique():
                    reason_key = reason if reason else "unknown"
                    self.time_by_reason[reason_key] = step_times.loc[
                        step_times["fp_stop_reason"] == reason, "iter_time"
                    ].sum()
            else:
                self.time_by_reason["unknown"] = self.total_time_sec
        
        # Final residual statistics
        if "proj_res" in final_rows.columns:
            residuals = final_rows["proj_res"].values
            self.final_residuals = residuals
            self.median_final_residual = np.median(residuals)
            self.max_final_residual = np.max(residuals)
    
    def is_valid(self) -> bool:
        return self.n_steps > 0


# =============================================================================
# Plotting
# =============================================================================

def plot_final_residuals(ax: plt.Axes, runs: list[RunMetrics]) -> None:
    """Plot (a): Final residual distribution per configuration.
    
    Uses log scale with a floor to handle near-machine-epsilon values.
    """
    valid_runs = [r for r in runs if r.is_valid() and r.final_residuals is not None]
    if not valid_runs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    positions = np.arange(len(valid_runs))
    
    # Floor for visibility (1e-14 to avoid machine epsilon issues)
    FLOOR = 1e-14
    
    # Strip plot (individual points)
    for i, r in enumerate(valid_runs):
        # Clamp residuals to floor
        residuals = np.maximum(r.final_residuals, FLOOR)
        
        # Jitter x positions
        jitter = np.random.uniform(-0.15, 0.15, len(residuals))
        ax.scatter(positions[i] + jitter, residuals, 
                   c=r.color, alpha=0.6, s=15, edgecolors='none', zorder=3)
        
        # Median line (also clamped)
        median = max(r.median_final_residual, FLOOR)
        ax.hlines(median, positions[i] - 0.25, positions[i] + 0.25, 
                  colors=r.color, linewidths=2, zorder=4)
        
        # Label: show real value even if below floor
        label_val = r.median_final_residual
        label_str = f"{label_val:.1e}" if label_val >= FLOOR else f"<{FLOOR:.0e}"
        ax.annotate(label_str, xy=(positions[i] + 0.3, median),
                    fontsize=6, va="center", color=r.color)
    
    # Tolerance line
    if valid_runs[0].config:
        tol = valid_runs[0].config.get("solver", {}).get("coupling_tol", 1e-6)
        ax.axhline(tol, color=TOL_COLORS["teal"], linestyle="--", linewidth=1.5, 
                   alpha=0.9, zorder=2, label=f"tol={tol:.0e}")
        ax.legend(loc="upper left", fontsize=6, frameon=False)
    
    ax.set_yscale("log")
    ax.set_ylim(bottom=FLOOR * 0.5)  # Set bottom limit
    ax.set_xticks(positions)
    ax.set_xticklabels([r.label for r in valid_runs], rotation=20, ha="right", fontsize=7)
    setup_axis_style(ax, xlabel="", ylabel="Final residual", title="(a) Final residual", loglog=False)


def plot_convergence_breakdown(ax: plt.Axes, runs: list[RunMetrics]) -> None:
    """Plot (b): Computation time breakdown by convergence outcome.
    
    Stacked bar: time spent in each stop_reason category (dynamic).
    """
    valid_runs = [r for r in runs if r.is_valid()]
    if not valid_runs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    x = np.arange(len(valid_runs))
    width = 0.6
    
    # Collect all unique stop reasons across all runs
    all_reasons = set()
    for r in valid_runs:
        if r.time_by_reason:
            all_reasons.update(r.time_by_reason.keys())
    
    # Define colors for known reasons, fallback for unknown
    reason_colors = {
        "converged": TOL_COLORS["teal"],
        "max_subiters": TOL_COLORS["red"],
        "no_progress": TOL_COLORS["orange"],
        "unknown": TOL_COLORS["magenta"],
    }
    reason_labels = {
        "converged": "Converged",
        "max_subiters": "Max iters",
        "no_progress": "Stalled",
        "unknown": "Unknown",
    }
    
    # Order: converged first, then failures
    ordered_reasons = []
    for r in ["converged", "max_subiters", "no_progress", "unknown"]:
        if r in all_reasons:
            ordered_reasons.append(r)
            all_reasons.discard(r)
    # Any remaining unknown reasons
    ordered_reasons.extend(sorted(all_reasons))
    
    # Build stacked bars
    bottoms = np.zeros(len(valid_runs))
    for reason in ordered_reasons:
        times = np.array([r.time_by_reason.get(reason, 0.0) if r.time_by_reason else 0.0 for r in valid_runs])
        color = reason_colors.get(reason, TOL_COLORS["grey"])
        label = reason_labels.get(reason, reason)
        ax.bar(x, times, width, bottom=bottoms, color=color, alpha=0.85, 
               edgecolor="white", linewidth=0.8, label=label)
        bottoms += times
    
    # Add total time labels above bars
    totals = np.array([r.total_time_sec for r in valid_runs])
    max_height = np.max(totals) if len(totals) > 0 else 1.0
    for i, r in enumerate(valid_runs):
        y_pos = totals[i] + 0.03 * max_height
        if totals[i] >= 60:
            time_str = f"{totals[i]/60:.1f} min"
        else:
            time_str = f"{totals[i]:.1f} s"
        ax.annotate(time_str, xy=(x[i], y_pos), ha="center", 
                    va="bottom", fontsize=7, fontweight="bold", color="#333333")
    
    ax.set_xticks(x)
    ax.set_xticklabels([r.label for r in valid_runs], rotation=20, ha="right", fontsize=7)
    ax.set_ylim(0, max_height * 1.25)
    ax.legend(loc="upper left", fontsize=6, frameon=False, ncol=len(ordered_reasons))
    setup_axis_style(ax, xlabel="", ylabel="Time (s)", title="(b) Computation time", loglog=False)


def plot_convergence_curves(ax: plt.Axes, runs: list[RunMetrics]) -> None:
    """Plot (c): Typical convergence curves (residual vs iteration).
    
    Shows one representative step from EACH configuration with distinct style.
    """
    valid_runs = [r for r in runs if r.is_valid() and r.subiters_df is not None]
    if not valid_runs:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    
    # Line styles to distinguish runs further
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    
    for idx, run in enumerate(valid_runs):
        df = run.subiters_df
        # Pick the step with MOST iterations (hardest convergence)
        iters_per_step = df.groupby("step").size()
        if len(iters_per_step) == 0:
            continue
        rep_step = iters_per_step.idxmax()
        
        step_data = df[df["step"] == rep_step].sort_values("iter")
        if "proj_res" not in step_data.columns or step_data.empty:
            continue
        
        iters = step_data["iter"].values
        res = step_data["proj_res"].values
        
        # Each run gets distinct color, linestyle, and sparse markers
        ax.semilogy(iters, res, color=run.color, 
                    linestyle=linestyles[idx % len(linestyles)],
                    linewidth=PLOT_LINEWIDTH, 
                    marker=markers[idx % len(markers)],
                    markevery=max(1, len(iters) // 6),
                    markersize=4,
                    label=run.label)
    
    # Tolerance line
    if valid_runs[0].config:
        tol = valid_runs[0].config.get("solver", {}).get("coupling_tol", 1e-6)
        ax.axhline(tol, color=TOL_COLORS["teal"], linestyle="--", linewidth=1.2, 
                   alpha=0.9, label=f"tol={tol:.0e}")
    
    ax.legend(loc="upper right", fontsize=6, frameon=False)
    ax.set_xlim(left=0)
    setup_axis_style(ax, xlabel="Iteration", ylabel="Residual", 
                     title="(c) Convergence curves", loglog=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("Anderson Acceleration Comparison: Physio vs Stiff")
    print("=" * 60)
    
    runs = [RunMetrics.from_directory(name, path) for name, path in RESULT_DIRS.items()]
    
    print("\nSummary Statistics (residual-based):")
    for r in runs:
        if r.is_valid():
            print(f"  {r.label}:")
            print(f"    Converged: {r.n_converged}/{r.n_steps} ({r.convergence_rate:.0f}%)")
            print(f"    Max iters: {r.n_max_subiters}, Stalled: {r.n_no_progress}")
            print(f"    Final residual: median={r.median_final_residual:.2e}, max={r.max_final_residual:.2e}")
        else:
            print(f"  {r.label}: [No data]")
    
    if not any(r.is_valid() for r in runs):
        print("\n[ERROR] No valid data. Run: python run_anderson_experiment.py")
        sys.exit(1)
    
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.5))
    
    plot_final_residuals(axes[0], runs)
    plot_convergence_breakdown(axes[1], runs)
    plot_convergence_curves(axes[2], runs)
    
    fig.tight_layout()
    output_path = save_manuscript_figure(fig, OUTPUT_FILE.name, dpi=PUBLICATION_DPI)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
