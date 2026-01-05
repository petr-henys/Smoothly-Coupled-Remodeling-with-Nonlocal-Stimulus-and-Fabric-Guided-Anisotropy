"""Anderson acceleration parameter sweep analysis: convergence metrics and visualization.

Loads solver metrics from Anderson sweep results (steps.csv, subiterations.csv),
computes convergence statistics, and visualizes parameter sensitivity.

Usage:
    python analysis/anderson_analysis.py
    
Inputs:
    results/anderson_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── steps.csv
    │   └── subiterations.csv
    ...

Outputs:
    results/anderson_sweep/
    ├── anderson_metrics.csv
    ├── anderson_sweep.png
    └── (manuscript/images/anderson_sweep.png)
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Imports from plot_utils (CMAME style)
# =============================================================================

from analysis.plot_utils import (
    apply_style,
    setup_axis_style,
    save_figure,
    save_manuscript_figure,
    PUBLICATION_DPI,
    PLOT_LINEWIDTH,
    PLOT_MARKERSIZE,
)


# =============================================================================
# Anderson Parameter Styling
# =============================================================================

ANDERSON_PARAM_COLORS = {
    "m": "#4C72B0",       # Steel blue (history)
    "beta": "#DD8452",    # Coral (mixing)
    "lam": "#55A868",     # Sage green (regularization)
}
ANDERSON_PARAM_LABELS = {
    "m": r"$m$ (history size)",
    "beta": r"$\beta$ (mixing)",
    "lam": r"$\lambda$ (regularization)",
}
ANDERSON_PARAM_MARKERS = {
    "m": "o",
    "beta": "s",
    "lam": "^",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AndersonMetrics:
    """Anderson acceleration performance metrics.
    
    Metrics characterizing convergence behavior:
    - total_subiters: Total fixed-point iterations across all timesteps
    - avg_subiters_per_step: Average iterations per timestep
    - convergence_rate: Estimated contraction ratio (median)
    - restarts: Number of Anderson restarts
    - failures: Number of timesteps that hit max_subiters
    """
    # Run identification
    run_hash: str
    output_dir: str
    
    # Anderson parameters
    m: int
    beta: float
    lam: float
    
    # Convergence metrics
    n_steps: int
    total_subiters: int
    avg_subiters_per_step: float
    max_subiters_per_step: int
    min_subiters_per_step: int
    
    # Contraction rate (geometric mean of r_k/r_{k-1})
    contraction_median: float
    contraction_mean: float
    
    # Stability metrics
    n_restarts: int
    n_failures: int  # Steps that hit max_subiters without converging
    
    # Condition number statistics
    cond_median: float
    cond_max: float
    
    # Total solve time (if available)
    total_solve_time: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "run_hash": self.run_hash,
            "output_dir": self.output_dir,
            "m": self.m,
            "beta": self.beta,
            "lam": self.lam,
            "n_steps": self.n_steps,
            "total_subiters": self.total_subiters,
            "avg_subiters_per_step": self.avg_subiters_per_step,
            "max_subiters_per_step": self.max_subiters_per_step,
            "min_subiters_per_step": self.min_subiters_per_step,
            "contraction_median": self.contraction_median,
            "contraction_mean": self.contraction_mean,
            "n_restarts": self.n_restarts,
            "n_failures": self.n_failures,
            "cond_median": self.cond_median,
            "cond_max": self.cond_max,
            "total_solve_time": self.total_solve_time,
        }


# =============================================================================
# Sweep Records Loading
# =============================================================================

def load_anderson_sweep_records(base_dir: Path) -> list[dict[str, Any]]:
    """Load Anderson sweep CSV records.
    
    Args:
        base_dir: Path to sweep output directory.
    
    Returns:
        List of sweep records sorted by Anderson parameters.
    """
    csv_file = base_dir / "sweep_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Sweep summary not found: {csv_file}")
    
    df_sweep = pd.read_csv(csv_file)
    
    # Sort by Anderson parameters
    sort_cols = []
    for col in ["solver.m", "solver.beta", "solver.lam"]:
        if col in df_sweep.columns:
            sort_cols.append(col)
    if sort_cols:
        df_sorted = df_sweep.sort_values(sort_cols)
    else:
        df_sorted = df_sweep
    
    return df_sorted.to_dict("records")


# =============================================================================
# Metrics Extraction from CSV
# =============================================================================

def load_solver_metrics(run_dir: Path) -> dict[str, Any]:
    """Load solver metrics from steps.csv and subiterations.csv.
    
    Args:
        run_dir: Path to simulation output directory.
    
    Returns:
        Dictionary with aggregated solver statistics.
    """
    steps_csv = run_dir / "steps.csv"
    subiters_csv = run_dir / "subiterations.csv"
    
    metrics = {
        "n_steps": 0,
        "total_subiters": 0,
        "subiters_per_step": [],
        "n_failures": 0,
        "contractions": [],
        "cond_values": [],
        "n_restarts": 0,
        "total_time": 0.0,
    }
    
    # Read steps.csv for per-step statistics
    if steps_csv.exists():
        with open(steps_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row.get("accepted", 1)):
                    metrics["n_steps"] += 1
                    num_subiters = int(row.get("num_subiters", 0))
                    metrics["total_subiters"] += num_subiters
                    metrics["subiters_per_step"].append(num_subiters)
                    
                    # Check for failures (non-convergence)
                    converged = row.get("converged", "1")
                    if converged in ("0", "False", "false"):
                        metrics["n_failures"] += 1
                    
                    # Accumulate solve time
                    for key in ["mech_time", "fab_time", "stim_time", "dens_time"]:
                        if key in row and row[key]:
                            try:
                                metrics["total_time"] += float(row[key])
                            except ValueError:
                                pass
    
    # Read subiterations.csv for detailed metrics
    if subiters_csv.exists():
        with open(subiters_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Contraction ratio
                contraction = row.get("contraction", "")
                if contraction and contraction not in ("", "nan", "inf"):
                    try:
                        c = float(contraction)
                        if 0 < c < 10:  # Filter outliers
                            metrics["contractions"].append(c)
                    except ValueError:
                        pass
                
                # Condition number
                cond = row.get("condH", "")
                if cond and cond not in ("", "nan", "inf"):
                    try:
                        c = float(cond)
                        if c > 0:
                            metrics["cond_values"].append(c)
                    except ValueError:
                        pass
                
                # Count restarts
                restart = row.get("restart", "")
                if restart in ("1", "True", "true"):
                    metrics["n_restarts"] += 1
    
    return metrics


def compute_anderson_metrics(
    record: dict[str, Any],
    solver_metrics: dict[str, Any],
) -> AndersonMetrics:
    """Compute Anderson metrics from solver data.
    
    Args:
        record: Sweep record with parameter values.
        solver_metrics: Aggregated solver statistics.
    
    Returns:
        AndersonMetrics dataclass.
    """
    n_steps = max(solver_metrics["n_steps"], 1)
    total_subiters = solver_metrics["total_subiters"]
    subiters_per_step = np.array(solver_metrics["subiters_per_step"])
    contractions = np.array(solver_metrics["contractions"])
    cond_values = np.array(solver_metrics["cond_values"])
    
    # Subiteration statistics
    if len(subiters_per_step) > 0:
        avg_subiters = float(np.mean(subiters_per_step))
        max_subiters = int(np.max(subiters_per_step))
        min_subiters = int(np.min(subiters_per_step))
    else:
        avg_subiters = 0.0
        max_subiters = 0
        min_subiters = 0
    
    # Contraction statistics
    if len(contractions) > 0:
        contraction_median = float(np.median(contractions))
        contraction_mean = float(np.mean(contractions))
    else:
        contraction_median = 1.0
        contraction_mean = 1.0
    
    # Condition number statistics
    if len(cond_values) > 0:
        cond_median = float(np.median(cond_values))
        cond_max = float(np.max(cond_values))
    else:
        cond_median = 1.0
        cond_max = 1.0
    
    return AndersonMetrics(
        run_hash=record.get("run_hash", ""),
        output_dir=record.get("output_dir", ""),
        m=int(record.get("solver.m", 0)),
        beta=float(record.get("solver.beta", 0.0)),
        lam=float(record.get("solver.lam", 0.0)),
        n_steps=n_steps,
        total_subiters=total_subiters,
        avg_subiters_per_step=avg_subiters,
        max_subiters_per_step=max_subiters,
        min_subiters_per_step=min_subiters,
        contraction_median=contraction_median,
        contraction_mean=contraction_mean,
        n_restarts=solver_metrics["n_restarts"],
        n_failures=solver_metrics["n_failures"],
        cond_median=cond_median,
        cond_max=cond_max,
        total_solve_time=solver_metrics["total_time"],
    )


# =============================================================================
# Analysis Pipeline
# =============================================================================

def analyze_sweep(
    sweep_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze all runs in an Anderson sweep.
    
    Args:
        sweep_dir: Path to sweep output directory.
        verbose: Print progress.
    
    Returns:
        DataFrame with Anderson metrics for all runs.
    """
    records = load_anderson_sweep_records(sweep_dir)
    if not records:
        raise ValueError(f"No sweep records found in {sweep_dir}")
    
    if verbose:
        print(f"Found {len(records)} sweep runs")
    
    metrics_list = []
    
    for idx, record in enumerate(records, start=1):
        output_dir = record["output_dir"]
        run_dir = sweep_dir / output_dir
        
        if verbose:
            m = record.get("solver.m", "?")
            beta = record.get("solver.beta", "?")
            lam = record.get("solver.lam", "?")
            print(f"  [{idx}/{len(records)}] m={m}, beta={beta}, lam={lam}")
        
        try:
            solver_metrics = load_solver_metrics(run_dir)
            metrics = compute_anderson_metrics(record, solver_metrics)
            metrics_list.append(metrics.to_dict())
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    return pd.DataFrame(metrics_list)


# =============================================================================
# Visualization
# =============================================================================

def create_diagnostic_figure(
    df: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Create diagnostic figure for Anderson sweep (1×4 row layout).
    
    CMAME-optimized single-row layout:
      (a) Heatmap: Iterations as f(m, β) - shows interaction and optimal region
      (b) Box plots: Marginal effect of each parameter on iterations
      (c) Heatmap: Contraction rate as f(m, β) - convergence quality
      (d) Stability: Restarts vs mixing β for different history sizes m
    
    Args:
        df: DataFrame with Anderson metrics.
        output_dir: Directory to save plots.
        metadata: Sweep metadata.
    """
    apply_style()
    
    # Get unique parameter values
    m_vals = sorted(df["m"].unique())
    beta_vals = sorted(df["beta"].unique())
    lam_vals = sorted(df["lam"].unique())
    
    # Find optimal lambda (lowest average iterations)
    lam_perf = df.groupby("lam")["avg_subiters_per_step"].mean()
    optimal_lam = lam_perf.idxmin()
    
    # Subset for heatmaps (fix lambda at optimal)
    df_lam_opt = df[df["lam"] == optimal_lam]
    
    # Create figure with 1×4 layout
    # Using 10.0 x 2.5 to fit 4 plots in a row nicely
    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5))
    
    # =========================================================================
    # Panel (a): Heatmap - Iterations as f(m, β)
    # =========================================================================
    ax = axes[0]
    _plot_heatmap(
        ax, df_lam_opt,
        x_col="m", y_col="beta",
        metric_col="avg_subiters_per_step",
        x_vals=m_vals, y_vals=beta_vals,
        xlabel=r"History $m$",
        ylabel=r"Mixing $\beta$",
        title=r"(a) Iterations/step",
        cmap="cividis_r",  # Modern, perceptually uniform
        show_values=True,
        fmt=".1f",
    )
    
    # =========================================================================
    # Panel (b): Box plots - Marginal parameter effects
    # =========================================================================
    ax = axes[1]
    _plot_marginal_boxplots(
        ax, df,
        metric_col="avg_subiters_per_step",
        param_cols=["m", "beta", "lam"],
        ylabel="Iterations/step",
        title="(b) Parameter sensitivity",
    )
    
    # =========================================================================
    # Panel (c): Heatmap - Contraction rate as f(m, β)
    # =========================================================================
    ax = axes[2]
    _plot_heatmap(
        ax, df_lam_opt,
        x_col="m", y_col="beta",
        metric_col="contraction_median",
        x_vals=m_vals, y_vals=beta_vals,
        xlabel=r"History $m$",
        ylabel=r"Mixing $\beta$",
        title=r"(c) Contraction $\rho$",
        cmap="RdYlBu",  # Diverging: red (bad) → blue (good)
        show_values=True,
        fmt=".2f",
        vmin=0.0, vmax=1.0,
    )
    
    # =========================================================================
    # Panel (d): Line plot - Restarts vs beta for different m
    # =========================================================================
    ax = axes[3]
    _plot_restarts_vs_beta(
        ax, df_lam_opt,
        beta_vals=beta_vals, m_vals=m_vals,
    )
    setup_axis_style(
        ax,
        xlabel=r"Mixing $\beta$",
        ylabel="Restarts",
        title="(d) Stability (Restarts)",
        grid=True,
    )
    
    plt.tight_layout()
    
    # Save figures
    save_manuscript_figure(fig, "anderson_sweep", dpi=PUBLICATION_DPI, close=False)
    save_figure(fig, output_dir / "anderson_sweep.png", dpi=PUBLICATION_DPI, close=True)


def _plot_heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    metric_col: str,
    x_vals: list,
    y_vals: list,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "viridis",
    show_values: bool = True,
    fmt: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Create a heatmap showing metric as f(x, y).
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with metrics.
        x_col, y_col: Column names for x and y axes.
        metric_col: Column name for the metric (color).
        x_vals, y_vals: Unique values for x and y.
        xlabel, ylabel, title: Axis labels.
        cmap: Colormap name.
        show_values: Annotate cells with values.
        fmt: Format string for annotations.
        vmin, vmax: Color scale limits.
    """
    # Pivot to create 2D grid
    pivot = df.pivot_table(
        values=metric_col, index=y_col, columns=x_col, aggfunc="mean"
    )
    
    # Ensure proper ordering
    pivot = pivot.reindex(index=y_vals[::-1], columns=x_vals)
    
    # Create heatmap
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in reversed(y_vals)])
    
    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=6)
    
    # Annotate cells with values
    if show_values:
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Choose text color based on background
                    text_color = "white" if val > pivot.values.mean() else "black"
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                           fontsize=6, color=text_color)
    
    # Mark optimal cell
    min_idx = np.unravel_index(np.nanargmin(pivot.values), pivot.shape)
    ax.add_patch(plt.Rectangle(
        (min_idx[1] - 0.5, min_idx[0] - 0.5), 1, 1,
        fill=False, edgecolor="red", linewidth=2
    ))


def _plot_marginal_boxplots(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    param_cols: list[str],
    ylabel: str,
    title: str,
) -> None:
    """Create grouped box plots showing marginal effect of each parameter.
    
    For each parameter, groups data by that parameter's values and shows
    the distribution of the metric. This reveals sensitivity to each parameter
    marginally (averaging over other parameters).
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with metrics.
        metric_col: Column to analyze.
        param_cols: List of parameter column names.
        ylabel, title: Axis labels.
    """
    import matplotlib.patches as mpatches
    
    positions = []
    labels = []
    colors = []
    all_data = []
    
    pos = 0
    for param in param_cols:
        unique_vals = sorted(df[param].unique())
        color = ANDERSON_PARAM_COLORS[param]
        
        for val in unique_vals:
            data = df.loc[df[param] == val, metric_col].values
            all_data.append(data)
            positions.append(pos)
            
            # Format label based on parameter type
            if param == "m":
                labels.append(str(int(val)))
            elif param == "lam":
                labels.append(f"{val:.0e}")
            else:
                labels.append(f"{val:.1f}")
            
            colors.append(color)
            pos += 1
        
        pos += 0.5  # Gap between parameter groups
    
    # Create box plots
    bp = ax.boxplot(
        all_data, positions=positions, widths=0.6, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.5),
    )
    
    # Color boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Set labels
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add parameter group labels at top
    param_centers = []
    param_labels_display = []
    idx = 0
    for param in param_cols:
        n_vals = len(df[param].unique())
        center = positions[idx] + (n_vals - 1) / 2
        param_centers.append(center)
        param_labels_display.append(ANDERSON_PARAM_LABELS[param])
        idx += n_vals
    
    # Add secondary x-axis for parameter names
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(param_centers)
    ax2.set_xticklabels(param_labels_display, fontsize=7)
    ax2.tick_params(length=0)
    
    # Remove top spine from secondary axis
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)


def _plot_restarts_vs_beta(
    ax: plt.Axes,
    df: pd.DataFrame,
    beta_vals: list,
    m_vals: list,
) -> None:
    """Plot number of restarts vs mixing parameter beta for different history sizes m.
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with metrics (filtered for optimal lambda).
        beta_vals: Mixing parameter values.
        m_vals: History size values.
    """
    from matplotlib.colors import Normalize
    
    # Color map for m values
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(m_vals), vmax=max(m_vals))
    
    for m in m_vals:
        subset = df[df["m"] == m]
        if subset.empty:
            continue
            
        # Sort by beta
        subset = subset.sort_values("beta")
        
        color = cmap(norm(m))
        ax.plot(
            subset["beta"], subset["n_restarts"],
            marker="o", color=color, linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE, label=f"$m={m}$",
        )
    
    # Add legend
    ax.legend(fontsize=6, loc="upper left", frameon=False, title="History $m$")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run Anderson sweep analysis."""
    sweep_dir = Path("results/anderson_sweep")
    
    print("=" * 70)
    print("ANDERSON ACCELERATION PARAMETER SWEEP ANALYSIS")
    print("=" * 70)
    print(f"Sweep directory: {sweep_dir}")
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        print("Run the sweep first: mpirun -n 4 python run_anderson_sweep.py")
        return
    
    # Load metadata
    metadata_file = sweep_dir / "sweep_summary.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Analyze sweep
    print("\nAnalyzing solver metrics...")
    df = analyze_sweep(sweep_dir, verbose=True)
    
    if df.empty:
        print("No valid runs found!")
        return
    
    # Save metrics CSV
    metrics_file = sweep_dir / "anderson_metrics.csv"
    df.to_csv(metrics_file, index=False)
    print(f"\nSaved metrics to {metrics_file}")
    
    # Print summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total runs analyzed: {len(df)}")
    print(f"Avg iterations/step: {df['avg_subiters_per_step'].mean():.2f} "
          f"(range: {df['avg_subiters_per_step'].min():.1f} - {df['avg_subiters_per_step'].max():.1f})")
    print(f"Median contraction: {df['contraction_median'].median():.3f}")
    print(f"Total failures: {df['n_failures'].sum()}")
    print(f"Total restarts: {df['n_restarts'].sum()}")
    
    # Best configuration
    best_idx = df["avg_subiters_per_step"].idxmin()
    best = df.loc[best_idx]
    print(f"\nBest configuration (fewest iterations):")
    print(f"  m={int(best['m'])}, beta={best['beta']:.2f}, lam={best['lam']:.1e}")
    print(f"  → {best['avg_subiters_per_step']:.2f} iterations/step")
    
    # Create diagnostic figure
    print("\nGenerating diagnostic figure...")
    create_diagnostic_figure(df, sweep_dir, metadata)
    
    print("\nAnalysis complete!")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Figure: {sweep_dir / 'anderson_sweep.png'}")


if __name__ == "__main__":
    main()
