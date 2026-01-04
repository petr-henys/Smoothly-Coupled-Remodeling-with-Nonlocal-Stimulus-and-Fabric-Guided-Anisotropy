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
    "m": "#0173B2",       # Blue (history)
    "beta": "#DE8F05",    # Orange (mixing)
    "lam": "#029E73",     # Green (regularization)
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

def _summarize(values: np.ndarray) -> tuple[float, float, float]:
    """Return (median, q25, q75) for a 1D array."""
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(med), float(q25), float(q75)


def create_diagnostic_figure(
    df: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Create diagnostic figure for Anderson sweep (1×3 layout).
    
    Layout:
      (a) Iterations per step - how many fixed-point iterations needed?
      (b) Contraction rate - how fast does residual decrease?
      (c) Condition number - how stable is the Gram matrix?
    
    Each panel shows sensitivity to all three Anderson parameters.
    
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
    
    # Baselines
    baseline_m = int(metadata.get("baseline_m", m_vals[len(m_vals) // 2]))
    baseline_beta = float(metadata.get("baseline_beta", beta_vals[len(beta_vals) // 2]))
    baseline_lam = float(metadata.get("baseline_lam", lam_vals[len(lam_vals) // 2]))
    
    # Create figure with 1×3 layout
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    
    # =========================================================================
    # Panel (a): Average Iterations per Step
    # =========================================================================
    ax = axes[0]
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="avg_subiters_per_step",
        param_values={
            "m": (m_vals, baseline_m),
            "beta": (beta_vals, baseline_beta),
            "lam": (lam_vals, baseline_lam),
        },
        use_log_x={"lam"},
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel="Iterations / step",
        title="(a) Convergence speed",
        grid=True,
    )
    ax.set_ylim(0, None)
    
    # =========================================================================
    # Panel (b): Contraction Rate
    # =========================================================================
    ax = axes[1]
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="contraction_median",
        param_values={
            "m": (m_vals, baseline_m),
            "beta": (beta_vals, baseline_beta),
            "lam": (lam_vals, baseline_lam),
        },
        use_log_x={"lam"},
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel=r"Contraction $\rho$",
        title="(b) Convergence rate",
        grid=True,
    )
    ax.set_ylim(0, 1)
    ax.axhline(y=1.0, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    
    # =========================================================================
    # Panel (c): Condition Number
    # =========================================================================
    ax = axes[2]
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="cond_median",
        param_values={
            "m": (m_vals, baseline_m),
            "beta": (beta_vals, baseline_beta),
            "lam": (lam_vals, baseline_lam),
        },
        use_log_x={"lam"},
        log_y=True,
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel=r"Condition $\kappa(H)$",
        title="(c) Numerical stability",
        grid=True,
    )
    
    # Add legend below the plots
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ANDERSON_PARAM_COLORS["m"], marker="o",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=ANDERSON_PARAM_LABELS["m"]),
        Line2D([0], [0], color=ANDERSON_PARAM_COLORS["beta"], marker="s",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=ANDERSON_PARAM_LABELS["beta"]),
        Line2D([0], [0], color=ANDERSON_PARAM_COLORS["lam"], marker="^",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=ANDERSON_PARAM_LABELS["lam"]),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    
    # Save figures
    save_manuscript_figure(fig, "anderson_sweep", dpi=PUBLICATION_DPI, close=False)
    save_figure(fig, output_dir / "anderson_sweep.png", dpi=PUBLICATION_DPI, close=True)


def _plot_parameter_sensitivity(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    param_values: dict[str, tuple[list, float]],
    use_log_x: set[str] | None = None,
    log_y: bool = False,
) -> None:
    """Plot metric sensitivity to each Anderson parameter.
    
    For each parameter, varies it while keeping others at baseline.
    Shows median line (IQR removed - it showed variability due to other
    parameters, not statistical uncertainty).
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with metrics.
        metric_col: Column name for the metric to plot.
        param_values: Dict mapping param names to (values, baseline).
        use_log_x: Set of param names to use log scale for normalization.
        log_y: Use log scale for y-axis.
    """
    if use_log_x is None:
        use_log_x = set()
    
    for param_name, (values, baseline) in param_values.items():
        color = ANDERSON_PARAM_COLORS[param_name]
        marker = ANDERSON_PARAM_MARKERS[param_name]
        
        # Normalize values by baseline
        if param_name in use_log_x:
            # For log-scale params, use log ratio
            x_normalized = np.log10(np.array(values)) - np.log10(baseline)
            x_normalized = 10 ** x_normalized  # Back to ratio
        else:
            x_normalized = np.array(values) / baseline
        
        # Compute median metric for each parameter value
        y_med = []
        
        for val in values:
            mask = df[param_name] == val
            metric_vals = df.loc[mask, metric_col].values
            med, _, _ = _summarize(metric_vals)
            y_med.append(med)
        
        y_med = np.array(y_med)
        
        # Plot median line with markers
        ax.plot(
            x_normalized, y_med,
            color=color, marker=marker, linestyle="-",
            linewidth=PLOT_LINEWIDTH, markersize=PLOT_MARKERSIZE + 2,
        )
    
    # Add vertical line at baseline (x=1)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    
    if log_y:
        ax.set_yscale("log")


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
