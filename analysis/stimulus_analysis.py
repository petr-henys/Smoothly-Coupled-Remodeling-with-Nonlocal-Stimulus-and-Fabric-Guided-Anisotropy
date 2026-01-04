"""Stimulus/mechanostat parameter sweep analysis: remodeling response metrics.

Loads checkpoints from stimulus sweep results, computes remodeling metrics
(density change, stimulus distribution, formation/resorption balance), and
visualizes mechanostat sensitivity.

Requires adios4dolfinx: pip install adios4dolfinx

Usage:
    mpirun -n 1 python analysis/stimulus_analysis.py
    
Inputs:
    results/stimulus_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/   <- contains rho, S, psi
    │   └── steps.csv
    ...

Outputs:
    results/stimulus_sweep/
    ├── stimulus_metrics.csv
    ├── stimulus_diagnostic.png
    └── (manuscript/images/stimulus_diagnostic.png)
"""

from __future__ import annotations

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
from mpi4py import MPI


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
# Stimulus Parameter Styling
# =============================================================================

STIMULUS_PARAM_COLORS = {
    "kappa": "#029E73",       # Green (stimulus)
    "delta0": "#DE8F05",      # Orange
    "psi_ref": "#0173B2",     # Blue
}
STIMULUS_PARAM_LABELS = {
    "kappa": r"$\kappa$ (saturation width)",
    "delta0": r"$\delta_0$ (lazy zone)",
    "psi_ref": r"$\psi_{\mathrm{ref}}$ (reference SED)",
}
STIMULUS_PARAM_MARKERS = {
    "kappa": "o",
    "delta0": "s",
    "psi_ref": "^",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StimulusMetrics:
    """Stimulus/mechanostat analysis metrics.
    
    Metrics characterizing remodeling response:
    - density_change: Relative change in average density from initial
    - formation_fraction: Fraction of domain with S > 0 (formation)
    - resorption_fraction: Fraction of domain with S < 0 (resorption)
    - lazy_fraction: Fraction in lazy zone (|S| < threshold)
    - stimulus statistics: mean, std of S field
    """
    # Run identification
    run_hash: str
    output_dir: str
    
    # Stimulus parameters
    kappa: float
    delta0: float
    psi_ref: float
    
    # Density metrics
    rho_initial: float
    rho_final_mean: float
    rho_final_std: float
    rho_final_min: float
    rho_final_max: float
    density_change_rel: float  # (rho_final - rho_initial) / rho_initial
    
    # Stimulus distribution
    S_mean: float
    S_std: float
    S_min: float
    S_max: float
    
    # Remodeling balance (fraction of domain)
    formation_fraction: float   # S > +threshold
    resorption_fraction: float  # S < -threshold
    lazy_fraction: float        # |S| < threshold
    
    # SED statistics
    psi_mean: float
    psi_std: float
    psi_max: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "run_hash": self.run_hash,
            "output_dir": self.output_dir,
            "kappa": self.kappa,
            "delta0": self.delta0,
            "psi_ref": self.psi_ref,
            "rho_initial": self.rho_initial,
            "rho_final_mean": self.rho_final_mean,
            "rho_final_std": self.rho_final_std,
            "rho_final_min": self.rho_final_min,
            "rho_final_max": self.rho_final_max,
            "density_change_rel": self.density_change_rel,
            "S_mean": self.S_mean,
            "S_std": self.S_std,
            "S_min": self.S_min,
            "S_max": self.S_max,
            "formation_fraction": self.formation_fraction,
            "resorption_fraction": self.resorption_fraction,
            "lazy_fraction": self.lazy_fraction,
            "psi_mean": self.psi_mean,
            "psi_std": self.psi_std,
            "psi_max": self.psi_max,
        }


# =============================================================================
# Sweep Records Loading
# =============================================================================

def load_stimulus_sweep_records(
    base_dir: Path,
    comm: MPI.Comm,
) -> list[dict[str, Any]]:
    """Load stimulus sweep CSV records (MPI-aware).
    
    Args:
        base_dir: Path to sweep output directory.
        comm: MPI communicator.
    
    Returns:
        List of sweep records sorted by stimulus parameters.
    """
    csv_file = base_dir / "sweep_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Sweep summary not found: {csv_file}")
    
    if comm.rank == 0:
        df_sweep = pd.read_csv(csv_file)
        # Sort by stimulus parameters
        sort_cols = []
        for col in ["stimulus.stimulus_kappa", "stimulus.stimulus_delta0", "stimulus.psi_ref_trab"]:
            if col in df_sweep.columns:
                sort_cols.append(col)
        if sort_cols:
            df_sorted = df_sweep.sort_values(sort_cols)
        else:
            df_sorted = df_sweep
        records = df_sorted.to_dict("records")
    else:
        records = None
    
    records = comm.bcast(records, root=0)
    return records


# =============================================================================
# Checkpoint Loading
# =============================================================================

def create_scalar_space(domain):
    """Create P1 scalar function space."""
    import basix.ufl
    from dolfinx import fem
    
    cell_name = domain.topology.cell_name()
    element = basix.ufl.element("Lagrange", cell_name, 1)
    return fem.functionspace(domain, element)


def create_dg0_space(domain):
    """Create DG0 scalar function space."""
    from dolfinx import fem
    return fem.functionspace(domain, ("DG", 0))


def load_fields_from_checkpoint(
    run_dir: Path,
    comm: MPI.Comm,
    final_time: float | None = None,
) -> tuple:
    """Load rho, S, psi from checkpoint.
    
    Args:
        run_dir: Path to simulation output directory.
        comm: MPI communicator.
        final_time: Time to load. If None, reads from config.json.
    
    Returns:
        Tuple of (rho, S, psi, domain, config).
    """
    checkpoint_path = run_dir / "checkpoint.bp"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Read config
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    if final_time is None:
        final_time = config.get("time", {}).get("total_time", 0.0)
    
    from analysis.analysis_utils import load_checkpoint_mesh, load_checkpoint_function
    
    domain = load_checkpoint_mesh(checkpoint_path, comm)
    
    # Create function spaces
    V_rho = create_scalar_space(domain)
    V_S = create_scalar_space(domain)
    V_psi = create_dg0_space(domain)
    
    # Load functions
    rho = load_checkpoint_function(checkpoint_path, "rho", V_rho, time=final_time)
    S = load_checkpoint_function(checkpoint_path, "S", V_S, time=final_time)
    psi = load_checkpoint_function(checkpoint_path, "psi", V_psi, time=final_time)
    
    return rho, S, psi, domain, config


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_stimulus_metrics(
    rho, S, psi,
    record: dict[str, Any],
    config: dict[str, Any],
    comm: MPI.Comm,
) -> StimulusMetrics:
    """Compute stimulus/remodeling metrics from checkpoint fields.
    
    Args:
        rho: Density function.
        S: Stimulus function.
        psi: SED function.
        record: Sweep record with parameter values.
        config: Simulation config.
        comm: MPI communicator.
    
    Returns:
        StimulusMetrics dataclass.
    """
    # Scatter forward
    rho.x.scatter_forward()
    S.x.scatter_forward()
    psi.x.scatter_forward()
    
    # Get local arrays
    n_local_rho = rho.function_space.dofmap.index_map.size_local
    rho_array = rho.x.array[:n_local_rho]
    
    n_local_S = S.function_space.dofmap.index_map.size_local
    S_array = S.x.array[:n_local_S]
    
    n_local_psi = psi.function_space.dofmap.index_map.size_local
    psi_array = psi.x.array[:n_local_psi]
    
    # Get initial density from config
    rho_initial = config.get("density", {}).get("rho0", 1.0)
    
    # Threshold for lazy zone detection
    lazy_threshold = 0.01  # |S| < 0.01 considered "in lazy zone"
    
    # Local statistics
    def local_stats(arr):
        if len(arr) == 0:
            return 0.0, 0.0, np.inf, -np.inf, 0
        return np.sum(arr), np.sum(arr**2), np.min(arr), np.max(arr), len(arr)
    
    rho_sum, rho_sq, rho_min, rho_max, rho_n = local_stats(rho_array)
    S_sum, S_sq, S_min, S_max, S_n = local_stats(S_array)
    psi_sum, psi_sq, psi_min, psi_max, psi_n = local_stats(psi_array)
    
    # Count fractions
    n_formation = np.sum(S_array > lazy_threshold)
    n_resorption = np.sum(S_array < -lazy_threshold)
    n_lazy = np.sum(np.abs(S_array) <= lazy_threshold)
    
    # Global reduction
    global_rho_sum = comm.allreduce(rho_sum, op=MPI.SUM)
    global_rho_sq = comm.allreduce(rho_sq, op=MPI.SUM)
    global_rho_min = comm.allreduce(rho_min, op=MPI.MIN)
    global_rho_max = comm.allreduce(rho_max, op=MPI.MAX)
    global_rho_n = comm.allreduce(rho_n, op=MPI.SUM)
    
    global_S_sum = comm.allreduce(S_sum, op=MPI.SUM)
    global_S_sq = comm.allreduce(S_sq, op=MPI.SUM)
    global_S_min = comm.allreduce(S_min, op=MPI.MIN)
    global_S_max = comm.allreduce(S_max, op=MPI.MAX)
    global_S_n = comm.allreduce(S_n, op=MPI.SUM)
    
    global_psi_sum = comm.allreduce(psi_sum, op=MPI.SUM)
    global_psi_sq = comm.allreduce(psi_sq, op=MPI.SUM)
    global_psi_max = comm.allreduce(psi_max, op=MPI.MAX)
    global_psi_n = comm.allreduce(psi_n, op=MPI.SUM)
    
    global_n_formation = comm.allreduce(n_formation, op=MPI.SUM)
    global_n_resorption = comm.allreduce(n_resorption, op=MPI.SUM)
    global_n_lazy = comm.allreduce(n_lazy, op=MPI.SUM)
    
    # Compute means and stds
    def mean_std(s, sq_s, n):
        if n == 0:
            return 0.0, 0.0
        mean = s / n
        var = max(0.0, sq_s / n - mean**2)
        return mean, np.sqrt(var)
    
    rho_mean, rho_std = mean_std(global_rho_sum, global_rho_sq, global_rho_n)
    S_mean, S_std = mean_std(global_S_sum, global_S_sq, global_S_n)
    psi_mean, psi_std = mean_std(global_psi_sum, global_psi_sq, global_psi_n)
    
    # Fractions
    total_S_n = max(global_S_n, 1)
    formation_frac = global_n_formation / total_S_n
    resorption_frac = global_n_resorption / total_S_n
    lazy_frac = global_n_lazy / total_S_n
    
    # Relative density change
    density_change = (rho_mean - rho_initial) / rho_initial if rho_initial > 0 else 0.0
    
    return StimulusMetrics(
        run_hash=record.get("run_hash", ""),
        output_dir=record.get("output_dir", ""),
        kappa=float(record.get("stimulus.stimulus_kappa", 0.0)),
        delta0=float(record.get("stimulus.stimulus_delta0", 0.0)),
        psi_ref=float(record.get("stimulus.psi_ref_trab", 0.0)),
        rho_initial=rho_initial,
        rho_final_mean=rho_mean,
        rho_final_std=rho_std,
        rho_final_min=global_rho_min,
        rho_final_max=global_rho_max,
        density_change_rel=density_change,
        S_mean=S_mean,
        S_std=S_std,
        S_min=global_S_min,
        S_max=global_S_max,
        formation_fraction=formation_frac,
        resorption_fraction=resorption_frac,
        lazy_fraction=lazy_frac,
        psi_mean=psi_mean,
        psi_std=psi_std,
        psi_max=global_psi_max,
    )


# =============================================================================
# Analysis Pipeline
# =============================================================================

def analyze_sweep(
    sweep_dir: Path,
    comm: MPI.Comm,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze all runs in a stimulus sweep.
    
    Args:
        sweep_dir: Path to sweep output directory.
        comm: MPI communicator.
        verbose: Print progress.
    
    Returns:
        DataFrame with stimulus metrics for all runs.
    """
    records = load_stimulus_sweep_records(sweep_dir, comm)
    if not records:
        raise ValueError(f"No sweep records found in {sweep_dir}")
    
    if verbose and comm.rank == 0:
        print(f"Found {len(records)} sweep runs")
    
    metrics_list = []
    
    for idx, record in enumerate(records, start=1):
        output_dir = record["output_dir"]
        run_dir = sweep_dir / output_dir
        
        if verbose and comm.rank == 0:
            kappa = record.get("stimulus.stimulus_kappa", "?")
            delta0 = record.get("stimulus.stimulus_delta0", "?")
            psi_ref = record.get("stimulus.psi_ref_trab", "?")
            print(f"  [{idx}/{len(records)}] kappa={kappa}, delta0={delta0}, psi_ref={psi_ref}")
        
        try:
            rho, S, psi, _, config = load_fields_from_checkpoint(run_dir, comm)
            metrics = compute_stimulus_metrics(rho, S, psi, record, config, comm)
            metrics_list.append(metrics.to_dict())
            
        except FileNotFoundError as e:
            if comm.rank == 0:
                print(f"    Warning: {e}")
            continue
        except Exception as e:
            if comm.rank == 0:
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
    """Create diagnostic figure for stimulus sweep (1×3 layout).
    
    Layout:
      (a) Density Change - net bone mass gain/loss from initial
      (b) Formation/Resorption Balance - fraction of domain in each state
      (c) Stimulus Activity - standard deviation of S (how active is remodeling?)
    
    Each panel shows sensitivity to all three stimulus parameters.
    
    Args:
        df: DataFrame with stimulus metrics.
        output_dir: Directory to save plots.
        metadata: Sweep metadata.
    """
    apply_style()
    
    # Get unique parameter values
    kappa_vals = sorted(df["kappa"].unique())
    delta0_vals = sorted(df["delta0"].unique())
    psi_ref_vals = sorted(df["psi_ref"].unique())
    
    # Baselines
    baseline_kappa = float(metadata.get("baseline_kappa", kappa_vals[len(kappa_vals) // 2]))
    baseline_delta0 = float(metadata.get("baseline_delta0", delta0_vals[len(delta0_vals) // 2]))
    baseline_psi_ref = float(metadata.get("baseline_psi_ref", psi_ref_vals[len(psi_ref_vals) // 2]))
    
    # Create figure with 1×3 layout
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    
    # =========================================================================
    # Panel (a): Density Change
    # =========================================================================
    ax = axes[0]
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="density_change_rel",
        param_values={
            "kappa": (kappa_vals, baseline_kappa),
            "delta0": (delta0_vals, baseline_delta0),
            "psi_ref": (psi_ref_vals, baseline_psi_ref),
        },
        use_log_x={"psi_ref"},
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel=r"$\Delta\rho/\rho_0$ [%]",
        title="(a) Net density change",
        grid=True,
    )
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    # Convert to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}'))
    
    # =========================================================================
    # Panel (b): Formation/Resorption Balance
    # =========================================================================
    ax = axes[1]
    # Plot formation fraction
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="formation_fraction",
        param_values={
            "kappa": (kappa_vals, baseline_kappa),
            "delta0": (delta0_vals, baseline_delta0),
            "psi_ref": (psi_ref_vals, baseline_psi_ref),
        },
        use_log_x={"psi_ref"},
        alpha=0.7,
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel="Domain fraction",
        title="(b) Remodeling balance",
        grid=True,
    )
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    
    # =========================================================================
    # Panel (c): Stimulus Activity
    # =========================================================================
    ax = axes[2]
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="S_std",
        param_values={
            "kappa": (kappa_vals, baseline_kappa),
            "delta0": (delta0_vals, baseline_delta0),
            "psi_ref": (psi_ref_vals, baseline_psi_ref),
        },
        use_log_x={"psi_ref"},
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel=r"$\mathrm{std}(S)$",
        title="(c) Remodeling activity",
        grid=True,
    )
    ax.set_ylim(0, None)
    
    # Add legend below the plots
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=STIMULUS_PARAM_COLORS["kappa"], marker="o",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=STIMULUS_PARAM_LABELS["kappa"]),
        Line2D([0], [0], color=STIMULUS_PARAM_COLORS["delta0"], marker="s",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=STIMULUS_PARAM_LABELS["delta0"]),
        Line2D([0], [0], color=STIMULUS_PARAM_COLORS["psi_ref"], marker="^",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=STIMULUS_PARAM_LABELS["psi_ref"]),
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
    save_manuscript_figure(fig, "stimulus_diagnostic", dpi=PUBLICATION_DPI, close=False)
    save_figure(fig, output_dir / "stimulus_diagnostic.png", dpi=PUBLICATION_DPI, close=True)


def _plot_parameter_sensitivity(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    param_values: dict[str, tuple[list, float]],
    use_log_x: set[str] | None = None,
    alpha: float = 1.0,
) -> None:
    """Plot metric sensitivity to each stimulus parameter.
    
    For each parameter, varies it while keeping others at baseline.
    Shows median line (IQR removed - it showed variability due to other
    parameters, not statistical uncertainty).
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with metrics.
        metric_col: Column name for the metric to plot.
        param_values: Dict mapping param names to (values, baseline).
        use_log_x: Set of param names to use log scale for normalization.
        alpha: Line alpha.
    """
    if use_log_x is None:
        use_log_x = set()
    
    for param_name, (values, baseline) in param_values.items():
        color = STIMULUS_PARAM_COLORS[param_name]
        marker = STIMULUS_PARAM_MARKERS[param_name]
        
        # Normalize values by baseline
        if param_name in use_log_x:
            x_normalized = np.log10(np.array(values)) - np.log10(baseline)
            x_normalized = 10 ** x_normalized
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
            alpha=alpha,
        )
    
    # Add vertical line at baseline (x=1)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run stimulus sweep analysis."""
    comm = MPI.COMM_WORLD
    
    sweep_dir = Path("results/stimulus_sweep")
    
    if comm.rank == 0:
        print("=" * 70)
        print("STIMULUS/MECHANOSTAT PARAMETER SWEEP ANALYSIS")
        print("=" * 70)
        print(f"Sweep directory: {sweep_dir}")
    
    if not sweep_dir.exists():
        if comm.rank == 0:
            print(f"Error: Sweep directory not found: {sweep_dir}")
            print("Run the sweep first: mpirun -n 4 python run_stimulus_sweep.py")
        return
    
    # Load metadata
    metadata_file = sweep_dir / "sweep_summary.json"
    if metadata_file.exists() and comm.rank == 0:
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    metadata = comm.bcast(metadata if comm.rank == 0 else None, root=0)
    
    # Analyze sweep
    if comm.rank == 0:
        print("\nAnalyzing checkpoint data...")
    
    df = analyze_sweep(sweep_dir, comm, verbose=True)
    
    if df.empty:
        if comm.rank == 0:
            print("No valid runs found!")
        return
    
    # Save metrics CSV
    if comm.rank == 0:
        metrics_file = sweep_dir / "stimulus_metrics.csv"
        df.to_csv(metrics_file, index=False)
        print(f"\nSaved metrics to {metrics_file}")
        
        # Print summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"Total runs analyzed: {len(df)}")
        print(f"Density change: {df['density_change_rel'].mean()*100:.2f}% "
              f"(range: {df['density_change_rel'].min()*100:.1f}% to {df['density_change_rel'].max()*100:.1f}%)")
        print(f"Formation fraction: {df['formation_fraction'].mean():.2f} (mean)")
        print(f"Resorption fraction: {df['resorption_fraction'].mean():.2f} (mean)")
        print(f"Lazy zone fraction: {df['lazy_fraction'].mean():.2f} (mean)")
    
    # Create diagnostic figure
    if comm.rank == 0:
        print("\nGenerating diagnostic figure...")
        create_diagnostic_figure(df, sweep_dir, metadata)
        print("\nAnalysis complete!")
        print(f"  - Metrics: {sweep_dir / 'stimulus_metrics.csv'}")
        print(f"  - Figure: {sweep_dir / 'stimulus_diagnostic.png'}")


if __name__ == "__main__":
    main()
