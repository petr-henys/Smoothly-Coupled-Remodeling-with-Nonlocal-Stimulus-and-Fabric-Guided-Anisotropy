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
    
    # Mechanostat error signal statistics (delta = (m - m_ref)/m_ref)
    delta_mean: float
    delta_std: float
    delta_q05: float
    delta_q95: float
    
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
            "delta_mean": self.delta_mean,
            "delta_std": self.delta_std,
            "delta_q05": self.delta_q05,
            "delta_q95": self.delta_q95,
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
    
    # Compute derived signal m = psi / rho
    # Note: rho and psi might be on different meshes/spaces in general, 
    # but here we assume compatible arrays or just use global stats.
    # For safety in this analysis script, we'll use the arrays directly if sizes match,
    # otherwise we skip the detailed delta calculation or interpolate.
    
    delta_stats = {"mean": 0.0, "std": 0.0, "q05": 0.0, "q95": 0.0}
    
    if len(rho_array) == len(psi_array):
        # Avoid division by zero
        rho_safe = np.maximum(rho_array, 1e-6)
        m_local = psi_array / rho_safe
        
        # Reference signal (simplified: using the trabecular ref from config)
        psi_ref = float(record.get("stimulus.psi_ref_trab", 0.01))
        m_ref = psi_ref  # Assuming m_ref ~ psi_ref for rho=1, simplified
        
        delta_local = (m_local - m_ref) / m_ref
        
        # Gather for stats (this might be heavy for large meshes, but okay for analysis)
        delta_global = comm.allgather(delta_local)
        delta_global = np.concatenate(delta_global)
        
        if len(delta_global) > 0:
            delta_stats["mean"] = float(np.mean(delta_global))
            delta_stats["std"] = float(np.std(delta_global))
            delta_stats["q05"] = float(np.quantile(delta_global, 0.05))
            delta_stats["q95"] = float(np.quantile(delta_global, 0.95))

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
        psi_ref=float(record.get("stimulus.psi_ref_trab", 0.01)),
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
        delta_mean=delta_stats["mean"],
        delta_std=delta_stats["std"],
        delta_q05=delta_stats["q05"],
        delta_q95=delta_stats["q95"],
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
        DataFrame with metrics for all runs.
    """
    # Load sweep records
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
            # Load fields from checkpoint
            rho, S, psi, domain, config = load_fields_from_checkpoint(run_dir, comm)
            
            # Compute metrics
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
    
    # Gather metrics from all ranks (though we only compute on rank 0 effectively if we used bcast, 
    # but here we are running in parallel so each rank does its part? 
    # Actually, the loop runs on all ranks, and compute_metrics does reductions.
    # So metrics_list is populated on all ranks with the same data.)
    
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
    """Create diagnostic figure for stimulus sweep.
    
    Layout (1x3):
      (a) Density Homeostasis: Density Change vs Psi_ref (grouped by Kappa).
      (b) Stability: Lazy Fraction vs Delta0 (grouped by Psi_ref).
      (c) Control Error: Mean Stimulus vs Kappa (grouped by Delta0).
    
    Args:
        df: DataFrame with metrics.
        output_dir: Directory to save plots.
        metadata: Sweep metadata.
    """
    apply_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    
    # =========================================================================
    # Panel (a): Density Homeostasis (Targeting)
    # =========================================================================
    ax = axes[0]
    _plot_interaction(
        ax, df,
        x_col="psi_ref", y_col="density_change_rel", group_col="kappa",
        cmap_name="Greens",
        legend_title=r"$\kappa$",
        log_x=True,
    )
    setup_axis_style(
        ax,
        xlabel=r"Reference Signal $\psi_{\mathrm{ref}}$",
        ylabel=r"$\Delta\rho/\rho_0$ [%]",
        title="(a) Density Homeostasis",
        grid=True,
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}'))
    
    # =========================================================================
    # Panel (b): Stability (Lazy Zone)
    # =========================================================================
    ax = axes[1]
    _plot_interaction(
        ax, df,
        x_col="delta0", y_col="lazy_fraction", group_col="psi_ref",
        cmap_name="Blues",
        legend_title=r"$\psi_{\mathrm{ref}}$",
        log_x=True,
    )
    setup_axis_style(
        ax,
        xlabel=r"Lazy Zone Width $\delta_0$",
        ylabel="Lazy Fraction",
        title="(b) Stability",
        grid=True,
    )
    ax.set_ylim(0, 1.0)
    
    # =========================================================================
    # Panel (c): Control Error (Drive)
    # =========================================================================
    ax = axes[2]
    _plot_interaction(
        ax, df,
        x_col="kappa", y_col="S_mean", group_col="delta0",
        cmap_name="Oranges",
        legend_title=r"$\delta_0$",
        log_x=True,
    )
    setup_axis_style(
        ax,
        xlabel=r"Saturation Width $\kappa$",
        ylabel=r"Mean Stimulus $\bar{S}$",
        title="(c) Control Drive",
        grid=True,
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    
    plt.tight_layout()
    
    # Save figures
    save_manuscript_figure(fig, "stimulus_diagnostic", dpi=PUBLICATION_DPI, close=False)
    save_figure(fig, output_dir / "stimulus_diagnostic.png", dpi=PUBLICATION_DPI, close=True)


def _plot_interaction(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    cmap_name: str,
    legend_title: str,
    log_x: bool = False,
) -> None:
    """Plot interaction between two parameters on a metric."""
    # Get unique values for grouping
    groups = sorted(df[group_col].unique())
    
    # Generate colors
    cmap = plt.get_cmap(cmap_name)
    # Avoid too light colors
    colors = [cmap(i) for i in np.linspace(0.4, 1.0, len(groups))]
    
    for i, group_val in enumerate(groups):
        # Filter data
        mask = np.isclose(df[group_col], group_val, rtol=1e-9)
        sub_df = df[mask].sort_values(x_col)
        
        # Aggregate if there are multiple points per x (e.g. varying third param)
        # We plot the mean line
        grouped = sub_df.groupby(x_col)[y_col].mean()
        x_vals = grouped.index.values
        y_vals = grouped.values
        
        label = f"{group_val:.1g}" if abs(group_val) < 1e-3 or abs(group_val) > 1e3 else f"{group_val:.2g}"
        
        ax.plot(
            x_vals, y_vals,
            color=colors[i],
            marker="o",
            linestyle="-",
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            label=label,
        )
    
    if log_x:
        ax.set_xscale("log")
    
    # Add legend
    ax.legend(title=legend_title, fontsize=8, title_fontsize=9, frameon=False)


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
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Figure: {sweep_dir / 'stimulus_diagnostic.png'}")


if __name__ == "__main__":
    main()
