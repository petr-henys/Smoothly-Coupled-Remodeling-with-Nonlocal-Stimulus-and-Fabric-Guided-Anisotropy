"""Diffusion regularization analysis: compute quality metrics and generate visualizations.

Loads checkpoints from diffusion sweep results, computes checkerboarding metrics
(total variation, gradient norms, roughness), and combines with solver performance
from steps.csv to produce Pareto analysis plots.

Requires adios4dolfinx: pip install adios4dolfinx

Usage:
    mpirun -n 1 python analysis/diffusion_analysis.py
    
Inputs:
    results/diffusion_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/
    │   └── steps.csv
    ...

Outputs:
    results/diffusion_sweep/
    ├── quality_metrics.csv
    ├── diffusion_heatmap.pdf
    ├── pareto_front.pdf
    └── diffusion_sweep_report.pdf
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
# Sweep Records Loading (diffusion-specific)
# =============================================================================

def load_diffusion_sweep_records(
    base_dir: Path,
    comm,
) -> list[dict[str, Any]]:
    """Load diffusion sweep CSV records (MPI-aware).
    
    Unlike convergence sweep, sorts by diffusion parameters.
    
    Args:
        base_dir: Path to sweep output directory.
        comm: MPI communicator.
    
    Returns:
        List of sweep records sorted by D_rho, stimulus_D, fabric_D.
    """
    csv_file = base_dir / "sweep_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Sweep summary not found: {csv_file}")
    
    if comm.rank == 0:
        df_sweep = pd.read_csv(csv_file)
        # Sort by diffusion parameters
        sort_cols = []
        for col in ["density.D_rho", "stimulus.stimulus_D", "fabric.fabric_D"]:
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
# Data Classes
# =============================================================================

@dataclass
class QualityMetrics:
    """Quality metrics for solution smoothness and checkerboarding detection.
    
    Metrics computed on the density field rho:
    - total_variation: TV(rho) = integral(|grad rho|)
    - roughness: sqrt(integral(|grad rho|^2)) normalized by mesh size
    - gradient_mean/max: Statistics of |grad rho|
    - range: rho_max - rho_min (spread of density values)
    """
    # Run identification
    run_hash: str
    output_dir: str
    
    # Diffusion parameters
    D_rho: float
    stimulus_D: float
    fabric_D: float
    
    # Field statistics
    rho_min: float
    rho_max: float
    rho_mean: float
    rho_std: float
    
    # Gradient-based smoothness metrics
    total_variation: float       # TV norm: ||grad rho||_L1
    gradient_l2: float          # H1 seminorm: ||grad rho||_L2
    gradient_mean: float        # Mean of |grad rho|
    gradient_max: float         # Max of |grad rho| (approx via L^p)
    
    # Normalized roughness (mesh-independent)
    roughness_ratio: float      # gradient_l2 / rho_mean (relative roughness)
    
    # Solver performance (from steps.csv)
    total_picard_iters: int
    avg_picard_per_step: float
    total_ksp_iters_dens: int
    total_ksp_iters_stim: int
    total_ksp_iters_mech: int
    total_ksp_iters_fab: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "run_hash": self.run_hash,
            "output_dir": self.output_dir,
            "D_rho": self.D_rho,
            "stimulus_D": self.stimulus_D,
            "fabric_D": self.fabric_D,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "rho_mean": self.rho_mean,
            "rho_std": self.rho_std,
            "total_variation": self.total_variation,
            "gradient_l2": self.gradient_l2,
            "gradient_mean": self.gradient_mean,
            "gradient_max": self.gradient_max,
            "roughness_ratio": self.roughness_ratio,
            "total_picard_iters": self.total_picard_iters,
            "avg_picard_per_step": self.avg_picard_per_step,
            "total_ksp_iters_dens": self.total_ksp_iters_dens,
            "total_ksp_iters_stim": self.total_ksp_iters_stim,
            "total_ksp_iters_mech": self.total_ksp_iters_mech,
            "total_ksp_iters_fab": self.total_ksp_iters_fab,
        }


# =============================================================================
# Checkpoint Loading
# =============================================================================

def create_scalar_space(domain: mesh.Mesh) -> fem.FunctionSpace:
    """Create P1 scalar function space."""
    import basix.ufl
    from dolfinx import fem

    cell_name = domain.topology.cell_name()
    element = basix.ufl.element("Lagrange", cell_name, 1)
    return fem.functionspace(domain, element)


def load_rho_from_checkpoint(
    run_dir: Path,
    comm,
    final_time: float | None = None,
) -> tuple[fem.Function, mesh.Mesh]:
    """Load density field rho from a run directory checkpoint.
    
    Args:
        run_dir: Path to simulation output directory.
        comm: MPI communicator.
        final_time: Time to load. If None, reads from config.json.
    
    Returns:
        Tuple of (rho function, mesh).
    
    Raises:
        FileNotFoundError: If checkpoint.bp not found.
    """
    checkpoint_path = run_dir / "checkpoint.bp"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Read final_time from config if not provided
    if final_time is None:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            final_time = cfg.get("time", {}).get("total_time", 0.0)
    
    from analysis.analysis_utils import load_checkpoint_mesh, load_checkpoint_function

    domain = load_checkpoint_mesh(checkpoint_path, comm)
    space = create_scalar_space(domain)
    rho = load_checkpoint_function(checkpoint_path, "rho", space, time=final_time)
    return rho, domain


# =============================================================================
# Solver Stats Loading
# =============================================================================

def load_solver_stats(run_dir: Path) -> dict[str, Any]:
    """Load solver statistics from steps.csv.
    
    Args:
        run_dir: Path to simulation output directory.
    
    Returns:
        Dictionary with aggregated solver statistics.
    """
    steps_csv = run_dir / "steps.csv"
    
    stats = {
        "total_picard_iters": 0,
        "n_steps": 0,
        "ksp_dens": 0,
        "ksp_stim": 0,
        "ksp_mech": 0,
        "ksp_fab": 0,
    }
    
    if not steps_csv.exists():
        print(f"  Warning: steps.csv not found in {run_dir}")
        return stats
    
    with open(steps_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only count accepted steps
            if int(row.get("accepted", 1)):
                stats["n_steps"] += 1
                stats["total_picard_iters"] += int(row.get("num_subiters", 0))
                stats["ksp_dens"] += int(row.get("dens_iters", 0))
                stats["ksp_stim"] += int(row.get("stim_iters", 0))
                stats["ksp_mech"] += int(row.get("mech_iters", 0))
                stats["ksp_fab"] += int(row.get("fab_iters", 0))
    
    return stats


# =============================================================================
# Quality Metrics Computation
# =============================================================================

def compute_quality_metrics(
    rho: fem.Function,
    record: dict[str, Any],
    solver_stats: dict[str, Any],
    comm,
) -> QualityMetrics:
    """Compute quality metrics for a density field.
    
    Uses UFL forms for mesh-integrated gradient norms.
    
    Args:
        rho: Density function.
        record: Sweep record with parameter values and run info.
        solver_stats: Aggregated solver statistics.
        comm: MPI communicator.
    
    Returns:
        QualityMetrics dataclass.
    """
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    domain = rho.function_space.mesh
    dx = ufl.Measure("dx", domain=domain)
    
    # Ensure ghost values are synchronized
    rho.x.scatter_forward()
    
    # Basic field statistics (local, then reduce)
    rho_array = rho.x.array[:rho.function_space.dofmap.index_map.size_local]
    
    local_min = float(np.min(rho_array)) if len(rho_array) > 0 else np.inf
    local_max = float(np.max(rho_array)) if len(rho_array) > 0 else -np.inf
    local_sum = float(np.sum(rho_array))
    local_count = len(rho_array)
    
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    
    rho_mean = global_sum / global_count if global_count > 0 else 0.0
    
    local_var_sum = float(np.sum((rho_array - rho_mean) ** 2))
    global_var_sum = comm.allreduce(local_var_sum, op=MPI.SUM)
    rho_std = np.sqrt(global_var_sum / global_count) if global_count > 0 else 0.0
    
    # Gradient-based metrics via UFL forms
    grad_rho = ufl.grad(rho)
    grad_mag = ufl.sqrt(ufl.inner(grad_rho, grad_rho) + 1e-12)  # Smooth abs
    
    # Total variation: L1 norm of gradient
    tv_form = fem.form(grad_mag * dx)
    tv_local = fem.assemble_scalar(tv_form)
    total_variation = comm.allreduce(float(tv_local), op=MPI.SUM)
    
    # Gradient L2 norm (H1 seminorm)
    grad_l2_sq_form = fem.form(ufl.inner(grad_rho, grad_rho) * dx)
    grad_l2_sq_local = fem.assemble_scalar(grad_l2_sq_form)
    gradient_l2 = np.sqrt(comm.allreduce(float(grad_l2_sq_local), op=MPI.SUM))
    
    # Domain volume for mean computation
    one = fem.Constant(domain, 1.0)
    vol_form = fem.form(one * dx)
    vol_local = fem.assemble_scalar(vol_form)
    volume = comm.allreduce(float(vol_local), op=MPI.SUM)
    
    gradient_mean = total_variation / volume if volume > 0 else 0.0
    
    # Approximate max gradient (via L^p norm with large p)
    p = 8  # Approximates L^inf
    grad_Lp_form = fem.form(ufl.inner(grad_rho, grad_rho) ** (p / 2) * dx)
    grad_Lp_local = fem.assemble_scalar(grad_Lp_form)
    grad_Lp = comm.allreduce(float(grad_Lp_local), op=MPI.SUM)
    gradient_max = (grad_Lp / volume) ** (1 / p) if volume > 0 else 0.0
    
    # Roughness ratio: relative gradient magnitude
    roughness_ratio = gradient_l2 / (rho_mean * np.sqrt(volume)) if rho_mean > 0 else 0.0
    
    # Extract solver statistics
    total_picard = solver_stats.get("total_picard_iters", 0)
    n_steps = max(solver_stats.get("n_steps", 1), 1)
    
    return QualityMetrics(
        run_hash=record.get("run_hash", ""),
        output_dir=record.get("output_dir", ""),
        D_rho=float(record.get("density.D_rho", 0.0)),
        stimulus_D=float(record.get("stimulus.stimulus_D", 0.0)),
        fabric_D=float(record.get("fabric.fabric_D", 0.0)),
        rho_min=global_min,
        rho_max=global_max,
        rho_mean=rho_mean,
        rho_std=rho_std,
        total_variation=total_variation,
        gradient_l2=gradient_l2,
        gradient_mean=gradient_mean,
        gradient_max=gradient_max,
        roughness_ratio=roughness_ratio,
        total_picard_iters=total_picard,
        avg_picard_per_step=total_picard / n_steps,
        total_ksp_iters_dens=solver_stats.get("ksp_dens", 0),
        total_ksp_iters_stim=solver_stats.get("ksp_stim", 0),
        total_ksp_iters_mech=solver_stats.get("ksp_mech", 0),
        total_ksp_iters_fab=solver_stats.get("ksp_fab", 0),
    )


# =============================================================================
# Analysis Pipeline
# =============================================================================

def analyze_sweep(
    sweep_dir: Path,
    comm,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze all runs in a diffusion sweep.
    
    Args:
        sweep_dir: Path to sweep output directory.
        comm: MPI communicator.
        verbose: Print progress.
    
    Returns:
        DataFrame with quality metrics for all runs.
    """
    # Load sweep records
    records = load_diffusion_sweep_records(sweep_dir, comm)
    if not records:
        raise ValueError(f"No sweep records found in {sweep_dir}")
    
    if verbose and comm.rank == 0:
        print(f"Found {len(records)} sweep runs")
    
    metrics_list = []
    
    for idx, record in enumerate(records, start=1):
        output_dir = record["output_dir"]
        run_dir = sweep_dir / output_dir
        
        if verbose and comm.rank == 0:
            D_rho = record.get("density.D_rho", "?")
            stim_D = record.get("stimulus.stimulus_D", "?")
            fab_D = record.get("fabric.fabric_D", "?")
            print(f"  [{idx}/{len(records)}] D_rho={D_rho}, stimulus_D={stim_D}, fabric_D={fab_D}")
        
        try:
            # Load rho from checkpoint
            rho, _ = load_rho_from_checkpoint(run_dir, comm)
            
            # Load solver stats
            solver_stats = load_solver_stats(run_dir)
            
            # Compute quality metrics
            metrics = compute_quality_metrics(rho, record, solver_stats, comm)
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
# Visualization (CMAME style from plot_utils.py)
# =============================================================================

from analysis.plot_utils import (
    apply_style,
    setup_axis_style,
    save_figure,
    save_manuscript_figure,
    PUBLICATION_DPI,
    PLOT_LINEWIDTH,
)

# Diffusion parameter styling (color-blind friendly)
DIFFUSION_COLORS = {
    "D_rho": "#DE8F05",      # Orange (density)
    "stimulus_D": "#029E73", # Green (stimulus)
    "fabric_D": "#CC78BC",   # Purple (fabric)
}
DIFFUSION_LABELS = {
    "D_rho": r"$D_\rho$",
    "stimulus_D": r"$D_S$",
    "fabric_D": r"$D_A$",
}
DIFFUSION_MARKERS = {
    "D_rho": "o",
    "stimulus_D": "s",
    "fabric_D": "^",
}

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
    """Create diagnostic figure for diffusion sweep.
    
    Layout (1×3):
      (a) Effect of density diffusion D_rho on smoothness and solver cost
      (b) Effect of stimulus diffusion D_S on smoothness and solver cost
      (c) Effect of fabric diffusion D_A on smoothness and solver cost
    
    Args:
        df: DataFrame with quality metrics.
        output_dir: Directory to save plots.
        metadata: Sweep metadata.
    """
    apply_style()
    
    # Get unique parameter values and baselines
    D_rho_vals = sorted(df["D_rho"].unique())
    stim_D_vals = sorted(df["stimulus_D"].unique())
    fab_D_vals = sorted(df["fabric_D"].unique())

    baseline_D_rho = float(metadata.get("baseline_D_rho", D_rho_vals[len(D_rho_vals) // 2]))
    baseline_stim_D = float(metadata.get("baseline_stimulus_D", stim_D_vals[len(stim_D_vals) // 2]))
    baseline_fab_D = float(metadata.get("baseline_fabric_D", fab_D_vals[len(fab_D_vals) // 2]))

    # Global y-limits for easy cross-panel comparison
    rough_min = float(df["roughness_ratio"].min())
    rough_max = float(df["roughness_ratio"].max())
    rough_pad = 0.05 * (rough_max - rough_min) if rough_max > rough_min else 0.0
    rough_ylim = (max(0.0, rough_min - rough_pad), rough_max + rough_pad)

    cost_min = float(df["avg_picard_per_step"].min())
    cost_max = float(df["avg_picard_per_step"].max())
    cost_pad = 0.05 * (cost_max - cost_min) if cost_max > cost_min else 0.0
    cost_ylim = (cost_min - cost_pad, cost_max + cost_pad)

    rng = np.random.default_rng(0)

    # Layout: 1×3 plots + a single legend row
    fig = plt.figure(figsize=(10.0, 3.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.18], hspace=0.25, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis("off")

    _plot_diffusion_impact_panel(
        ax0,
        df,
        param_name="D_rho",
        baseline=baseline_D_rho,
        title=r"(a) Density diffusion $D_\rho$",
        color=DIFFUSION_COLORS["D_rho"],
        rng=rng,
        rough_ylim=rough_ylim,
        cost_ylim=cost_ylim,
        show_ylabel=True,
        show_cost_ylabel=False,
    )
    _plot_diffusion_impact_panel(
        ax1,
        df,
        param_name="stimulus_D",
        baseline=baseline_stim_D,
        title=r"(b) Stimulus diffusion $D_S$",
        color=DIFFUSION_COLORS["stimulus_D"],
        rng=rng,
        rough_ylim=rough_ylim,
        cost_ylim=cost_ylim,
        show_ylabel=False,
        show_cost_ylabel=False,
    )
    _plot_diffusion_impact_panel(
        ax2,
        df,
        param_name="fabric_D",
        baseline=baseline_fab_D,
        title=r"(c) Fabric diffusion $D_A$",
        color=DIFFUSION_COLORS["fabric_D"],
        rng=rng,
        rough_ylim=rough_ylim,
        cost_ylim=cost_ylim,
        show_ylabel=False,
        show_cost_ylabel=True,
    )

    # Unified legend for all panels (single place, no repetition)
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="#555555", marker="o", linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=4,
               label="Roughness ratio (mean)"),
        Line2D([0], [0], color="black", marker="o", linestyle="--", linewidth=PLOT_LINEWIDTH, markersize=3,
               label="Picard iters/step (mean)"),
        Line2D([0], [0], color="#555555", marker="o", linestyle="", markersize=4, alpha=0.35,
               label="Individual runs"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, alpha=0.8,
               label=r"Baseline ($D=D_{\mathrm{ref}}$)"),
    ]
    ax_leg.legend(handles=legend_handles, loc="center", ncol=4, frameon=False)

    # Save to manuscript/images (required) and keep a copy in the sweep directory.
    save_manuscript_figure(fig, "diffusion_diagnostic", dpi=PUBLICATION_DPI, close=False)
    save_figure(fig, output_dir / "diffusion_diagnostic.png", dpi=PUBLICATION_DPI, close=True)


def _plot_diffusion_impact_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    param_name: str,
    baseline: float,
    title: str,
    color: str,
    rng: np.random.Generator,
    rough_ylim: tuple[float, float],
    cost_ylim: tuple[float, float],
    show_ylabel: bool,
    show_cost_ylabel: bool,
) -> None:
    """Plot marginal impact of a diffusion coefficient on (i) smoothness and (ii) solver cost.

    - Primary axis (left): roughness_ratio (scatter + mean)
    - Secondary axis (right): avg_picard_per_step (mean)
    """
    values = sorted(df[param_name].unique())
    factors = np.array([float(v) / baseline for v in values], dtype=float)

    rough_mean = np.zeros_like(factors)
    cost_mean = np.zeros_like(factors)

    # Scatter all combinations for this parameter level (jittered in log-x)
    for i, v in enumerate(values):
        subset = df[np.isclose(df[param_name], v, rtol=1e-12, atol=0.0)]

        rough_vals = subset["roughness_ratio"].to_numpy(dtype=float)
        cost_vals = subset["avg_picard_per_step"].to_numpy(dtype=float)

        rough_mean[i] = np.mean(rough_vals)
        cost_mean[i] = np.mean(cost_vals)

        # Multiplicative jitter in log space (stable and scale-invariant)
        jitter = 10.0 ** rng.uniform(-0.02, 0.02, size=rough_vals.size)
        ax.scatter(
            factors[i] * jitter,
            rough_vals,
            s=18,
            color=color,
            alpha=0.4,  # Slightly more opaque since we removed the band
            edgecolors="none",
            zorder=2,
        )

    # Mean line (roughness)
    ax.plot(
        factors,
        rough_mean,
        color=color,
        marker=DIFFUSION_MARKERS.get(param_name, "o"),
        linewidth=PLOT_LINEWIDTH,
        markersize=4,
        zorder=3,
    )

    # Secondary axis: solver cost (mean)
    ax2 = ax.twinx()
    ax2.plot(
        factors,
        cost_mean,
        color="black",
        linestyle="--",
        marker="o",
        linewidth=PLOT_LINEWIDTH,
        markersize=3,
        zorder=3,
    )
    ax2.grid(False)
    ax2.set_ylim(*cost_ylim)
    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel("Picard iters/step" if show_cost_ylabel else "")
    if not show_cost_ylabel:
        ax2.set_yticklabels([])

    # Baseline marker
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.6, linewidth=1.0, zorder=0)

    setup_axis_style(
        ax,
        xlabel=r"Factor $D/D_{\mathrm{ref}}$",
        ylabel="Roughness ratio" if show_ylabel else "",
        title=title,
        grid=True,
    )
    if not show_ylabel:
        ax.set_yticklabels([])

    ax.set_xscale("log")
    ax.set_xlim(0.08, 12.5)
    ax.set_xticks([0.1, 1.0, 10.0])
    ax.set_xticklabels(["0.1x", "1x", "10x"])
    ax.set_ylim(*rough_ylim)


def generate_report(
    df: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Generate summary report."""
    report_path = output_dir / "diffusion_report.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DIFFUSION REGULARIZATION SWEEP REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("SWEEP CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Element size h:        {metadata.get('element_size_mm', '?'):.3f} mm\n")
        f.write(f"Regularization length: {metadata.get('regularization_length_mm', '?'):.3f} mm\n")
        f.write(f"Alpha (ℓ/h):           {metadata.get('alpha', 1.5)}\n")
        f.write(f"Total runs:            {len(df)}\n\n")
        
        f.write("BASELINE DIFFUSIVITIES (ℓ = 1.5h)\n")
        f.write("-" * 40 + "\n")
        f.write(f"D_rho:      {metadata.get('baseline_D_rho', '?'):.4f} mm²/day\n")
        f.write(f"stimulus_D: {metadata.get('baseline_stimulus_D', '?'):.4f} mm²/day\n")
        f.write(f"fabric_D:   {metadata.get('baseline_fabric_D', '?'):.4f} mm²/day\n\n")
        
        f.write("QUALITY METRICS SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Roughness ratio:       min={df['roughness_ratio'].min():.4f}, max={df['roughness_ratio'].max():.4f}\n")
        f.write(f"Total variation:       min={df['total_variation'].min():.2f}, max={df['total_variation'].max():.2f}\n")
        f.write(f"Gradient L2:           min={df['gradient_l2'].min():.4f}, max={df['gradient_l2'].max():.4f}\n\n")
        
        f.write("SOLVER PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Picard iters/step:     min={df['avg_picard_per_step'].min():.1f}, max={df['avg_picard_per_step'].max():.1f}\n")
        f.write(f"Total Picard iters:    min={df['total_picard_iters'].min()}, max={df['total_picard_iters'].max()}\n\n")
        
        f.write("BEST CONFIGURATIONS\n")
        f.write("-" * 40 + "\n")
        
        # Best quality (smoothest)
        best_quality = df.loc[df["roughness_ratio"].idxmin()]
        f.write(f"Smoothest (lowest roughness):\n")
        f.write(f"  D_rho={best_quality['D_rho']:.4f}, stimulus_D={best_quality['stimulus_D']:.4f}, fabric_D={best_quality['fabric_D']:.4f}\n")
        f.write(f"  roughness={best_quality['roughness_ratio']:.4f}, picard={best_quality['avg_picard_per_step']:.1f}/step\n\n")
        
        # Fastest solver
        best_speed = df.loc[df["avg_picard_per_step"].idxmin()]
        f.write(f"Fastest (lowest iterations):\n")
        f.write(f"  D_rho={best_speed['D_rho']:.4f}, stimulus_D={best_speed['stimulus_D']:.4f}, fabric_D={best_speed['fabric_D']:.4f}\n")
        f.write(f"  roughness={best_speed['roughness_ratio']:.4f}, picard={best_speed['avg_picard_per_step']:.1f}/step\n\n")
        
    print(f"Saved report: {report_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run diffusion sweep analysis."""
    sweep_dir = Path("results/diffusion_sweep")
    
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        print("Run sweep first: mpirun -n 4 python run_diffusion_sweep.py")
        return
    
    print("=" * 70)
    print("DIFFUSION REGULARIZATION ANALYSIS")
    print("=" * 70)
    
    # Load metadata
    metadata_path = sweep_dir / "sweep_summary.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            sweep_json = json.load(f)
            metadata = sweep_json.get("metadata", {})

    # Prefer precomputed metrics (plot-only mode does not require MPI/DOLFINx).
    metrics_path = sweep_dir / "quality_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        comm_rank = 0
        print(f"\nLoaded precomputed metrics: {metrics_path}")
    else:
        # Analyze sweep - compute quality metrics from checkpoints (requires MPI + DOLFINx stack).
        try:
            from mpi4py import MPI
        except ModuleNotFoundError as e:
            print("Error: mpi4py is required to compute metrics from checkpoints.")
            print("Either install mpi4py and run via mpirun, or provide precomputed quality_metrics.csv.")
            print(f"Details: {e}")
            return

        comm = MPI.COMM_WORLD
        comm_rank = comm.rank

        if comm_rank == 0:
            print("\nComputing quality metrics from checkpoints...")
        df = analyze_sweep(sweep_dir, comm, verbose=(comm_rank == 0))

        if df.empty:
            if comm_rank == 0:
                print("Error: No metrics computed. Check checkpoints.")
            return

        # Save quality metrics
        if comm_rank == 0:
            df.to_csv(metrics_path, index=False)
            print(f"\nSaved quality metrics: {metrics_path}")
    
    # Generate visualizations (rank 0 only)
    if comm_rank == 0:
        print("\nGenerating visualizations...")
        try:
            create_diagnostic_figure(df, sweep_dir, metadata)
            generate_report(df, sweep_dir, metadata)
        except ImportError as e:
            print(f"Warning: Visualization skipped (missing matplotlib): {e}")
        except Exception as e:
            import traceback
            print(f"Warning: Visualization failed: {e}")
            traceback.print_exc()
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
