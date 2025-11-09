"""Plot algorithm performance metrics from convergence sweep.

Analyzes per-timestep telemetry data from run_convergence.py sweep and plots
key performance indicators vs DOFs to demonstrate scalability.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI

from postprocessor import SweepLoader
from analysis.plot_utils import (
    DT_COLORS, DT_MARKERS, setup_axis_style, remove_all_legends,
    create_unified_legend, save_figure, print_banner, format_dt_label,
)


# ============================================================================
# Configuration
# ============================================================================

METRICS_CONFIG = {
    "subiterations": {
        "ylabel": "Subiterations per step",
        "title": "Fixed-Point Iterations",
        "data_key": "num_subiters",
        "aggregate": "mean",
    },
    "ksp_iterations": {
        "ylabel": "Total KSP iterations per step",
        "title": "Linear Solver Iterations",
        "data_key": ["mech_iters", "stim_iters", "dens_iters", "dir_iters"],
        "aggregate": "sum",
    },
    "memory": {
        "ylabel": "Memory [MB]",
        "title": "Memory Consumption",
        "data_key": "memory_mb",
        "aggregate": "mean",
    },
    "walltime": {
        "ylabel": "Wall time [s]",
        "title": "Wall-Clock Time",
        "data_key": ["mech_time", "stim_time", "dens_time", "dir_time"],
        "aggregate": "sum",
    },
}


# ============================================================================
# Data extraction
# ============================================================================

def compute_dofs_per_field(N: int) -> dict[str, int]:
    """Compute number of DOFs for each field type on N×N×N mesh.
    
    Args:
        N: Mesh resolution (cells per dimension)
        
    Returns:
        Dictionary with DOFs per field and total
    """
    # P1 elements on unit cube with N×N×N hexahedral cells
    num_vertices = (N + 1) ** 3
    
    return {
        "u": 3 * num_vertices,      # Vector (3 components)
        "rho": num_vertices,         # Scalar
        "S": num_vertices,           # Scalar
        "A": 9 * num_vertices,       # Tensor (3×3 components)
        "total": 14 * num_vertices,  # Total system DOFs
    }


def extract_per_timestep_metrics(
    loader,
    metric_config: dict,
) -> float:
    """Extract and aggregate metric over all timesteps for a single run.
    
    Args:
        loader: SimulationLoader instance
        metric_config: Metric configuration dict
        
    Returns:
        Aggregated metric value (mean or sum over timesteps)
    """
    subiters_df = loader.get_subiterations_metrics()
    
    # Group by timestep
    grouped = subiters_df.groupby("step")
    
    data_key = metric_config["data_key"]
    aggregate = metric_config["aggregate"]
    
    # Extract metric per timestep
    if isinstance(data_key, list):
        # Sum multiple columns first, then aggregate over steps
        per_step_values = []
        for step, group in grouped:
            step_sum = sum(group[key].sum() for key in data_key if key in group.columns)
            per_step_values.append(step_sum)
    else:
        # Single column - aggregate per step first
        if data_key == "num_subiters":
            # Special case: count subiterations per step
            per_step_values = [len(group) for _, group in grouped]
        elif data_key in subiters_df.columns:
            # Average within each step, then aggregate over steps
            per_step_values = [group[data_key].mean() for _, group in grouped]
        else:
            return np.nan
    
    # Aggregate over all timesteps
    if aggregate == "mean":
        return np.mean(per_step_values)
    elif aggregate == "sum":
        return np.sum(per_step_values)
    else:
        raise ValueError(f"Unknown aggregate type: {aggregate}")


def collect_performance_data(
    sweep_loader: SweepLoader,
    dt_values: list[float],
    N_values: list[int],
) -> dict[str, pd.DataFrame]:
    """Collect performance metrics for all runs in sweep.
    
    Returns dictionary mapping metric names to DataFrames with columns:
    N, total_dofs, dt_days, metric_value
    """
    summary = sweep_loader.get_summary()
    
    results = {metric: [] for metric in METRICS_CONFIG.keys()}
    
    for _, row in summary.iterrows():
        N = int(row["N"])
        dt = float(row["dt_days"])
        
        # Skip if not in requested ranges
        if N not in N_values or dt not in dt_values:
            continue
        
        loader = sweep_loader.get_loader(row["output_dir"])
        dofs_info = compute_dofs_per_field(N)
        
        # Extract each metric
        for metric_name, metric_config in METRICS_CONFIG.items():
            value = extract_per_timestep_metrics(loader, metric_config)
            
            results[metric_name].append({
                "N": N,
                "total_dofs": dofs_info["total"],
                "dt_days": dt,
                "value": value,
            })
    
    # Convert to DataFrames
    return {
        metric: pd.DataFrame(data) for metric, data in results.items()
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_performance_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_config: dict,
    dt_values: list[float],
) -> None:
    """Plot single performance metric vs total DOFs."""
    for dt in sorted(dt_values):
        dt_data = df[df["dt_days"] == dt].sort_values("N")
        
        if len(dt_data) == 0:
            continue
        
        ax.plot(
            dt_data["total_dofs"].values,
            dt_data["value"].values,
            marker=DT_MARKERS.get(dt, "o"),
            color=DT_COLORS.get(dt, "gray"),
            label=format_dt_label(dt),
            linewidth=2,
            markersize=8,
        )
    
    setup_axis_style(
        ax, "Total DOFs", metric_config["ylabel"],
        metric_config["title"], loglog=True
    )


def create_performance_figure(
    sweep_loader: SweepLoader,
    output_file: Path,
) -> None:
    """Create 4-subplot performance figure.
    
    Args:
        sweep_loader: SweepLoader instance
        output_file: Output figure path
    """
    # Get available sweep parameters
    summary = sweep_loader.get_summary()
    dt_values = sorted(summary["dt_days"].unique())
    N_values = sorted(summary["N"].unique())
    
    print(f"Found {len(N_values)} mesh resolutions: {N_values}")
    print(f"Found {len(dt_values)} timestep sizes: {dt_values}")
    
    # Collect all performance data
    print("\nExtracting performance metrics from telemetry...")
    perf_data = collect_performance_data(sweep_loader, dt_values, N_values)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Algorithm Performance: Bone Remodeling Simulator",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
    # Plot each metric
    metric_names = list(METRICS_CONFIG.keys())
    for idx, (metric_name, metric_config) in enumerate(METRICS_CONFIG.items()):
        row = idx // 2
        col = idx % 2
        
        print(f"Plotting {metric_name}...")
        df = perf_data[metric_name]
        plot_performance_metric(axes[row, col], df, metric_config, dt_values)
    
    # Create unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    remove_all_legends(axes)
    create_unified_legend(fig, handles, labels, ncol=len(dt_values))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    save_figure(fig, output_file)
    print(f"\n✓ Performance figure saved to {output_file}")
    
    # Print summary statistics
    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        for metric_name, df in perf_data.items():
            print(f"\n{METRICS_CONFIG[metric_name]['title']}:")
            
            # Group by N and show range across dt values
            for N in sorted(N_values):
                N_data = df[df["N"] == N]
                if len(N_data) > 0:
                    min_val = N_data["value"].min()
                    max_val = N_data["value"].max()
                    mean_val = N_data["value"].mean()
                    dofs = compute_dofs_per_field(N)["total"]
                    print(f"  N={N:2d} (DOFs={dofs:7d}): "
                          f"mean={mean_val:8.2f}, range=[{min_val:8.2f}, {max_val:8.2f}]")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    sweep_dir = Path("results/convergence_sweep")
    output_file = Path("manuscript/images/performance_plot.png")
    
    if not sweep_dir.exists():
        if comm.rank == 0:
            print(f"ERROR: Sweep directory not found: {sweep_dir}")
            print("Run 'mpirun -np 8 python analysis/run_convergence.py' first.")
        sys.exit(1)
    
    if comm.rank == 0:
        print_banner("PERFORMANCE PLOTTING")
    
    # Load sweep
    sweep_loader = SweepLoader(sweep_dir, comm, verbose=(comm.rank == 0))
    
    # Create figure
    if comm.rank == 0:
        create_performance_figure(sweep_loader, output_file)
        print_banner("PERFORMANCE PLOTTING COMPLETE")
