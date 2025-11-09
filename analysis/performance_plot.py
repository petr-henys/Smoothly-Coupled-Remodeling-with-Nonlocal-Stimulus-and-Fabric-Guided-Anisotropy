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
    SUBSOLVER_COLORS, SUBSOLVER_MARKERS, SUBSOLVER_LABELS, SUBSOLVER_NAMES,
    DT_COLORS, DT_MARKERS,
    FIGSIZE_DOUBLE_COLUMN, PUBLICATION_DPI,
    PLOT_LINEWIDTH, PLOT_MARKERSIZE,
    setup_axis_style, save_figure, print_banner, add_subplot_legend,
    LEGEND_FONTSIZE,
)


# ============================================================================
# Configuration
# ============================================================================

# Fixed dt for all plots
FIXED_DT = 25.0

# Memory metric styling (grayscale for print)
MEMORY_METRICS = ["total", "max"]
MEMORY_LABELS = {
    "total": "Total (sum all ranks)",
    "max": "Maximum per rank",
}
MEMORY_COLORS = {
    "total": "#000000",  # Black
    "max": "#606060",    # Dark gray
}
MEMORY_MARKERS = {
    "total": "o",
    "max": "s",
}
MEMORY_LINESTYLES = {
    "total": "-",
    "max": "--",
}

def format_dt_label(dt: float) -> str:
    """Format dt value for legend labels."""
    if dt == int(dt):
        return r"$\Delta t = " + f"{int(dt)}" + r"$ days"
    else:
        return r"$\Delta t = " + f"{dt:.2f}" + r"$ days"

METRICS_CONFIG = {
    "subiterations": {
        "ylabel": r"Subiterations per timestep",
        "title": r"(a) Fixed-point iterations",
        "data_key": "num_subiters",
        "aggregate": "mean",
        "by_dt": True,
        "vary_both": True,  # Vary both N and dt
    },
    "ksp_iterations": {
        "ylabel": r"KSP iterations per timestep",
        "title": r"(b) Linear solver iterations ($\Delta t = {dt_value}$ days)",
        "data_keys": {
            "mech": "mech_iters",
            "stim": "stim_iters",
            "dens": "dens_iters",
            "dir": "dir_iters",
        },
        "aggregate": "sum",
        "by_subsolver": True,
    },
    "memory": {
        "ylabel": r"Memory [MB]",
        "title": r"(c) Memory consumption ($\Delta t = {dt_value}$ days)",
        "data_keys": {
            "total": "memory_mb",      # Sum across all ranks
            "max": "memory_mb_max",    # Maximum per rank
        },
        "aggregate": "mean",
        "by_memory_metric": True,
    },
    "walltime": {
        "ylabel": r"Walltime per timestep [s]",
        "title": r"(d) Computational cost ($\Delta t = {dt_value}$ days)",
        "data_keys": {
            "mech": "mech_time",
            "stim": "stim_time",
            "dens": "dens_time",
            "dir": "dir_time",
        },
        "aggregate": "sum",
        "by_subsolver": True,
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
    subsolver: str = None,
    memory_metric: str = None,
) -> float:
    """Extract and aggregate metric over all timesteps for a single run.
    
    Args:
        loader: SimulationLoader instance
        metric_config: Metric configuration dict
        subsolver: Subsolver name if by_subsolver=True
        memory_metric: Memory metric type if by_memory_metric=True
        
    Returns:
        Aggregated metric value (mean or sum over timesteps)
    """
    subiters_df = loader.get_subiterations_metrics()
    
    # Group by timestep
    grouped = subiters_df.groupby("step")
    
    aggregate = metric_config["aggregate"]
    by_subsolver = metric_config.get("by_subsolver", False)
    by_memory_metric = metric_config.get("by_memory_metric", False)
    
    # Extract metric per timestep
    if by_subsolver:
        # Per-subsolver metric
        data_keys = metric_config["data_keys"]
        data_key = data_keys[subsolver]
        
        per_step_values = []
        for step, group in grouped:
            if data_key in group.columns:
                step_val = group[data_key].sum()  # Sum over subiterations
                per_step_values.append(step_val)
    elif by_memory_metric:
        # Per-memory-metric
        data_keys = metric_config["data_keys"]
        data_key = data_keys[memory_metric]
        
        per_step_values = []
        for step, group in grouped:
            if data_key in group.columns:
                step_val = group[data_key].mean()  # Average within step
                per_step_values.append(step_val)
            else:
                return np.nan
    else:
        # Global metric
        data_key = metric_config["data_key"]
        
        if data_key == "num_subiters":
            # Special case: count subiterations per step
            per_step_values = [len(group) for _, group in grouped]
        elif data_key in subiters_df.columns:
            # Average within each step
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
    dt_value: float,
    N_values: list[int],
    dt_values: list[float],
) -> dict[str, pd.DataFrame]:
    """Collect performance metrics for all runs.
    
    Returns dictionary mapping metric names to DataFrames.
    For by_subsolver metrics: columns are N, total_dofs, subsolver, value
    For by_dt metrics: columns are N, total_dofs, dt_days, value
    For global metrics: columns are N, total_dofs, value
    """
    summary = sweep_loader.get_summary()
    
    results = {metric: [] for metric in METRICS_CONFIG.keys()}
    
    for _, row in summary.iterrows():
        N = int(row["N"])
        dt = float(row["dt_days"])
        
        loader = sweep_loader.get_loader(row["output_dir"])
        dofs_info = compute_dofs_per_field(N)
        
        # Extract each metric
        for metric_name, metric_config in METRICS_CONFIG.items():
            by_subsolver = metric_config.get("by_subsolver", False)
            by_memory_metric = metric_config.get("by_memory_metric", False)
            by_dt = metric_config.get("by_dt", False)
            vary_both = metric_config.get("vary_both", False)
            
            # Determine if we should collect this combination
            if by_dt and vary_both:
                # For by_dt plots that vary both: collect all N and dt combinations
                if N not in N_values or dt not in dt_values:
                    continue
            elif by_dt:
                # For by_dt plots with fixed N: use fixed N, vary dt
                fixed_N = metric_config.get("fixed_N")
                if N != fixed_N or dt not in dt_values:
                    continue
            else:
                # For other plots: use fixed dt, vary N
                if N not in N_values or dt != dt_value:
                    continue
            
            if by_subsolver:
                # Per-subsolver metrics
                for subsolver in SUBSOLVER_NAMES:
                    value = extract_per_timestep_metrics(loader, metric_config, subsolver=subsolver)
                    results[metric_name].append({
                        "N": N,
                        "total_dofs": dofs_info["total"],
                        "subsolver": subsolver,
                        "value": value,
                    })
            elif by_memory_metric:
                # Per-memory-metric
                for mem_metric in MEMORY_METRICS:
                    value = extract_per_timestep_metrics(loader, metric_config, memory_metric=mem_metric)
                    results[metric_name].append({
                        "N": N,
                        "total_dofs": dofs_info["total"],
                        "memory_metric": mem_metric,
                        "value": value,
                    })
            elif by_dt:
                # By dt metric
                value = extract_per_timestep_metrics(loader, metric_config)
                results[metric_name].append({
                    "N": N,
                    "total_dofs": dofs_info["total"],
                    "dt_days": dt,
                    "value": value,
                })
            else:
                # Global metrics
                value = extract_per_timestep_metrics(loader, metric_config)
                results[metric_name].append({
                    "N": N,
                    "total_dofs": dofs_info["total"],
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
) -> None:
    """Plot single performance metric vs total DOFs.
    
    For by_subsolver metrics, plots one line per subsolver.
    For by_memory_metric, plots one line per memory metric.
    For by_dt metrics, plots one line per dt value.
    For global metrics, plots single line.
    """
    by_subsolver = metric_config.get("by_subsolver", False)
    by_memory_metric = metric_config.get("by_memory_metric", False)
    by_dt = metric_config.get("by_dt", False)
    
    if by_subsolver:
        # Plot per subsolver
        for subsolver in SUBSOLVER_NAMES:
            subsolver_data = df[df["subsolver"] == subsolver].sort_values("N")
            
            if len(subsolver_data) == 0:
                continue
            
            ax.plot(
                subsolver_data["total_dofs"].values,
                subsolver_data["value"].values,
                marker=SUBSOLVER_MARKERS[subsolver],
                color=SUBSOLVER_COLORS[subsolver],
                label=SUBSOLVER_LABELS[subsolver],
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKERSIZE,
            )
    elif by_memory_metric:
        # Plot per memory metric
        for mem_metric in MEMORY_METRICS:
            mem_data = df[df["memory_metric"] == mem_metric].sort_values("N")
            
            if len(mem_data) == 0:
                continue
            
            ax.plot(
                mem_data["total_dofs"].values,
                mem_data["value"].values,
                marker=MEMORY_MARKERS[mem_metric],
                color=MEMORY_COLORS[mem_metric],
                label=MEMORY_LABELS[mem_metric],
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKERSIZE,
            )
    elif by_dt:
        # Plot per dt value (vs DOFs if vary_both, vs dt otherwise)
        vary_both = metric_config.get("vary_both", False)
        for dt in sorted(df["dt_days"].unique()):
            dt_data = df[df["dt_days"] == dt].sort_values("N" if vary_both else "dt_days")
            
            if len(dt_data) == 0:
                continue
            
            if vary_both:
                # Plot vs DOFs (varying N)
                x_vals = dt_data["total_dofs"].values
            else:
                # Plot vs dt
                x_vals = dt_data["dt_days"].values
            
            ax.plot(
                x_vals,
                dt_data["value"].values,
                marker=DT_MARKERS.get(dt, "o"),
                color=DT_COLORS.get(dt, "gray"),
                label=format_dt_label(dt),
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKERSIZE,
            )
    else:
        # Plot global metric (single line)
        df_sorted = df.sort_values("N")
        ax.plot(
            df_sorted["total_dofs"].values,
            df_sorted["value"].values,
            marker="o",
            color="C0",
            label="Total",
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
        )
    
    # Set axis labels and style
    by_dt = metric_config.get("by_dt", False)
    vary_both = metric_config.get("vary_both", False)
    
    if by_dt and not vary_both:
        xlabel = r"Time step $\Delta t$ [days]"
    else:
        xlabel = "Total DOFs"
    
    setup_axis_style(
        ax, xlabel, metric_config["ylabel"],
        metric_config["title"], loglog=True
    )


def create_performance_figure(
    sweep_loader: SweepLoader,
    dt_value: float,
    output_file: Path,
) -> None:
    """Create 4-subplot performance figure.
    
    Args:
        sweep_loader: SweepLoader instance
        dt_value: Fixed timestep size to use (for most plots)
        output_file: Output figure path
    """
    # Get available sweep parameters
    summary = sweep_loader.get_summary()
    N_values = sorted(summary["N"].unique())
    dt_values = sorted(summary["dt_days"].unique())
    
    print(f"Using dt={dt_value} days for most plots")
    print(f"Found {len(N_values)} mesh resolutions: {N_values}")
    print(f"Found {len(dt_values)} timestep sizes: {dt_values}")
    
    # Collect all performance data
    print("\nExtracting performance metrics from telemetry...")
    perf_data = collect_performance_data(sweep_loader, dt_value, N_values, dt_values)
    
    # Create figure with CMAME double-column width
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_DOUBLE_COLUMN)
    
    # Plot each metric
    metric_names = list(METRICS_CONFIG.keys())
    for idx, (metric_name, metric_config) in enumerate(METRICS_CONFIG.items()):
        row = idx // 2
        col = idx % 2
        
        print(f"Plotting {metric_name}...")
        df = perf_data[metric_name]
        
        # Format title with dt_value if needed
        title = metric_config["title"]
        if "{dt_value}" in title:
            title = title.format(dt_value=dt_value)
        
        # Temporarily update config with formatted title
        plot_config = metric_config.copy()
        plot_config["title"] = title
        
        plot_performance_metric(axes[row, col], df, plot_config)
        
        # Add individual legend to each subplot
        add_subplot_legend(axes[row, col], loc="upper left")
    
    plt.tight_layout()
    save_figure(fig, output_file, dpi=PUBLICATION_DPI)
    print(f"\n✓ Performance figure saved to {output_file}")
    
    # Print summary statistics
    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "=" * 80)
        print(f"PERFORMANCE SUMMARY (dt={dt_value} days)")
        print("=" * 80)
        
        for metric_name, df in perf_data.items():
            title = METRICS_CONFIG[metric_name]['title']
            if "{dt_value}" in title:
                title = title.format(dt_value=dt_value)
            print(f"\n{title}:")
            by_subsolver = METRICS_CONFIG[metric_name].get("by_subsolver", False)
            by_memory_metric = METRICS_CONFIG[metric_name].get("by_memory_metric", False)
            by_dt = METRICS_CONFIG[metric_name].get("by_dt", False)
            
            if by_dt:
                vary_both = METRICS_CONFIG[metric_name].get("vary_both", False)
                if vary_both:
                    # Show per-N and per-dt values
                    for dt in sorted(df["dt_days"].unique()):
                        print(f"  dt={dt:5.2f} days:")
                        dt_df = df[df["dt_days"] == dt]
                        for N in sorted(dt_df["N"].unique()):
                            N_data = dt_df[dt_df["N"] == N]
                            if len(N_data) > 0:
                                val = N_data["value"].values[0]
                                dofs = compute_dofs_per_field(N)["total"]
                                print(f"    N={N:2d} (DOFs={dofs:7d}): {val:.2f}")
                else:
                    # Show per-dt values only
                    for dt in sorted(df["dt_days"].unique()):
                        dt_data = df[df["dt_days"] == dt]
                        if len(dt_data) > 0:
                            val = dt_data["value"].values[0]
                            print(f"  dt={dt:5.2f} days: {val:.2f}")
            else:
                # Show per-N values
                for N in sorted(N_values):
                    N_data = df[df["N"] == N]
                    if len(N_data) == 0:
                        continue
                    
                    dofs = compute_dofs_per_field(N)["total"]
                    
                    if by_subsolver:
                        # Show per-subsolver values
                        subsolver_vals = []
                        for subsolver in SUBSOLVER_NAMES:
                            s_data = N_data[N_data["subsolver"] == subsolver]
                            if len(s_data) > 0:
                                subsolver_vals.append(f"{subsolver}={s_data['value'].values[0]:.2f}")
                        print(f"  N={N:2d} (DOFs={dofs:7d}): {', '.join(subsolver_vals)}")
                    elif by_memory_metric:
                        # Show per-memory-metric values
                        mem_vals = []
                        for mem_metric in MEMORY_METRICS:
                            m_data = N_data[N_data["memory_metric"] == mem_metric]
                            if len(m_data) > 0:
                                mem_vals.append(f"{mem_metric}={m_data['value'].values[0]:.2f}")
                        print(f"  N={N:2d} (DOFs={dofs:7d}): {', '.join(mem_vals)}")
                    else:
                        # Show single value
                        val = N_data["value"].values[0]
                        print(f"  N={N:2d} (DOFs={dofs:7d}): {val:.2f}")


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
    
    # Create figure using fixed dt
    if comm.rank == 0:
        create_performance_figure(sweep_loader, FIXED_DT, output_file)
        print_banner("PERFORMANCE PLOTTING COMPLETE")
