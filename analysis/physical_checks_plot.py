"""Plot physical conservation checks from convergence sweep.

Visualizes conservation/balance residuals for mechanics, stimulus, density, and direction
equations to verify physical correctness across different timestep sizes.
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
    setup_axis_style, save_figure, print_banner,
    DT_COLORS, DT_MARKERS, DT_LINESTYLES,
    SUBSOLVER_COLORS,
    FIGSIZE_TALL, PUBLICATION_DPI,
    PLOT_LINEWIDTH, PLOT_MARKERSIZE,
    LEGEND_FONTSIZE, LEGEND_FRAMEALPHA, LEGEND_EDGECOLOR, LEGEND_FANCYBOX,
    add_subplot_legend,
)


# ============================================================================
# Configuration
# ============================================================================

# Fixed N for physical checks comparison
FIXED_N = 36

def format_dt_label(dt: float) -> str:
    """Format dt value for legend labels."""
    if dt == int(dt):
        return r"$\Delta t = " + f"{int(dt)}" + r"$ days"
    else:
        return r"$\Delta t = " + f"{dt:.2f}" + r"$ days"


# Color intensity modulation for different dt values (lighter to darker)
# Maps dt to alpha value for color intensity
DT_ALPHAS = {
    6.25: 0.4,   # Lightest
    12.5: 0.5,
    25.0: 0.7,
    50.0: 0.85,
    100.0: 1.0,  # Darkest (full intensity)
}

# Physical checks configuration
PHYSICAL_CHECKS = {
    "energy": {
        "metrics": ["energy_Wint", "energy_Wext"],
        "labels": [r"$W_{\mathrm{int}}$", r"$W_{\mathrm{ext}}$"],
        "ylabel": "Energy [N·mm]",
        "title": r"(a) Mechanical energy balance",
        "loglog": False,
        "subsolver": "mech",  # Mechanics (blue)
    },
    "energy_residual": {
        "metrics": ["energy_res_rel"],
        "labels": ["Relative residual"],
        "ylabel": "Relative residual",
        "title": r"(b) Mechanical energy error",
        "loglog": True,
        "subsolver": "mech",  # Mechanics (blue)
    },
    "power": {
        "metrics": ["power_res_abs", "power_res_rel"],
        "labels": ["Absolute", "Relative"],
        "ylabel": "Power residual",
        "title": r"(c) Stimulus power balance error",
        "loglog": True,
        "subsolver": "stim",  # Stimulus (green)
    },
    "mass": {
        "metrics": ["mass_res_abs", "mass_res_rel"],
        "labels": ["Absolute", "Relative"],
        "ylabel": "Mass residual",
        "title": r"(d) Density mass balance error",
        "loglog": True,
        "subsolver": "dens",  # Density (orange)
    },
    "trace": {
        "metrics": ["trace_A_avg", "trace_Mhat_avg"],
        "labels": [r"$\mathrm{tr}(\mathbf{A})$", r"$\mathrm{tr}(\hat{\mathbf{M}})$"],
        "ylabel": "Trace value",
        "title": r"(e) Direction trace conservation",
        "loglog": False,
        "subsolver": "dir",  # Direction (purple)
    },
    "trace_residual": {
        "metrics": ["trace_res"],
        "labels": ["Trace residual"],
        "ylabel": "Trace residual",
        "title": r"(f) Direction trace error",
        "loglog": True,
        "subsolver": "dir",  # Direction (purple)
    },
}


# ============================================================================
# Data extraction
# ============================================================================

def collect_physical_checks(
    sweep_loader: SweepLoader,
    N_value: int,
    dt_values: list[float],
) -> dict[str, pd.DataFrame]:
    """Collect physical check metrics for fixed N, varying dt.
    
    Returns dictionary mapping metric names to DataFrames with columns:
    dt_days, step, time_days, metric_value
    """
    summary = sweep_loader.get_summary()
    
    results = {check: [] for check in PHYSICAL_CHECKS.keys()}
    
    for _, row in summary.iterrows():
        N = int(row["N"])
        dt = float(row["dt_days"])
        
        # Only use specified N and dt values
        if N != N_value or dt not in dt_values:
            continue
        
        loader = sweep_loader.get_loader(row["output_dir"])
        
        # Load subiterations metrics (contains physical checks)
        subiters_df = loader.get_subiterations_metrics()
        
        # For each check type, extract relevant metrics
        for check_name, check_config in PHYSICAL_CHECKS.items():
            metric_names = check_config["metrics"]
            
            # Group by timestep and get last subiteration (converged values)
            grouped = subiters_df.groupby("step")
            
            for step, group in grouped:
                # Take last subiteration (converged state)
                last_iter = group.iloc[-1]
                
                for metric_name in metric_names:
                    if metric_name in last_iter:
                        results[check_name].append({
                            "dt_days": dt,
                            "step": int(step),
                            "time_days": float(last_iter.get("time_days", np.nan)),
                            "metric": metric_name,
                            "value": float(last_iter[metric_name]),
                        })
    
    # Convert to DataFrames
    return {
        check: pd.DataFrame(data) for check, data in results.items()
    }


# ============================================================================
# Plotting
# ============================================================================

def plot_physical_check(
    ax: plt.Axes,
    df: pd.DataFrame,
    check_config: dict,
    dt_values: list[float],
    is_bottom_row: bool = False,
) -> None:
    """Plot physical check metric vs time for different dt values."""
    metric_names = check_config["metrics"]
    metric_labels = check_config["labels"]
    subsolver = check_config["subsolver"]
    base_color = SUBSOLVER_COLORS[subsolver]
    
    # Plot each dt value with varying intensity
    for dt in sorted(dt_values):
        dt_data = df[df["dt_days"] == dt].copy()
        
        if len(dt_data) == 0:
            continue
        
        # Sort by time
        dt_data = dt_data.sort_values("time_days")
        
        # Get alpha for this dt (intensity variation)
        alpha = DT_ALPHAS.get(dt, 0.7)
        
        # If multiple metrics, plot them with different line styles
        if len(metric_names) == 1:
            # Single metric - one line per dt
            metric_data = dt_data[dt_data["metric"] == metric_names[0]]
            ax.plot(
                metric_data["time_days"].values,
                metric_data["value"].values,
                marker=DT_MARKERS.get(dt, "o"),
                color=base_color,
                alpha=alpha,
                label=format_dt_label(dt),
                linestyle=DT_LINESTYLES.get(dt, "-"),
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKERSIZE,
                markevery=max(1, len(metric_data) // 10),  # Show fewer markers
            )
        else:
            # Multiple metrics - plot all for each dt
            for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
                metric_data = dt_data[dt_data["metric"] == metric_name]
                
                if len(metric_data) == 0:
                    continue
                
                # Use solid line for first metric, dashed for second
                linestyle = "-" if idx == 0 else "--"
                label = f"{format_dt_label(dt)} ({metric_label})"
                
                ax.plot(
                    metric_data["time_days"].values,
                    metric_data["value"].values,
                    marker=DT_MARKERS.get(dt, "o"),
                    color=base_color,
                    alpha=alpha,
                    label=label,
                    linewidth=PLOT_LINEWIDTH,
                    linestyle=linestyle,
                    markersize=PLOT_MARKERSIZE,
                    markevery=max(1, len(metric_data) // 10),
                )
    
    # Set axis style
    if check_config["loglog"]:
        ax.set_yscale("log")
    
    # Only show x-axis label on bottom row
    xlabel = "Time [days]" if is_bottom_row else ""
    setup_axis_style(
        ax, xlabel, check_config["ylabel"],
        check_config["title"], loglog=False, grid=True
    )


def create_physical_checks_figure(
    sweep_loader: SweepLoader,
    N_value: int,
    output_file: Path,
) -> None:
    """Create 6-subplot physical checks figure.
    
    Args:
        sweep_loader: SweepLoader instance
        N_value: Fixed mesh resolution to use
        output_file: Output figure path
    """
    # Get available sweep parameters
    summary = sweep_loader.get_summary()
    dt_values = sorted(summary["dt_days"].unique())
    
    print(f"Using N={N_value}")
    print(f"Found {len(dt_values)} timestep sizes: {dt_values}")
    
    # Collect physical check data
    print("\nExtracting physical checks from telemetry...")
    physical_data = collect_physical_checks(sweep_loader, N_value, dt_values)
    
    # Create figure (2x3 grid) with equal aspect ratio subplots and shared x-axis
    # Each subplot: 2.5" wide × 2.0" tall → total: 7.5" × 4.0"
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.0), sharex=True)
    
    # Plot each check
    check_names = list(PHYSICAL_CHECKS.keys())
    for idx, (check_name, check_config) in enumerate(PHYSICAL_CHECKS.items()):
        row = idx // 3
        col = idx % 3
        is_bottom_row = (row == 1)  # Bottom row for 2x3 grid
        
        print(f"Plotting {check_name}...")
        df = physical_data[check_name]
        
        if len(df) > 0:
            plot_physical_check(axes[row, col], df, check_config, dt_values, is_bottom_row)
        else:
            axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
            axes[row, col].set_title(check_config["title"])
    
    # Create unified legend with neutral grayscale colors matching dt intensities
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_handles = []
    legend_labels = []
    
    # Add dt values
    for dt in sorted(dt_values):
        alpha = DT_ALPHAS.get(dt, 0.7)
        # Use neutral gray color with varying intensity
        handle = Line2D([0], [0], 
                       color='gray', 
                       alpha=alpha,
                       linewidth=PLOT_LINEWIDTH * 1.5,
                       marker=DT_MARKERS.get(dt, "o"),
                       markersize=PLOT_MARKERSIZE * 1.2,
                       linestyle=DT_LINESTYLES.get(dt, "-"))
        legend_handles.append(handle)
        legend_labels.append(format_dt_label(dt))
    
    # Add separator
    legend_handles.append(Patch(facecolor='none', edgecolor='none'))
    legend_labels.append('')
    
    # Add line style indicators for metrics with absolute/relative
    legend_handles.append(Line2D([0], [0], color='black', linewidth=PLOT_LINEWIDTH, linestyle='-'))
    legend_labels.append('Absolute')
    legend_handles.append(Line2D([0], [0], color='black', linewidth=PLOT_LINEWIDTH, linestyle='--'))
    legend_labels.append('Relative')
    
    # Apply tight_layout first, then adjust for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend (increased from 0.12)
    
    # Add unified legend below all subplots (outside plot area)
    fig.legend(
        legend_handles, legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(dt_values) + 3,  # dt values + separator + 2 line styles
        fontsize=LEGEND_FONTSIZE,
        framealpha=LEGEND_FRAMEALPHA,
        frameon=True,
        edgecolor=LEGEND_EDGECOLOR,
        fancybox=LEGEND_FANCYBOX,
    )
    save_figure(fig, output_file, dpi=PUBLICATION_DPI)
    print(f"\n✓ Physical checks figure saved to {output_file}")
    
    # Print summary statistics
    if MPI.COMM_WORLD.rank == 0:
        print("\n" + "=" * 80)
        print(f"PHYSICAL CHECKS SUMMARY (N={N_value})")
        print("=" * 80)
        
        for check_name, df in physical_data.items():
            if len(df) == 0:
                continue
            
            print(f"\n{PHYSICAL_CHECKS[check_name]['title']}:")
            
            for dt in sorted(dt_values):
                dt_data = df[df["dt_days"] == dt]
                if len(dt_data) == 0:
                    continue
                
                print(f"  dt={dt:5.2f} days:")
                
                for metric in PHYSICAL_CHECKS[check_name]["metrics"]:
                    metric_data = dt_data[dt_data["metric"] == metric]
                    if len(metric_data) > 0:
                        values = metric_data["value"].values
                        mean_val = np.mean(values)
                        max_val = np.max(np.abs(values))
                        min_val = np.min(values)
                        print(f"    {metric}: mean={mean_val:.3e}, "
                              f"min={min_val:.3e}, max={max_val:.3e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    sweep_dir = Path("results/convergence_sweep")
    output_file = Path("manuscript/images/physical_checks_plot.png")
    
    if not sweep_dir.exists():
        if comm.rank == 0:
            print(f"ERROR: Sweep directory not found: {sweep_dir}")
            print("Run 'mpirun -np 8 python analysis/run_convergence.py' first.")
        sys.exit(1)
    
    if comm.rank == 0:
        print_banner("PHYSICAL CHECKS PLOTTING")
    
    # Load sweep
    sweep_loader = SweepLoader(sweep_dir, comm, verbose=(comm.rank == 0))
    
    # Create figure
    if comm.rank == 0:
        create_physical_checks_figure(sweep_loader, FIXED_N, output_file)
        print_banner("PHYSICAL CHECKS PLOTTING COMPLETE")
