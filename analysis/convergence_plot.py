"""Plot convergence analysis results from convergence_data.xlsx.

Creates 4-subplot figure:
- Top row: Spatial convergence (L2 and H1 errors vs h)
- Bottom row: Temporal convergence (L2 and H1 errors vs dt)
All fields shown in each subplot with reference lines for O(h), O(h²), O(dt).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.plot_utils import (
    FIELD_NAMES, FIELD_LABELS, FIELD_COLORS, FIELD_MARKERS,
    REFERENCE_COLOR, REFERENCE_ALPHA, REFERENCE_LINESTYLE, REFERENCE_LINEWIDTH,
    estimate_convergence_order, add_reference_line, setup_axis_style,
    remove_all_legends, create_unified_legend, save_figure, print_banner,
)


# ============================================================================
# Helper functions
# ============================================================================


def load_spatial_data(xlsx_file: Path, dt_value: float) -> dict[str, pd.DataFrame]:
    """Load spatial convergence data for a specific dt."""
    data = {}
    for field in FIELD_NAMES:
        sheet_name = f"spatial_{field}_dt{dt_value}"
        try:
            df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
            data[field] = df
        except Exception as e:
            print(f"Warning: Could not load {sheet_name}: {e}")
    return data


def load_temporal_data(xlsx_file: Path, N_value: int) -> dict[str, pd.DataFrame]:
    """Load temporal convergence data for a specific N."""
    data = {}
    for field in FIELD_NAMES:
        sheet_name = f"temporal_{field}_N{N_value}"
        try:
            df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
            data[field] = df
        except Exception as e:
            print(f"Warning: Could not load {sheet_name}: {e}")
    return data


def plot_spatial_convergence(
    ax: plt.Axes,
    spatial_data: dict[str, pd.DataFrame],
    error_type: str,
    dt_value: float,
) -> None:
    """Plot spatial convergence (varying h, fixed dt)."""
    ylabel = r"$L^2$ error" if error_type == "L2_error" else r"$H^1$ error"
    title_prefix = "Spatial L²" if error_type == "L2_error" else "Spatial H¹"
    
    # Plot all fields
    for field in FIELD_NAMES:
        if field not in spatial_data:
            continue
        
        df = spatial_data[field]
        h = df["h"].values
        error = df[error_type].values
        
        order = estimate_convergence_order(h, error)
        label = f"{FIELD_LABELS[field]} (p={order:.2f})"
        
        ax.loglog(
            h, error,
            marker=FIELD_MARKERS[field],
            color=FIELD_COLORS[field],
            label=label,
            linewidth=2,
            markersize=8,
        )
    
    # Add reference lines
    if len(spatial_data) > 0:
        first_field = next(iter(spatial_data.values()))
        h_vals = first_field["h"].values
        h_range = (h_vals.min(), h_vals.max())
        
        # Scale reference lines to median error
        mid_errors = [df[error_type].iloc[len(df) // 2] for df in spatial_data.values()]
        ref_scale = np.median(mid_errors)
        
        add_reference_line(ax, h_range, 1.0, ref_scale, r"$O(h)$")
        add_reference_line(ax, h_range, 2.0, ref_scale, r"$O(h^2)$", linestyle=":")
    
    setup_axis_style(ax, r"Mesh size $h$", ylabel, 
                     f"{title_prefix} convergence (dt={dt_value} days)", loglog=True)


def plot_temporal_convergence(
    ax: plt.Axes,
    temporal_data: dict[str, pd.DataFrame],
    error_type: str,
    N_value: int,
) -> None:
    """Plot temporal convergence (varying dt, fixed N)."""
    ylabel = r"$L^2$ error" if error_type == "L2_error" else r"$H^1$ error"
    title_prefix = "Temporal L²" if error_type == "L2_error" else "Temporal H¹"
    
    # Plot all fields
    for field in FIELD_NAMES:
        if field not in temporal_data:
            continue
        
        df = temporal_data[field]
        dt = df["dt_days"].values
        error = df[error_type].values
        
        # Use finest temporal points (smallest dt, from start)
        order = estimate_convergence_order(dt, error, from_start=True)
        label = f"{FIELD_LABELS[field]} (p={order:.2f})"
        
        ax.loglog(
            dt, error,
            marker=FIELD_MARKERS[field],
            color=FIELD_COLORS[field],
            label=label,
            linewidth=2,
            markersize=8,
        )
    
    # Add reference line O(dt)
    if len(temporal_data) > 0:
        first_field = next(iter(temporal_data.values()))
        dt_vals = first_field["dt_days"].values
        dt_range = (dt_vals.min(), dt_vals.max())
        
        # Scale reference line to median error
        mid_errors = [df[error_type].iloc[len(df) // 2] for df in temporal_data.values()]
        ref_scale = np.median(mid_errors)
        
        add_reference_line(ax, dt_range, 1.0, ref_scale, r"$O(\Delta t)$")
    
    setup_axis_style(ax, r"Time step $\Delta t$ [days]", ylabel,
                     f"{title_prefix} convergence (N={N_value})", loglog=True)


# ============================================================================
# Main plotting
# ============================================================================

def create_convergence_figure(
    xlsx_file: Path,
    dt_spatial: float,
    N_temporal: int,
    output_file: Path,
) -> None:
    """Create 4-subplot convergence figure.
    
    Args:
        xlsx_file: Path to convergence_data.xlsx
        dt_spatial: dt value for spatial convergence plots
        N_temporal: N value for temporal convergence plots
        output_file: Output figure path
    """
    # Load data
    print(f"Loading spatial data (dt={dt_spatial})...")
    spatial_data = load_spatial_data(xlsx_file, dt_spatial)
    
    print(f"Loading temporal data (N={N_temporal})...")
    temporal_data = load_temporal_data(xlsx_file, N_temporal)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Convergence Analysis: Bone Remodeling Simulator",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    
    # Top row: Spatial convergence
    print("Plotting spatial L2 convergence...")
    plot_spatial_convergence(axes[0, 0], spatial_data, "L2_error", dt_spatial)
    
    print("Plotting spatial H1 convergence...")
    plot_spatial_convergence(axes[0, 1], spatial_data, "H1_error", dt_spatial)
    
    # Bottom row: Temporal convergence
    print("Plotting temporal L2 convergence...")
    plot_temporal_convergence(axes[1, 0], temporal_data, "L2_error", N_temporal)
    
    print("Plotting temporal H1 convergence...")
    plot_temporal_convergence(axes[1, 1], temporal_data, "H1_error", N_temporal)
    
    # Keep individual legends on each subplot to show both spatial and temporal orders
    # Just add them in a cleaner way
    
    # Style spatial legends (top row)
    for ax in axes[0, :]:
        legend = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        legend.set_title("Spatial", prop={"size": 9, "weight": "bold"})
    
    # Style temporal legends (bottom row)
    for ax in axes[1, :]:
        legend = ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        legend.set_title("Temporal", prop={"size": 9, "weight": "bold"})
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    save_figure(fig, output_file)
    print(f"✓ Convergence figure saved to {output_file}")


if __name__ == "__main__":
    xlsx_file = Path("analysis/convergence_analysis/convergence_data.xlsx")
    output_file = Path("manuscript/images/convergence_plot.png")
    
    if not xlsx_file.exists():
        print(f"ERROR: {xlsx_file} not found!")
        print("Run 'mpirun -np 8 python analysis/convergence_errors.py' first.")
        sys.exit(1)
    
    print_banner("CONVERGENCE PLOTTING")
    
    # Read first sheet to determine available parameters
    xl = pd.ExcelFile(xlsx_file)
    sheet_names = xl.sheet_names
    
    # Extract available dt and N values from sheet names
    dt_values = set()
    N_values = set()
    for sheet in sheet_names:
        if sheet.startswith("spatial_"):
            # Extract dt from "spatial_u_dt50.0"
            dt_str = sheet.split("_dt")[1]
            dt_values.add(float(dt_str))
        elif sheet.startswith("temporal_"):
            # Extract N from "temporal_u_N36"
            N_str = sheet.split("_N")[1]
            N_values.add(int(N_str))
    
    dt_values = sorted(dt_values)
    N_values = sorted(N_values)
    
    print(f"Available dt values: {dt_values}")
    print(f"Available N values: {N_values}")
    
    # Use middle values as defaults
    dt_spatial = dt_values[len(dt_values) // 2] if dt_values else 50.0
    N_temporal = N_values[len(N_values) // 2] if N_values else 36
    
    print(f"\nUsing dt={dt_spatial} for spatial plots")
    print(f"Using N={N_temporal} for temporal plots")
    print()
    
    # Create figure
    create_convergence_figure(xlsx_file, dt_spatial, N_temporal, output_file)
    
    print_banner("CONVERGENCE PLOTTING COMPLETE")
