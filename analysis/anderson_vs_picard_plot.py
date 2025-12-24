"""Compare Anderson acceleration vs Picard iteration convergence rates.

Reads sweep output from run_anderson_sweep.py and generates publication plots.

Usage:
    python analysis/anderson_vs_picard_plot.py

Input:
    results/anderson_sweep/
    ├── sweep_summary.csv
    └── <hash>/
        ├── steps.csv
        └── subiterations.csv

Output:
    manuscript/images/anderson_vs_picard.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from postprocessor import SweepLoader
from analysis.plot_utils import (
    FIGSIZE_FULL_WIDTH,
    LEGEND_EDGECOLOR,
    LEGEND_FONTSIZE,
    LEGEND_FRAMEALPHA,
    PLOT_ALPHA_OVERLAY,
    PLOT_LINEWIDTH,
    PUBLICATION_DPI,
    print_banner,
    save_figure,
    setup_axis_style,
)


# Acceleration type styling (grayscale for print)
ACCEL_COLORS = {
    "picard": "#000000",    # Black
    "anderson": "#0173B2",  # Blue (colorblind-friendly)
}
ACCEL_LINESTYLES = {
    "picard": "-",
    "anderson": "--",
}
ACCEL_LABELS = {
    "picard": "Picard",
    "anderson": "Anderson",
}


def plot_convergence_curves(
    ax: plt.Axes,
    df: pd.DataFrame,
    color: str,
    linestyle: str,
    alpha: float = PLOT_ALPHA_OVERLAY,
) -> None:
    """Plot convergence curves (residual vs subiteration) for all timesteps.
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with columns: step, iter, proj_res.
        color: Line color.
        linestyle: Line style.
        alpha: Line transparency (for overlapping curves).
    """
    for step in sorted(df["step"].unique()):
        step_data = df[df["step"] == step].sort_values("iter")
        ax.plot(
            step_data["iter"].values,
            step_data["proj_res"].values,
            linewidth=PLOT_LINEWIDTH * 0.8,
            alpha=alpha,
            color=color,
            linestyle=linestyle,
        )


def plot_subiter_statistics(
    ax: plt.Axes,
    df_steps: pd.DataFrame,
    color: str,
    marker: str,
    label: str,
) -> None:
    """Plot subiteration count per timestep.
    
    Args:
        ax: Matplotlib axis.
        df_steps: DataFrame with columns: step, subiters.
        color: Marker/line color.
        marker: Marker style.
        label: Legend label.
    """
    ax.plot(
        df_steps["step"].values,
        df_steps["subiters"].values,
        marker=marker,
        color=color,
        linewidth=PLOT_LINEWIDTH,
        markersize=4,
        label=label,
    )


def main() -> None:
    """Generate Anderson vs Picard comparison plots."""
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        print_banner("ANDERSON VS PICARD COMPARISON")
    
    # Load sweep data
    sweep_dir = Path("results/anderson_sweep")
    if not sweep_dir.exists():
        if comm.rank == 0:
            print(f"Error: Sweep directory not found: {sweep_dir}")
            print("Run: mpirun -n 4 python run_anderson_sweep.py")
        return
    
    simulations = SweepLoader(str(sweep_dir), comm)
    summary = simulations.get_summary()
    
    if summary.empty:
        if comm.rank == 0:
            print("Error: No simulation data found.")
        return
    
    # Organize data by dt and accel_type
    data_by_dt: dict[float, dict[str, dict]] = {}
    for _, row in summary.iterrows():
        dt = float(row["dt_days"])
        accel = str(row["accel_type"])
        loader = simulations.get_loader(row["output_dir"])
        
        if dt not in data_by_dt:
            data_by_dt[dt] = {}
        
        data_by_dt[dt][accel] = {
            "subiter": loader.get_subiterations_metrics(),
            "steps": loader.get_step_metrics(),
        }
    
    # Only rank 0 plots
    if comm.rank != 0:
        return
    
    dt_values = sorted(data_by_dt.keys())
    n_dt = len(dt_values)
    
    # Layout: 2 rows x n_dt cols
    # Row 1: Convergence curves (residual vs iteration)
    # Row 2: Subiterations per timestep
    fig = plt.figure(figsize=(min(3.5 * n_dt, FIGSIZE_FULL_WIDTH[0]), 5.5))
    gs = GridSpec(
        3, n_dt,
        figure=fig,
        height_ratios=[1.0, 0.08, 1.0],  # plots, legend gap, plots
        hspace=0.35,
        wspace=0.35,
    )
    
    subplot_labels = [chr(97 + i) for i in range(26)]  # a, b, c, ...
    
    # Row 1: Convergence curves
    for idx, dt in enumerate(dt_values):
        ax = fig.add_subplot(gs[0, idx])
        
        for accel_type in ["picard", "anderson"]:
            if accel_type in data_by_dt[dt]:
                df = data_by_dt[dt][accel_type]["subiter"]
                if df is not None and not df.empty:
                    plot_convergence_curves(
                        ax, df,
                        color=ACCEL_COLORS[accel_type],
                        linestyle=ACCEL_LINESTYLES[accel_type],
                    )
        
        label = f"({subplot_labels[idx]})"
        setup_axis_style(
            ax,
            xlabel="Subiteration" if idx == n_dt // 2 else "",
            ylabel="Residual" if idx == 0 else "",
            title=rf"{label} $\Delta t = {dt:.0f}$ days",
            loglog=False,
            grid=True,
        )
        ax.set_yscale("log")
        
        # Consistent y-limits across panels
        ax.set_ylim(1e-8, 1e1)
    
    # Row 2: Subiterations per step
    for idx, dt in enumerate(dt_values):
        ax = fig.add_subplot(gs[2, idx])
        
        for accel_type in ["picard", "anderson"]:
            if accel_type in data_by_dt[dt]:
                df = data_by_dt[dt][accel_type]["steps"]
                if df is not None and not df.empty:
                    plot_subiter_statistics(
                        ax, df,
                        color=ACCEL_COLORS[accel_type],
                        marker="o" if accel_type == "picard" else "s",
                        label=ACCEL_LABELS[accel_type],
                    )
        
        label = f"({subplot_labels[n_dt + idx]})"
        setup_axis_style(
            ax,
            xlabel="Timestep" if idx == n_dt // 2 else "",
            ylabel="Subiterations" if idx == 0 else "",
            title=rf"{label} $\Delta t = {dt:.0f}$ days",
            loglog=False,
            grid=True,
        )
    
    # Create shared legend below the first row
    legend_handles = [
        plt.Line2D([0], [0], color=ACCEL_COLORS["picard"],
                   linestyle=ACCEL_LINESTYLES["picard"],
                   linewidth=PLOT_LINEWIDTH, label="Picard"),
        plt.Line2D([0], [0], color=ACCEL_COLORS["anderson"],
                   linestyle=ACCEL_LINESTYLES["anderson"],
                   linewidth=PLOT_LINEWIDTH, label="Anderson"),
    ]
    
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.52),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        framealpha=LEGEND_FRAMEALPHA,
        edgecolor=LEGEND_EDGECOLOR,
        frameon=True,
        fancybox=False,
    )
    
    # Save figure
    output_path = Path("manuscript/images/anderson_vs_picard.png")
    save_figure(fig, output_path, dpi=PUBLICATION_DPI)
    
    print_banner("ANDERSON VS PICARD PLOTTING COMPLETE")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

