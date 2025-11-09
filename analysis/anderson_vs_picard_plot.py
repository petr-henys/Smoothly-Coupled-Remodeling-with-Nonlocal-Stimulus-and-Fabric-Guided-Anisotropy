"""Compare Anderson acceleration vs Picard iteration convergence rates."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from mpi4py import MPI

from postprocessor import SweepLoader
from analysis.plot_utils import (
    setup_axis_style, save_figure, print_banner,
    PUBLICATION_DPI, PLOT_LINEWIDTH, PLOT_ALPHA_OVERLAY,
    add_subplot_legend,
)


def plot_convergence_curves(ax, df, color, linestyle):
    """Plot all convergence curves from a dataframe."""
    for step in sorted(df["step"].unique()):
        step_data = df[df["step"] == step].sort_values("iter")
        ax.plot(
            step_data["iter"].values,
            step_data["proj_res"].values,
            linewidth=PLOT_LINEWIDTH * 0.8,  # Slightly thinner for overlapping
            alpha=PLOT_ALPHA_OVERLAY,
            color=color,
            linestyle=linestyle
        )


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        print_banner("ANDERSON VS PICARD COMPARISON")
    
    # Load sweep data
    simulations = SweepLoader("results/anderson_sweep", comm)
    summary = simulations.get_summary()
    
    data_by_dt = {}
    for _, row in summary.iterrows():
        dt, accel = row["dt_days"], row["accel_type"]
        loader = simulations.get_loader(row["output_dir"])
        
        if dt not in data_by_dt:
            data_by_dt[dt] = {}
        data_by_dt[dt][accel] = loader.get_subiterations_metrics()
    
    # Plot on rank 0
    if comm.rank == 0:
        dt_values = sorted(data_by_dt.keys())
        # Create figure with appropriate width
        fig_width = min(3.5 * len(dt_values), 8.0)  # Max A4 width
        fig, axes = plt.subplots(
            1, len(dt_values), 
            figsize=(fig_width, 2.8)
        )
        
        # Ensure axes is iterable (handle single subplot case)
        if len(dt_values) == 1:
            axes = [axes]
        
        # Subplot labels (a), (b), (c), ...
        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        
        for idx, (ax, dt) in enumerate(zip(axes, dt_values)):
            picard_df = data_by_dt[dt].get("picard")
            anderson_df = data_by_dt[dt].get("anderson")
            
            if picard_df is not None:
                plot_convergence_curves(ax, picard_df, "#000000", "-")
            
            if anderson_df is not None:
                plot_convergence_curves(ax, anderson_df, "#606060", ":")
            
            # Add subplot label to title
            label = subplot_labels[idx] if idx < len(subplot_labels) else f"({chr(97+idx)})"
            setup_axis_style(
                ax,
                xlabel="Subiteration",
                ylabel="Residual",
                title=rf"{label} $\Delta t = {dt:.0f}$ days",
                loglog=False,
                grid=True,
            )
            ax.set_yscale("log")
        
        # Add legend to first plot
        axes[0].plot([], [], "#000000", linestyle="-", linewidth=PLOT_LINEWIDTH, label="Picard")
        axes[0].plot([], [], "#606060", linestyle=":", linewidth=PLOT_LINEWIDTH, label="Anderson")
        add_subplot_legend(axes[0], loc="upper right")
        
        plt.tight_layout()
        save_figure(fig, Path("manuscript/images/anderson_vs_picard.png"), dpi=PUBLICATION_DPI)
        
        print_banner("ANDERSON VS PICARD PLOTTING COMPLETE")

