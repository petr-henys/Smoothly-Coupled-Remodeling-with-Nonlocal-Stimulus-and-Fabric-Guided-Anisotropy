import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from postprocessor import SweepLoader
from mpi4py import MPI


def plot_convergence_curves(ax, df, color, linestyle):
    """Plot all convergence curves from a dataframe."""
    for step in sorted(df["step"].unique()):
        step_data = df[df["step"] == step].sort_values("iter")
        ax.plot(
            step_data["iter"].values,
            step_data["proj_res"].values,
            linewidth=1.5,
            alpha=0.6,
            color=color,
            linestyle=linestyle
        )


# Load sweep data
simulations = SweepLoader("results/anderson_sweep", MPI.COMM_WORLD)
summary = simulations.get_summary()

data_by_dt = {}
for _, row in summary.iterrows():
    dt, accel = row["dt_days"], row["accel_type"]
    loader = simulations.get_loader(row["output_dir"])
    
    if dt not in data_by_dt:
        data_by_dt[dt] = {}
    data_by_dt[dt][accel] = loader.get_subiterations_metrics()

# Plot on rank 0
if MPI.COMM_WORLD.rank == 0:
    dt_values = sorted(data_by_dt.keys())
    fig, axes = plt.subplots(
        1, len(dt_values), 
        figsize=(6 * len(dt_values), 5)
    )
    
    for ax, dt in zip(axes, dt_values):
        picard_df = data_by_dt[dt].get("picard")
        anderson_df = data_by_dt[dt].get("anderson")
        
        if picard_df is not None:
            plot_convergence_curves(ax, picard_df, "C0", "-")
        
        if anderson_df is not None:
            plot_convergence_curves(ax, anderson_df, "C1", ":")
        
        ax.set_xlabel("Subiteration", fontsize=12)
        ax.set_ylabel("Residual", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_title(f"dt = {dt:.0f} days", fontsize=12, fontweight="bold")
    
    # Add legend to first plot
    axes[0].plot([], [], "C0-", linewidth=2, label="Picard")
    axes[0].plot([], [], "C1:", linewidth=2, label="Anderson")
    axes[0].legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig("manuscript/images/anderson_vs_picard.png", dpi=300, bbox_inches="tight")

