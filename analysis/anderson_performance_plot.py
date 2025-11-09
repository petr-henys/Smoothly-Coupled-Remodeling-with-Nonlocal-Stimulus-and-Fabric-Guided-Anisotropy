"""Anderson acceleration performance metrics and sensitivity analysis.

Two-panel figure:
- Left: Time evolution of Anderson metrics at fixed dt=25 days for different (m, β)
  - Condition number κ(H) as line plot (log scale)
  - Rejection rate, backtracking, history depth as composed bars/events
- Right: Sensitivity analysis showing how final-state metrics depend on (dt, m, β)
  - Heatmaps or grouped bars showing parameter influence

Input:
    results/anderson_sweep/ containing Anderson runs with telemetry enabled

Output:
    manuscript/images/anderson_performance_vs_time.png

Usage:
    python3 analysis/anderson_performance_plot.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from postprocessor import SweepLoader
from analysis.plot_utils import (
    setup_axis_style, save_figure, print_banner,
    PUBLICATION_DPI, PLOT_LINEWIDTH, PLOT_MARKERSIZE,
    DT_COLORS, DT_MARKERS, DT_LINESTYLES, format_dt_label,
    add_subplot_legend, FIGSIZE_DOUBLE_COLUMN, PLOT_ALPHA_OVERLAY,
)


def compute_windowed_metrics(df, window_size_days=25.0):
    """Compute windowed statistics for Anderson metrics over simulation time.
    
    Groups subiterations into time windows and computes:
    - Rejection rate (fraction of rejected iterations)
    - Average backtracking events per iteration
    - Average condition number
    - Average history depth
    
    Args:
        df: DataFrame with columns [time_days, accepted, backtracks, condH, aa_hist]
        window_size_days: Time window size for averaging
        
    Returns:
        DataFrame with columns [time_center, rejection_rate, avg_backtracks, avg_condH, avg_hist]
    """
    # Define time bins
    t_min = df["time_days"].min()
    t_max = df["time_days"].max()
    bins = np.arange(t_min, t_max + window_size_days, window_size_days)
    
    # Bin data
    df = df.copy()
    df["time_bin"] = pd.cut(df["time_days"], bins=bins, include_lowest=True)
    
    # Compute windowed metrics
    grouped = df.groupby("time_bin", observed=True)
    
    results = []
    for interval, group in grouped:
        if len(group) == 0:
            continue
            
        time_center = (interval.left + interval.right) / 2
        rejection_rate = 1.0 - group["accepted"].mean()  # Fraction rejected
        avg_backtracks = group["backtracks"].mean()
        avg_condH = group["condH"].mean()
        avg_hist = group["aa_hist"].mean()
        
        results.append({
            "time_center": time_center,
            "rejection_rate": rejection_rate,
            "avg_backtracks": avg_backtracks,
            "avg_condH": avg_condH,
            "avg_hist": avg_hist,
        })
    
    return pd.DataFrame(results)





def plot_time_evolution_cond_only(ax, data_by_config, fixed_dt=25.0, window_size_days=25.0):
    """Plot condition number time evolution at fixed dt.
    
    Args:
        ax: Matplotlib axis
        data_by_config: Dict {(dt, m, beta): dataframe}
        fixed_dt: dt value to plot (days)
        window_size_days: Time window for averaging
    """
    
    # Filter for fixed dt
    configs_at_dt = [(m, beta, df) for (dt, m, beta), df in data_by_config.items() if dt == fixed_dt]
    
    if len(configs_at_dt) == 0:
        return
    
    # Color map for different m values (colorblind-friendly)
    m_colors = {4: "#E69F00", 8: "#56B4E9", 12: "#009E73"}
    beta_styles = {0.5: "-", 1.0: "--"}
    
    # Plot individual condition number curves
    for m, beta, df in sorted(configs_at_dt):
        metrics = compute_windowed_metrics(df, window_size_days=window_size_days)
        
        if len(metrics) == 0:
            continue
        
        time = metrics["time_center"].values
        color = m_colors.get(m, "#000000")
        linestyle = beta_styles.get(beta, "-")
        label = f"m={m}, β={beta:.1f}"
        
        # Condition number (continuous)
        cond = metrics["avg_condH"].values
        ax.plot(time, cond, linewidth=PLOT_LINEWIDTH, linestyle=linestyle,
                color=color, alpha=0.8, label=label)


def compute_sensitivity_metrics(data_by_config):
    """Compute final-state metrics for sensitivity analysis.
    
    Returns:
        DataFrame with columns [dt, m, beta, avg_rejection_rate, avg_cond, avg_hist, total_backtracks]
    """
    results = []
    
    for (dt, m, beta), df in data_by_config.items():
        # Use last 200 days for "steady state" metrics
        last_time = df["time_days"].max()
        steady_df = df[df["time_days"] >= last_time - 200]
        
        if len(steady_df) == 0:
            continue
        
        avg_rejection_rate = 1.0 - steady_df["accepted"].mean()
        avg_cond = steady_df["condH"].mean()
        avg_hist = steady_df["aa_hist"].mean()
        total_backtracks = steady_df["backtracks"].sum()
        
        results.append({
            "dt": dt,
            "m": m,
            "beta": beta,
            "avg_rejection_rate": avg_rejection_rate * 100,  # As percentage
            "avg_cond": avg_cond,
            "avg_hist": avg_hist,
            "total_backtracks": total_backtracks,
        })
    
    return pd.DataFrame(results)


def plot_sensitivity_analysis(axes, sensitivity_df):
    """Plot sensitivity of metrics to (dt, m, beta).
    
    Args:
        axes: Array of 3 axes [ax_rej, ax_cond, ax_hist]
        sensitivity_df: DataFrame from compute_sensitivity_metrics
    """
    ax_rej, ax_cond, ax_hist = axes
    
    # Group by dt and (m, beta)
    dt_values = sorted(sensitivity_df["dt"].unique())
    m_values = sorted(sensitivity_df["m"].unique())
    beta_values = sorted(sensitivity_df["beta"].unique())
    
    # Create grouped bar positions
    n_groups = len(dt_values)
    n_bars_per_group = len(m_values) * len(beta_values)
    bar_width = 0.8 / n_bars_per_group
    
    m_colors = {4: "#E69F00", 8: "#56B4E9", 12: "#009E73"}
    beta_hatches = {0.5: "", 1.0: "///"}
    
    for metric_idx, (ax, metric_name, ylabel) in enumerate([
        (ax_rej, "avg_rejection_rate", "Rejection rate (%)"),
        (ax_cond, "avg_cond", r"$\kappa(\mathbf{H})$"),
        (ax_hist, "avg_hist", "History depth"),
    ]):
        for group_idx, dt in enumerate(dt_values):
            dt_data = sensitivity_df[sensitivity_df["dt"] == dt]
            
            bar_idx = 0
            for m in m_values:
                for beta in beta_values:
                    row = dt_data[(dt_data["m"] == m) & (dt_data["beta"] == beta)]
                    
                    if len(row) == 0:
                        bar_idx += 1
                        continue
                    
                    value = row[metric_name].values[0]
                    x_pos = group_idx + (bar_idx - n_bars_per_group/2 + 0.5) * bar_width
                    
                    label = f"m={m}, β={beta:.1f}" if group_idx == 0 else None
                    
                    ax.bar(x_pos, value, bar_width, 
                           color=m_colors[m], hatch=beta_hatches[beta],
                           alpha=0.7, edgecolor='black', linewidth=0.5,
                           label=label)
                    
                    bar_idx += 1
        
        # Styling
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels([f"{int(dt)}" for dt in dt_values])
        ax.set_xlabel("Timestep dt (days)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Log scale for condition number
        if metric_name == "avg_cond":
            ax.set_yscale("log")


if __name__ == "__main__":
    import pandas as pd
    
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        print_banner("ANDERSON PERFORMANCE METRICS VS TIME")
    
    # Load sweep data - filter for Anderson only
    simulations = SweepLoader("results/anderson_sweep", comm)
    summary = simulations.get_summary()
    anderson_runs = summary[summary["accel_type"] == "anderson"]
    
    if len(anderson_runs) == 0:
        if comm.rank == 0:
            print("ERROR: No Anderson runs found in sweep!")
        sys.exit(1)
    
    # Organize data by configuration (dt, m, beta)
    data_by_config = {}
    for _, row in anderson_runs.iterrows():
        dt = row["dt_days"]
        loader = simulations.get_loader(row["output_dir"])
        
        # Get configuration parameters
        cfg = loader.get_config()
        m = cfg.get("m")
        beta = cfg.get("beta")
        
        # Skip if parameters not available
        if m is None or beta is None:
            if comm.rank == 0:
                print(f"WARNING: Missing m or beta for {row['output_dir']}, skipping")
            continue
        
        subiters_df = loader.get_subiterations_metrics()
        
        # Verify required columns exist
        required_cols = ["time_days", "accepted", "backtracks", "condH", "aa_hist"]
        missing = [col for col in required_cols if col not in subiters_df.columns]
        if missing:
            if comm.rank == 0:
                print(f"WARNING: Missing columns {missing} for dt={dt}, skipping")
            continue
        
        # Store by unique configuration
        config_key = (dt, m, beta)
        data_by_config[config_key] = subiters_df
    
    if len(data_by_config) == 0:
        if comm.rank == 0:
            print("ERROR: No valid Anderson data found!")
        sys.exit(1)
    
    # Create plots on rank 0
    if comm.rank == 0:
        # Create figure: 1 row × 4 columns, increased height
        fig = plt.figure(figsize=(FIGSIZE_DOUBLE_COLUMN[0], FIGSIZE_DOUBLE_COLUMN[1]*0.4))
        
        # All in one row: time evolution + sensitivity analysis
        ax_cond = plt.subplot(1, 4, 1)       # Time evolution (condition number)
        ax_sens_rej = plt.subplot(1, 4, 2)   # Sensitivity: rejection rate
        ax_sens_cond = plt.subplot(1, 4, 3)  # Sensitivity: condition number
        ax_sens_hist = plt.subplot(1, 4, 4)  # Sensitivity: history depth
        
        window_size = 25.0
        fixed_dt = 25.0
        
        # LEFT: Time evolution at dt=25 (condition number only)
        plot_time_evolution_cond_only(
            ax_cond,
            data_by_config,
            fixed_dt=fixed_dt,
            window_size_days=window_size,
        )
        
        # Style time evolution panel
        ax_cond.set_xlabel("Simulation time (days)", fontsize=8)
        ax_cond.set_ylabel(r"Condition number $\kappa(\mathbf{H})$", fontsize=8)
        ax_cond.set_yscale("log")
        ax_cond.grid(True, alpha=0.3)
        ax_cond.set_title(f"(a) Condition (dt={fixed_dt:.0f}d)", fontweight="bold", fontsize=8)
        # No legend in subplot - will add unified legend below
        
        # RIGHT: Sensitivity analysis
        sensitivity_df = compute_sensitivity_metrics(data_by_config)
        plot_sensitivity_analysis(
            [ax_sens_rej, ax_sens_cond, ax_sens_hist],
            sensitivity_df
        )
        
        # Add title to sensitivity panel
        ax_sens_rej.set_title("(b) Rejection sensitivity", fontweight="bold", fontsize=8)
        ax_sens_cond.set_title("(c) Condition sensitivity", fontweight="bold", fontsize=8)
        ax_sens_hist.set_title("(d) History sensitivity", fontweight="bold", fontsize=8)
        
        plt.tight_layout(rect=[0, 0.15, 1, 1])  # Leave space at bottom for legend
        
        # Unified legend below all plots
        # Collect handles from time evolution plot (for m, beta combinations)
        handles_time, labels_time = ax_cond.get_legend_handles_labels()
        # Collect handles from first sensitivity plot (for bar groups)
        handles_sens, labels_sens = ax_sens_rej.get_legend_handles_labels()
        
        # Combine: time evolution legend on left, sensitivity legend on right
        fig.legend(handles_time, labels_time, 
                   loc='lower left', bbox_to_anchor=(0.02, 0.0), 
                   ncol=3, fontsize=5, frameon=True, framealpha=0.95,
                   title="Time evolution (a)", title_fontsize=6)
        
        fig.legend(handles_sens, labels_sens,
                   loc='lower right', bbox_to_anchor=(0.98, 0.0),
                   ncol=3, fontsize=5, frameon=True, framealpha=0.95,
                   title="Parameter configs (b-d)", title_fontsize=6)
        
        # Save figure
        output_path = Path("manuscript/images/anderson_performance_vs_time.png")
        save_figure(fig, output_path, dpi=PUBLICATION_DPI)
        
        print_banner("ANDERSON PERFORMANCE PLOTTING COMPLETE")
        print(f"Output: {output_path}")
        print(f"Window size: {window_size} days")
        print(f"Number of configurations: {len(data_by_config)}")
        
        # Group by dt for summary
        configs_by_dt = {}
        for (dt, m, beta) in data_by_config.keys():
            if dt not in configs_by_dt:
                configs_by_dt[dt] = []
            configs_by_dt[dt].append((m, beta))
        
        for dt in sorted(configs_by_dt.keys()):
            configs = configs_by_dt[dt]
            print(f"  dt={dt:.1f} days: {len(configs)} configurations")
            for m, beta in sorted(configs):
                print(f"    m={m}, β={beta:.1f}")
