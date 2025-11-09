"""Visualize contractivity properties of Anderson acceleration: spectral radius and interaction gains over time.

This script analyzes the coupling structure of the fixed-point iteration by examining:
1. Spectral radius (rhoJ) evolution over time for different timesteps
2. Individual interaction gains (J_gs matrix entries) showing how subsolvers couple
3. Distribution of spectral radius values across different configurations

Requires: coupling_each_iter=True in simulation config to record J_gs and rhoJ.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpi4py import MPI

from postprocessor import SweepLoader
from analysis.plot_utils import (
    setup_axis_style, save_figure, print_banner,
    PUBLICATION_DPI, PLOT_LINEWIDTH, PLOT_MARKERSIZE,
    SUBSOLVER_COLORS, SUBSOLVER_LABELS, 
    DT_COLORS, DT_MARKERS, format_dt_label,
    FIGSIZE_FULL_WIDTH, add_subplot_legend,
)


def parse_jacobian_matrix(j_gs_str):
    """Parse J_gs JSON string to numpy array.
    
    Args:
        j_gs_str: JSON string representation of 4x4 matrix
        
    Returns:
        4x4 numpy array or None if parsing fails
    """
    if pd.isna(j_gs_str) or j_gs_str == "":
        return None
    try:
        return np.array(json.loads(j_gs_str))
    except (json.JSONDecodeError, ValueError):
        return None


def extract_coupling_data(loader):
    """Extract spectral radius and gains data from simulation.
    
    Args:
        loader: SimulationLoader instance
        
    Returns:
        DataFrame with columns: time_days, step, iter, rhoJ, J_gs (parsed)
    """
    df = loader.get_subiterations_metrics()
    
    # Filter rows with coupling data
    df_coupling = df[df["rhoJ"].notna()].copy()
    
    if df_coupling.empty:
        return None
    
    # Parse J_gs strings to matrices
    df_coupling["J_gs_matrix"] = df_coupling["J_gs"].apply(parse_jacobian_matrix)
    
    # Remove rows where parsing failed
    df_coupling = df_coupling[df_coupling["J_gs_matrix"].notna()].copy()
    
    return df_coupling


def plot_spectral_radius_vs_time(ax, data_by_dt, dt_values):
    """Plot spectral radius (rhoJ) evolution over simulation time.
    
    Shows contractivity: rhoJ < 1 indicates fixed-point contraction.
    Lower values mean faster convergence per iteration.
    
    Plots max(rhoJ) per timestep since rhoJ increases during subiterations
    as the system approaches the fixed point (Jacobian evaluated at different states).
    """
    for dt in dt_values:
        if dt not in data_by_dt or data_by_dt[dt] is None:
            continue
        
        df = data_by_dt[dt]
        
        # CRITICAL: Take MAXIMUM rhoJ per timestep (not first!)
        # rhoJ increases during subiterations as fixed-point iteration converges
        # The highest value shows worst-case contractivity for that timestep
        grouped = df.groupby("step").agg({
            "time_days": "first",
            "rhoJ": ["max", "mean", "min"]
        }).reset_index()
        
        times = grouped["time_days"]["first"].values
        rho_max = grouped["rhoJ"]["max"].values
        rho_mean = grouped["rhoJ"]["mean"].values
        rho_min = grouped["rhoJ"]["min"].values
        
        color = DT_COLORS.get(dt, "#000000")
        marker = DT_MARKERS.get(dt, "o")
        
        # Plot max trajectory (worst-case contractivity)
        ax.plot(
            times, rho_max,
            color=color,
            marker=marker,
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            label=format_dt_label(dt),
        )
    
    # Add reference line at rhoJ = 1 (contraction threshold)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label=r"$\rho(J) = 1$ (threshold)")
    
    setup_axis_style(
        ax,
        xlabel="Time (days)",
        ylabel=r"Spectral radius $\rho(J)$",
        title=r"(a) Contractivity Evolution",
        loglog=False,
        grid=True,
    )
    add_subplot_legend(ax, loc="upper right")


def identify_strongest_couplings(J_matrix, top_n=5):
    """Identify strongest coupling terms from Jacobian matrix.
    
    Args:
        J_matrix: 4x4 interaction gains matrix
        top_n: Number of strongest terms to return
        
    Returns:
        List of (i, j, value, description) tuples sorted by magnitude
    """
    subsolver_names = ["u (mechanics)", "S (stimulus)", "ρ (density)", "A (direction)"]
    
    couplings = []
    for i in range(4):
        for j in range(4):
            val = J_matrix[i, j]
            desc = f"J[{i},{j}]: {subsolver_names[j]} → {subsolver_names[i]}"
            couplings.append((i, j, val, desc))
    
    # Sort by magnitude (descending)
    couplings.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return couplings[:top_n]


def plot_gains_heatmap(ax, data_by_dt, dt_selected, step_selected=None):
    """Plot interaction gains matrix J_gs as heatmap for selected configuration.
    
    Shows coupling strength between subsolvers:
    J[i,j] = sensitivity of subsolver i to changes in subsolver j
    """
    if dt_selected not in data_by_dt or data_by_dt[dt_selected] is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    
    df = data_by_dt[dt_selected]
    
    # Select representative timestep
    if step_selected is None:
        # Use middle timestep
        step_selected = sorted(df["step"].unique())[len(df["step"].unique()) // 2]
    
    # Get J_gs matrices for this step
    step_data = df[df["step"] == step_selected]
    if step_data.empty:
        ax.text(0.5, 0.5, "No data for selected step", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    
    # Average over subiterations (typically only 1-2 per step with coupling_each_iter)
    matrices = np.array(step_data["J_gs_matrix"].tolist())
    J_avg = np.mean(matrices, axis=0)
    
    # Create heatmap
    im = ax.imshow(J_avg, cmap="YlOrRd", aspect="auto", vmin=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gain magnitude", rotation=270, labelpad=15)
    
    # Annotate cells with values
    for i in range(4):
        for j in range(4):
            text_color = "white" if J_avg[i, j] > 0.5 * J_avg.max() else "black"
            # Annotation font derived from global font size for consistency
            ax.text(j, i, f"{J_avg[i, j]:.2e}",
                   ha="center", va="center", color=text_color,
                   fontsize=max(plt.rcParams.get('font.size', 8) - 1, 6))
    
    # Set ticks and labels
    subsolver_names = ["u", "S", "ρ", "A"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(subsolver_names)
    ax.set_yticklabels(subsolver_names)
    ax.set_xlabel("Input field")
    ax.set_ylabel("Output field")
    
    time_days = step_data["time_days"].iloc[0]
    ax.set_title(f"(b) Gains Matrix J (t={time_days:.0f} days, dt={dt_selected:.0f} days)", fontweight="bold")
    
    return J_avg


def plot_gains_timeseries(ax, data_by_dt, dt_selected):
    """Plot individual gain components J[i,j] evolution over time.
    
    Shows how coupling between specific subsolver pairs changes during simulation.
    Focuses on strongest couplings identified from heatmap analysis.
    """
    if dt_selected not in data_by_dt or data_by_dt[dt_selected] is None:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return
    
    df = data_by_dt[dt_selected]
    
    # Extract specific gains of interest (strongest couplings from typical heatmaps)
    # J[i,j]: effect of field j on field i (i = output, j = input)
    # Indices: 0=u, 1=S, 2=rho, 3=A
    # Strongest typically: ρ→ρ (self), ρ→u (density affects mechanics), ρ→A (density affects fabric)
    
    gains_to_plot = [
        (2, 2, r"$J_{\rho,\rho}$ (ρ→ρ)", "density self-coupling", SUBSOLVER_COLORS["dens"]),
        (0, 2, r"$J_{u,\rho}$ (ρ→u)", "density→mechanics", SUBSOLVER_COLORS["mech"]),
        (3, 2, r"$J_{A,\rho}$ (ρ→A)", "density→direction", SUBSOLVER_COLORS["dir"]),
        (1, 2, r"$J_{S,\rho}$ (ρ→S)", "density→stimulus", SUBSOLVER_COLORS["stim"]),
    ]
    
    for i, j, label, desc, color in gains_to_plot:
        gains = []
        times = []
        
        for _, row in df.iterrows():
            J = row["J_gs_matrix"]
            if J is not None and J.shape == (4, 4):
                gains.append(J[i, j])
                times.append(row["time_days"])
        
        if gains:
            ax.plot(
                times, gains,
                color=color,
                linewidth=PLOT_LINEWIDTH,
                marker="o",
                markersize=PLOT_MARKERSIZE * 0.8,
                label=label,
                alpha=0.85,
            )
    
    setup_axis_style(
        ax,
        xlabel="Time (days)",
        ylabel="Gain magnitude",
        title=f"(c) Key Interaction Gains (dt={dt_selected:.0f} days)",
        loglog=False,
        grid=True,
    )
    ax.set_yscale("log")
    add_subplot_legend(ax, loc="upper right")


def plot_spectral_radius_distribution(ax, data_by_dt, dt_values):
    """Plot distribution of spectral radius across timesteps for each dt.
    
    Shows statistical properties using max(rhoJ) per timestep.
    """
    positions = []
    data_list = []
    labels = []
    colors = []
    
    for idx, dt in enumerate(dt_values):
        if dt not in data_by_dt or data_by_dt[dt] is None:
            continue
        
        df = data_by_dt[dt]
        # Use max rhoJ per timestep (consistent with evolution plot)
        rho_max_per_step = df.groupby("step")["rhoJ"].max().values
        
        positions.append(idx)
        data_list.append(rho_max_per_step)
        labels.append(format_dt_label(dt))
        colors.append(DT_COLORS.get(dt, "#808080"))
    
    if not data_list:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return
    
    # Create violin plot
    parts = ax.violinplot(
        data_list,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True,
    )
    
    # Color each violin
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add reference line
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, alpha=0.5, label=r"$\rho(J) = 1$")
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"Spectral radius $\rho(J)$")
    ax.set_title(r"(d) Spectral Radius Distribution", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)


def plot_dominant_coupling_direction(ax, data_by_dt, dt_values):
    """Plot which subsolver pair has strongest coupling over time.
    
    Identifies dominant interaction by max off-diagonal gain.
    """
    subsolver_names = ["u", "S", "ρ", "A"]
    
    for dt in dt_values:
        if dt not in data_by_dt or data_by_dt[dt] is None:
            continue
        
        df = data_by_dt[dt]
        
        times = []
        max_gains = []
        
        for time_group in df.groupby("time_days"):
            time_val = time_group[0]
            matrices = time_group[1]["J_gs_matrix"].tolist()
            
            # Average matrices at this timestep
            J_avg = np.mean(np.array(matrices), axis=0)
            
            # Find max off-diagonal (exclude self-coupling)
            J_offdiag = J_avg.copy()
            np.fill_diagonal(J_offdiag, 0)
            max_gain = np.max(J_offdiag)
            
            times.append(time_val)
            max_gains.append(max_gain)
        
        color = DT_COLORS.get(dt, "#000000")
        marker = DT_MARKERS.get(dt, "o")
        
        ax.plot(
            times, max_gains,
            color=color,
            marker=marker,
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            label=format_dt_label(dt),
        )
    
    setup_axis_style(
        ax,
        xlabel="Time (days)",
        ylabel="Max off-diagonal gain",
        title=r"(e) Strongest Coupling Strength",
        loglog=False,
        grid=True,
    )
    ax.set_yscale("log")
    add_subplot_legend(ax, loc="upper right")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    
    if comm.rank == 0:
        print_banner("CONTRACTIVITY AND INTERACTION GAINS ANALYSIS")
    
    # Load sweep data
    sweep_dir = Path("results/anderson_sweep")
    simulations = SweepLoader(str(sweep_dir), comm)
    summary = simulations.get_summary()
    
    # Filter for Anderson acceleration only
    anderson_runs = summary[summary["accel_type"] == "anderson"]
    
    # Extract coupling data for each dt
    # Use specific Anderson configuration (m=8, beta=1.0) for consistency
    data_by_dt = {}
    dt_values = sorted(anderson_runs["dt_days"].unique())
    
    for dt in dt_values:
        dt_runs = anderson_runs[anderson_runs["dt_days"] == dt]
        
        if dt_runs.empty:
            continue
        
        # Filter for m=8, beta=1.0 (representative middle configuration)
        preferred = dt_runs[(dt_runs["m"] == 8) & (dt_runs["beta"] == 1.0)]
        if preferred.empty:
            # Fallback to first available
            preferred = dt_runs.iloc[[0]]
        
        row = preferred.iloc[0]
        loader = simulations.get_loader(row["output_dir"])
        
        df_coupling = extract_coupling_data(loader)
        if df_coupling is not None and not df_coupling.empty:
            data_by_dt[dt] = df_coupling
            if comm.rank == 0:
                print(f"Loaded coupling data for dt={dt:.1f} days (m={row['m']}, beta={row['beta']}): {len(df_coupling)} records")
        else:
            if comm.rank == 0:
                print(f"WARNING: No coupling data for dt={dt:.1f} days (set coupling_each_iter=True)")
    
    # Generate plots on rank 0
    if comm.rank == 0:
        if not data_by_dt:
            print("\nERROR: No coupling data found!")
            print("Ensure simulations were run with cfg.coupling_each_iter = True")
            sys.exit(1)
        
        # Create comprehensive figure with 2×3 grid
        fig = plt.figure(figsize=(FIGSIZE_FULL_WIDTH[0] * 1.2, FIGSIZE_FULL_WIDTH[1] * 1.5))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])   # Top row: spectral radius vs time (full width)
        ax2 = fig.add_subplot(gs[1, 0])   # Middle left: gains heatmap
        ax3 = fig.add_subplot(gs[1, 1])   # Middle right: gains timeseries
        ax4 = fig.add_subplot(gs[2, 0])   # Bottom left: spectral radius distribution
        ax5 = fig.add_subplot(gs[2, 1])   # Bottom right: dominant coupling
        
        # Select dt=25 for detailed plots (as requested)
        dt_repr = 25.0
        if dt_repr not in dt_values:
            dt_repr = dt_values[0]  # Fallback
        
        print(f"\nGenerating plots (representative dt={dt_repr:.1f} days)...")
        
        plot_spectral_radius_vs_time(ax1, data_by_dt, dt_values)
        J_matrix = plot_gains_heatmap(ax2, data_by_dt, dt_repr)
        plot_gains_timeseries(ax3, data_by_dt, dt_repr)
        plot_spectral_radius_distribution(ax4, data_by_dt, dt_values)
        plot_dominant_coupling_direction(ax5, data_by_dt, dt_values)
        
        # Print strongest coupling analysis
        if J_matrix is not None:
            print(f"\n" + "="*80)
            print(f"STRONGEST COUPLING TERMS (dt={dt_repr:.1f} days)")
            print("="*80)
            strongest = identify_strongest_couplings(J_matrix, top_n=8)
            for rank, (i, j, val, desc) in enumerate(strongest, 1):
                coupling_type = "DIAGONAL (self)" if i == j else "off-diagonal"
                print(f"{rank}. {desc:<50} = {val:.4e}  [{coupling_type}]")
            print("="*80)
            print("\nInterpretation:")
            print("- J[i,j]: Sensitivity of field i to changes in field j")
            print("- Large diagonal terms: Strong self-coupling (field depends on itself)")
            print("- Large off-diagonal: Strong cross-coupling between subsolvers")
            print("- Typically ρ-dominated due to remodeling transport equations")
            print("="*80)
        
        # Save figure
        output_path = Path("manuscript/images/contractivity_analysis.png")
        save_figure(fig, output_path, dpi=PUBLICATION_DPI, close=False)
        
        print(f"\nSaved: {output_path}")
        print_banner("CONTRACTIVITY ANALYSIS COMPLETE")
        
        plt.show()
