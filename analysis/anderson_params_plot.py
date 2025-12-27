"""Anderson parameter sensitivity: residual vs subiteration plot.

Single-panel figure showing convergence curves for different Anderson
parameter combinations (m, beta, lam).

Input:
    results/anderson_params_sweep/ containing sweep results

Output:
    manuscript/images/anderson_params_sensitivity.png

Usage:
    python3 analysis/anderson_params_plot.py [sweep_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from postprocessor import SweepLoader
from analysis.plot_utils import (
    apply_style,
    setup_axis_style,
    save_manuscript_figure,
    FIGSIZE_SINGLE_COLUMN,
    PUBLICATION_DPI,
    PLOT_LINEWIDTH,
)


# ==============================================================================
# Styling for Anderson parameter plots
# ==============================================================================

# Color palette for history size m
M_COLORS = {
    2: "#E69F00",   # Orange
    4: "#56B4E9",   # Sky blue  
    6: "#009E73",   # Teal
    8: "#CC79A7",   # Pink
}

# Line styles for beta values
BETA_LINESTYLES = {
    0.5: ":",
    0.7: "--",
    1.0: "-",
}


def main() -> None:
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/anderson_params_sweep")
    output_file = Path("manuscript/images/anderson_params_sensitivity.png")

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    from mpi4py import MPI
    comm = MPI.COMM_SELF  # Serial postprocessing

    sweep = SweepLoader(str(sweep_dir), comm)
    summary = sweep.get_summary()
    
    if summary.empty:
        raise RuntimeError("No runs found in sweep summary.")

    # Verify expected columns
    for col in ("m", "beta", "lam", "output_dir"):
        if col not in summary.columns:
            raise ValueError(f"Missing '{col}' in sweep_summary.csv. Columns: {list(summary.columns)}")

    # Load data grouped by (m, beta, lam)
    runs: dict[tuple[int, float, float], pd.DataFrame] = {}
    for _, row in summary.iterrows():
        m = int(row["m"])
        beta = float(row["beta"])
        lam = float(row["lam"])
        key = (m, beta, lam)
        
        if key in runs:
            continue
        
        try:
            loader = sweep.get_loader(str(row["output_dir"]))
            subiters = loader.get_subiterations_metrics()
            runs[key] = subiters
        except Exception as e:
            print(f"Warning: Failed to load {row['output_dir']}: {e}")
            continue

    if not runs:
        raise RuntimeError("No valid runs loaded.")

    # Console summary
    print("\n=== ANDERSON PARAMETER SENSITIVITY ===")
    print(f"Loaded {len(runs)} parameter combinations")
    
    m_vals = sorted(set(k[0] for k in runs.keys()))
    beta_vals = sorted(set(k[1] for k in runs.keys()))
    lam_vals = sorted(set(k[2] for k in runs.keys()))
    
    print(f"m values: {m_vals}")
    print(f"beta values: {beta_vals}")
    print(f"lam values: {lam_vals}")

    # ----------------- plotting -----------------
    apply_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)
    
    # Fixed lam for cleaner plot (use middle value)
    fixed_lam = lam_vals[len(lam_vals) // 2] if lam_vals else 0.05
    
    # Also fix m and beta for single-run visualization (show ALL timesteps from one config)
    fixed_m = m_vals[len(m_vals) // 2] if m_vals else 4
    fixed_beta = beta_vals[len(beta_vals) // 2] if beta_vals else 0.7
    
    # Find the run with fixed parameters
    target_key = None
    for key in runs.keys():
        m, beta, lam = key
        if m == fixed_m and np.isclose(beta, fixed_beta) and np.isclose(lam, fixed_lam):
            target_key = key
            break
    
    if target_key is None:
        # Fallback to first run
        target_key = list(runs.keys())[0]
        fixed_m, fixed_beta, fixed_lam = target_key
    
    subiters = runs[target_key]
    
    # Get all timesteps
    steps = sorted(subiters["step"].unique())
    n_steps = len(steps)
    
    print(f"Plotting {n_steps} timesteps for m={fixed_m}, β={fixed_beta}, λ={fixed_lam}")
    
    # Use viridis colormap for timestep progression
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=n_steps - 1)
    
    # Plot ALL timesteps
    for i, step in enumerate(steps):
        step_data = subiters[subiters["step"] == step].sort_values("iter")
        
        if step_data.empty or "proj_res" not in step_data.columns:
            continue
        
        iters = step_data["iter"].values
        residuals = step_data["proj_res"].values
        
        color = cmap(norm(i))
        
        ax.semilogy(
            iters, residuals,
            color=color,
            linestyle="-",
            linewidth=PLOT_LINEWIDTH,
            alpha=0.7,
        )
    
    setup_axis_style(
        ax,
        xlabel="Subiteration",
        ylabel="Residual",
        title=f"Convergence (m={fixed_m}, β={fixed_beta}, λ={fixed_lam})",
        loglog=False,
    )
    
    # Add colorbar for timestep progression
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label("Simulation progress", fontsize=7)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Start", "", "End"])
    
    fig.tight_layout()
    
    save_manuscript_figure(fig, output_file.name, dpi=PUBLICATION_DPI)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
