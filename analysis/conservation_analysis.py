"""Conservation analysis: visualize mass/energy balance from simulation results.

Loads steps.csv from simulation output, extracts conservation metrics, and
generates diagnostic plots showing:
  (a) Total bone mass evolution M(t)
  (b) Mass balance: dM/dt vs source integral
  (c) Strain energy evolution W(t)

These plots verify that the numerical scheme respects conservation laws:
- Mass should only change due to formation/resorption source terms
- Mass balance error should be small (consistent discretization)
- Energy should evolve smoothly with loading

Usage:
    python analysis/conservation_analysis.py

Inputs:
    results/<run>/
    ├── steps.csv   <- Contains conservation columns (total_mass_g, etc.)
    └── config.json

Outputs:
    results/<run>/conservation_diagnostic.png
    (or manuscript/images/conservation_diagnostic.png)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plot_utils import (
    apply_style,
    save_figure,
    PUBLICATION_DPI,
    PLOT_LINEWIDTH,
    COLOR_PALETTE,
)


def load_steps_data(run_dir: Path) -> pd.DataFrame:
    """Load steps.csv with conservation metrics.
    
    Args:
        run_dir: Path to simulation output directory.
    
    Returns:
        DataFrame with timestep data including conservation columns.
    
    Raises:
        FileNotFoundError: If steps.csv not found.
        ValueError: If conservation columns are missing.
    """
    steps_file = run_dir / "steps.csv"
    if not steps_file.exists():
        raise FileNotFoundError(f"steps.csv not found in {run_dir}")
    
    df = pd.read_csv(steps_file)
    
    # Check for conservation columns
    required_cols = ["total_mass_g", "mass_rate_g_day", "source_integral_g_day"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Conservation columns missing from steps.csv: {missing}. "
            "Run a simulation with conservation monitoring enabled."
        )
    
    # Filter to accepted steps only (for plotting time series)
    if "accepted" in df.columns:
        df = df[df["accepted"] == 1].copy()
    
    return df


def load_config(run_dir: Path) -> dict[str, Any]:
    """Load simulation config.json."""
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


def create_conservation_figure(
    df: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Create 1×3 conservation diagnostic figure.
    
    Layout:
      (a) Total mass M(t) - absolute and relative change
      (b) Mass balance - dM/dt vs source, with balance error
      (c) Strain energy W(t) - total stored elastic energy
    
    Args:
        df: DataFrame with steps data and conservation columns.
        config: Simulation config (for initial values).
        output_path: Where to save the figure.
        title_suffix: Optional suffix for figure title.
    """
    apply_style()
    
    # Extract time series
    time = df["time_days"].values
    mass = df["total_mass_g"].values
    mass_rate = df["mass_rate_g_day"].values
    source = df["source_integral_g_day"].values
    
    # Recompute balance error (normalized by total mass)
    # This represents the spurious mass generation rate as a fraction of system mass.
    scale = np.maximum(mass, 1e-9)
    balance_error = np.abs(mass_rate - source) / scale
    
    energy = df["total_energy_mJ"].values if "total_energy_mJ" in df.columns else None
    
    # Initial mass for relative change
    M0 = mass[0] if len(mass) > 0 else 1.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    
    # =========================================================================
    # Panel (a): Total Mass Evolution
    # =========================================================================
    ax = axes[0]
    
    # Primary axis: absolute mass
    color1 = COLOR_PALETTE["primary"]
    line1, = ax.plot(time, mass, color=color1, linewidth=PLOT_LINEWIDTH, label="$M(t)$")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Total mass $M$ [g]", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    
    # Secondary axis: relative change
    ax2 = ax.twinx()
    color2 = COLOR_PALETTE["secondary"]
    rel_change = (mass - M0) / M0 * 100  # Percentage
    line2, = ax2.plot(time, rel_change, color=color2, linewidth=PLOT_LINEWIDTH, 
             linestyle="--", alpha=0.8, label=r"$\Delta M/M_0$")
    ax2.set_ylabel(r"Relative change $\Delta M/M_0$ [\%]", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    # Combined legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best", fontsize=8)
    
    ax.set_title("(a) Total bone mass", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel (b): Mass Balance
    # =========================================================================
    ax = axes[1]
    
    # Plot dM/dt and source term
    # Use distinct styles: Rate = solid thick, Source = dashed/dotted
    color_rate = "#2c3e50" # Dark blue/grey
    color_src = "#e67e22"  # Orange
    
    l1, = ax.plot(time, mass_rate, color=color_rate, 
            linewidth=PLOT_LINEWIDTH, label=r"$dM/dt$")
    l2, = ax.plot(time, source, color=color_src, 
            linewidth=PLOT_LINEWIDTH, linestyle="--", label="Source $Q$")
    
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mass rate [g/day]")
    
    # Add balance error on secondary axis
    if balance_error is not None:
        ax3 = ax.twinx()
        color_err = "#c0392b" # Red
        # Scale to % for readability
        err_pct = balance_error * 100
        # Use line + fill for error to make it visible even if small
        l3, = ax3.plot(time, err_pct, color=color_err, linewidth=0.5, alpha=0.5, label="Error")
        ax3.fill_between(time, 0, err_pct, color=color_err, alpha=0.15)
        
        ax3.set_ylabel("Balance error [% $M$/day]", color=color_err, fontsize=9)
        ax3.tick_params(axis="y", labelcolor=color_err, labelsize=8)
        # Auto-scale but ensure 0 is included
        # If error is huge (e.g. 100%), let it show. If tiny (1e-6), let it show.
        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(0, max(ymax, 1e-6))
        
        # Combined legend
        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=7)
    else:
        ax.legend(loc="upper right", fontsize=8)
    
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title("(b) Mass balance", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Panel (c): Strain Energy
    # =========================================================================
    ax = axes[2]
    
    color_w = COLOR_PALETTE["accent"] # Green/Teal
    color_rate = "#8e44ad" # Purple
    
    if energy is not None and len(energy) > 0:
        l1, = ax.plot(time, energy, color=color_w, 
                linewidth=PLOT_LINEWIDTH, label="$W(t)$")
        ax.set_ylabel("Strain energy $W$ [mJ]", color=color_w)
        ax.tick_params(axis="y", labelcolor=color_w)
        
        # Add energy rate on secondary axis
        energy_rate = df["energy_rate_mJ_day"].values if "energy_rate_mJ_day" in df.columns else None
        if energy_rate is not None:
            ax4 = ax.twinx()
            l2, = ax4.plot(time, energy_rate, color=color_rate, 
                     linewidth=PLOT_LINEWIDTH * 0.8, linestyle="--", alpha=0.8, label=r"$dW/dt$")
            ax4.set_ylabel("Energy rate [mJ/day]", color=color_rate, fontsize=9)
            ax4.tick_params(axis="y", labelcolor=color_rate, labelsize=8)
            
            # Legend
            lines = [l1, l2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="best", fontsize=8)
        else:
            ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Energy data\nnot available", 
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
    
    ax.set_xlabel("Time [days]")
    ax.set_title("(c) Elastic strain energy", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Finalize
    plt.tight_layout()
    
    # Save
    save_figure(fig, output_path, dpi=PUBLICATION_DPI, close=True)
    print(f"Saved: {output_path}")


def create_mass_balance_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compute summary statistics for mass balance.
    
    Returns:
        Dictionary with mass balance metrics.
    """
    mass = df["total_mass_g"].values
    
    # Recompute balance error (same logic as in plot)
    mass_rate = df["mass_rate_g_day"].values
    source = df["source_integral_g_day"].values
    scale = np.maximum(mass, 1e-9)
    balance_error = np.abs(mass_rate - source) / scale
    
    summary = {
        "M_initial_g": float(mass[0]) if len(mass) > 0 else 0.0,
        "M_final_g": float(mass[-1]) if len(mass) > 0 else 0.0,
        "M_change_g": float(mass[-1] - mass[0]) if len(mass) > 0 else 0.0,
        "M_change_percent": float((mass[-1] - mass[0]) / mass[0] * 100) if len(mass) > 0 and mass[0] > 0 else 0.0,
    }
    
    summary["balance_error_mean"] = float(np.mean(balance_error))
    summary["balance_error_max"] = float(np.max(balance_error))
    summary["balance_error_p95"] = float(np.percentile(balance_error, 95))
    
    return summary


def main() -> None:
    """Run conservation analysis on default simulation results."""
    # Default paths - adjust as needed
    # Map run directory -> output name suffix
    run_dirs = {
        Path(".physio_results_box"): "box_physio",
        Path(".stiff_results_box"): "box_stiff",
        Path("results/box_physio"): "box_physio", # Fallback
        Path("results/box_stiff"): "box_stiff",   # Fallback
    }
    
    output_base = Path("manuscript/images")
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CONSERVATION ANALYSIS")
    print("=" * 60)
    
    processed = set()

    for run_dir, run_name in run_dirs.items():
        if not run_dir.exists():
            continue
        
        if run_name in processed:
            continue

        print(f"\nAnalyzing: {run_dir} -> {run_name}")
        
        try:
            df = load_steps_data(run_dir)
            config = load_config(run_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipped: {e}")
            continue
        
        print(f"  Loaded {len(df)} accepted timesteps")
        
        # Compute summary
        summary = create_mass_balance_summary(df)
        print(f"  Mass: {summary['M_initial_g']:.4f} → {summary['M_final_g']:.4f} g "
              f"({summary['M_change_percent']:+.2f}%)")
        if "balance_error_mean" in summary:
            print(f"  Balance error: mean={summary['balance_error_mean']:.2e}, "
                  f"max={summary['balance_error_max']:.2e}")
        
        # Generate figure
        output_path = output_base / f"conservation_diagnostic_{run_name}.png"
        create_conservation_figure(df, config, output_path)
        processed.add(run_name)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
