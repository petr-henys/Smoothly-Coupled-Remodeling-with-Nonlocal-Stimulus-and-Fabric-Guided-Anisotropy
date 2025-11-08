"""Generate Roache GCI convergence plots from precomputed QoI data.

This script loads QoI data exported by convergence_roache.py and creates
publication-quality plots (similar structure to convergence_plots.py).
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def create_roache_spatial_plot(
    qoi_file: Path,
    dt_fixed: float,
    output_file: Path,
) -> None:
    """Create spatial Roache plot with 1x2 subplots (all QoIs on each)."""
    # Load QoI data
    df = pd.read_excel(qoi_file)
    
    # QoI definitions (matching convergence_roache.py output)
    qois = [
        ("mean_ux_mm", "Mean $u_x$ [mm]", "o-"),
        ("mean_S", "Mean $S$ [–]", "s-"),
        ("mean_rho", "Mean $\\rho$ [–]", "^-"),
        ("anisotropy", "Anisotropy", "d-"),
    ]
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Roache QoI Convergence - Spatial (dt = {dt_fixed} days)", fontsize=14)
    
    ax_abs, ax_rel = axes
    
    h = df["h"].values
    
    # Plot all QoIs on both subplots
    for qoi_name, qoi_label, marker in qois:
        y = df[qoi_name].values
        y_abs = np.abs(y)  # Absolute values for log scale
        
        # Absolute convergence (left)
        ax_abs.loglog(h, y_abs, marker, label=qoi_label, linewidth=2, markersize=6)
        
        # Relative change from finest (right)
        if len(y) > 1:
            y_finest = y[-1]
            rel_change = np.abs((y - y_finest) / y_finest) if y_finest != 0 else np.abs(y - y_finest)
            # Skip finest point (zero relative change)
            ax_rel.loglog(h[:-1], rel_change[:-1], marker, label=qoi_label, linewidth=2, markersize=6)
    
    # Add reference slopes
    if len(h) > 1:
        # Use first QoI for reference line position
        y_ref = np.abs(df[qois[0][0]].values)
        ref_line_1 = y_ref[0] * (h / h[0]) ** 1
        ref_line_2 = y_ref[0] * (h / h[0]) ** 2
        ax_abs.loglog(h, ref_line_1, "k--", alpha=0.4, linewidth=1.5, label="O(h)")
        ax_abs.loglog(h, ref_line_2, "k:", alpha=0.4, linewidth=1.5, label="O(h²)")
        
        # Relative change reference
        if len(h) > 1:
            rel_ref = rel_change[0] * (h[:-1] / h[0]) ** 2
            ax_rel.loglog(h[:-1], rel_ref, "k:", alpha=0.4, linewidth=1.5, label="O(h²)")
    
    # Configure left subplot (absolute values)
    ax_abs.set_xlabel("Mesh size h", fontsize=12)
    ax_abs.set_ylabel("QoI Value (absolute)", fontsize=12)
    ax_abs.set_title("Absolute QoI Values", fontsize=13)
    ax_abs.legend(loc="best", fontsize=9)
    ax_abs.grid(True, alpha=0.3, which="both")
    
    # Configure right subplot (relative change)
    ax_rel.set_xlabel("Mesh size h", fontsize=12)
    ax_rel.set_ylabel("Relative Change from Finest", fontsize=12)
    ax_rel.set_title("Convergence to Finest Grid", fontsize=13)
    ax_rel.legend(loc="best", fontsize=9)
    ax_rel.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Spatial Roache plot saved to {output_file}")
    plt.close()


def create_roache_temporal_plot(
    qoi_file: Path,
    N_fixed: int,
    output_file: Path,
) -> None:
    """Create temporal Roache plot with 1x2 subplots (all QoIs on each)."""
    # Load QoI data
    df = pd.read_excel(qoi_file)
    
    # QoI definitions (matching convergence_roache.py output)
    qois = [
        ("mean_ux_mm", "Mean $u_x$ [mm]", "o-"),
        ("mean_S", "Mean $S$ [–]", "s-"),
        ("mean_rho", "Mean $\\rho$ [–]", "^-"),
        ("anisotropy", "Anisotropy", "d-"),
    ]
    
    # Create figure with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Roache QoI Convergence - Temporal (N = {N_fixed})", fontsize=14)
    
    ax_abs, ax_rel = axes
    
    dt = df["dt_days"].values
    
    # Plot all QoIs on both subplots
    for qoi_name, qoi_label, marker in qois:
        y = df[qoi_name].values
        y_abs = np.abs(y)  # Absolute values for log scale
        
        # Absolute convergence (left)
        ax_abs.loglog(dt, y_abs, marker, label=qoi_label, linewidth=2, markersize=6)
        
        # Relative change from finest (right)
        if len(y) > 1:
            y_finest = y[-1]
            rel_change = np.abs((y - y_finest) / y_finest) if y_finest != 0 else np.abs(y - y_finest)
            # Skip finest point (zero relative change)
            ax_rel.loglog(dt[:-1], rel_change[:-1], marker, label=qoi_label, linewidth=2, markersize=6)
    
    # Add reference slopes
    if len(dt) > 1:
        # Use first QoI for reference line position
        y_ref = np.abs(df[qois[0][0]].values)
        ref_line_1 = y_ref[0] * (dt / dt[0]) ** 1
        ax_abs.loglog(dt, ref_line_1, "k--", alpha=0.4, linewidth=1.5, label="O(dt)")
        
        # Relative change reference
        if len(dt) > 1:
            rel_ref = rel_change[0] * (dt[:-1] / dt[0]) ** 1
            ax_rel.loglog(dt[:-1], rel_ref, "k--", alpha=0.4, linewidth=1.5, label="O(dt)")
    
    # Configure left subplot (absolute values)
    ax_abs.set_xlabel("Time step dt (days)", fontsize=12)
    ax_abs.set_ylabel("QoI Value (absolute)", fontsize=12)
    ax_abs.set_title("Absolute QoI Values", fontsize=13)
    ax_abs.legend(loc="best", fontsize=9)
    ax_abs.grid(True, alpha=0.3, which="both")
    
    # Configure right subplot (relative change)
    ax_rel.set_xlabel("Time step dt (days)", fontsize=12)
    ax_rel.set_ylabel("Relative Change from Finest", fontsize=12)
    ax_rel.set_title("Convergence to Finest Time Step", fontsize=13)
    ax_rel.legend(loc="best", fontsize=9)
    ax_rel.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Temporal Roache plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    roache_file = Path("analysis/convergence_analysis/roache_convergence.xlsx")
    output_dir = Path("manuscript/images")
    
    # Parameters (for titles)
    dt_fixed = 25.0
    N_fixed = 81
    
    # Verify file exists
    if not roache_file.exists():
        print(f"ERROR: Missing {roache_file}")
        print("Run convergence_roache.py first to generate Roache GCI data.")
        sys.exit(1)
    
    print("=" * 80)
    print("Generating Roache QoI Convergence Plots")
    print("=" * 80)
    print(f"Loading data from {roache_file}")
    print()
    
    # Load QoI data from XLSX
    spatial_qoi_df = pd.read_excel(roache_file, sheet_name="spatial_qoi")
    temporal_qoi_df = pd.read_excel(roache_file, sheet_name="temporal_qoi")
    
    # Save individual sheets as temporary XLSX for plotting functions
    temp_dir = Path("analysis/convergence_analysis/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    spatial_temp = temp_dir / "spatial_qoi.xlsx"
    temporal_temp = temp_dir / "temporal_qoi.xlsx"
    
    spatial_qoi_df.to_excel(spatial_temp, index=False)
    temporal_qoi_df.to_excel(temporal_temp, index=False)
    
    # Create spatial convergence plot
    print(f"Creating spatial QoI plot (dt={dt_fixed})...")
    create_roache_spatial_plot(
        qoi_file=spatial_temp,
        dt_fixed=dt_fixed,
        output_file=output_dir / f"roache_spatial_dt{dt_fixed}.png",
    )
    
    # Create temporal convergence plot
    print(f"Creating temporal QoI plot (N={N_fixed})...")
    create_roache_temporal_plot(
        qoi_file=temporal_temp,
        N_fixed=N_fixed,
        output_file=output_dir / f"roache_temporal_N{N_fixed}.png",
    )
    
    # Cleanup temp files
    spatial_temp.unlink()
    temporal_temp.unlink()
    temp_dir.rmdir()
    
    print()
    print("=" * 80)
    print("Roache QoI plotting complete!")
    print("=" * 80)
