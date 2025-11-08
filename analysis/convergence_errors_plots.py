"""Generate convergence plots from precomputed XLSX data.

This script loads convergence data exported by convergence_analysis.py
and creates publication-quality plots without recomputing errors.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def create_spatial_convergence_plot(
    xlsx_file: Path,
    dt_fixed: float,
    output_file: Path,
) -> None:
    """Create spatial convergence plot from XLSX data with L2 and H1 subplots."""
    # Field definitions
    fields = [
        ("u", "Displacement", "o-"),
        ("rho", "Density", "s-"),
        ("S", "Stimulus", "^-"),
        ("A", "Orientation", "d-"),
    ]
    
    # Load data from XLSX
    results = {}
    for field_name, field_label, _ in fields:
        sheet_name = f"spatial_{field_name}_dt{dt_fixed}"
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
        results[field_name] = {
            "h": df["h"].values,
            "l2": df["L2_error"].values,
            "h1": df["H1_error"].values,
            "label": field_label,
        }
    
    # Create plot with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Spatial Convergence (dt = {dt_fixed} days)", fontsize=14)
    
    ax_l2, ax_h1 = axes
    
    # Plot all fields on each subplot
    for field_name, field_label, marker in fields:
        data = results[field_name]
        h = data["h"]
        l2 = data["l2"]
        h1 = data["h1"]
        
        # L2 norm (left subplot)
        ax_l2.loglog(h, l2, marker, label=field_label, linewidth=2, markersize=6)
        
        # H1 seminorm (right subplot)
        ax_h1.loglog(h, h1, marker, label=field_label, linewidth=2, markersize=6)
    
    # Add reference slopes to L2 plot
    h = results["u"]["h"]
    l2_ref = results["u"]["l2"][0]
    if len(h) > 1:
        ax_l2.loglog(h, l2_ref * (h / h[0]) ** 1, "k--", alpha=0.4, linewidth=1.5, label="O(h)")
        ax_l2.loglog(h, l2_ref * (h / h[0]) ** 2, "k:", alpha=0.4, linewidth=1.5, label="O(h²)")
    
    # Add reference slopes to H1 plot
    h1_ref = results["u"]["h1"][0]
    if len(h) > 1:
        ax_h1.loglog(h, h1_ref * (h / h[0]) ** 1, "k--", alpha=0.4, linewidth=1.5, label="O(h)")
        ax_h1.loglog(h, h1_ref * (h / h[0]) ** 2, "k:", alpha=0.4, linewidth=1.5, label="O(h²)")
    
    # Configure L2 subplot
    ax_l2.set_xlabel("Mesh size h", fontsize=12)
    ax_l2.set_ylabel("L2 Error", fontsize=12)
    ax_l2.set_title("L2 Norm", fontsize=13)
    ax_l2.legend(loc="best", fontsize=10)
    ax_l2.grid(True, alpha=0.3, which="both")
    
    # Configure H1 subplot
    ax_h1.set_xlabel("Mesh size h", fontsize=12)
    ax_h1.set_ylabel("H1 Seminorm Error", fontsize=12)
    ax_h1.set_title("H1 Seminorm", fontsize=13)
    ax_h1.legend(loc="best", fontsize=10)
    ax_h1.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Spatial convergence plot saved to {output_file}")
    plt.close()


def create_temporal_convergence_plot(
    xlsx_file: Path,
    N_fixed: int,
    output_file: Path,
) -> None:
    """Create temporal convergence plot from XLSX data with L2 and H1 subplots."""
    # Field definitions
    fields = [
        ("u", "Displacement", "o-"),
        ("rho", "Density", "s-"),
        ("S", "Stimulus", "^-"),
        ("A", "Orientation", "d-"),
    ]
    
    # Load data from XLSX
    results = {}
    for field_name, field_label, _ in fields:
        sheet_name = f"temporal_{field_name}_N{N_fixed}"
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
        results[field_name] = {
            "dt": df["dt_days"].values,
            "l2": df["L2_error"].values,
            "h1": df["H1_error"].values,
            "label": field_label,
        }
    
    # Create plot with 1x2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Temporal Convergence (N = {N_fixed})", fontsize=14)
    
    ax_l2, ax_h1 = axes
    
    # Plot all fields on each subplot
    for field_name, field_label, marker in fields:
        data = results[field_name]
        dt = data["dt"]
        l2 = data["l2"]
        h1 = data["h1"]
        
        # L2 norm (left subplot)
        ax_l2.loglog(dt, l2, marker, label=field_label, linewidth=2, markersize=6)
        
        # H1 seminorm (right subplot)
        ax_h1.loglog(dt, h1, marker, label=field_label, linewidth=2, markersize=6)
    
    # Add reference slopes to L2 plot
    dt = results["u"]["dt"]
    l2_ref = results["u"]["l2"][0]
    if len(dt) > 1:
        ax_l2.loglog(dt, l2_ref * (dt / dt[0]) ** 1, "k--", alpha=0.4, linewidth=1.5, label="O(dt)")
        ax_l2.loglog(dt, l2_ref * (dt / dt[0]) ** 2, "k:", alpha=0.4, linewidth=1.5, label="O(dt²)")
    
    # Add reference slopes to H1 plot
    h1_ref = results["u"]["h1"][0]
    if len(dt) > 1:
        ax_h1.loglog(dt, h1_ref * (dt / dt[0]) ** 1, "k--", alpha=0.4, linewidth=1.5, label="O(dt)")
        ax_h1.loglog(dt, h1_ref * (dt / dt[0]) ** 2, "k:", alpha=0.4, linewidth=1.5, label="O(dt²)")
    
    # Configure L2 subplot
    ax_l2.set_xlabel("Time step dt (days)", fontsize=12)
    ax_l2.set_ylabel("L2 Error", fontsize=12)
    ax_l2.set_title("L2 Norm", fontsize=13)
    ax_l2.legend(loc="best", fontsize=10)
    ax_l2.grid(True, alpha=0.3, which="both")
    
    # Configure H1 subplot
    ax_h1.set_xlabel("Time step dt (days)", fontsize=12)
    ax_h1.set_ylabel("H1 Seminorm Error", fontsize=12)
    ax_h1.set_title("H1 Seminorm", fontsize=13)
    ax_h1.legend(loc="best", fontsize=10)
    ax_h1.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Temporal convergence plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    xlsx_file = Path("analysis/convergence_analysis/convergence_data.xlsx")
    output_dir = Path("manuscript/images")
    
    # Choose specific dt and N for main plots (adjust as needed)
    dt_fixed = 25.0  # days
    N_fixed = 36
    
    # Verify XLSX exists
    if not xlsx_file.exists():
        raise FileNotFoundError(
            f"Convergence data not found: {xlsx_file}\n"
            "Run convergence_analysis.py first to generate the data."
        )
    
    print(f"Loading convergence data from {xlsx_file}")
    print(f"Plotting spatial convergence for dt={dt_fixed}")
    print(f"Plotting temporal convergence for N={N_fixed}")
    
    # Create plots
    create_spatial_convergence_plot(
        xlsx_file=xlsx_file,
        dt_fixed=dt_fixed,
        output_file=output_dir / f"spatial_convergence_dt{dt_fixed}.png",
    )
    
    create_temporal_convergence_plot(
        xlsx_file=xlsx_file,
        N_fixed=N_fixed,
        output_file=output_dir / f"temporal_convergence_N{N_fixed}.png",
    )
    
    print("\nConvergence plots complete!")
    print("Note: To plot other dt/N combinations, edit dt_fixed/N_fixed at the top of this script")
