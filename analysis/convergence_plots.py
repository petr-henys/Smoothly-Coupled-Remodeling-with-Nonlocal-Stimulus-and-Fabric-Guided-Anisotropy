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
    """Create spatial convergence plot from XLSX data."""
    # Field definitions
    fields = [
        ("u", "Displacement"),
        ("rho", "Density"),
        ("S", "Stimulus"),
        ("A", "Orientation"),
    ]
    
    # Load data from XLSX
    results = {}
    for field_name, field_label in fields:
        sheet_name = f"spatial_{field_name}_dt{dt_fixed}"
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
        results[field_name] = {
            "h": df["h"].values,
            "l2": df["L2_error"].values,
            "h1": df["H1_error"].values,
            "label": field_label,
        }
    
    # Create plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Spatial Convergence (dt = {dt_fixed} days)", fontsize=14)
    
    for idx, (field_name, _) in enumerate(fields):
        data = results[field_name]
        h = data["h"]
        l2 = data["l2"]
        h1 = data["h1"]
        label = data["label"]
        
        # L2 norm
        ax_l2 = axes[0, idx]
        ax_l2.loglog(h, l2, "o-", label=f"{label} L2", linewidth=2, markersize=6)
        # Reference slopes
        if len(h) > 1:
            ax_l2.loglog(h, l2[0] * (h / h[0]) ** 1, "--", alpha=0.5, label="O(h)")
            ax_l2.loglog(h, l2[0] * (h / h[0]) ** 2, ":", alpha=0.5, label="O(h²)")
        ax_l2.set_xlabel("h")
        ax_l2.set_ylabel("L2 Error")
        ax_l2.set_title(f"{label} - L2 Norm")
        ax_l2.legend()
        ax_l2.grid(True, alpha=0.3)
        
        # H1 seminorm
        ax_h1 = axes[1, idx]
        ax_h1.loglog(h, h1, "s-", label=f"{label} H1", linewidth=2, markersize=6)
        # Reference slopes
        if len(h) > 1:
            ax_h1.loglog(h, h1[0] * (h / h[0]) ** 1, "--", alpha=0.5, label="O(h)")
            ax_h1.loglog(h, h1[0] * (h / h[0]) ** 2, ":", alpha=0.5, label="O(h²)")
        ax_h1.set_xlabel("h")
        ax_h1.set_ylabel("H1 Error")
        ax_h1.set_title(f"{label} - H1 Seminorm")
        ax_h1.legend()
        ax_h1.grid(True, alpha=0.3)
    
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
    """Create temporal convergence plot from XLSX data."""
    # Field definitions
    fields = [
        ("u", "Displacement"),
        ("rho", "Density"),
        ("S", "Stimulus"),
        ("A", "Orientation"),
    ]
    
    # Load data from XLSX
    results = {}
    for field_name, field_label in fields:
        sheet_name = f"temporal_{field_name}_N{N_fixed}"
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
        results[field_name] = {
            "dt": df["dt_days"].values,
            "l2": df["L2_error"].values,
            "h1": df["H1_error"].values,
            "label": field_label,
        }
    
    # Create plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Temporal Convergence (N = {N_fixed})", fontsize=14)
    
    for idx, (field_name, _) in enumerate(fields):
        data = results[field_name]
        dt = data["dt"]
        l2 = data["l2"]
        h1 = data["h1"]
        label = data["label"]
        
        # L2 norm
        ax_l2 = axes[0, idx]
        ax_l2.loglog(dt, l2, "o-", label=f"{label} L2", linewidth=2, markersize=6)
        # Reference slopes
        if len(dt) > 1:
            ax_l2.loglog(dt, l2[0] * (dt / dt[0]) ** 1, "--", alpha=0.5, label="O(dt)")
            ax_l2.loglog(dt, l2[0] * (dt / dt[0]) ** 2, ":", alpha=0.5, label="O(dt²)")
        ax_l2.set_xlabel("dt (days)")
        ax_l2.set_ylabel("L2 Error")
        ax_l2.set_title(f"{label} - L2 Norm")
        ax_l2.legend()
        ax_l2.grid(True, alpha=0.3)
        
        # H1 seminorm
        ax_h1 = axes[1, idx]
        ax_h1.loglog(dt, h1, "s-", label=f"{label} H1", linewidth=2, markersize=6)
        # Reference slopes
        if len(dt) > 1:
            ax_h1.loglog(dt, h1[0] * (dt / dt[0]) ** 1, "--", alpha=0.5, label="O(dt)")
            ax_h1.loglog(dt, h1[0] * (dt / dt[0]) ** 2, ":", alpha=0.5, label="O(dt²)")
        ax_h1.set_xlabel("dt (days)")
        ax_h1.set_ylabel("H1 Error")
        ax_h1.set_title(f"{label} - H1 Seminorm")
        ax_h1.legend()
        ax_h1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Temporal convergence plot saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    xlsx_file = Path("results/convergence_analysis/convergence_data.xlsx")
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
