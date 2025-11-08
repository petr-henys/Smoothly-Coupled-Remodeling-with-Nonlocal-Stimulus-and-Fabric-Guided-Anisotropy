"""Generate performance dashboard from convergence sweep results.

Analyzes solver performance metrics (time, KSP iterations, memory) vs. DOFs
for fixed dt, varying mesh resolution.

Usage:
    python3 analysis/convergence_performance_plots.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_performance_data(
    base_dir: Path,
    dt_fixed: float,
) -> pd.DataFrame:
    """Load performance metrics from convergence sweep for fixed dt.
    
    Returns DataFrame with columns:
    - N: mesh resolution
    - total_dofs: total DOFs
    - mech_time_median: median mechanics solver time [s]
    - stim_time_median: median stimulus solver time [s]
    - dens_time_median: median density solver time [s]
    - dir_time_median: median direction solver time [s]
    - mech_iters_median: median KSP iterations (mechanics)
    - stim_iters_median: median KSP iterations (stimulus)
    - dens_iters_median: median KSP iterations (density)
    - dir_iters_median: median KSP iterations (direction)
    - memory_mb_max: max RSS memory [MB]
    """
    # Load sweep summary
    sweep_csv = base_dir / "sweep_summary.csv"
    df_sweep = pd.read_csv(sweep_csv)
    
    # Filter by dt
    df_filtered = df_sweep[np.abs(df_sweep["dt_days"] - dt_fixed) < 1e-6].copy()
    df_filtered = df_filtered.sort_values("N")
    
    performance_data = []
    
    for _, row in df_filtered.iterrows():
        output_dir = row["output_dir"]
        N = row["N"]
        run_dir = base_dir / output_dir
        
        # Load run summary
        summary_file = run_dir / "run_summary.json"
        if not summary_file.exists():
            print(f"Warning: Missing {summary_file}")
            continue
        
        with open(summary_file, "r") as f:
            summary = json.load(f)
        
        # Load subiterations for median KSP iterations
        subiters_file = run_dir / "subiterations.csv"
        if not subiters_file.exists():
            print(f"Warning: Missing {subiters_file}")
            continue
        
        df_sub = pd.read_csv(subiters_file)
        
        # Extract performance metrics
        perf = {
            "N": N,
            "total_dofs": summary["dofs"]["total"],
            "mech_time_median": summary["median_step_times_s"]["mech"],
            "stim_time_median": summary["median_step_times_s"]["stim"],
            "dens_time_median": summary["median_step_times_s"]["dens"],
            "dir_time_median": summary["median_step_times_s"]["dir"],
            "mech_iters_median": df_sub["mech_iters"].median(),
            "stim_iters_median": df_sub["stim_iters"].median(),
            "dens_iters_median": df_sub["dens_iters"].median(),
            "dir_iters_median": df_sub["dir_iters"].median(),
            "memory_mb_max": summary["memory_mb"]["rss_max"],
        }
        
        performance_data.append(perf)
    
    return pd.DataFrame(performance_data)


def create_performance_dashboard(
    df: pd.DataFrame,
    dt_fixed: float,
    output_file: Path,
) -> None:
    """Create performance dashboard with 3 subplots (1x3).
    
    Left: Solver times vs DOFs
    Middle: KSP iterations vs DOFs
    Right: Memory vs DOFs
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Performance Dashboard (dt = {dt_fixed} days)", fontsize=14, fontweight="bold")
    
    dofs = df["total_dofs"].values
    
    # ========================================================================
    # LEFT: Solver times vs DOFs
    # ========================================================================
    ax = axes[0]
    
    solvers = [
        ("mech_time_median", "Mechanics", "o-"),
        ("stim_time_median", "Stimulus", "s-"),
        ("dens_time_median", "Density", "^-"),
        ("dir_time_median", "Direction", "d-"),
    ]
    
    for field, label, marker in solvers:
        times = df[field].values
        ax.loglog(dofs, times, marker, label=label, linewidth=2, markersize=6)
    
    # Reference slopes
    if len(dofs) > 1:
        t_ref = df["mech_time_median"].values[0]
        # O(N) for linear scaling
        ref_n1 = t_ref * (dofs / dofs[0]) ** 1.0
        ax.loglog(dofs, ref_n1, "k--", alpha=0.4, linewidth=1.5, label="O(N)")
        # O(N^1.5) for typical sparse direct
        ref_n15 = t_ref * (dofs / dofs[0]) ** 1.5
        ax.loglog(dofs, ref_n15, "k:", alpha=0.4, linewidth=1.5, label="O(N$^{1.5}$)")
    
    ax.set_xlabel("Total DOFs", fontsize=12)
    ax.set_ylabel("Median Solver Time [s]", fontsize=12)
    ax.set_title("Solver Time Scaling", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    
    # ========================================================================
    # MIDDLE: KSP iterations vs DOFs
    # ========================================================================
    ax = axes[1]
    
    ksp_fields = [
        ("mech_iters_median", "Mechanics", "o-"),
        ("stim_iters_median", "Stimulus", "s-"),
        ("dens_iters_median", "Density", "^-"),
        ("dir_iters_median", "Direction", "d-"),
    ]
    
    for field, label, marker in ksp_fields:
        iters = df[field].values
        ax.semilogx(dofs, iters, marker, label=label, linewidth=2, markersize=6)
    
    ax.set_xlabel("Total DOFs", fontsize=12)
    ax.set_ylabel("Median KSP Iterations", fontsize=12)
    ax.set_title("KSP Iteration Scaling", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    
    # ========================================================================
    # RIGHT: Memory vs DOFs
    # ========================================================================
    ax = axes[2]
    
    memory = df["memory_mb_max"].values
    ax.loglog(dofs, memory, "o-", label="Max RSS", linewidth=2, markersize=6, color="tab:red")
    
    # Reference slopes
    if len(dofs) > 1:
        mem_ref = memory[0]
        # O(N) for linear memory scaling
        ref_n1 = mem_ref * (dofs / dofs[0]) ** 1.0
        ax.loglog(dofs, ref_n1, "k--", alpha=0.4, linewidth=1.5, label="O(N)")
    
    ax.set_xlabel("Total DOFs", fontsize=12)
    ax.set_ylabel("Max RSS Memory [MB]", fontsize=12)
    ax.set_title("Memory Scaling", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Performance dashboard saved to {output_file}")
    plt.close()


def export_performance_table(
    df: pd.DataFrame,
    dt_fixed: float,
    output_file: Path,
) -> None:
    """Export performance data as CSV table."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, float_format="%.6f")
    print(f"Performance table saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    base_dir = Path("results/convergence_sweep")
    output_dir = Path("manuscript/images")
    tables_dir = Path("manuscript/tables")
    
    # Fixed dt for performance analysis
    dt_fixed = 25.0  # days
    
    print("=" * 80)
    print("Performance Dashboard Generation")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Fixed dt: {dt_fixed} days (varying N)")
    print()
    
    # Load performance data
    print("Loading performance data...")
    df = load_performance_data(base_dir, dt_fixed)
    
    if df.empty:
        print("ERROR: No performance data found!")
        sys.exit(1)
    
    print(f"Loaded {len(df)} performance records:")
    print(f"  N range: {df['N'].min()} - {df['N'].max()}")
    print(f"  DOFs range: {df['total_dofs'].min()} - {df['total_dofs'].max()}")
    print()
    
    # Create dashboard plot
    print("Creating performance dashboard...")
    create_performance_dashboard(
        df=df,
        dt_fixed=dt_fixed,
        output_file=output_dir / f"performance_dashboard_dt{dt_fixed}.png",
    )
    
    # Export table
    print("Exporting performance table...")
    export_performance_table(
        df=df,
        dt_fixed=dt_fixed,
        output_file=tables_dir / f"performance_data_dt{dt_fixed}.csv",
    )
    
    print()
    print("=" * 80)
    print("Performance dashboard generation complete!")
    print("=" * 80)
