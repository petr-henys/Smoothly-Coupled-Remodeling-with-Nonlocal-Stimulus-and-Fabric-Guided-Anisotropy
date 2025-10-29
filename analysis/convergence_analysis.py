"""Convergence analysis for spatial and temporal refinement.

Phase 2 script: Loads convergence sweep results, computes L2/H1 errors,
and generates convergence plots for spatial (fixed dt) and temporal (fixed N) refinement.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from mpi4py import MPI

# Import matplotlib only on rank 0 to avoid MPI issues
comm = MPI.COMM_WORLD
if comm.rank == 0:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

from analysis.utils import (
    load_sweep_records,
    load_field_from_npz,
    compute_l2_h1_errors,
)


def analyze_field_errors(
    records: List[Dict[str, Any]],
    field_name: str,
    field_type: str,
    comm: MPI.Comm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute L2/H1 errors for a single field across all sweep points.
    
    Returns N_values, h_values, l2_errors, h1_errors arrays.
    """
    N_values = []
    h_values = []
    l2_errors = []
    h1_errors = []
    
    prev_field = None
    prev_N = None
    
    for record in records:
        run_dir = Path(record["output_path"])
        N = int(record["N"])
        
        domain, field = load_field_from_npz(
            run_dir, comm, N, field_name, field_type
        )
        
        if prev_field is not None:
            # Compute errors on finer mesh
            l2_err, h1_err = compute_l2_h1_errors(prev_field, field, domain)
            N_values.append(prev_N)
            h_values.append(1.0 / prev_N)
            l2_errors.append(l2_err)
            h1_errors.append(h1_err)
        
        prev_field = field
        prev_N = N
    
    return (
        np.array(N_values),
        np.array(h_values),
        np.array(l2_errors),
        np.array(h1_errors),
    )

import sys
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI

from analysis.utils import (
    load_sweep_records,
    load_field_from_npz,
    compute_l2_h1_errors,
)


def analyze_field_errors(
    records: List[Dict[str, Any]],
    field_name: str,
    field_type: str,
    comm: MPI.Comm,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute L2/H1 errors for a single field across all sweep points.
    
    Returns N_values, h_values, l2_errors, h1_errors arrays.
    """
    N_values = []
    h_values = []
    l2_errors = []
    h1_errors = []
    
    prev_field = None
    prev_N = None
    
    for record in records:
        run_dir = Path(record["output_path"])
        N = int(record["N"])
        
        domain, field = load_field_from_npz(
            run_dir, comm, N, field_name, field_type
        )
        
        if prev_field is not None:
            # Compute errors on finer mesh
            l2_err, h1_err = compute_l2_h1_errors(prev_field, field, domain)
            N_values.append(prev_N)
            h_values.append(1.0 / prev_N)
            l2_errors.append(l2_err)
            h1_errors.append(h1_err)
        
        prev_field = field
        prev_N = N
    
    return (
        np.array(N_values),
        np.array(h_values),
        np.array(l2_errors),
        np.array(h1_errors),
    )


def filter_records_by_dt(
    records: List[Dict[str, Any]], dt_value: float
) -> List[Dict[str, Any]]:
    """Filter records by dt_days and sort by N."""
    filtered = [r for r in records if abs(r["dt_days"] - dt_value) < 1e-6]
    return sorted(filtered, key=lambda r: r["N"])


def filter_records_by_N(
    records: List[Dict[str, Any]], N_value: int
) -> List[Dict[str, Any]]:
    """Filter records by N and sort by dt_days."""
    filtered = [r for r in records if r["N"] == N_value]
    return sorted(filtered, key=lambda r: r["dt_days"])


def create_spatial_convergence_plot(
    base_dir: Path,
    dt_fixed: float,
    output_file: Path,
    comm: MPI.Comm,
) -> None:
    """Create spatial convergence plot (varying N, fixed dt)."""
    records = load_sweep_records(base_dir, comm)
    records_filtered = filter_records_by_dt(records, dt_fixed)
    
    if comm.rank == 0:
        print(f"Spatial convergence: dt={dt_fixed}, {len(records_filtered)} meshes")
    
    # Field definitions
    fields = [
        ("u", "vector", "Displacement"),
        ("rho", "scalar", "Density"),
        ("S", "scalar", "Stimulus"),
        ("A", "tensor", "Orientation"),
    ]
    
    # Compute errors for all fields
    results = {}
    for field_name, field_type, field_label in fields:
        N_vals, h_vals, l2_errs, h1_errs = analyze_field_errors(
            records_filtered, field_name, field_type, comm
        )
        results[field_name] = {
            "N": N_vals,
            "h": h_vals,
            "l2": l2_errs,
            "h1": h1_errs,
            "label": field_label,
        }
    
    if comm.rank != 0:
        return
    
    # Create plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Spatial Convergence (dt = {dt_fixed} days)", fontsize=14)
    
    for idx, (field_name, _, _) in enumerate(fields):
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
    base_dir: Path,
    N_fixed: int,
    output_file: Path,
    comm: MPI.Comm,
) -> None:
    """Create temporal convergence plot (varying dt, fixed N)."""
    records = load_sweep_records(base_dir, comm)
    records_filtered = filter_records_by_N(records, N_fixed)
    
    if comm.rank == 0:
        print(f"Temporal convergence: N={N_fixed}, {len(records_filtered)} timesteps")
    
    # Field definitions
    fields = [
        ("u", "vector", "Displacement"),
        ("rho", "scalar", "Density"),
        ("S", "scalar", "Stimulus"),
        ("A", "tensor", "Orientation"),
    ]
    
    # Compute errors for all fields
    results = {}
    for field_name, field_type, field_label in fields:
        N_vals, _, l2_errs, h1_errs = analyze_field_errors(
            records_filtered, field_name, field_type, comm
        )
        # For temporal, use dt values not h
        dt_vals = np.array([r["dt_days"] for r in records_filtered[:-1]])
        results[field_name] = {
            "dt": dt_vals,
            "l2": l2_errs,
            "h1": h1_errs,
            "label": field_label,
        }
    
    if comm.rank != 0:
        return
    
    # Create plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Temporal Convergence (N = {N_fixed})", fontsize=14)
    
    for idx, (field_name, _, _) in enumerate(fields):
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
    comm = MPI.COMM_WORLD
    
    base_dir = Path("results/convergence_sweep")
    
    # Spatial convergence (fixed dt=25.0 days, varying N)
    create_spatial_convergence_plot(
        base_dir=base_dir,
        dt_fixed=25.0,
        output_file=Path("manuscript/images/spatial_convergence.png"),
        comm=comm,
    )
    
    # Temporal convergence (fixed N=36, varying dt)
    create_temporal_convergence_plot(
        base_dir=base_dir,
        N_fixed=36,
        output_file=Path("manuscript/images/temporal_convergence.png"),
        comm=comm,
    )
    
    if comm.rank == 0:
        print("\nConvergence analysis complete!")
