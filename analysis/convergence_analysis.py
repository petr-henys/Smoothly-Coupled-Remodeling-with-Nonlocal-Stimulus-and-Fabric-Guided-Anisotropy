"""Convergence analysis for spatial and temporal refinement.

Phase 2 script: Loads convergence sweep results, computes L2/H1 errors,
and exports detailed XLSX tables for spatial (fixed dt) and temporal (fixed N) refinement.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
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
    base_dir: Path,
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
        output_dir = record["output_dir"]
        run_dir = base_dir / output_dir
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


def compute_spatial_convergence_data(
    base_dir: Path,
    dt_value: float,
    comm: MPI.Comm,
) -> Dict[str, pd.DataFrame]:
    """Compute spatial convergence data for a specific dt (varying N, fixed dt).
    
    Returns dictionary mapping field names to DataFrames with columns:
    N, h, L2_error, H1_error
    """
    records = load_sweep_records(base_dir, comm)
    records_filtered = filter_records_by_dt(records, dt_value)
    
    if comm.rank == 0:
        print(f"Spatial convergence: dt={dt_value}, {len(records_filtered)} meshes")
    
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
            records_filtered, field_name, field_type, comm, base_dir
        )
        
        df = pd.DataFrame({
            "N": N_vals,
            "h": h_vals,
            "L2_error": l2_errs,
            "H1_error": h1_errs,
        })
        results[field_name] = df
    
    return results


def compute_temporal_convergence_data(
    base_dir: Path,
    N_value: int,
    comm: MPI.Comm,
) -> Dict[str, pd.DataFrame]:
    """Compute temporal convergence data for a specific N (varying dt, fixed N).
    
    Returns dictionary mapping field names to DataFrames with columns:
    dt_days, L2_error, H1_error
    """
    records = load_sweep_records(base_dir, comm)
    records_filtered = filter_records_by_N(records, N_value)
    
    if comm.rank == 0:
        print(f"Temporal convergence: N={N_value}, {len(records_filtered)} timesteps")
    
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
            records_filtered, field_name, field_type, comm, base_dir
        )
        # For temporal, use dt values not h
        dt_vals = np.array([r["dt_days"] for r in records_filtered[:-1]])
        
        df = pd.DataFrame({
            "dt_days": dt_vals,
            "L2_error": l2_errs,
            "H1_error": h1_errs,
        })
        results[field_name] = df
    
    return results


def save_convergence_xlsx(
    all_spatial_data: Dict[float, Dict[str, pd.DataFrame]],
    all_temporal_data: Dict[int, Dict[str, pd.DataFrame]],
    output_file: Path,
) -> None:
    """Save all spatial and temporal convergence data to XLSX with multiple sheets.
    
    Args:
        all_spatial_data: Dict mapping dt_value -> {field_name -> DataFrame}
        all_temporal_data: Dict mapping N_value -> {field_name -> DataFrame}
        output_file: Output XLSX path
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Spatial sheets (one per dt × field combination)
        for dt_value, spatial_data in all_spatial_data.items():
            for field_name, df in spatial_data.items():
                sheet_name = f"spatial_{field_name}_dt{dt_value}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Temporal sheets (one per N × field combination)
        for N_value, temporal_data in all_temporal_data.items():
            for field_name, df in temporal_data.items():
                sheet_name = f"temporal_{field_name}_N{N_value}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Convergence data saved to {output_file}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    
    base_dir = Path("results/convergence_sweep")
    output_dir = Path("analysis/convergence_analysis")
    
    # Load all sweep records
    all_records = load_sweep_records(base_dir, comm)
    
    # Extract unique dt and N values
    dt_values = sorted(set(r["dt_days"] for r in all_records))
    N_values = sorted(set(r["N"] for r in all_records))
    
    if comm.rank == 0:
        print(f"Found {len(dt_values)} unique dt values: {dt_values}")
        print(f"Found {len(N_values)} unique N values: {N_values}")
    
    # Compute spatial convergence for all dt values
    all_spatial_data = {}
    for dt_val in dt_values:
        if comm.rank == 0:
            print(f"\n=== Processing spatial convergence for dt={dt_val} ===")
        spatial_data = compute_spatial_convergence_data(
            base_dir=base_dir,
            dt_value=dt_val,
            comm=comm,
        )
        all_spatial_data[dt_val] = spatial_data
    
    # Compute temporal convergence for all N values
    all_temporal_data = {}
    for N_val in N_values:
        if comm.rank == 0:
            print(f"\n=== Processing temporal convergence for N={N_val} ===")
        temporal_data = compute_temporal_convergence_data(
            base_dir=base_dir,
            N_value=N_val,
            comm=comm,
        )
        all_temporal_data[N_val] = temporal_data
    
    # Save all data to XLSX (rank 0 only)
    if comm.rank == 0:
        save_convergence_xlsx(
            all_spatial_data=all_spatial_data,
            all_temporal_data=all_temporal_data,
            output_file=output_dir / "convergence_data.xlsx",
        )
        print("\nConvergence analysis complete!")
        print(f"Generated {len(dt_values)} spatial convergence datasets")
        print(f"Generated {len(N_values)} temporal convergence datasets")
