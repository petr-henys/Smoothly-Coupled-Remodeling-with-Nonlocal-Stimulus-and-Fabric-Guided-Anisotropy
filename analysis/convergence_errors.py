"""Convergence analysis for spatial and temporal refinement.

Loads convergence sweep results from checkpoints, computes L2/H1 errors,
and exports detailed XLSX tables for spatial (fixed dt) and temporal (fixed N) refinement.

Requires adios4dolfinx: pip install adios4dolfinx

Usage:
    mpirun -n 1 python convergence_errors.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from mpi4py import MPI

from dolfinx import mesh, fem
import basix.ufl

from analysis.analysis_utils import (
    load_sweep_records,
    compute_l2_h1_errors,
    load_checkpoint_mesh,
    load_checkpoint_function,
)


def create_function_space(
    domain: mesh.Mesh,
    field_type: str,
) -> fem.FunctionSpace:
    """Create appropriate function space for field type.
    
    Supported types:
    - "scalar": P1 Lagrange scalar
    - "scalar_dg0": DG0 scalar (piecewise constant)
    - "vector": P1 Lagrange vector (3D)
    - "tensor": P1 Lagrange tensor (3x3)
    - "tensor_dg0": DG0 tensor (3x3, piecewise constant)
    """
    cell_name = domain.topology.cell_name()
    
    if field_type == "vector":
        element = basix.ufl.element("Lagrange", cell_name, 1, shape=(3,))
    elif field_type == "scalar":
        element = basix.ufl.element("Lagrange", cell_name, 1)
    elif field_type == "scalar_dg0":
        element = basix.ufl.element("DG", cell_name, 0)
    elif field_type == "tensor":
        element = basix.ufl.element("Lagrange", cell_name, 1, shape=(3, 3))
    elif field_type == "tensor_dg0":
        element = basix.ufl.element("DG", cell_name, 0, shape=(3, 3))
    else:
        raise ValueError(f"Unknown field type: {field_type}")
    
    return fem.functionspace(domain, element)


def load_field_from_checkpoint(
    run_dir: Path,
    field_name: str,
    field_type: str,
    comm: MPI.Comm,
    final_time: float | None = None,
) -> tuple[fem.Function, mesh.Mesh]:
    """Load a field from a run directory using adios4dolfinx checkpoint.
    
    Args:
        run_dir: Path to simulation output directory.
        field_name: Name of field to load (e.g., "rho", "u").
        field_type: "scalar", "vector", or "tensor".
        comm: MPI communicator.
        final_time: Time to load. If None, reads from config.json.
    
    Returns:
        Tuple of (function, mesh).
    
    Raises:
        FileNotFoundError: If checkpoint.bp not found in run_dir.
    """
    checkpoint_path = run_dir / "checkpoint.bp"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Run requires adios4dolfinx checkpoint."
        )
    
    # If final_time not specified, read from config.json
    if final_time is None:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            final_time = cfg.get("time", {}).get("total_time", 0.0)
    
    domain, _ = load_checkpoint_mesh(checkpoint_path, comm)
    space = create_function_space(domain, field_type)
    func = load_checkpoint_function(
        checkpoint_path, field_name, space, time=final_time, comm=comm
    )
    return func, domain


def analyze_field_errors(
    records: List[Dict[str, Any]],
    field_name: str,
    field_type: str,
    comm: MPI.Comm,
    base_dir: Path,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute L2/H1 errors for a single field across all sweep points.
    
    Loads fields from checkpoints (adios4dolfinx preferred, NPZ fallback),
    computes pairwise errors between consecutive refinement levels.
    
    Args:
        records: List of sweep records sorted by refinement level.
        field_name: Name of field to analyze (e.g., "rho", "u").
        field_type: "scalar", "vector", or "tensor".
        comm: MPI communicator.
        base_dir: Base directory containing sweep outputs.
        verbose: Print progress messages.
    
    Returns:
        Tuple of (N_values, h_values, l2_errors, h1_errors) as numpy arrays.
    """
    N_values = []
    h_values = []
    l2_errors = []
    h1_errors = []
    
    prev_field = None
    prev_N = None
    
    total = len(records)
    for idx, record in enumerate(records, start=1):
        output_dir = record["output_dir"]
        run_dir = Path(base_dir) / output_dir
        N = int(record["N"])
        
        if verbose and comm.rank == 0:
            print(f"  [{idx}/{total}] Loading {field_name} (N={N})...", flush=True)
        
        # Load field using available backend
        try:
            field, domain = load_field_from_checkpoint(
                run_dir, field_name, field_type, comm
            )
        except FileNotFoundError:
            if verbose and comm.rank == 0:
                print(f"      (skip) Field '{field_name}' not found in {run_dir}")
            return (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
        
        if prev_field is not None:
            if verbose and comm.rank == 0:
                print(f"  [{idx}/{total}] Computing errors (N_prev={prev_N} → N={N})...", flush=True)
            
            # Compute errors on the finer mesh
            l2_err, h1_err = compute_l2_h1_errors(prev_field, field, domain)
            N_values.append(prev_N)
            # Use a dimensional mesh size estimate based on the box extent.
            # For the box sweep we set nx=ny=N and nz scaled by aspect ratio, so
            # h ~ Lx/N. We infer Lx from the mesh bounding box.
            x_local = domain.geometry.x
            if x_local.size == 0:
                Lx = 1.0
            else:
                x_min = domain.comm.allreduce(float(np.min(x_local[:, 0])), op=MPI.MIN)
                x_max = domain.comm.allreduce(float(np.max(x_local[:, 0])), op=MPI.MAX)
                Lx = max(x_max - x_min, 1e-14)
            h_values.append(Lx / float(prev_N))
            l2_errors.append(l2_err)
            h1_errors.append(h1_err)
            if verbose and comm.rank == 0:
                print(f"      L2 error: {l2_err:.6e}, H1 error: {h1_err:.6e}", flush=True)
        
        prev_field = field
        prev_N = N
    
    return (
        np.array(N_values, dtype=float),
        np.array(h_values, dtype=float),
        np.array(l2_errors, dtype=float),
        np.array(h1_errors, dtype=float),
    )


def load_run_performance(run_dir: Path) -> dict[str, float]:
    """Load performance metrics from steps.csv (rank 0 written).

    Returns totals over the full run.
    """
    steps_path = run_dir / "steps.csv"
    if not steps_path.exists():
        return {
            "mech_iters": 0.0,
            "fab_iters": 0.0,
            "stim_iters": 0.0,
            "dens_iters": 0.0,
            "mech_time": 0.0,
            "fab_time": 0.0,
            "stim_time": 0.0,
            "dens_time": 0.0,
            "memory_mb": 0.0,
        }

    df = pd.read_csv(steps_path)
    # Totals over accepted steps
    totals = {
        "mech_iters": float(df.get("mech_iters", 0).sum()),
        "fab_iters": float(df.get("fab_iters", 0).sum()),
        "stim_iters": float(df.get("stim_iters", 0).sum()),
        "dens_iters": float(df.get("dens_iters", 0).sum()),
        "mech_time": float(df.get("mech_time", 0.0).sum()),
        "fab_time": float(df.get("fab_time", 0.0).sum()),
        "stim_time": float(df.get("stim_time", 0.0).sum()),
        "dens_time": float(df.get("dens_time", 0.0).sum()),
        "memory_mb": float(df.get("memory_mb", 0.0).max()) if "memory_mb" in df else 0.0,
    }
    return totals


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
        print(f"\n==> Spatial convergence: dt={dt_value}, {len(records_filtered)} meshes")
    
    # Field definitions (must match checkpoint contents from run_convergence_sweep.py)
    # Note: psi is cycle-averaged SED (DG0), more meaningful than u which is recomputed per loading case
    fields = [
        ("psi", "scalar_dg0", "SED (psi)"),
        ("rho", "scalar", "Density"),
        ("S", "scalar", "Stimulus"),
        ("L", "tensor", "Log-fabric"),
    ]
    
    # Compute errors for all fields
    results = {}
    total_fields = len(fields)
    for field_idx, (field_name, field_type, field_label) in enumerate(fields, start=1):
        if comm.rank == 0:
            print(f"\n  Field [{field_idx}/{total_fields}]: {field_label} ({field_name})")
        
        N_vals, h_vals, l2_errs, h1_errs = analyze_field_errors(
            records_filtered, field_name, field_type, comm, base_dir, verbose=True
        )
        
        if N_vals.size > 0:
            df = pd.DataFrame({
                "N": N_vals,
                "h": h_vals,
                "L2_error": l2_errs,
                "H1_error": h1_errs,
            })
            results[field_name] = df
        
        if comm.rank == 0:
            print(f"  ✓ Completed {field_label}")
    
    return results


def compute_spatial_performance_data(
    base_dir: Path,
    dt_value: float,
    comm: MPI.Comm,
) -> pd.DataFrame:
    """Aggregate per-run performance metrics for fixed dt, varying N."""
    records = load_sweep_records(base_dir, comm)
    records_filtered = filter_records_by_dt(records, dt_value)

    rows: list[dict[str, float]] = []
    for record in records_filtered:
        run_dir = Path(base_dir) / record["output_dir"]
        perf = load_run_performance(run_dir)
        rows.append({
            "N": float(record["N"]),
            "dt_days": float(record["dt_days"]),
            **perf,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Compute a dimensional h estimate like in error analysis.
    # Use the finest run's mesh for Lx.
    try:
        finest_dir = Path(base_dir) / records_filtered[-1]["output_dir"]
        checkpoint_path = finest_dir / "checkpoint.bp"
        if checkpoint_path.exists():
            domain, _ = load_checkpoint_mesh(checkpoint_path, comm)
            x_local = domain.geometry.x
            x_min = domain.comm.allreduce(float(np.min(x_local[:, 0])), op=MPI.MIN)
            x_max = domain.comm.allreduce(float(np.max(x_local[:, 0])), op=MPI.MAX)
            Lx = max(x_max - x_min, 1e-14)
        else:
            Lx = 1.0
    except Exception:
        Lx = 1.0

    df["h"] = Lx / df["N"].astype(float)
    return df.sort_values("N")


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
        print(f"\n==> Temporal convergence: N={N_value}, {len(records_filtered)} timesteps")
    
    # Field definitions (must match checkpoint contents from run_convergence_sweep.py)
    # Note: psi is cycle-averaged SED (DG0), more meaningful than u which is recomputed per loading case
    fields = [
        ("psi", "scalar_dg0", "SED (psi)"),
        ("rho", "scalar", "Density"),
        ("S", "scalar", "Stimulus"),
        ("L", "tensor", "Log-fabric"),
    ]
    
    # Compute errors for all fields
    results = {}
    total_fields = len(fields)
    for field_idx, (field_name, field_type, field_label) in enumerate(fields, start=1):
        if comm.rank == 0:
            print(f"\n  Field [{field_idx}/{total_fields}]: {field_label} ({field_name})")
        
        # Temporal refinement: compare coarse dt -> finer dt on the same mesh.
        # Sort dt from largest to smallest so each pair is (prev=coarser, current=finer).
        records_dt = sorted(records_filtered, key=lambda r: r["dt_days"], reverse=True)

        dt_coarse: list[float] = []
        l2_errs: list[float] = []
        h1_errs: list[float] = []

        prev_field = None
        prev_dt = None
        total = len(records_dt)

        for idx_rec, record in enumerate(records_dt, start=1):
            run_dir = Path(base_dir) / record["output_dir"]
            dt = float(record["dt_days"])
            if comm.rank == 0:
                print(f"  [{idx_rec}/{total}] Loading {field_name} (dt={dt})...", flush=True)

            try:
                field, domain = load_field_from_checkpoint(
                    run_dir, field_name, field_type, comm
                )
            except FileNotFoundError:
                if comm.rank == 0:
                    print(f"      (skip) Field '{field_name}' not found in {run_dir}")
                prev_field = None
                prev_dt = None
                break

            if prev_field is not None and prev_dt is not None:
                if comm.rank == 0:
                    print(f"  [{idx_rec}/{total}] Computing errors (dt={prev_dt} → dt={dt})...", flush=True)
                l2_err, h1_err = compute_l2_h1_errors(prev_field, field, domain)
                dt_coarse.append(prev_dt)
                l2_errs.append(l2_err)
                h1_errs.append(h1_err)
                if comm.rank == 0:
                    print(f"      L2 error: {l2_err:.6e}, H1 error: {h1_err:.6e}", flush=True)

            prev_field = field
            prev_dt = dt

        if len(dt_coarse) > 0:
            df = pd.DataFrame({
                "dt_days": np.array(dt_coarse, dtype=float),
                "L2_error": np.array(l2_errs, dtype=float),
                "H1_error": np.array(h1_errs, dtype=float),
            })
            results[field_name] = df
        
        if comm.rank == 0:
            print(f"  ✓ Completed {field_label}")
    
    return results


def compute_temporal_performance_data(
    base_dir: Path,
    N_value: int,
    comm: MPI.Comm,
) -> pd.DataFrame:
    """Aggregate per-run performance metrics for fixed N, varying dt."""
    records = load_sweep_records(base_dir, comm)
    records_filtered = filter_records_by_N(records, N_value)
    rows: list[dict[str, float]] = []
    for record in records_filtered:
        run_dir = Path(base_dir) / record["output_dir"]
        perf = load_run_performance(run_dir)
        rows.append({
            "N": float(record["N"]),
            "dt_days": float(record["dt_days"]),
            **perf,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("dt_days", ascending=False)


def save_convergence_xlsx(
    all_spatial_data: Dict[float, Dict[str, pd.DataFrame]],
    all_temporal_data: Dict[int, Dict[str, pd.DataFrame]],
    all_spatial_perf: Dict[float, pd.DataFrame],
    all_temporal_perf: Dict[int, pd.DataFrame],
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

        for dt_value, df_perf in all_spatial_perf.items():
            sheet_name = f"spatial_perf_dt{dt_value}"
            df_perf.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Temporal sheets (one per N × field combination)
        for N_value, temporal_data in all_temporal_data.items():
            for field_name, df in temporal_data.items():
                sheet_name = f"temporal_{field_name}_N{N_value}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        for N_value, df_perf in all_temporal_perf.items():
            sheet_name = f"temporal_perf_N{N_value}"
            df_perf.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Convergence data saved to {output_file}")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    
    base_dir = Path("results/convergence_sweep")
    output_dir = Path("analysis/convergence_analysis")
    
    if comm.rank == 0:
        print("=" * 80)
        print("CONVERGENCE ANALYSIS - Phase 2")
        print("=" * 80)
    
    # Load all sweep records
    if comm.rank == 0:
        print("\n[1/4] Loading sweep records...")
    all_records = load_sweep_records(base_dir, comm)
    
    # Extract unique dt and N values
    dt_values = sorted(set(r["dt_days"] for r in all_records))
    N_values = sorted(set(r["N"] for r in all_records))
    
    if comm.rank == 0:
        print(f"  ✓ Found {len(all_records)} total simulation runs")
        print(f"  ✓ {len(dt_values)} unique dt values: {dt_values}")
        print(f"  ✓ {len(N_values)} unique N values: {N_values}")
    
    # Compute spatial convergence for all dt values
    if comm.rank == 0:
        print(f"\n[2/4] Computing spatial convergence ({len(dt_values)} refinement series)...")
    
    all_spatial_data = {}
    all_spatial_perf = {}
    for idx, dt_val in enumerate(dt_values, start=1):
        if comm.rank == 0:
            print(f"\n{'─' * 80}")
            print(f"SPATIAL [{idx}/{len(dt_values)}]: dt = {dt_val} days")
            print(f"{'─' * 80}")
        spatial_data = compute_spatial_convergence_data(
            base_dir=base_dir,
            dt_value=dt_val,
            comm=comm,
        )
        all_spatial_data[dt_val] = spatial_data

        perf_df = compute_spatial_performance_data(
            base_dir=base_dir,
            dt_value=dt_val,
            comm=comm,
        )
        all_spatial_perf[dt_val] = perf_df
        if comm.rank == 0:
            print(f"\n✓ Completed spatial convergence for dt={dt_val}")
    
    # Compute temporal convergence for all N values
    if comm.rank == 0:
        print(f"\n[3/4] Computing temporal convergence ({len(N_values)} refinement series)...")
    
    all_temporal_data = {}
    all_temporal_perf = {}
    for idx, N_val in enumerate(N_values, start=1):
        if comm.rank == 0:
            print(f"\n{'─' * 80}")
            print(f"TEMPORAL [{idx}/{len(N_values)}]: N = {N_val}")
            print(f"{'─' * 80}")
        temporal_data = compute_temporal_convergence_data(
            base_dir=base_dir,
            N_value=N_val,
            comm=comm,
        )
        all_temporal_data[N_val] = temporal_data

        perf_df = compute_temporal_performance_data(
            base_dir=base_dir,
            N_value=N_val,
            comm=comm,
        )
        all_temporal_perf[N_val] = perf_df
        if comm.rank == 0:
            print(f"\n✓ Completed temporal convergence for N={N_val}")
    
    # Save all data to XLSX (rank 0 only)
    if comm.rank == 0:
        print(f"\n[4/4] Saving results to XLSX...")
        save_convergence_xlsx(
            all_spatial_data=all_spatial_data,
            all_temporal_data=all_temporal_data,
            all_spatial_perf=all_spatial_perf,
            all_temporal_perf=all_temporal_perf,
            output_file=output_dir / "convergence_data.xlsx",
        )
        print("\n" + "=" * 80)
        print("CONVERGENCE ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"✓ Generated {len(dt_values)} spatial convergence datasets")
        print(f"✓ Generated {len(N_values)} temporal convergence datasets")
        print(f"✓ Output saved to: {output_dir / 'convergence_data.xlsx'}")
        print("=" * 80)
