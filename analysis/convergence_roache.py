"""Roache GCI analysis for quantities of interest (QoI).

Phase 2 script: Loads convergence sweep results, computes domain-averaged QoIs,
applies Richardson extrapolation (using utils.py), and exports roache_convergence.xlsx.

QoIs (as per manuscript Section 4.2):
- Mean ux [mm]: Domain-averaged x-displacement  
- Mean S [–]: Domain-averaged stimulus
- Mean rho [–]: Domain-averaged density  
- Anisotropy [–]: Domain-average tr(A^T A)

Usage:
    mpirun -np <N> python3 analysis/convergence_roache.py
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
from dolfinx import fem

from analysis.utils import (
    load_sweep_records,
    load_field_from_npz,
    compute_richardson_triplets_qoi,
)


def compute_qoi_from_fields(
    u: fem.Function,
    S: fem.Function,
    rho: fem.Function,
    A: fem.Function,
    domain,
) -> Dict[str, float]:
    """Compute scalar QoIs from solution fields.
    
    Uses direct array operations to avoid JIT compilation issues.
    Returns dimensional QoIs matching manuscript definitions.
    """
    # Dimensional scales (from Config defaults)
    u_c = 1e-3  # m
    
    comm = domain.comm
    
    # Ghost updates
    u.x.scatter_forward()
    S.x.scatter_forward()
    rho.x.scatter_forward()
    A.x.scatter_forward()
    
    # Get DOF arrays (owned DOFs only)
    u_arr = u.x.array
    S_arr = S.x.array
    rho_arr = rho.x.array
    A_arr = A.x.array
    
    # Mean x-displacement [mm]
    # u is vector function, take every 3rd value starting at 0 (x-component)
    ux_vals = u_arr[0::3]
    mean_ux_nd_local = np.mean(ux_vals) if len(ux_vals) > 0 else 0.0
    mean_ux_nd = comm.allreduce(mean_ux_nd_local, op=MPI.SUM) / comm.size
    mean_ux_mm = mean_ux_nd * u_c * 1000.0
    
    # Mean stimulus [–]
    mean_S_local = np.mean(S_arr) if len(S_arr) > 0 else 0.0
    mean_S = comm.allreduce(mean_S_local, op=MPI.SUM) / comm.size
    
    # Mean density [–]
    mean_rho_local = np.mean(rho_arr) if len(rho_arr) > 0 else 0.0
    mean_rho = comm.allreduce(mean_rho_local, op=MPI.SUM) / comm.size
    
    # Anisotropy: mean tr(A^T A) = mean ||A||_F^2
    # A is stored as flattened symmetric tensor (6 components per DOF for 3D)
    # For symmetric 3x3: [A11, A22, A33, A12, A13, A23] per DOF
    n_dofs = len(A_arr) // 6
    aniso_vals = []
    for i in range(n_dofs):
        idx = i * 6
        A11, A22, A33 = A_arr[idx], A_arr[idx+1], A_arr[idx+2]
        A12, A13, A23 = A_arr[idx+3], A_arr[idx+4], A_arr[idx+5]
        # Frobenius norm squared: ||A||_F^2 = sum of all squared components
        # For symmetric: A11^2 + A22^2 + A33^2 + 2*(A12^2 + A13^2 + A23^2)
        frob_sq = A11**2 + A22**2 + A33**2 + 2.0*(A12**2 + A13**2 + A23**2)
        aniso_vals.append(frob_sq)
    
    aniso_local = np.mean(aniso_vals) if aniso_vals else 0.0
    anisotropy = comm.allreduce(aniso_local, op=MPI.SUM) / comm.size
    
    return {
        "mean_ux_mm": float(mean_ux_mm),
        "mean_S": float(mean_S),
        "mean_rho": float(mean_rho),
        "anisotropy": float(anisotropy),
    }


def compute_spatial_qoi_data(
    base_dir: Path,
    dt_value: float,
    comm: MPI.Comm,
) -> pd.DataFrame:
    """Compute QoIs for spatial refinement (fixed dt, varying N)."""
    records = load_sweep_records(base_dir, comm)
    spatial_records = [r for r in records if abs(r["dt_days"] - dt_value) < 1e-6]
    spatial_records = sorted(spatial_records, key=lambda r: r["N"])
    
    if comm.rank == 0:
        print(f"\n==> Spatial QoI: dt={dt_value}, {len(spatial_records)} levels")
    
    qoi_data = []
    
    for idx, record in enumerate(spatial_records, start=1):
        N = record["N"]
        h = 1.0 / N
        output_dir = record["output_dir"]
        run_dir = base_dir / output_dir
        
        if comm.rank == 0:
            print(f"  [{idx}/{len(spatial_records)}] N={N}, h={h:.4f}...", flush=True)
        
        # Load final-time fields
        domain, u = load_field_from_npz(run_dir, comm, N, "u", "vector")
        _, S = load_field_from_npz(run_dir, comm, N, "S", "scalar")
        _, rho = load_field_from_npz(run_dir, comm, N, "rho", "scalar")
        _, A = load_field_from_npz(run_dir, comm, N, "A", "tensor")
        
        # Compute QoIs
        qois = compute_qoi_from_fields(u, S, rho, A, domain)
        
        qoi_data.append({
            "N": N,
            "h": h,
            **qois,
        })
        
        if comm.rank == 0:
            print(f"      QoIs: ux={qois['mean_ux_mm']:.6e} mm, "
                  f"S={qois['mean_S']:.6e}, "
                  f"rho={qois['mean_rho']:.6e}, "
                  f"aniso={qois['anisotropy']:.6e}")
    
    return pd.DataFrame(qoi_data)


def compute_temporal_qoi_data(
    base_dir: Path,
    N_value: int,
    comm: MPI.Comm,
) -> pd.DataFrame:
    """Compute QoIs for temporal refinement (fixed N, varying dt)."""
    records = load_sweep_records(base_dir, comm)
    temporal_records = [r for r in records if r["N"] == N_value]
    temporal_records = sorted(temporal_records, key=lambda r: r["dt_days"], reverse=True)
    
    if comm.rank == 0:
        print(f"\n==> Temporal QoI: N={N_value}, {len(temporal_records)} levels")
    
    qoi_data = []
    
    for idx, record in enumerate(temporal_records, start=1):
        N = record["N"]
        dt_days = record["dt_days"]
        output_dir = record["output_dir"]
        run_dir = base_dir / output_dir
        
        if comm.rank == 0:
            print(f"  [{idx}/{len(temporal_records)}] dt={dt_days} days...", flush=True)
        
        # Load final-time fields
        domain, u = load_field_from_npz(run_dir, comm, N, "u", "vector")
        _, S = load_field_from_npz(run_dir, comm, N, "S", "scalar")
        _, rho = load_field_from_npz(run_dir, comm, N, "rho", "scalar")
        _, A = load_field_from_npz(run_dir, comm, N, "A", "tensor")
        
        # Compute QoIs
        qois = compute_qoi_from_fields(u, S, rho, A, domain)
        
        qoi_data.append({
            "dt_days": dt_days,
            **qois,
        })
        
        if comm.rank == 0:
            print(f"      QoIs: ux={qois['mean_ux_mm']:.6e} mm, "
                  f"S={qois['mean_S']:.6e}, "
                  f"rho={qois['mean_rho']:.6e}, "
                  f"aniso={qois['anisotropy']:.6e}")
    
    return pd.DataFrame(qoi_data)


def apply_richardson_to_qois(
    df: pd.DataFrame,
    refine_var: str,
    qoi_names: List[str],
) -> Dict[str, pd.DataFrame]:
    """Apply Richardson/GCI to each QoI using utils.compute_richardson_triplets_qoi.
    
    Args:
        df: DataFrame with refinement variable and QoI columns
        refine_var: "h" for spatial, "dt_days" for temporal
        qoi_names: List of QoI column names
    
    Returns:
        Dictionary mapping QoI name to DataFrame with Richardson analysis
    """
    h_values = df[refine_var].tolist()
    
    results = {}
    for qoi_name in qoi_names:
        q_values = df[qoi_name].tolist()
        triplets = compute_richardson_triplets_qoi(h_values, q_values)
        results[qoi_name] = pd.DataFrame(triplets)
    
    return results


def export_roache_xlsx(
    spatial_qoi_df: pd.DataFrame,
    temporal_qoi_df: pd.DataFrame,
    spatial_rich: Dict[str, pd.DataFrame],
    temporal_rich: Dict[str, pd.DataFrame],
    output_file: Path,
    comm: MPI.Comm,
) -> None:
    """Export all Roache data to single XLSX workbook (rank 0 only)."""
    if comm.rank != 0:
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Raw QoI data
        spatial_qoi_df.to_excel(writer, sheet_name="spatial_qoi", index=False)
        temporal_qoi_df.to_excel(writer, sheet_name="temporal_qoi", index=False)
        
        # Richardson triplets for each QoI (spatial)
        for qoi_name, df_rich in spatial_rich.items():
            sheet_name = f"spatial_{qoi_name}"[:31]  # Excel limit
            df_rich.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Richardson triplets for each QoI (temporal)
        for qoi_name, df_rich in temporal_rich.items():
            sheet_name = f"temporal_{qoi_name}"[:31]  # Excel limit
            df_rich.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Summary statistics (spatial)
        summary_spatial = []
        for qoi_name, df_rich in spatial_rich.items():
            if df_rich.empty:
                continue
            summary_spatial.append({
                "qoi": qoi_name,
                "mean_p": df_rich["p"].mean(),
                "mean_GCI32_percent": df_rich["GCI32_percent"].mean(),
                "mean_beta": df_rich["beta"].mean(),
                "num_triplets": len(df_rich),
                "all_monotone": int(df_rich["monotone"].all()),
            })
        pd.DataFrame(summary_spatial).to_excel(writer, sheet_name="summary_spatial", index=False)
        
        # Summary statistics (temporal)
        summary_temporal = []
        for qoi_name, df_rich in temporal_rich.items():
            if df_rich.empty:
                continue
            summary_temporal.append({
                "qoi": qoi_name,
                "mean_p": df_rich["p"].mean(),
                "mean_GCI32_percent": df_rich["GCI32_percent"].mean(),
                "mean_beta": df_rich["beta"].mean(),
                "num_triplets": len(df_rich),
                "all_monotone": int(df_rich["monotone"].all()),
            })
        pd.DataFrame(summary_temporal).to_excel(writer, sheet_name="summary_temporal", index=False)
    
    print(f"\n✓ Roache GCI data saved to {output_file}")


def main():
    """Main entry point."""
    comm = MPI.COMM_WORLD
    
    # Configuration
    base_dir = Path("results/convergence_sweep")
    output_file = Path("analysis/convergence_analysis/roache_convergence.xlsx")
    
    # Analysis parameters (match convergence_errors.py)
    dt_fixed = 25.0  # days
    N_fixed = 81
    
    if comm.rank == 0:
        print("=" * 80)
        print("Roache GCI Analysis for QoIs")
        print("=" * 80)
        print(f"Base directory: {base_dir}")
        print(f"Spatial: dt={dt_fixed} days (varying N)")
        print(f"Temporal: N={N_fixed} (varying dt)")
        print()
    
    # ========================================================================
    # SPATIAL CONVERGENCE
    # ========================================================================
    if comm.rank == 0:
        print("-" * 80)
        print("SPATIAL CONVERGENCE")
        print("-" * 80)
    
    spatial_qoi_df = compute_spatial_qoi_data(base_dir, dt_fixed, comm)
    
    qoi_names = ["mean_ux_mm", "mean_S", "mean_rho", "anisotropy"]
    
    if comm.rank == 0:
        print("\n  Applying Richardson extrapolation...")
    spatial_rich = apply_richardson_to_qois(spatial_qoi_df, "h", qoi_names)
    
    # ========================================================================
    # TEMPORAL CONVERGENCE
    # ========================================================================
    if comm.rank == 0:
        print()
        print("-" * 80)
        print("TEMPORAL CONVERGENCE")
        print("-" * 80)
    
    temporal_qoi_df = compute_temporal_qoi_data(base_dir, N_fixed, comm)
    
    if comm.rank == 0:
        print("\n  Applying Richardson extrapolation...")
    temporal_rich = apply_richardson_to_qois(temporal_qoi_df, "dt_days", qoi_names)
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    export_roache_xlsx(
        spatial_qoi_df, temporal_qoi_df,
        spatial_rich, temporal_rich,
        output_file, comm
    )
    
    if comm.rank == 0:
        print()
        print("=" * 80)
        print("Roache GCI Analysis Complete")
        print("=" * 80)


if __name__ == "__main__":
    main()
