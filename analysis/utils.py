"""
Utilities for convergence-analysis post-processing.

Provides MPI-independent NPZ I/O, cross-mesh interpolation, error norms, and
Richardson/GCI helpers used by the convergence scripts and plotting.

Note on NPZ snapshots
---------------------
Fields are stored with DOF coordinates and element metadata. Loading uses
KDTree matching to handle MPI-partition reordering, making snapshots fully
MPI-independent (any rank count can load).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

import numpy as np
import pandas as pd
from mpi4py import MPI
from dolfinx import mesh, fem
import basix.ufl
import ufl
from scipy.spatial import cKDTree


# ============================================================================
# Configuration
# ============================================================================

# Quadrature degree for error integration
QUADRATURE_DEGREE = 6

# Raise evaluation-space polynomial degree by this amount
ERROR_SPACE_RAISE = 2  # P1 -> P3 by default

# Roache safety factor: 1.25 for >=3 grids, 3.0 for 2 grids
GCI_SAFETY_FACTOR = 1.25

# Numerical tolerances for beta denominator / saturation checks
BETA_DENOM_ABS_TOL = 1e-12
BETA_DENOM_REL_TOL = 1e-8


# ============================================================================
# NPZ I/O (MPI-independent with coordinate matching)
# ============================================================================

def save_function_npz(func: fem.Function, path: Path, comm: MPI.Comm) -> None:
    """Save function to NPZ with DOF coordinates and element metadata.
    
    Stores:
    - DOF coordinates (for KDTree matching on load)
    - DOF values
    - Element family, degree, shape
    - Block size
    
    Rank 0 gathers all data and writes single NPZ file.
    Loading is MPI-independent via coordinate-based DOF matching.
    """
    space = func.function_space
    element = space.element
    index_map = space.dofmap.index_map
    bs = space.dofmap.index_map_bs
    
    # Get DOF coordinates for owned DOFs
    owned_dofs = index_map.size_local
    dof_coords = space.tabulate_dof_coordinates()[:owned_dofs]
    owned_values = func.x.array[:owned_dofs * bs].copy()
    
    # Gather to rank 0
    all_coords = comm.gather(dof_coords, root=0)
    all_values = comm.gather(owned_values, root=0)
    
    if comm.rank != 0:
        return
    
    # Concatenate all gathered data
    global_coords = np.vstack(all_coords)
    global_values = np.concatenate(all_values)
    
    # Extract element metadata
    basix_element = element.basix_element
    family_int = basix_element.family
    degree = basix_element.degree
    value_shape = element.value_shape
    
    # Save everything
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coords=global_coords,
        values=global_values,
        bs=np.int32(bs),
        family=np.int32(family_int),
        degree=np.int32(degree),
        value_shape=np.array(value_shape, dtype=np.int32),
    )


def load_npz_field(comm: MPI.Comm, npz_file: Path, target: fem.Function) -> None:
    """Load NPZ snapshot into target function using coordinate matching.
    
    Uses KDTree to match stored DOF coordinates with target space DOFs,
    making loading MPI-independent (works with any rank count/partition).
    
    Validates element compatibility (family, degree, shape).
    """
    # Rank 0 loads data
    if comm.rank == 0:
        with np.load(npz_file) as data:
            stored_coords = data["coords"]
            stored_values = data["values"]
            stored_bs = int(data["bs"])
            stored_family = int(data["family"])
            stored_degree = int(data["degree"])
            stored_shape = tuple(data["value_shape"])
    else:
        stored_coords = stored_values = None
        stored_bs = stored_family = stored_degree = stored_shape = None
    
    # Broadcast metadata
    stored_bs = comm.bcast(stored_bs, root=0)
    stored_family = comm.bcast(stored_family, root=0)
    stored_degree = comm.bcast(stored_degree, root=0)
    stored_shape = comm.bcast(stored_shape, root=0)
    
    # Validate element compatibility
    space = target.function_space
    element = space.element
    basix_element = element.basix_element
    bs = space.dofmap.index_map_bs
    
    if bs != stored_bs:
        raise RuntimeError(f"Block size mismatch: stored={stored_bs}, target={bs}")
    if basix_element.family != stored_family:
        raise RuntimeError(
            f"Element family mismatch: stored={stored_family}, target={basix_element.family}"
        )
    if basix_element.degree != stored_degree:
        raise RuntimeError(
            f"Element degree mismatch: stored={stored_degree}, target={basix_element.degree}"
        )
    
    # Compare value shapes (handle tuples with arrays)
    target_shape = element.value_shape
    if len(target_shape) != len(stored_shape):
        raise RuntimeError(
            f"Element shape mismatch: stored={stored_shape}, target={target_shape}"
        )
    for i, (s_stored, s_target) in enumerate(zip(stored_shape, target_shape)):
        if s_stored != s_target:
            raise RuntimeError(
                f"Element shape mismatch: stored={stored_shape}, target={target_shape}"
            )
    
    # Get local DOF coordinates
    index_map = space.dofmap.index_map
    owned_dofs = index_map.size_local
    local_coords = space.tabulate_dof_coordinates()[:owned_dofs]
    
    # Build KDTree on rank 0 and query all ranks' coordinates
    if comm.rank == 0:
        kdtree = cKDTree(stored_coords)
    else:
        kdtree = None
    
    # Broadcast stored data to all ranks (needed for querying)
    stored_coords = comm.bcast(stored_coords, root=0)
    stored_values = comm.bcast(stored_values, root=0)
    
    # Query KDTree for each local DOF
    distances, indices = cKDTree(stored_coords).query(local_coords)
    
    # Validate matching (should be exact up to floating-point tolerance)
    max_dist = np.max(distances)
    if max_dist > 1e-10:
        raise RuntimeError(
            f"DOF coordinate matching failed: max distance = {max_dist:.2e} > 1e-10\n"
            f"Stored and target meshes may be incompatible."
        )
    
    # Map values using matched indices
    for local_dof_idx in range(owned_dofs):
        stored_dof_idx = indices[local_dof_idx]
        for component in range(bs):
            target.x.array[local_dof_idx * bs + component] = stored_values[stored_dof_idx * bs + component]
    
    target.x.scatter_forward()


# ============================================================================
# Interpolation and integration helpers
# ============================================================================

def transfer_function_to_space(source: fem.Function, target_space: fem.FunctionSpace) -> fem.Function:
    """Interpolate ``source`` into ``target_space`` (possibly cross-mesh)."""
    tgt_mesh = target_space.mesh
    tgt_dim = tgt_mesh.topology.dim
    tgt_mesh.topology.create_connectivity(tgt_dim, tgt_dim)

    num_owned_cells = tgt_mesh.topology.index_map(tgt_dim).size_local
    cells = np.arange(num_owned_cells, dtype=np.int32)
    interp_data = fem.create_interpolation_data(target_space, source.function_space, cells=cells)

    target = fem.Function(target_space, name=source.name)
    target.interpolate_nonmatching(source, cells, interp_data)
    target.x.scatter_forward()
    return target


def mpi_scalar_integral(
    integrand: ufl.core.expr.Expr, 
    domain: mesh.Mesh,
) -> float:
    """MPI-parallel integral of a scalar UFL expression on ``domain``."""
    quadrature_degree = QUADRATURE_DEGREE
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": quadrature_degree})
    form = fem.form(integrand * dx)
    local_val = fem.assemble_scalar(form)
    return domain.comm.allreduce(local_val, op=MPI.SUM)


# ============================================================================
# Richardson / GCI utilities
# ============================================================================

def solve_order_from_ratios(
    R: float,
    r21: float,
    r32: float,
) -> float:
    """Solve for order ``p`` from non-uniform three-grid relation.

    Relation: ``R = (r21**p - 1) / (r32**p - 1)``, where ``R`` is the ratio of
    successive errors (or QoI differences) and ``r21 = h1/h2``, ``r32 = h2/h3``.
    """
    p_lo = 0.1
    p_hi = 10.0
    tol = 1e-12
    max_iter = 200
    
    if R <= 0 or not np.isfinite(R):
        return np.nan
    if r21 <= 1.0 or r32 <= 1.0:
        return np.nan

    # Handle (almost) uniform refinement separately to avoid degeneracy
    if np.isclose(r21, r32, rtol=1e-8, atol=1e-12):
        r = 0.5 * (r21 + r32)
        if r > 1.0:
            order = np.log(R) / np.log(r)
            return order if np.isfinite(order) else np.nan
        return np.nan

    def f(p: float) -> float:
        """Residual with basic overflow protection."""
        log_r21, log_r32 = np.log(r21), np.log(r32)
        if p * log_r21 > 700 or p * log_r32 > 700:
            return np.inf if p > 0 else -np.inf
        num = (r21**p - 1.0)
        den = (r32**p - 1.0)
        if abs(den) < 1e-15:
            return np.inf
        return num / den - R

    lo, hi = p_lo, p_hi
    flo, fhi = f(lo), f(hi)

    # Expand interval if root not bracketed
    expand = 0
    while flo * fhi > 0 and expand < 10:
        lo = max(1e-6, lo * 0.5)
        hi = hi * 2.0
        flo, fhi = f(lo), f(hi)
        expand += 1
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if not np.isfinite(fmid):
            return np.nan
        if abs(hi - lo) < tol or abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)


def _compute_gci_and_beta(
    Q1: float,
    Q2: float,
    Q3: float,
    Q_ext: float,
    r32: float,
    p: float,
    safety_factor: float,
    use_extrapolation: bool,
    p_for_beta: float,
    h1: float,
    h2: float,
    h3: float,
) -> Tuple[float, float, float]:
    """Compute GCI (fine/coarse) and asymptotic indicator ``beta``.

    If ``use_extrapolation`` is False, returns relative-change indicators instead
    of strict GCI (useful for error-norm sequences without a reference solution).

    Beta is normalized for non-uniform triplets:
        raw      = |Q2 - Q1| / |Q3 - Q2|
        expected = |h2^{p_beta} - h1^{p_beta}| / |h3^{p_beta} - h2^{p_beta}|  (if h1,h2,h3 given)
                   else r32^{p_beta}  (uniform fallback)
        beta     = raw / expected
    Additional guards:
        - sign flip of differences (oscillatory) -> beta = NaN
        - saturated denominator |Q3-Q2| ~ 0 -> beta = NaN
    """
    # Differences
    d21 = Q2 - Q1
    d32 = Q3 - Q2

    # GCI values (Roache-style wrt extrapolated limit, else relative-change)
    if use_extrapolation and np.isfinite(Q_ext) and abs(Q_ext) > 0:
        GCI32 = safety_factor * abs(Q3 - Q_ext) / abs(Q_ext)
        GCI21 = safety_factor * abs(Q2 - Q_ext) / abs(Q_ext)
    else:
        GCI32 = safety_factor * (abs(d32) / (abs(Q3) + 1e-16))
        GCI21 = safety_factor * (abs(d21) / (abs(Q2) + 1e-16))

    # Beta computation
    p_beta = p_for_beta if np.isfinite(p_for_beta) else p
    beta = np.nan
    
    # Oscillatory triplet -> not asymptotic
    if d21 * d32 <= 0:
        return GCI32, GCI21, np.nan
    
    # Saturated denominator
    scale = max(abs(Q2), abs(Q3), 1.0)
    if abs(d32) <= (BETA_DENOM_ABS_TOL + BETA_DENOM_REL_TOL * scale):
        return GCI32, GCI21, np.nan

    eps = 1e-300
    raw = abs(d21) / (abs(d32) + eps)

    expected = np.nan
    if np.isfinite(p_beta) and (h1 > 0) and (h2 > 0) and (h3 > 0):
        num = abs((h2 ** p_beta) - (h1 ** p_beta))
        den = abs((h3 ** p_beta) - (h2 ** p_beta))
        if den > 0:
            expected = num / den
    
    if not np.isfinite(expected) and np.isfinite(r32) and np.isfinite(p_beta) and (r32 > 0):
        expected = r32 ** p_beta
    
    if np.isfinite(expected) and expected > 0:
        beta = raw / expected

    return GCI32, GCI21, beta


# ============================================================================
# Error norms
# ============================================================================

def compute_l2_h1_errors(
    field_coarse: fem.Function,
    field_fine: fem.Function,
    domain_fine: mesh.Mesh,
) -> Tuple[float, float]:
    """L2 and H1-seminorm of difference on the fine mesh.

    Builds a higher-order evaluation space on ``domain_fine`` matching the
    value shape of ``field_fine`` for accurate integration.
    """
    V_fine = field_fine.function_space
    ufl_el = V_fine.ufl_element()
    degree = ufl_el.degree + ERROR_SPACE_RAISE
    family = ufl_el.family_name

    value_shape = field_fine.ufl_shape  # (), (n,), (n,m), ...
    cell_name = domain_fine.topology.cell_name()

    element_high = basix.ufl.element(family, cell_name, degree, shape=tuple(value_shape))
    space_high = fem.functionspace(domain_fine, element_high)

    fine_high = transfer_function_to_space(field_fine, space_high)
    coarse_high = transfer_function_to_space(field_coarse, space_high)

    diff = fine_high - coarse_high
    l2_error = np.sqrt(mpi_scalar_integral(ufl.inner(diff, diff), domain_fine))
    h1_error = np.sqrt(mpi_scalar_integral(ufl.inner(ufl.grad(diff), ufl.grad(diff)), domain_fine))
    return l2_error, h1_error


# ============================================================================
# Field loading helpers
# ============================================================================

def load_field_from_npz(
    run_dir: Path,
    comm: MPI.Comm,
    N: int,
    field_name: str,
    field_type: str,
) -> Tuple[mesh.Mesh, fem.Function]:
    """Load a field NPZ from a hash-named output directory into a freshly created mesh and space.
    
    Args:
        run_dir: Path to the hash-named output directory (e.g., results/convergence_sweep/407c0bfb)
        comm: MPI communicator
        N: Mesh resolution (number of cells per dimension)
        field_name: Field name (e.g., "u", "rho", "S", "A")
        field_type: Field type ("scalar", "vector", or "tensor")
    
    Returns:
        Tuple of (domain, field) where domain is the mesh and field is the loaded Function
    """
    npz_file = run_dir / f"{field_name}.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_file}")

    domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
    gdim = domain.geometry.dim

    if field_type == "scalar":
        element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    elif field_type == "vector":
        element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(gdim,))
    elif field_type == "tensor":
        element = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(gdim, gdim))
    else:
        raise ValueError(f"Unknown field_type: {field_type}")

    space = fem.functionspace(domain, element)
    field = fem.Function(space, name=field_name)
    load_npz_field(comm, npz_file, field)
    field.x.scatter_forward()
    return domain, field


# ============================================================================
# Sweep loading and generic analysis helpers
# ============================================================================

def load_sweep_records(
    base_dir: Path,
    comm: MPI.Comm,
) -> List[Dict[str, Any]]:
    """Load sweep CSV records (MPI-aware).

    Returns all sweep records sorted by N and dt_days.
    """
    csv_file = base_dir / "sweep_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Sweep summary not found: {csv_file}")

    if comm.rank == 0:
        df_sweep = pd.read_csv(csv_file)
        df_sorted = df_sweep.sort_values(["N", "dt_days"])
        records = df_sorted.to_dict("records")
    else:
        records = None

    records = comm.bcast(records, root=0)
    return records


def analyze_solver_convergence(
    base_dir: Path,
    dt_days: float,
    comm: MPI.Comm,
    field_name: str,
) -> pd.DataFrame:
    """Generic field-error analysis across meshes for fixed ``dt_days``."""
    all_records = load_sweep_records(base_dir, comm)
    records = [r for r in all_records if abs(r["dt_days"] - dt_days) < 1e-6]
    records = sorted(records, key=lambda r: r["N"])
    
    if len(records) < 3:
        raise ValueError(f"Need ≥3 mesh sizes for dt={dt_days}, got {len(records)}")
    
    N_values = np.array([r["N"] for r in records])
    h_values = 1.0 / N_values

    l2_errors: List[float] = []
    h1_errors: List[float] = []
    prev_field = None

    for record in records:
        # Use output_dir from sweep_summary.csv to construct path
        output_dir = record["output_dir"]
        run_dir = base_dir / output_dir
        N = int(record["N"])

        if field_name in ("rho", "S"):
            field_type = "scalar"
        elif field_name == "u":
            field_type = "vector"
        elif field_name == "A":
            field_type = "tensor"
        else:
            raise ValueError(f"Unknown field: {field_name}")

        domain, field = load_field_from_npz(run_dir, comm, N, field_name, field_type)
        if prev_field is not None:
            l2_err, h1_err = compute_l2_h1_errors(prev_field, field, domain)
            l2_errors.append(l2_err)
            h1_errors.append(h1_err)
        prev_field = field

    return build_error_dataframe(N_values, h_values, np.asarray(l2_errors), np.asarray(h1_errors))


# ============================================================================
# Pairwise error post-processing
# ============================================================================

def compute_convergence_rates(errors: np.ndarray, h_values: np.ndarray) -> List[float]:
    """Compute local convergence orders from an error sequence and mesh sizes.

    For triplets (h_i, h_{i+1}, h_{i+2}) with pairwise errors E_i, E_{i+1} we
    solve for p using the non-uniform three-grid relation.
    """
    rates: List[float] = []
    n = len(errors)
    for i in range(n - 1):
        if i + 2 >= len(h_values):
            rates.append(np.nan)
            continue
        Ei, Ei1 = float(errors[i]), float(errors[i + 1])
        if not (np.isfinite(Ei) and np.isfinite(Ei1)) or Ei1 <= 0.0 or Ei <= 0.0:
            rates.append(np.nan)
            continue
        r21 = float(h_values[i] / h_values[i + 1])
        r32 = float(h_values[i + 1] / h_values[i + 2])
        R = Ei / Ei1
        p_i = solve_order_from_ratios(R, r21, r32)
        rates.append(p_i if np.isfinite(p_i) else np.nan)
    return rates


def build_error_dataframe(
    N_values: np.ndarray,
    h_values: np.ndarray,
    l2_errors: np.ndarray,
    h1_errors: np.ndarray,
) -> pd.DataFrame:
    """Build error results table with local convergence rates."""
    l2_rates = compute_convergence_rates(l2_errors, h_values)
    h1_rates = compute_convergence_rates(h1_errors, h_values)

    error_results: List[Dict[str, Any]] = []
    for i in range(len(l2_errors)):
        error_results.append(
            {
                "pair": f"{N_values[i]}-{N_values[i+1]}",
                "N_coarse": N_values[i],
                "N_fine": N_values[i + 1],
                "h_coarse": h_values[i],
                "h_fine": h_values[i + 1],
                "l2_error": l2_errors[i],
                "h1_error": h1_errors[i],
                "l2_rate": l2_rates[i] if i < len(l2_rates) else np.nan,
                "h1_rate": h1_rates[i] if i < len(h1_rates) else np.nan,
            }
        )
    return pd.DataFrame(error_results)


def compute_richardson_triplets(
    h_values: np.ndarray,
    qoi_pair_errors: List[float],
) -> List[Dict[str, Any]]:
    """Three-grid analysis for pairwise error sequences.

    Uses relative-change indicators (no extrapolation) suitable for error norms
    where an exact solution is unavailable.
    """
    safety_factor = GCI_SAFETY_FACTOR
    rows: List[Dict[str, Any]] = []
    E = [float(e) for e in qoi_pair_errors]
    nE = len(E)
    nH = len(h_values)

    for i in range(nE - 2):
        if i + 2 >= nH:
            break

        Q1, Q2, Q3 = E[i], E[i + 1], E[i + 2]
        h1, h2, h3 = float(h_values[i]), float(h_values[i + 1]), float(h_values[i + 2])
        r21, r32 = h1 / h2, h2 / h3

        R = np.inf if Q2 == 0 else abs(Q1) / abs(Q2)
        p = solve_order_from_ratios(R, r21, r32)

        GCI32, GCI21, beta = _compute_gci_and_beta(
            Q1, Q2, Q3, Q_ext=np.nan, r32=r32, p=p, safety_factor=safety_factor,
            use_extrapolation=False, p_for_beta=p, h1=h1, h2=h2, h3=h3
        )

        monotone = int((Q1 >= Q2 >= Q3) or (Q1 <= Q2 <= Q3))
        rows.append(
            {
                "p": float(p) if np.isfinite(p) else np.nan,
                "Q1": float(Q1),
                "Q2": float(Q2),
                "Q3": float(Q3),
                "Q_ext": float("nan"),
                "GCI32": float(GCI32) if np.isfinite(GCI32) else np.nan,
                "GCI21": float(GCI21) if np.isfinite(GCI21) else np.nan,
                "GCI32_percent": float(100.0 * GCI32) if np.isfinite(GCI32) else np.nan,
                "GCI21_percent": float(100.0 * GCI21) if np.isfinite(GCI21) else np.nan,
                "beta": float(beta) if np.isfinite(beta) else np.nan,
                "monotone": int(monotone),
                "triplet": f"{i}:{i+1}:{i+2}  |  h=({h1:.6g},{h2:.6g},{h3:.6g})  r=({r21:.6g},{r32:.6g})",
            }
        )

    return rows


# ============================================================================
# Excel export (optional utility)
# ============================================================================

def save_convergence_results_to_excel(
    error_results: Dict[float, pd.DataFrame],
    richardson_results: Dict[float, pd.DataFrame],
    output_file: Path,
    comm: MPI.Comm,
) -> None:
    """Save error/Richardson tables into an Excel workbook (rank 0 only)."""
    if comm.rank != 0:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # QoI summary
        summary_data = []
        for dt_val, df_rich in richardson_results.items():
            monotone_series = df_rich.get("monotone", pd.Series(dtype=int))
            all_monotone = int(monotone_series.fillna(0).astype(bool).all()) if len(df_rich) else np.nan
            summary_data.append(
                {
                    "dt": dt_val,
                    "num_triplets": len(df_rich),
                    "mean_p": df_rich["p"].mean(),
                    "mean_GCI32_percent": df_rich["GCI32_percent"].mean(),
                    "mean_beta": df_rich["beta"].mean(),
                    "all_monotone": all_monotone,
                }
            )
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary_QoI", index=False)

        # Detailed sheets
        for dt_val, df_rich in richardson_results.items():
            df_rich.to_excel(writer, sheet_name=f"QoI_dt_{dt_val:.2f}".replace(".", "_"), index=False)
        for dt_val, df_err in error_results.items():
            df_err.to_excel(writer, sheet_name=f"Errors_dt_{dt_val:.2f}".replace(".", "_"), index=False)
    print(f"Results saved to {output_file}")


# ============================================================================
# QoI triplets (with Richardson extrapolation)
# ============================================================================

def compute_global_pref(h_values: List[float], q_values: List[float]) -> float:
    """Global reference order ``p_ref`` via log–log fit of successive differences."""
    h = np.asarray(h_values, float)
    q = np.asarray(q_values, float)
    diffs = np.abs(np.diff(q))
    h_for_diffs = h[:-1]
    m = np.isfinite(h_for_diffs) & np.isfinite(diffs) & (h_for_diffs > 0) & (diffs > 0)
    if m.sum() < 2:
        return float("nan")
    X = np.log(h_for_diffs[m])
    Y = np.log(diffs[m])
    p_ref = np.polyfit(X, Y, 1)[0]
    return float(p_ref)


def compute_richardson_triplets_qoi(
    h_values: List[float],
    q_values: List[float],
) -> List[Dict[str, Any]]:
    """Richardson extrapolation and Roache GCI for QoI sequences.

    For each consecutive triplet (h1,h2,h3) with QoIs (Q1,Q2,Q3), estimate
    order p, extrapolate Q_ext, compute GCI21/GCI32 and beta.
    """
    safety_factor = GCI_SAFETY_FACTOR
    
    if len(h_values) != len(q_values):
        raise ValueError("h_values and q_values must have the same length")
    if len(h_values) < 3:
        return []

    p_ref = compute_global_pref(h_values, q_values)

    h = np.asarray(h_values, dtype=float)
    Q = np.asarray(q_values, dtype=float)
    rows: List[Dict[str, Any]] = []

    for i in range(len(h) - 2):
        h1, h2, h3 = float(h[i]), float(h[i + 1]), float(h[i + 2])
        Q1, Q2, Q3 = float(Q[i]), float(Q[i + 1]), float(Q[i + 2])
        r21, r32 = h1 / h2, h2 / h3
        d21, d32 = Q2 - Q1, Q3 - Q2

        R = abs(d21) / abs(d32) if abs(d32) > 0 else np.inf
        p = solve_order_from_ratios(R, r21, r32)

        Q_ext = np.nan
        if np.isfinite(p):
            denom = r32 ** p - 1.0
            if abs(denom) > 1e-14:
                Q_ext = Q3 + (Q3 - Q2) / denom

        GCI32, GCI21, beta = _compute_gci_and_beta(
            Q1=Q1, Q2=Q2, Q3=Q3, Q_ext=Q_ext, r32=r32, p=p, safety_factor=safety_factor,
            use_extrapolation=True, p_for_beta=p_ref, h1=h1, h2=h2, h3=h3
        )

        monotone = int(d21 * d32 > 0.0)
        rows.append(
            {
                "i_start": i,
                "i_mid": i + 1,
                "i_end": i + 2,
                "h1": h1,
                "h2": h2,
                "h3": h3,
                "r21": r21,
                "r32": r32,
                "Q1": Q1,
                "Q2": Q2,
                "Q3": Q3,
                "d21": d21,
                "d32": d32,
                "p": float(p) if np.isfinite(p) else np.nan,
                "Q_ext": float(Q_ext) if np.isfinite(Q_ext) else np.nan,
                "GCI21": float(GCI21) if np.isfinite(GCI21) else np.nan,
                "GCI32": float(GCI32) if np.isfinite(GCI32) else np.nan,
                "GCI21_percent": float(100.0 * GCI21) if np.isfinite(GCI21) else np.nan,
                "GCI32_percent": float(100.0 * GCI32) if np.isfinite(GCI32) else np.nan,
                "beta": float(beta) if np.isfinite(beta) else np.nan,
                "monotone": int(monotone),
                "triplet": f"{i}:{i + 1}:{i + 2}  |  h=({h1:.6g},{h2:.6g},{h3:.6g})  r=({r21:.6g},{r32:.6g})",
            }
        )

    return rows
