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
    
    # Broadcast stored data to all ranks (needed for coordinate matching)
    stored_coords = comm.bcast(stored_coords, root=0)
    stored_values = comm.bcast(stored_values, root=0)
    
    # Build KDTree from stored coords and query local DOF coordinates
    kdtree = cKDTree(stored_coords)
    distances, indices = kdtree.query(local_coords)
    
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






