"""Utilities for convergence-analysis post-processing.

Provides:
- Checkpoint loading via adios4dolfinx (preferred) or legacy NPZ
- Cross-mesh interpolation and error norms (L2, H1)
- Richardson extrapolation and GCI helpers

Checkpoint Loading (adios4dolfinx)
----------------------------------
Uses adios4dolfinx for MPI-independent checkpoint loading. This is the
preferred approach as it:
- Uses a single file format (ADIOS2/BP)
- Supports N-to-M process count changes
- Is maintained by the DOLFINx team

Legacy NPZ Loading
------------------
For backwards compatibility, NPZ files with DOF coordinate matching are
still supported. This approach stores DOF coordinates + values and uses
KDTree matching for MPI-independent loading.

Note: NPZ is deprecated - prefer adios4dolfinx checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Any, List

import numpy as np
import pandas as pd
from mpi4py import MPI
from dolfinx import mesh, fem
import basix.ufl
import ufl
from scipy.spatial import cKDTree

# Check for adios4dolfinx availability
try:
    import adios4dolfinx as adx
    HAS_ADIOS4DOLFINX = True
except ImportError:
    HAS_ADIOS4DOLFINX = False


# ============================================================================
# Configuration
# ============================================================================

# Quadrature degree for error integration
QUADRATURE_DEGREE = 6

# Raise evaluation-space polynomial degree by this amount
ERROR_SPACE_RAISE = 2  # P1 -> P3 by default


# ============================================================================
# Checkpoint Loading (adios4dolfinx - preferred)
# ============================================================================

def load_checkpoint_mesh(
    checkpoint_path: Path,
    comm: MPI.Comm,
) -> Tuple[mesh.Mesh, mesh.MeshTags | None]:
    """Load mesh from adios4dolfinx checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp directory.
        comm: MPI communicator.
    
    Returns:
        Tuple of (mesh, meshtags) where meshtags may be None.
    
    Raises:
        ImportError: If adios4dolfinx is not installed.
    """
    if not HAS_ADIOS4DOLFINX:
        raise ImportError(
            "adios4dolfinx required for checkpoint loading. "
            "Install with: pip install adios4dolfinx"
        )
    
    # adios4dolfinx API: read_mesh(filename, comm, ...)
    domain = adx.read_mesh(checkpoint_path, comm)
    
    try:
        # adios4dolfinx API: read_meshtags(filename, mesh, meshtag_name=...)
        facet_tags = adx.read_meshtags(checkpoint_path, domain, meshtag_name="meshtags")
    except (KeyError, RuntimeError):
        facet_tags = None
    
    return domain, facet_tags


def load_checkpoint_function(
    checkpoint_path: Path,
    name: str,
    function_space: fem.FunctionSpace,
    time: float | None = None,
    comm: MPI.Comm | None = None,
) -> fem.Function:
    """Load function from adios4dolfinx checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp directory.
        name: Function name as stored in checkpoint.
        function_space: Target function space.
        time: Time value to load. If None, loads latest.
        comm: MPI communicator (unused, kept for API compatibility).
    
    Returns:
        Loaded fem.Function.
    """
    if not HAS_ADIOS4DOLFINX:
        raise ImportError(
            "adios4dolfinx required for checkpoint loading. "
            "Install with: pip install adios4dolfinx"
        )
    
    func = fem.Function(function_space, name=name)
    
    # adios4dolfinx API: read_function(filename, u, ..., time=..., name=...)
    if time is not None:
        adx.read_function(checkpoint_path, func, time=time, name=name)
    else:
        # Load latest timestep
        adx.read_function(checkpoint_path, func, name=name)
    
    return func


def get_checkpoint_final_time(
    checkpoint_path: Path,
    comm: MPI.Comm,
) -> float:
    """Get the final (largest) time value in a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp directory.
        comm: MPI communicator.
    
    Returns:
        Final time value as float.
    """
    if not HAS_ADIOS4DOLFINX:
        raise ImportError("adios4dolfinx required")
    
    # Read time values from checkpoint metadata
    # adios4dolfinx stores time as step attributes
    import adios2
    
    final_time = 0.0
    if comm.rank == 0:
        try:
            with adios2.open(str(checkpoint_path), "r", comm=MPI.COMM_SELF) as fh:
                for step in fh:
                    # Get time from any available function
                    available = fh.available_variables()
                    for var_name in available:
                        if "/values" in var_name or var_name.endswith("_values"):
                            try:
                                time_attr = step.read_attribute("time")
                                if time_attr is not None:
                                    final_time = max(final_time, float(time_attr))
                                break
                            except (KeyError, RuntimeError):
                                continue
        except Exception:
            pass
    
    final_time = comm.bcast(final_time, root=0)
    return final_time



# ============================================================================
# Legacy NPZ I/O (deprecated - use adios4dolfinx checkpoints instead)
# ============================================================================

def save_function_npz(func: fem.Function, path: Path, comm: MPI.Comm) -> None:
    """Save a function snapshot to NPZ (rank 0 writes).
    
    DEPRECATED: Use simulation.checkpoint.CheckpointStorage instead.

    Stores owned-DOF coordinates and values plus element metadata so loads can
    match DOFs by coordinates and remain MPI-independent.
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
    """Load an NPZ snapshot into `target` via DOF coordinate matching.
    
    DEPRECATED: Use load_checkpoint_function() with adios4dolfinx instead.

    Validates element compatibility and uses a KDTree to map stored coordinates
    onto the current MPI partition.
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
    if len(distances) > 0:
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





