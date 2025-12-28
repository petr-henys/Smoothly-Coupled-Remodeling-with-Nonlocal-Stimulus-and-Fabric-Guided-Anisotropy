"""Utilities for convergence-analysis post-processing.

Provides:
- Checkpoint loading via adios4dolfinx
- Cross-mesh interpolation and error norms (L2, H1)
- Richardson extrapolation and GCI helpers

Checkpoint Loading (adios4dolfinx)
----------------------------------
Uses adios4dolfinx for MPI-independent checkpoint loading:
- Uses a single file format (ADIOS2/BP)
- Supports N-to-M process count changes
- Is maintained by the DOLFINx team
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
import adios4dolfinx as adx


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
) -> mesh.Mesh:
    """Load mesh from adios4dolfinx checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp directory.
        comm: MPI communicator.
    
    Returns:
        Loaded mesh.
    """
    return adx.read_mesh(checkpoint_path, comm)


def load_checkpoint_meshtags(
    checkpoint_path: Path,
    domain: mesh.Mesh,
    meshtag_name: str = "meshtags",
) -> mesh.MeshTags:
    """Load meshtags from adios4dolfinx checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp directory.
        domain: Mesh the meshtags belong to.
        meshtag_name: Name of the meshtag in the checkpoint.
    
    Returns:
        Loaded meshtags.
        
    Raises:
        KeyError/RuntimeError: If meshtags not found in checkpoint.
    """
    return adx.read_meshtags(checkpoint_path, domain, meshtag_name=meshtag_name)


def load_checkpoint_function(
    checkpoint_path: Path,
    name: str,
    function_space: fem.FunctionSpace,
    time: float | None = None,
) -> fem.Function:
    """Load function from adios4dolfinx checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp directory.
        name: Function name as stored in checkpoint.
        function_space: Target function space.
        time: Time value to load. If None, loads latest.
    
    Returns:
        Loaded fem.Function.
    """
    
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





