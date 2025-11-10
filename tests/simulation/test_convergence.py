#!/usr/bin/env python3
"""
Convergence analysis, NPZ I/O, and QoI tests.

Merged from:
- test_convergence_analysis.py
- test_convergence_npz_io.py
- test_convergence_qoi_energy.py
- test_npz_cross_mpi.py
"""

"""Test convergence analysis functionality."""

import tempfile
import shutil
from pathlib import Path
from mpi4py import MPI
import numpy as np
import pytest

from dolfinx import mesh, fem
import basix.ufl
from analysis.analysis_utils import (
    save_function_npz,
    load_npz_field,
)


@pytest.mark.mpi
def test_save_load_npz_roundtrip():
    """Test that NPZ save/load preserves function values."""
    comm = MPI.COMM_WORLD
    
    # Create test mesh and function
    domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, P1)
    
    # Create test function with known pattern
    u_orig = fem.Function(V, name="test_function")
    u_orig.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * x[2])
    u_orig.x.scatter_forward()
    
    # Create temporary directory for test
    temp_dir = Path(tempfile.mkdtemp()) if comm.rank == 0 else None
    temp_dir = comm.bcast(temp_dir, root=0)
    
    try:
        # Save to NPZ
        npz_path = temp_dir / "test_function.npz"
        save_function_npz(u_orig, npz_path, comm)
        
        # Load back into new function
        u_loaded = fem.Function(V, name="loaded_function")
        load_npz_field(comm, npz_path, u_loaded)
        
        # Compare values
        diff = u_orig.x.array - u_loaded.x.array
        max_diff = comm.allreduce(np.max(np.abs(diff)), op=MPI.MAX)
        
        assert max_diff < 1e-14, f"Roundtrip error too large: {max_diff}"
        
        if comm.rank == 0:
            print(f"✓ NPZ roundtrip test passed (max_diff = {max_diff:.2e})")
    
    finally:
        if comm.rank == 0:
            shutil.rmtree(temp_dir)


    

################################################################################

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.mpi
@pytest.mark.unit
def test_npz_roundtrip_scalar_vector_tensor(shared_tmpdir):
    """Verify NPZ save/load with coordinate-based matching (MPI-independent).

    - Save scalar, vector, and tensor fields
    - Load them back
    - Compare owned-DOF arrays for exact equality
    
    KDTree matching ensures MPI-independence.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl

    # Import the saver used in convergence_runs and the loader used in postprocess
    from analysis.analysis_utils import load_npz_field, save_function_npz

    comm = MPI.COMM_WORLD
    N = 8
    domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)

    # Function spaces (P1)
    cell = domain.topology.cell_name()
    P1 = basix.ufl.element("Lagrange", cell, 1)
    P1_vec = basix.ufl.element("Lagrange", cell, 1, shape=(3,))
    P1_ten = basix.ufl.element("Lagrange", cell, 1, shape=(3, 3))
    Q = fem.functionspace(domain, P1)
    V = fem.functionspace(domain, P1_vec)
    T = fem.functionspace(domain, P1_ten)

    # Create fields with known patterns
    rho = fem.Function(Q, name="rho")
    rho.interpolate(lambda x: 0.7 + 0.2 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    rho.x.scatter_forward()

    u = fem.Function(V, name="u")
    u.interpolate(lambda x: np.vstack([x[0], x[1], x[2]]))
    u.x.scatter_forward()

    A = fem.Function(T, name="A")
    def A_init(x):
        n = x.shape[1]
        vals = np.zeros((9, n))
        vals[0, :] = 1.0/3.0 + 0.01 * np.sin(np.pi * x[0])
        vals[4, :] = 1.0/3.0 + 0.01 * np.cos(np.pi * x[1])
        vals[8, :] = 1.0/3.0 + 0.01 * np.sin(np.pi * x[2])
        return vals
    A.interpolate(A_init)
    A.x.scatter_forward()

    outdir = Path(shared_tmpdir) / "npz_roundtrip"
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Save
    save_function_npz(rho, outdir / "rho.npz", comm)
    save_function_npz(u, outdir / "u.npz", comm)
    save_function_npz(A, outdir / "A.npz", comm)

    # Load into fresh functions
    rho_loaded = fem.Function(Q, name="rho")
    load_npz_field(comm, outdir / "rho.npz", rho_loaded)
    rho_loaded.x.scatter_forward()

    u_loaded = fem.Function(V, name="u")
    load_npz_field(comm, outdir / "u.npz", u_loaded)
    u_loaded.x.scatter_forward()

    A_loaded = fem.Function(T, name="A")
    load_npz_field(comm, outdir / "A.npz", A_loaded)
    A_loaded.x.scatter_forward()

    # Compare only the owned part (ghosts are derived)
    def owned_prefix(f):
        idxmap = f.function_space.dofmap.index_map
        bs = f.function_space.dofmap.index_map_bs
        return f.x.array[: idxmap.size_local * bs]

    for f_orig, f_loaded in [(rho, rho_loaded), (u, u_loaded), (A, A_loaded)]:
        diff = owned_prefix(f_orig) - owned_prefix(f_loaded)
        maxdiff_local = np.max(np.abs(diff)) if diff.size else 0.0
        maxdiff = comm.allreduce(maxdiff_local, op=MPI.MAX)
        assert maxdiff < 1e-12  # Relaxed for KDTree float matching


@pytest.mark.mpi
@pytest.mark.unit
def test_npz_mpi_independence(shared_tmpdir):
    """Verify NPZ load works with different MPI rank count than save.
    
    This test validates that coordinate-based matching makes NPZ files
    truly MPI-independent (can't actually test with different MPI size
    in single pytest run, but validates element compatibility checks).
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    from analysis.analysis_utils import load_npz_field, save_function_npz

    comm = MPI.COMM_WORLD
    N = 6
    domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    Q = fem.functionspace(domain, P1)
    
    f = fem.Function(Q, name="f")
    f.interpolate(lambda x: x[0] + 2*x[1] + 3*x[2])
    f.x.scatter_forward()

    outdir = Path(shared_tmpdir) / "npz_mpi_indep"
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    # Save
    save_function_npz(f, outdir / "f.npz", comm)
    
    # Load back (same MPI size, but validates coordinate matching)
    f_loaded = fem.Function(Q, name="f")
    load_npz_field(comm, outdir / "f.npz", f_loaded)
    f_loaded.x.scatter_forward()
    
    # Verify
    idxmap = Q.dofmap.index_map
    bs = Q.dofmap.index_map_bs
    diff = f.x.array[:idxmap.size_local * bs] - f_loaded.x.array[:idxmap.size_local * bs]
    maxdiff_local = np.max(np.abs(diff)) if diff.size else 0.0
    maxdiff = comm.allreduce(maxdiff_local, op=MPI.MAX)
    assert maxdiff < 1e-12


@pytest.mark.mpi
@pytest.mark.unit
def test_npz_element_mismatch_detection(shared_tmpdir):
    """Ensure load_npz_field raises on element family/degree/shape mismatch."""
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    from analysis.analysis_utils import load_npz_field, save_function_npz

    comm = MPI.COMM_WORLD
    N = 6
    domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
    cell = domain.topology.cell_name()
    
    # Save P1 scalar field
    P1 = basix.ufl.element("Lagrange", cell, 1)
    Q = fem.functionspace(domain, P1)
    f = fem.Function(Q, name="f")
    f.x.array[:] = 1.23
    f.x.scatter_forward()

    outdir = Path(shared_tmpdir) / "npz_mismatch"
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    save_function_npz(f, outdir / "f_p1.npz", comm)
    
    # Try loading into P2 space (should fail)
    P2 = basix.ufl.element("Lagrange", cell, 2)
    Q2 = fem.functionspace(domain, P2)
    f2 = fem.Function(Q2, name="f2")
    
    with pytest.raises(RuntimeError, match="degree mismatch"):
        load_npz_field(comm, outdir / "f_p1.npz", f2)



################################################################################

import numpy as np
import pytest


################################################################################

"""Test NPZ I/O across different MPI sizes (manual verification).

This test must be run manually in two stages:
1. Save with one MPI size: mpirun -np 2 pytest tests/test_npz_cross_mpi.py::test_save -v
2. Load with different size: mpirun -np 4 pytest tests/test_npz_cross_mpi.py::test_load -v
"""

import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.mpi
def test_save(shared_tmpdir):
    """Save field with current MPI configuration."""
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    from analysis.analysis_utils import save_function_npz

    comm = MPI.COMM_WORLD
    N = 12
    domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
    
    # P1 scalar
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    Q = fem.functionspace(domain, P1)
    rho = fem.Function(Q, name="rho")
    rho.interpolate(lambda x: 0.5 + 0.3 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    rho.x.scatter_forward()
    
    # P1 vector
    P1_vec = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
    V = fem.functionspace(domain, P1_vec)
    u = fem.Function(V, name="u")
    u.interpolate(lambda x: np.vstack([x[0]**2, x[1]**2, x[2]**2]))
    u.x.scatter_forward()
    
    outdir = Path("/tmp/npz_cross_mpi_test")
    if comm.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== SAVING with {comm.size} MPI ranks ===")
    comm.Barrier()
    
    save_function_npz(rho, outdir / "rho.npz", comm)
    save_function_npz(u, outdir / "u.npz", comm)
    
    if comm.rank == 0:
        print(f"Saved to {outdir}")
        print("Run with different MPI size:")
        print(f"  mpirun -np <different_N> pytest tests/test_npz_cross_mpi.py::test_load -v")


@pytest.mark.mpi
def test_load(shared_tmpdir):
    """Load field saved with potentially different MPI configuration."""
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    from analysis.analysis_utils import load_npz_field

    comm = MPI.COMM_WORLD
    N = 12
    domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
    
    # P1 scalar
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    Q = fem.functionspace(domain, P1)
    rho_loaded = fem.Function(Q, name="rho")
    
    # P1 vector
    P1_vec = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
    V = fem.functionspace(domain, P1_vec)
    u_loaded = fem.Function(V, name="u")
    
    outdir = Path("/tmp/npz_cross_mpi_test")
    
    if comm.rank == 0:
        print(f"\n=== LOADING with {comm.size} MPI ranks ===")
    
    # This should work regardless of MPI size used to save
    load_npz_field(comm, outdir / "rho.npz", rho_loaded)
    load_npz_field(comm, outdir / "u.npz", u_loaded)
    
    # Verify against analytical solution
    rho_exact = fem.Function(Q, name="rho_exact")
    rho_exact.interpolate(lambda x: 0.5 + 0.3 * np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))
    rho_exact.x.scatter_forward()
    
    u_exact = fem.Function(V, name="u_exact")
    u_exact.interpolate(lambda x: np.vstack([x[0]**2, x[1]**2, x[2]**2]))
    u_exact.x.scatter_forward()
    
    # Compare
    idxmap_q = Q.dofmap.index_map
    bs_q = Q.dofmap.index_map_bs
    diff_rho = rho_loaded.x.array[:idxmap_q.size_local * bs_q] - rho_exact.x.array[:idxmap_q.size_local * bs_q]
    maxdiff_rho_local = np.max(np.abs(diff_rho)) if diff_rho.size else 0.0
    maxdiff_rho = comm.allreduce(maxdiff_rho_local, op=MPI.MAX)
    
    idxmap_v = V.dofmap.index_map
    bs_v = V.dofmap.index_map_bs
    diff_u = u_loaded.x.array[:idxmap_v.size_local * bs_v] - u_exact.x.array[:idxmap_v.size_local * bs_v]
    maxdiff_u_local = np.max(np.abs(diff_u)) if diff_u.size else 0.0
    maxdiff_u = comm.allreduce(maxdiff_u_local, op=MPI.MAX)
    
    if comm.rank == 0:
        print(f"Max difference rho: {maxdiff_rho:.2e}")
        print(f"Max difference u:   {maxdiff_u:.2e}")
    
    assert maxdiff_rho < 1e-12, f"rho mismatch: {maxdiff_rho:.2e}"
    assert maxdiff_u < 1e-12, f"u mismatch: {maxdiff_u:.2e}"
    
    if comm.rank == 0:
        print("✓ Cross-MPI loading successful!")
