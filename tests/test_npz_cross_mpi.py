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
    from analysis.utils import save_function_npz

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
    from analysis.utils import load_npz_field

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
