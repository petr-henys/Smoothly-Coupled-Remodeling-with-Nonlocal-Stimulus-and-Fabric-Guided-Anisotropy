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
    from analysis.utils import load_npz_field, save_function_npz

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
    from analysis.utils import load_npz_field, save_function_npz

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
    from analysis.utils import load_npz_field, save_function_npz

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

