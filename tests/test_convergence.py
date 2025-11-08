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
from analysis.utils import (
    save_function_npz,
    load_field_from_npz,
    compute_richardson_triplets_qoi
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
        domain_loaded, u_loaded = load_field_from_npz(temp_dir, comm, 8, "test_function", "scalar")
        
        # Compare values
        diff = u_orig.x.array - u_loaded.x.array
        max_diff = comm.allreduce(np.max(np.abs(diff)), op=MPI.MAX)
        
        assert max_diff < 1e-14, f"Roundtrip error too large: {max_diff}"
        
        if comm.rank == 0:
            print(f"✓ NPZ roundtrip test passed (max_diff = {max_diff:.2e})")
    
    finally:
        if comm.rank == 0:
            shutil.rmtree(temp_dir)


@pytest.mark.unit
def test_richardson_extrapolation():
    """Test Richardson extrapolation with known convergence."""
    # Create synthetic data with known convergence order p=2
    h_values = [0.1, 0.05, 0.025, 0.0125]
    exact_value = 1.0
    p_true = 2.0
    
    # Generate QoI values with p=2 convergence: Q(h) = Q_exact + C*h^p
    C = 0.1
    qoi_values = [exact_value + C * h**p_true for h in h_values]
    
    # Apply Richardson extrapolation
    richardson_data = compute_richardson_triplets_qoi(h_values, qoi_values)
    
    assert len(richardson_data) == 2, "Should have 2 Richardson triplets"
    
    # Check estimated convergence orders
    p_estimates = [r["p"] for r in richardson_data]
    for p_est in p_estimates:
        assert abs(p_est - p_true) < 0.1, f"Convergence order estimate {p_est} too far from true value {p_true}"
    
    # Check extrapolated values
    q_ext_estimates = [r["Q_ext"] for r in richardson_data]
    for q_ext in q_ext_estimates:
        if np.isfinite(q_ext):
            assert abs(q_ext - exact_value) < 0.05, f"Extrapolated value {q_ext} too far from exact {exact_value}"
    
    print(f"✓ Richardson test passed: p_estimates = {p_estimates}, Q_ext = {q_ext_estimates}")


@pytest.mark.unit  
def test_gci_computation():
    """Test Grid Convergence Index computation."""
    # Test data with good convergence
    h_values = [0.08, 0.04, 0.02] 
    qoi_values = [1.1, 1.025, 1.00625]  # Converging to 1.0 with p≈2
    
    richardson_data = compute_richardson_triplets_qoi(h_values, qoi_values)
    
    assert len(richardson_data) == 1, "Should have 1 Richardson triplet"
    
    result = richardson_data[0]
    
    # Check GCI values are reasonable (should be small for good convergence)
    assert 0 < result["GCI32_percent"] < 10, f"GCI32 should be reasonable: {result['GCI32_percent']}"
    assert 0 < result["GCI21_percent"] < 10, f"GCI21 should be reasonable: {result['GCI21_percent']}"
    
    # Check beta (should be close to 1 for consistent convergence)
    if np.isfinite(result["beta"]):
        assert 0.5 < result["beta"] < 2.0, f"Beta should be close to 1: {result['beta']}"
    
    print(f"✓ GCI test passed: GCI32={result['GCI32_percent']:.2f}%, GCI21={result['GCI21_percent']:.2f}%, beta={result['beta']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

    

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



################################################################################

import numpy as np
import pytest


@pytest.mark.mpi
@pytest.mark.unit
def test_qoi_dirichlet_energy_scalar_richardson():
    """Use Dirichlet energy of a smooth scalar field as QoI.

    Q(h) = 1/2 ∫ |∇f_h|^2 dx converges to a limit with mesh refinement.
    We apply Richardson/GCI on Q(h) across uniform refinements and assert
    sensible p > 0, decreasing GCIs, and ratio consistency.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    import ufl
    from analysis.utils import (
        mpi_scalar_integral,
        compute_richardson_triplets_qoi,
    )

    comm = MPI.COMM_WORLD
    # Uniform-ish ratio to fit solver assumptions (r ≈ 1.5)
    N_list = [8, 12, 18, 27]  # 4 levels → 2 triplets

    def make_field(N):
        domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        Q = fem.functionspace(domain, P1)
        f = fem.Function(Q, name="f")
        # Quadratic manufactured solution (nontrivial gradient)
        f.interpolate(
            lambda x: (
                x[0] * x[0]
                - 0.5 * x[1] * x[1]
                + 0.75 * x[2] * x[2]
                + 0.3 * x[0] * x[1]
                + 0.2 * x[1] * x[2]
                + 0.15 * x[0] * x[2]
            )
        )
        f.x.scatter_forward()
        return domain, f

    domains, fields = zip(*(make_field(N) for N in N_list))
    h_values = 1.0 / np.asarray(N_list, dtype=float)

    # QoI: Dirichlet energy 0.5 * ∫ |∇f_h|^2
    Q_vals = []
    for dom, f in zip(domains, fields):
        e_density = 0.5 * ufl.inner(ufl.grad(f), ufl.grad(f))
        Q_vals.append(mpi_scalar_integral(e_density, dom))

    rows = compute_richardson_triplets_qoi(h_values.tolist(), Q_vals)
    assert len(rows) == len(N_list) - 2

    for i, row in enumerate(rows):
        assert np.isfinite(row["p"]) and row["p"] > 0.5 and row["p"] < 3.5
        # Finer level should be closer to Q_ext → smaller GCI32
        assert row["GCI32_percent"] <= row["GCI21_percent"] + 1e-8
        # Consistency: GCI32/GCI21 ≈ (h3/h2)^p = (1/r32)^p
        ratio = row["GCI32"] / row["GCI21"] if row["GCI21"] > 0 else np.nan
        expected = (1.0 / row["r32"]) ** row["p"] if row["r32"] > 0 else np.nan
        if np.isfinite(ratio) and np.isfinite(expected):
            assert abs(ratio - expected) / max(1e-12, expected) < 0.25


@pytest.mark.mpi
@pytest.mark.unit
def test_qoi_dirichlet_energy_vector_richardson():
    """Use Dirichlet energy of a smooth vector field as QoI.

    Q(h) = 1/2 ∫ |∇u_h|^2 dx with u quadratic in each component.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    import ufl
    from analysis.utils import (
        mpi_scalar_integral,
        compute_richardson_triplets_qoi,
    )

    comm = MPI.COMM_WORLD
    N_list = [8, 12, 18, 27]

    def make_field(N):
        domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
        V = fem.functionspace(
            domain, basix.ufl.element("Lagrange", domain.topology.cell_name(), 1, shape=(3,))
        )
        u = fem.Function(V, name="u")
        u.interpolate(lambda x: np.vstack([x[0] ** 2, 0.5 * x[1] ** 2, 0.25 * x[2] ** 2]))
        u.x.scatter_forward()
        return domain, u

    domains, fields = zip(*(make_field(N) for N in N_list))
    h_values = 1.0 / np.asarray(N_list, dtype=float)

    Q_vals = []
    for dom, u in zip(domains, fields):
        e_density = 0.5 * ufl.inner(ufl.grad(u), ufl.grad(u))
        Q_vals.append(mpi_scalar_integral(e_density, dom))

    rows = compute_richardson_triplets_qoi(h_values.tolist(), Q_vals)
    assert len(rows) == len(N_list) - 2

    for row in rows:
        assert np.isfinite(row["p"]) and row["p"] > 0.5 and row["p"] < 3.5
        assert row["GCI32_percent"] <= row["GCI21_percent"] + 1e-8
        ratio = row["GCI32"] / row["GCI21"] if row["GCI21"] > 0 else np.nan
        expected = (1.0 / row["r32"]) ** row["p"] if row["r32"] > 0 else np.nan
        if np.isfinite(ratio) and np.isfinite(expected):
            assert abs(ratio - expected) / max(1e-12, expected) < 0.25


@pytest.mark.mpi
@pytest.mark.unit
def test_qoi_gradient_norm_scalar_richardson():
    """Use L2 norm of gradient as QoI (not variance).

    Q(h) = ∫ |∇f_h|^2 dx behaves similarly to energy and is meaningful for GCI.
    """
    from mpi4py import MPI
    from dolfinx import mesh, fem
    import basix.ufl
    import ufl
    from analysis.utils import (
        mpi_scalar_integral,
        compute_richardson_triplets_qoi,
    )

    comm = MPI.COMM_WORLD
    N_list = [8, 12, 18, 27]

    def make_field(N):
        domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        Q = fem.functionspace(domain, P1)
        f = fem.Function(Q, name="f")
        f.interpolate(lambda x: np.sin(1.3 * np.pi * x[0]) * np.cos(0.7 * np.pi * x[1]) + 0.2 * x[2])
        f.x.scatter_forward()
        return domain, f

    domains, fields = zip(*(make_field(N) for N in N_list))
    h_values = 1.0 / np.asarray(N_list, dtype=float)

    Q_vals = []
    for dom, f in zip(domains, fields):
        Q_vals.append(mpi_scalar_integral(ufl.inner(ufl.grad(f), ufl.grad(f)), dom))

    rows = compute_richardson_triplets_qoi(h_values.tolist(), Q_vals)
    assert len(rows) == len(N_list) - 2

    for row in rows:
        assert np.isfinite(row["p"]) and row["p"] > 0.5
        assert row["GCI32_percent"] <= row["GCI21_percent"] + 1e-8
        ratio = row["GCI32"] / row["GCI21"] if row["GCI21"] > 0 else np.nan
        expected = (1.0 / row["r32"]) ** row["p"] if row["r32"] > 0 else np.nan
        if np.isfinite(ratio) and np.isfinite(expected):
            assert abs(ratio - expected) / max(1e-12, expected) < 0.35



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
