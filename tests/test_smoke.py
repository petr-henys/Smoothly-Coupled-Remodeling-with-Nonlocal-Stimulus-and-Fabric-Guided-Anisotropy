#!/usr/bin/env python3
"""
Smoke tests - fast sanity checks that basic functionality works.

All tests here should complete in < 5 seconds each, ideally < 1 second.
Total suite runtime target: < 30 seconds.

These tests are designed to:
- Catch catastrophic failures quickly
- Run on every commit in CI
- Provide fast feedback to developers
- Use minimal computational resources (small meshes, simple problems)
"""

import pytest
pytestmark = pytest.mark.smoke  # Mark all tests in this file as smoke tests

pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
import basix
from pathlib import Path

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.model import Remodeller
from simulation.storage import UnifiedStorage
from simulation.logger import get_logger

comm = MPI.COMM_WORLD


# =============================================================================
# Configuration and Initialization Smoke Tests
# =============================================================================

@pytest.mark.unit
def test_config_instantiation():
    """Config should instantiate without errors on tiny mesh."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)

    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    assert cfg is not None
    assert cfg.domain is not None
    assert float(cfg.E0_nd) > 0
    assert cfg.dx is not None


@pytest.mark.unit
def test_mesh_creation():
    """Basic mesh creation should work."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)

    assert domain.topology.dim == 3
    assert domain.geometry.dim == 3
    comm.Barrier()


@pytest.mark.unit
def test_function_spaces_creation():
    """Function space creation should work."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    # Ensure spaces have positive scalar DOF counts
    assert V.dofmap.index_map.size_global * V.dofmap.index_map_bs > 0
    assert Q.dofmap.index_map.size_global * Q.dofmap.index_map_bs > 0
    assert T.dofmap.index_map.size_global * T.dofmap.index_map_bs > 0


@pytest.mark.unit
def test_boundary_condition_creation():
    """Dirichlet BC creation should work."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    V = functionspace(domain, P1_vec)

    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

    assert len(bc_mech) > 0
    assert all(isinstance(bc, fem.DirichletBC) for bc in bc_mech)


# =============================================================================
# Solver Initialization Smoke Tests
# =============================================================================

@pytest.mark.unit
def test_mechanics_solver_initialization():
    """MechanicsSolver should initialize without errors."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    rho = Function(Q, name="rho")
    rho.x.array[:] = 0.5
    rho.x.scatter_forward()

    A = Function(T, name="A")
    A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    A.x.scatter_forward()

    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

    mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)

    assert mech is not None
    mech.destroy()


@pytest.mark.integration
def test_simple_mechanics_solve():
    """Mechanics solver should converge on tiny mesh with no load."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    u = Function(V, name="u")
    rho = Function(Q, name="rho")
    rho.x.array[:] = 0.5
    rho.x.scatter_forward()

    A = Function(T, name="A")
    A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    A.x.scatter_forward()

    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

    mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)
    mech.solver_setup()
    its, reason = mech.solve(u)

    assert its < 100, f"Solver used {its} iterations (expected < 100)"
    assert reason > 0, "Solver did not converge"

    mech.destroy()


@pytest.mark.unit
def test_stimulus_solver_initialization():
    """StimulusSolver should initialize."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
    cfg.set_dt_dim(1.0)

    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    Q = functionspace(domain, P1)

    S_old = Function(Q, name="S_old")

    stim = StimulusSolver(Q, S_old, cfg)

    assert stim is not None


@pytest.mark.unit
def test_density_solver_initialization():
    """DensitySolver should initialize."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    rho_old = Function(Q, name="rho_old")
    A = Function(T, name="A")
    A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    S = Function(Q, name="S")

    dens = DensitySolver(Q, rho_old, A, S, cfg)

    assert dens is not None


@pytest.mark.unit
def test_direction_solver_initialization():
    """DirectionSolver should initialize."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
    T = functionspace(domain, P1_ten)

    A_old = Function(T, name="A_old")

    dirn = DirectionSolver(T, A_old, cfg)

    assert dirn is not None


# =============================================================================
# Storage Smoke Tests
# =============================================================================

@pytest.mark.unit
def test_storage_initialization(shared_tmpdir):
    """UnifiedStorage should initialize."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                results_dir=shared_tmpdir / "smoke_storage",
                verbose=False, enable_telemetry=False)

    storage = UnifiedStorage(cfg)

    assert storage is not None
    assert storage.fields is not None
    assert storage.metrics is not None

    storage.close()


# =============================================================================
# Model Integration Smoke Tests
# =============================================================================

@pytest.mark.integration
def test_remodeller_initialization(shared_tmpdir):
    """Remodeller should initialize on tiny mesh."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                results_dir=shared_tmpdir / "smoke_remodeller",
                verbose=False, enable_telemetry=False)

    rem = Remodeller(cfg)

    assert rem is not None
    assert rem.u is not None
    assert rem.rho is not None
    assert rem.A is not None
    assert rem.S is not None

    rem.close()


@pytest.mark.integration
def test_remodeller_single_step(shared_tmpdir):
    """Remodeller should complete one timestep on tiny mesh."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                results_dir=shared_tmpdir / "smoke_step",
                max_subiters=10, coupling_tol=1e-3,
                verbose=False, enable_telemetry=False)

    with Remodeller(cfg) as rem:
        # Single step with relaxed tolerance for speed
        rem.step(dt=1.0)

        # Check fields were updated (aggregate across ranks since small mesh may not have DOFs on all ranks)
        local_positive = np.any(rem.rho.x.array > 0) if len(rem.rho.x.array) > 0 else False
        global_positive = comm.allreduce(local_positive, op=MPI.LOR)
        assert global_positive, "rho field should have positive values after step"


# =============================================================================
# Logger Smoke Test
# =============================================================================

@pytest.mark.unit
def test_logger_creation():
    """Logger should initialize."""
    logger = get_logger(comm, verbose=True, name="SmokeTest")

    assert logger is not None

    # Basic logging should not crash
    if comm.rank == 0:
        logger.info("Smoke test log message")


# =============================================================================
# Utility Function Smoke Tests
# =============================================================================

@pytest.mark.unit
def test_smooth_functions_basic():
    """Smooth functions should work on scalar inputs."""
    from simulation.subsolvers import smooth_abs, smooth_max, smooth_plus, smooth_heaviside

    eps = 1e-6

    # Test basic values
    assert smooth_abs(-5.0, eps) > 0
    assert smooth_max(0.3, 0.5, eps) >= 0.5
    assert smooth_plus(2.0, eps) > 0
    assert 0 <= smooth_heaviside(1.0, eps) <= 1


@pytest.mark.unit
def test_psd_tensor_enforcement():
    """PSD tensor enforcement should work."""
    from simulation.subsolvers import unittrace_psd
    import ufl

    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
    T = functionspace(domain, P1_ten)

    A = Function(T, name="A")
    A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))

    # Apply PSD enforcement
    A_hat = unittrace_psd(A, 3, 1e-6)

    # Check trace
    assert ufl.tr(A_hat) is not None  # Just check it doesn't crash


# All smoke tests should complete in < 30 seconds total
# Run with: pytest tests/test_smoke.py -v -m smoke
