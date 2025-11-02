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

REFACTORED: Reduced from 20+ tests to 8 critical smoke tests
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
from simulation.subsolvers import MechanicsSolver
from simulation.model import Remodeller
from simulation.storage import UnifiedStorage
from simulation.logger import get_logger

comm = MPI.COMM_WORLD


# =============================================================================
# Core Smoke Tests (8 critical tests, reduced from 20+)
# =============================================================================

@pytest.mark.unit
def test_config_and_mesh_creation():
    """Config and mesh creation should work without errors."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    # Verify basic mesh properties
    assert domain.topology.dim == 3
    assert domain.geometry.dim == 3
    
    # Verify config initialized correctly
    assert cfg is not None
    assert cfg.domain is not None
    assert float(cfg.E0_nd) > 0
    assert cfg.dx is not None
    
    comm.Barrier()

@pytest.mark.unit
def test_function_spaces_and_bc_creation():
    """Function spaces and boundary conditions should initialize."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)

    # Create function spaces
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    # Verify spaces have positive DOF counts
    assert V.dofmap.index_map.size_global * V.dofmap.index_map_bs > 0
    assert Q.dofmap.index_map.size_global * Q.dofmap.index_map_bs > 0
    assert T.dofmap.index_map.size_global * T.dofmap.index_map_bs > 0
    
    # Create boundary conditions
    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    assert len(bc_mech) > 0
    assert all(isinstance(bc, fem.DirichletBC) for bc in bc_mech)


@pytest.mark.unit
def test_solver_initialization():
    """All subsolver types should initialize without errors."""
    from simulation.subsolvers import StimulusSolver, DensitySolver, DirectionSolver
    
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
    cfg.set_dt_dim(1.0)

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    # Initialize fields
    rho = Function(Q, name="rho")
    rho.x.array[:] = 0.5
    rho.x.scatter_forward()

    A = Function(T, name="A")
    A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    A.x.scatter_forward()

    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    
    S_old = Function(Q, name="S_old")
    rho_old = Function(Q, name="rho_old")
    A_old = Function(T, name="A_old")

    # Test all solver types
    mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)
    assert mech is not None
    mech.destroy()
    
    stim = StimulusSolver(Q, S_old, cfg)
    assert stim is not None
    
    dens = DensitySolver(Q, rho_old, A, Function(Q, name="S"), cfg)
    assert dens is not None
    
    dirn = DirectionSolver(T, A_old, cfg)
    assert dirn is not None


@pytest.mark.integration
def test_simple_mechanics_solve():
    """Mechanics solver should converge quickly on tiny mesh."""
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
def test_storage_initialization(shared_tmpdir):
    """Storage system should initialize correctly."""
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


@pytest.mark.integration
def test_remodeller_initialization(shared_tmpdir):
    """Remodeller should initialize all fields and solvers."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                results_dir=shared_tmpdir / "smoke_remodeller",
                verbose=False, enable_telemetry=False)

    rem = Remodeller(cfg)

    # Verify all critical fields exist
    assert rem is not None
    assert rem.u is not None
    assert rem.rho is not None
    assert rem.A is not None
    assert rem.S is not None
    
    # Verify solvers initialized
    assert rem.mechsolver is not None
    assert rem.stimsolver is not None
    assert rem.densolver is not None
    assert rem.dirsolver is not None

    rem.close()


@pytest.mark.integration
def test_remodeller_single_timestep(shared_tmpdir):
    """Remodeller should complete one timestep successfully."""
    domain = mesh.create_unit_cube(comm, 2, 2, 2, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                results_dir=shared_tmpdir / "smoke_step",
                max_subiters=10, coupling_tol=1e-3,  # Relaxed for speed
                verbose=False, enable_telemetry=False)

    with Remodeller(cfg) as rem:
        # Single step with relaxed tolerance for speed
        rem.step(dt=1.0)

        # Check fields were updated (aggregate across ranks for small mesh)
        local_positive = np.any(rem.rho.x.array > 0) if len(rem.rho.x.array) > 0 else False
        global_positive = comm.allreduce(local_positive, op=MPI.LOR)
        assert global_positive, "rho field should have positive values after step"


@pytest.mark.unit
def test_utility_functions_basic():
    """Basic utility functions should work correctly."""
    from simulation.subsolvers import smooth_abs, smooth_max, smooth_plus, smooth_heaviside

    eps = 1e-6

    # Test smooth functions
    assert smooth_abs(-5.0, eps) > 0
    assert smooth_max(0.3, 0.5, eps) >= 0.5
    assert smooth_plus(2.0, eps) > 0
    assert 0 <= smooth_heaviside(1.0, eps) <= 1
    
    # Test logger creation
    logger = get_logger(comm, verbose=True, name="SmokeTest")
    assert logger is not None


# Total: 8 smoke tests (reduced from 20+)
# Expected runtime: < 30 seconds on small mesh
