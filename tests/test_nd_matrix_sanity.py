
#!/usr/bin/env python3
"""
Sanity checks around matrix assembly scaling.
We keep these very light to avoid duplicating the heavy invariance tests.
"""
import numpy as np
import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")

from mpi4py import MPI
from dolfinx import mesh, fem
import basix

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver

def _mat_action_norm(A, x_petsc):
    """Compute ||A x||_2 without converting to dense."""
    y = x_petsc.duplicate()
    A.mult(x_petsc, y)
    return float(y.norm())

@pytest.mark.unit
def test_stimulus_matrix_changes_with_dt():
    """Changing dt (dimensional) and calling update_lhs should change the LHS norm significantly."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 4, 4, 4, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    # Q space
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    Q = fem.functionspace(domain, P1)

    S_old = fem.Function(Q, name="S_old")
    solver = StimulusSolver(Q, S_old, cfg)
    solver.solver_setup()

    n1 = solver.A.norm()
    cfg.set_dt_dim(50.0)
    solver.update_lhs()
    n2 = solver.A.norm()
    # Require a noticeable change
    assert abs(n2 - n1) / max(1.0, n1) > 1e-6, "Stimulus LHS norm did not change with dt"

@pytest.mark.unit
def test_mechanics_operator_action_nonzero():
    """Mechanics K should produce nonzero action on a random vector when rho>0."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 4, 4, 4, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    # Spaces
    P1v = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    V = fem.functionspace(domain, P1v)

    Q = fem.functionspace(domain, ("P", 1))
    T = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,3)))

    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho"); rho.x.array[:] = 0.5; rho.x.scatter_forward()
    A = fem.Function(T, name="A"); A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:,None] * np.ones((1, x.shape[1]))); A.x.scatter_forward()
    bcs = build_dirichlet_bcs(V, facet_tags)
    mech = MechanicsSolver(V, rho, A, bcs, [], cfg)
    mech.solver_setup()

    z = u.x.petsc_vec.duplicate()
    z.setRandom()
    norm = _mat_action_norm(mech.A, z)
    assert np.isfinite(norm) and norm > 0.0, "Mechanics operator action is zero or non-finite"

@pytest.mark.unit
def test_density_matrix_psd_action():
    """x^T A x >= 0 for the density solver (discrete PSD check)."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 4, 4, 4, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    Q = fem.functionspace(domain, ("P", 1))
    T = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,3)))

    rho_old = fem.Function(Q, name="rho_old"); rho_old.x.array[:] = 0.5; rho_old.x.scatter_forward()
    A = fem.Function(T, name="A"); A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:,None] * np.ones((1, x.shape[1]))); A.x.scatter_forward()
    S = fem.Function(Q, name="S"); S.x.array[:] = 0.0; S.x.scatter_forward()

    dens = DensitySolver(Q, rho_old, A, S, cfg)
    dens.solver_setup(); dens.update_system()

    z = fem.Function(Q).x.petsc_vec.duplicate()
    z.setRandom()
    y = z.duplicate()
    dens.A.mult(z, y)
    energy = z.dot(y)
    assert energy >= -1e-12, f"Density operator not PSD: x^T A x = {energy}"
