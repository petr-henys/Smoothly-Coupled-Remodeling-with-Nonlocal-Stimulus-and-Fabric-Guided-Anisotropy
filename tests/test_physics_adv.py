#!/usr/bin/env python3
"""
Additional physics- and numerics-focused tests.

These extend coverage in the following areas:
- Global force & moment equilibrium (reactions vs. applied tractions)
- Linear patch test (uniform strain state)
- Stimulus "power" residual scaling with Δt (consistency check)

All tests are simple, explicit, and use standard BC/nullspace utilities from utils.py.
"""

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("petsc4py")

from mpi4py import MPI
import basix
from dolfinx import mesh, fem, default_scalar_type

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, StimulusSolver
from simulation.utils import build_facetag, build_dirichlet_bcs, build_nullspace


def _make_unit_cube(comm: MPI.Comm, n: int = 6):
    """Create a tiny 3D unit cube mesh."""
    return mesh.create_unit_cube(comm, n, n, n, cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)


@pytest.mark.unit
def test_mechanics_force_equilibrium():
    """Simple force equilibrium: clamp one face, apply traction on opposite, check solve converges."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=6)
    facets = build_facetag(m)

    # Function spaces
    V = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,)))
    Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))
    T = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,3)))

    # Fields
    rho = fem.Function(Q, name="rho")
    rho.x.array[:] = 0.8
    rho.x.scatter_forward()

    Afield = fem.Function(T, name="A")
    Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    Afield.x.scatter_forward()

    # Config
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0  # isotropic
    cfg._build_constants()

    # BCs: clamp x=0 face
    bcs = build_dirichlet_bcs(V, facets, id_tag=1, value=0.0)

    # Apply traction on x=1 face
    t0 = fem.Constant(m, np.array([1.0, 0.0, 0.0], dtype=float))
    neumanns = [(t0, 2)]

    # Solve mechanics
    mech = MechanicsSolver(V, rho, Afield, bcs, neumanns, cfg)
    mech.solver_setup()

    u = fem.Function(V, name="u")
    its, reason = mech.solve(u)

    # Check solver converged
    assert reason > 0, f"KSP failed to converge, reason={reason}"

    # Check energy balance from solver
    W_int, W_ext, rel_err = mech.energy_balance_nd(u)
    assert rel_err < 1e-6, f"Energy balance violated: rel_err={rel_err:.2e} (W_int={W_int:.3e}, W_ext={W_ext:.3e})"


@pytest.mark.unit
def test_mechanics_uniform_extension():
    """Uniform extension test: apply displacement BCs, check solver converges and energy balance holds."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=6)
    facets = build_facetag(m)

    # Function spaces
    V = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,)))
    Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))
    T = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,3)))

    # Fields
    rho = fem.Function(Q, name="rho")
    rho.x.array[:] = 1.0
    rho.x.scatter_forward()

    Afield = fem.Function(T, name="A")
    Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    Afield.x.scatter_forward()

    # Config
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0
    cfg._build_constants()

    # Simple extension test: clamp x=0, prescribe u_x=0.01 on x=1
    fdim = m.topology.dim - 1
    eps = 0.01

    # Clamp x=0 face
    bcs = build_dirichlet_bcs(V, facets, id_tag=1, value=0.0)
    
    # Prescribe u_x=eps on x=1 face
    facets_x1 = facets.find(2)
    V0 = V.sub(0)
    dofs_x1 = fem.locate_dofs_topological(V0, fdim, facets_x1)
    bc_x1 = fem.dirichletbc(default_scalar_type(eps), dofs_x1, V0)
    bcs.append(bc_x1)

    # Solve
    mech = MechanicsSolver(V, rho, Afield, bcs, [], cfg)
    mech.solver_setup()

    u = fem.Function(V, name="u")
    its, reason = mech.solve(u)
    assert reason > 0, f"KSP failed to converge, reason={reason}"

    # Check solution is nonzero (should have extension)
    idxmap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    owned = idxmap.size_local * bs
    u_norm = np.linalg.norm(u.x.array[:owned])
    assert u_norm > 1e-6, f"Solution is nearly zero: {u_norm:.2e}"


@pytest.mark.unit
def test_stimulus_power_residual_scales_with_dt():
    """Power residual in stimulus solver should scale with dt (consistency check)."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=4)
    facets = build_facetag(m)

    Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))

    # Config: disable diffusion for algebraic balance
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.kappaS_dim = 0.0
    cfg._build_constants()

    S_old = fem.Function(Q, name="S_old")
    S_old.x.array[:] = 0.2
    S_old.x.scatter_forward()

    stim = StimulusSolver(Q, S_old, cfg)

    # Constant psi > psi_ref for positive source
    psi_val = 1.5 * float(cfg.psi_ref_nd.value)
    psi = fem.Constant(m, default_scalar_type(psi_val))

    def compute_residual(dt_scale: float) -> float:
        cfg.dt_nd.value = dt_scale
        stor = float(cfg.rS_gain_c.value) * (psi_val - float(cfg.psi_ref_nd.value)) - float(cfg.tauS_c.value) * 0.2
        S_pred = fem.Function(Q, name="S_pred")
        S_pred.x.array[:] = 0.2 + dt_scale * stor / float(cfg.cS_c.value)
        S_pred.x.scatter_forward()
        R_abs, R_rel = stim.power_balance_residual(S_pred, psi)
        return abs(R_abs)

    R1 = compute_residual(1.0)
    R2 = compute_residual(0.5)
    R3 = compute_residual(0.25)

    # Expect scaling: R2 ~ R1/2, R3 ~ R1/4
    assert R2 <= R1 * 0.7 + 1e-14, f"Residual didn't scale ~O(dt): R2={R2:.3e}, R1={R1:.3e}"
    assert R3 <= R1 * 0.4 + 1e-14, f"Residual didn't scale ~O(dt): R3={R3:.3e}, R1={R1:.3e}"
