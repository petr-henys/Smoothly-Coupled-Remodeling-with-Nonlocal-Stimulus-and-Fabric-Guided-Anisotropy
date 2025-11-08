#!/usr/bin/env python3
"""
Additional physics- and numerics-focused tests.

These extend coverage in the following areas:
- Global force & moment equilibrium (reactions vs. applied tractions)
- Linear patch test (uniform strain state)
- Mechanics matrix symmetry / (semi)definiteness (Rayleigh check)
- Stimulus "power" residual scaling with Δt (consistency check)

All tests are small and skip automatically if dolfinx is missing.
"""

import math
import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("petsc4py")

from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, create_matrix
from petsc4py import PETSc

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, StimulusSolver
from simulation.utils import build_facetag, build_dirichlet_bcs

dtype = PETSC_SCALAR = PETSc.ScalarType


def _make_unit_cube(comm: MPI.Comm, n: int = 6):
    """Create a tiny 3D unit cube mesh."""
    return mesh.create_unit_cube(comm, n, n, n, mesh.CellType.hexahedron, mesh.GhostMode.shared_facet)


def _vector_L2_integral(mesh_, expr, ds_meas) -> np.ndarray:
    """Integrate a 3-vector expression over boundary: ∫ expr ds -> R^3"""
    vals = []
    for i in range(3):
        form = fem.form(expr[i] * ds_meas)
        val_local = fem.assemble_scalar(form)
        vals.append(float(mesh_.comm.allreduce(val_local, op=MPI.SUM)))
    return np.array(vals, dtype=float)


def _matrix_norm(A: PETSc.Mat) -> float:
    """Operator 2-norm (via power iteration surrogate using random vectors)."""
    # PETSc has no direct spectral norm; use a few random actions as a proxy
    v = A.createVecLeft(); v.setRandom()
    w = A.createVecLeft()
    for _ in range(4):
        A.mult(v, w)
        n = w.norm()
        if n > 0:
            w.scale(1.0 / n)
        v, w = w, v
    A.mult(v, w)
    return w.norm()


@pytest.mark.unit
def test_mechanics_global_force_and_moment_equilibrium():
    """∫ σ·n ds + ∫ t ds ≈ 0 and ∫ x×(σ·n) ds + ∫ x×t ds ≈ 0 (no body forces)."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=6)
    facets = build_facetag(m)

    # Function spaces
    P1_vec = mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,))
    V = fem.functionspace(m, P1_vec)
    P1 = mesh.ufl.element("Lagrange", m.topology.cell_name(), 1)
    Q = fem.functionspace(m, P1)

    # Fields
    rho = fem.Function(Q, name="rho"); rho.x.array[:] = 0.8; rho.x.scatter_forward()
    A = fem.Function(fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,3)))),  # dummy, not used if xi_aniso=0
    # Minimal config
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0  # isotropic to simplify
    cfg._build_constants()

    # Mechanics: clamp x=0, apply traction on x=1
    bcs = build_dirichlet_bcs(V, facets, id_tag=1, value=0.0)  # x=0
    t0 = fem.Constant(m, default_scalar_type((1.0, 0.0, 0.0)))
    neumanns = [(t0, 2)]  # x=1
    # Use identity-like A (unit trace) so anisotropy is off even if xi_aniso>0
    T = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,3)))
    Afield = fem.Function(T, name="A")
    Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1]))); Afield.x.scatter_forward()

    mech = MechanicsSolver(V, rho, Afield, bcs, neumanns, cfg)
    mech.solver_setup()
    u = fem.Function(V, name="u")
    its, reason = mech.solve(u)
    assert reason in (0, 4), "KSP did not converge"

    # Build quantities for equilibrium checks
    n = ufl.FacetNormal(m)
    sigma = mech.sigma(u, rho)
    traction_int = ufl.dot(sigma, n)             # internal traction σ·n
    traction_ext = t0                            # applied traction (Neumann)

    # Force balance over the *whole* boundary
    F_int = _vector_L2_integral(m, traction_int, cfg.ds)
    F_ext = _vector_L2_integral(m, traction_ext, cfg.ds(2))  # only tag=2
    force_res = np.linalg.norm(F_int + F_ext) / max(1e-12, np.linalg.norm(F_ext))
    assert force_res < 5e-3, f"Global force imbalance too large: {force_res:.2e}"

    # Moment balance about origin
    x = ufl.SpatialCoordinate(m)
    M_int = _vector_L2_integral(m, ufl.cross(x, traction_int), cfg.ds)
    M_ext = _vector_L2_integral(m, ufl.cross(x, traction_ext), cfg.ds(2))
    moment_res = np.linalg.norm(M_int + M_ext) / max(1e-12, np.linalg.norm(M_ext))
    assert moment_res < 5e-3, f"Global moment imbalance too large: {moment_res:.2e}"


@pytest.mark.unit
def test_mechanics_linear_patch():
    """Uniform strain state: linear displacement field is reproduced exactly."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=4)
    facets = build_facetag(m)

    # V, Q, T
    V = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,)))
    Q = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1))
    T = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,3)))

    # Fields
    rho = fem.Function(Q, name="rho"); rho.x.array[:] = 1.0; rho.x.scatter_forward()
    Afield = fem.Function(T, name="A"); Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1]))); Afield.x.scatter_forward()

    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0
    cfg._build_constants()

    # Prescribe linear displacement on *whole boundary*: u = [α x, β y, γ z]
    alpha, beta, gamma = 0.01, -0.02, 0.015
    u_exact = fem.Function(V, name="u_exact")
    u_exact.interpolate(lambda x: np.vstack((alpha*x[0], beta*x[1], gamma*x[2])))

    # Dirichlet on all facets (1..4); no loads
    bcs = []
    for tag in (1,2,3,4):
        bcs.extend(build_dirichlet_bcs(V, facets, id_tag=tag, value=None))
    # Apply nonzero values via interpolation
    for bc in bcs:
        dofs, first_ghost = bc.dof_indices()
        u_vec = u_exact.x.array[:]
        bc.set_value(u_vec)

    mech = MechanicsSolver(V, rho, Afield, bcs, [], cfg)
    mech.solver_setup()
    u = fem.Function(V, name="u")
    its, reason = mech.solve(u)
    assert reason in (0, 4)

    # Compare with exact linear field (L2-ish discrete norm over owned DOFs)
    idxmap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    owned = idxmap.size_local * bs
    err = np.linalg.norm(u.x.array[:owned] - u_exact.x.array[:owned]) / max(1e-16, np.linalg.norm(u_exact.x.array[:owned]))
    assert err < 1e-10, f"Patch test failed: rel error {err:.2e}"


@pytest.mark.unit
def test_mechanics_matrix_symmetry_and_psd():
    """Check ||K-Kᵀ||/||K|| ≪ 1 and zᵀKz ≥ 0 (Rayleigh check)."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=6)
    facets = build_facetag(m)

    V = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,)))
    Q = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1))
    T = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1, shape=(3,3)))

    rho = fem.Function(Q, name="rho"); rho.x.array[:] = 0.7; rho.x.scatter_forward()
    Afield = fem.Function(T, name="A"); Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1]))); Afield.x.scatter_forward()

    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0
    cfg._build_constants()

    bcs = build_dirichlet_bcs(V, facets, id_tag=1, value=0.0)

    mech = MechanicsSolver(V, rho, Afield, bcs, [], cfg)
    A = create_matrix(mech.a_form); assemble_matrix(A, mech.a_form, bcs=bcs); A.assemble()

    # Symmetry check
    AT = A.transpose()
    # Compute a proxy for ||A|| via power iteration
    nA = _matrix_norm(A) or 1.0
    diff = A.copy(); diff.axpy(-1.0, AT, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN); diff.assemble()
    nDiff = _matrix_norm(diff)
    assert nDiff / nA < 1e-10, f"Mechanics stiffness is not symmetric enough: {nDiff/nA:.3e}"

    # Rayleigh PSD check
    z = A.createVecLeft(); z.setRandom()
    y = A.createVecLeft(); A.mult(z, y)
    energy = z.dot(y)
    assert energy >= -1e-10, f"z^T K z must be >= 0, got {energy:.3e}"


@pytest.mark.unit
def test_stimulus_power_residual_scales_with_dt():
    """For an explicit Euler predictor S* = Sold + dt*(source - decay)/cS (kappa=0), residual ~ O(dt)."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=4)
    facets = build_facetag(m)

    Q = fem.functionspace(m, mesh.ufl.element("Lagrange", m.topology.cell_name(), 1))

    # Config: disable diffusion to keep algebraic balance
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.kappaS_dim = 0.0
    cfg._build_constants()

    S_old = fem.Function(Q, name="S_old"); S_old.x.array[:] = 0.2; S_old.x.scatter_forward()
    stim = StimulusSolver(Q, S_old, cfg)

    # Constant psi > psi_ref to have positive source
    psi_val = 1.5 * float(cfg.psi_ref_nd.value)
    psi = fem.Constant(m, default_scalar_type(psi_val))

    def predictor(dt_scale: float) -> float:
        # set dt
        cfg.dt_nd.value = dt_scale
        # S* = S_old + dt/cS * (rS*(psi-psi_ref) - tau*S_old)
        stor = float(cfg.rS_gain_c.value) * (psi_val - float(cfg.psi_ref_nd.value)) - float(cfg.tauS_c.value) * 0.2
        S_pred = fem.Function(Q, name="S_pred"); S_pred.x.array[:] = 0.2 + dt_scale * stor / float(cfg.cS_c.value); S_pred.x.scatter_forward()
        R_abs, R_rel = stim.power_balance_residual(S_pred, psi)
        return abs(R_abs)

    R1 = predictor(1.0)
    R2 = predictor(0.5)
    R3 = predictor(0.25)

    # Expect near-linear scaling: R2 ~ R1/2, R3 ~ R1/4 (allow loose factor)
    assert R2 <= R1 * 0.7 + 1e-14, f"Residual didn't scale ~O(dt): R2={R2:.3e}, R1={R1:.3e}"
    assert R3 <= R1 * 0.4 + 1e-14, f"Residual didn't scale ~O(dt): R3={R3:.3e}, R1={R1:.3e}"
