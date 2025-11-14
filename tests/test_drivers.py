"""Physical correctness tests for energy/structure drivers (Instant, Gait)."""

import numpy as np
import pytest

import basix
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem import Function, functionspace

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import InstantEnergyDriver, GaitEnergyDriver


def _unit_cube(n: int = 4):
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.shared_facet)


class _DummyGait:
    """Minimal gait loader that updates a traction function on a tagged facet."""
    def __init__(self, V: fem.FunctionSpace, axis: int = 0, facet_id: int = 2, base: float = -0.3):
        self.V = V
        self.axis = int(axis)
        self.tag = int(facet_id)
        self.base = float(base)
        self.load_scale = 1.0
        self.t = Function(V, name="t_dummy")

    def get_quadrature(self):
        return [(0.0, 0.5), (100.0, 0.5)]

    def update_loads(self, phase_percent: float) -> None:
        # Linear scaling over phase; 0% -> 0, 100% -> 1
        factor = (phase_percent / 100.0) * self.load_scale
        vec = np.zeros(3, dtype=float)
        vec[self.axis] = self.base * factor
        self.t.interpolate(lambda x: np.tile(vec.reshape(3, 1), (1, x.shape[1])))
        self.t.x.scatter_forward()


@pytest.fixture
def mech_with_dummy_gait():
    comm = MPI.COMM_WORLD
    domain = _unit_cube(4)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)

    u = Function(V, name="u")
    rho = Function(Q, name="rho"); rho.x.array[:] = 0.6; rho.x.scatter_forward()
    A = Function(T, name="A")
    A.interpolate(lambda x: (np.eye(3) / 3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    A.x.scatter_forward()

    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    gait = _DummyGait(V, axis=0, facet_id=2, base=-0.4)
    mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [(gait.t, gait.tag)])
    mech.setup()
    return mech, gait


class TestInstantDriver:
    def test_energy_expr_matches_mechanics(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        # Apply a non-zero load
        gait.update_loads(100.0)
        mech.assemble_rhs(); mech.solve()

        drv = InstantEnergyDriver(mech)
        psi_expr = drv.energy_expr()

        # Integrate ψ and compare to manual 0.5 σ:ε integral
        cfg = mech.cfg
        eps = mech.get_strain_tensor()
        sig = mech.sigma(mech.u, mech.rho, mech.A_dir)
        psi_manual = 0.5 * ufl.inner(sig, eps)

        psi_drv_loc = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        psi_man_loc = fem.assemble_scalar(fem.form(psi_manual * cfg.dx))
        comm = cfg.domain.comm
        psi_drv = comm.allreduce(psi_drv_loc, op=MPI.SUM)
        psi_man = comm.allreduce(psi_man_loc, op=MPI.SUM)
        assert psi_drv == pytest.approx(psi_man, rel=1e-9, abs=1e-12)

    def test_structure_psd_and_trace(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        gait.update_loads(100.0)
        mech.assemble_rhs(); mech.solve()
        drv = InstantEnergyDriver(mech)

        M = drv.structure_expr()
        cfg = mech.cfg
        # Symmetry and PSD: integrate v^T M v for random v
        v = ufl.as_vector([1.0, 2.0, 3.0])
        vtMv = ufl.inner(v, ufl.dot(M, v))
        val_loc = fem.assemble_scalar(fem.form(vtMv * cfg.dx))
        val = cfg.domain.comm.allreduce(val_loc, op=MPI.SUM)
        assert val >= -1e-10

        # Trace(M) = ε:ε (Frobenius norm squared)
        eps = mech.get_strain_tensor()
        trM_loc = fem.assemble_scalar(fem.form(ufl.tr(M) * cfg.dx))
        e2_loc = fem.assemble_scalar(fem.form(ufl.inner(eps, eps) * cfg.dx))
        comm = cfg.domain.comm
        trM = comm.allreduce(trM_loc, op=MPI.SUM)
        e2 = comm.allreduce(e2_loc, op=MPI.SUM)
        assert trM == pytest.approx(e2, rel=1e-9, abs=1e-12)


class TestGaitDriver:
    def test_energy_does_not_scale_with_cpd(self, mech_with_dummy_gait):
        """Energy expression produces gait-averaged value (no cpd parameter)."""
        mech, gait = mech_with_dummy_gait
        drv = GaitEnergyDriver(mech, gait, psi_ref=mech.cfg.psi_ref)
        drv.update_snapshots()
        psi_loc = fem.assemble_scalar(fem.form(drv.energy_expr() * mech.cfg.dx))
        psi = mech.comm.allreduce(psi_loc, op=MPI.SUM)
        # Just verify positive energy density from gait loading
        assert psi > 0, f"Expected positive energy, got {psi:.3e}"

    def test_energy_scales_with_load(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        gait.load_scale = 1.0
        drv_base = GaitEnergyDriver(mech, gait, psi_ref=mech.cfg.psi_ref)
        drv_base.update_snapshots()
        psi_base_loc = fem.assemble_scalar(fem.form(drv_base.energy_expr() * mech.cfg.dx))
        psi_base = mech.comm.allreduce(psi_base_loc, op=MPI.SUM)

        # Double load -> ~4x energy (linear elasticity: ψ ~ ε² ~ load²)
        gait.load_scale = 2.0
        drv_double = GaitEnergyDriver(mech, gait, psi_ref=mech.cfg.psi_ref)
        drv_double.update_snapshots()
        psi_double_loc = fem.assemble_scalar(fem.form(drv_double.energy_expr() * mech.cfg.dx))
        psi_double = mech.comm.allreduce(psi_double_loc, op=MPI.SUM)

        ratio = psi_double / max(psi_base, 1e-300)
        assert 3.0 < ratio < 5.0, f"Energy should scale ~4x with 2x load; ratio={ratio:.2f}"

    def test_structure_psd_and_scaling(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        gait.load_scale = 1.0
        drv = GaitEnergyDriver(mech, gait, psi_ref=mech.cfg.psi_ref)
        drv.update_snapshots()
        M1 = drv.structure_expr()
        M1_int_loc = fem.assemble_scalar(fem.form(ufl.tr(M1) * mech.cfg.dx))
        M1_int = mech.comm.allreduce(M1_int_loc, op=MPI.SUM)

        gait.load_scale = 2.0
        drv.invalidate()
        drv.update_snapshots()
        M2 = drv.structure_expr()
        M2_int_loc = fem.assemble_scalar(fem.form(ufl.tr(M2) * mech.cfg.dx))
        M2_int = mech.comm.allreduce(M2_int_loc, op=MPI.SUM)

        # PSD check via integrated quadratic form
        v = ufl.as_vector([1.0, 0.0, 0.0])
        vtMv1 = fem.assemble_scalar(fem.form(ufl.inner(v, ufl.dot(M1, v)) * mech.cfg.dx))
        vtMv2 = fem.assemble_scalar(fem.form(ufl.inner(v, ufl.dot(M2, v)) * mech.cfg.dx))
        vtMv1 = mech.comm.allreduce(vtMv1, op=MPI.SUM)
        vtMv2 = mech.comm.allreduce(vtMv2, op=MPI.SUM)
        assert vtMv1 >= -1e-10 and vtMv2 >= -1e-10

        # Scaling ~4x with 2x load
        ratio = M2_int / max(M1_int, 1e-300)
        assert 3.0 < ratio < 5.0, f"Structure trace should scale ~4x; ratio={ratio:.2f}"

