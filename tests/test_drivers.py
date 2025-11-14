"""Unified tests for remodeling drivers (Instant and Gait)."""

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
from tests.test_model import _DummyGaitLoader, _unit_cube


def _unit_cube_local(n: int = 4):
    """Local unit cube mesh helper for this module."""
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.shared_facet)


class _DummyGait:
    """Minimal gait loader that produces three traction fields for hip + muscles."""

    def __init__(self, V: fem.FunctionSpace, facet_id: int = 2, base: float = -0.3):
        self.V = V
        self.tag = int(facet_id)
        self.base = float(base)
        self.load_scale = 1.0
        self.t_hip = Function(V, name="t_hip_dummy")
        self.t_glmed = Function(V, name="t_glmed_dummy")
        self.t_glmax = Function(V, name="t_glmax_dummy")

    def get_quadrature(self):
        return [(0.0, 0.5), (100.0, 0.5)]

    def update_loads(self, phase_percent: float) -> None:
        factor = (phase_percent / 100.0) * self.load_scale
        hip_vec = np.array([self.base * factor, -0.5 * self.base * factor, -0.3 * self.base * factor])
        glmed_vec = np.array([0.2 * self.base * factor, 0.8 * self.base * factor, -0.1 * self.base * factor])
        glmax_vec = np.array([0.1 * self.base * factor, -0.2 * self.base * factor, -0.9 * self.base * factor])
        for field, vec in zip((self.t_hip, self.t_glmed, self.t_glmax), (hip_vec, glmed_vec, glmax_vec)):
            field.interpolate(lambda x, vec=vec: np.tile(vec.reshape(3, 1), (1, x.shape[1])))
            field.x.scatter_forward()


@pytest.fixture
def mech_with_dummy_gait():
    """Mechanics solver on unit cube with simple dummy gait loading."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube_local(4)
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
    gait = _DummyGait(V, facet_id=2, base=-0.4)
    neumann_bcs = [
        (gait.t_hip, gait.tag),
        (gait.t_glmed, gait.tag),
        (gait.t_glmax, gait.tag),
    ]
    mech = MechanicsSolver(u, rho, A, cfg, bc_mech, neumann_bcs)
    mech.setup()
    return mech, gait


class TestInstantDriver:
    def test_energy_expr_matches_mechanics(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        gait.update_loads(100.0)
        mech.assemble_rhs(); mech.solve()

        drv = InstantEnergyDriver(mech)
        psi_expr = drv.energy_expr()

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
        v = ufl.as_vector([1.0, 2.0, 3.0])
        vtMv = ufl.inner(v, ufl.dot(M, v))
        val_loc = fem.assemble_scalar(fem.form(vtMv * cfg.dx))
        val = cfg.domain.comm.allreduce(val_loc, op=MPI.SUM)
        assert val >= -1e-10

        eps = mech.get_strain_tensor()
        trM_loc = fem.assemble_scalar(fem.form(ufl.tr(M) * cfg.dx))
        e2_loc = fem.assemble_scalar(fem.form(ufl.inner(eps, eps) * cfg.dx))
        comm = cfg.domain.comm
        trM = comm.allreduce(trM_loc, op=MPI.SUM)
        e2 = comm.allreduce(e2_loc, op=MPI.SUM)
        assert trM == pytest.approx(e2, rel=1e-9, abs=1e-12)


class TestGaitDriverUnitCube:
    def test_energy_does_not_scale_with_cpd(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        drv = GaitEnergyDriver(mech, gait, mech.cfg)
        drv.update_snapshots()
        psi_loc = fem.assemble_scalar(fem.form(drv.energy_expr() * mech.cfg.dx))
        psi = mech.comm.allreduce(psi_loc, op=MPI.SUM)
        assert psi > 0, f"Expected positive energy, got {psi:.3e}"

    def test_energy_scales_with_load(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        gait.load_scale = 1.0
        drv_base = GaitEnergyDriver(mech, gait, mech.cfg)
        drv_base.update_snapshots()
        psi_base_loc = fem.assemble_scalar(fem.form(drv_base.energy_expr() * mech.cfg.dx))
        psi_base = mech.comm.allreduce(psi_base_loc, op=MPI.SUM)

        gait.load_scale = 2.0
        drv_double = GaitEnergyDriver(mech, gait, mech.cfg)
        drv_double.update_snapshots()
        psi_double_loc = fem.assemble_scalar(fem.form(drv_double.energy_expr() * mech.cfg.dx))
        psi_double = mech.comm.allreduce(psi_double_loc, op=MPI.SUM)

        ratio = psi_double / max(psi_base, 1e-300)
        expected = 2.0 ** (2.0 * mech.cfg.n_power)
        assert 0.5 * expected < ratio < 1.5 * expected, (
            "Energy scaling should follow load^(2*n_power); "
            f"expected≈{expected:.2f}, ratio={ratio:.2f}"
        )

    def test_structure_psd_and_scaling(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        gait.load_scale = 1.0
        drv = GaitEnergyDriver(mech, gait, mech.cfg)
        drv.update_snapshots()
        M1 = drv.structure_expr()
        M1_int_loc = fem.assemble_scalar(fem.form(ufl.tr(M1) * mech.cfg.dx))
        M1_int = mech.comm.allreduce(M1_int_loc, op=MPI.SUM)

        gait.load_scale = 2.0
        drv = GaitEnergyDriver(mech, gait, mech.cfg)
        drv.update_snapshots()
        M2 = drv.structure_expr()
        M2_int_loc = fem.assemble_scalar(fem.form(ufl.tr(M2) * mech.cfg.dx))
        M2_int = mech.comm.allreduce(M2_int_loc, op=MPI.SUM)

        v = ufl.as_vector([1.0, 0.0, 0.0])
        vtMv1 = fem.assemble_scalar(fem.form(ufl.inner(v, ufl.dot(M1, v)) * mech.cfg.dx))
        vtMv2 = fem.assemble_scalar(fem.form(ufl.inner(v, ufl.dot(M2, v)) * mech.cfg.dx))
        vtMv1 = mech.comm.allreduce(vtMv1, op=MPI.SUM)
        vtMv2 = mech.comm.allreduce(vtMv2, op=MPI.SUM)
        assert vtMv1 >= -1e-10 and vtMv2 >= -1e-10

        ratio = M2_int / max(M1_int, 1e-300)
        assert 3.0 < ratio < 5.0, f"Structure trace should scale ~4x; ratio={ratio:.2f}"


class TestGaitDriverFemur:
    def test_driver_produces_nonzero_energy(self, tmp_path):
        """Check if GaitEnergyDriver produces non-zero energy expression.

        This is a light-weight version of the original debug test.
        """
        comm = MPI.COMM_WORLD
        domain = _unit_cube(4)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True, results_dir=str(tmp_path))

        gdim = domain.geometry.dim
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim, gdim))

        V = fem.functionspace(domain, P1_vec)
        Q = fem.functionspace(domain, P1)
        T = fem.functionspace(domain, P1_ten)

        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A = fem.Function(T, name="dir_tensor")

        rho.x.array[:] = cfg.rho0

        def _A_const(x):
            n = x.shape[1]
            vals = (np.eye(gdim, dtype=np.float64) / gdim).reshape(gdim * gdim, 1)
            return np.tile(vals, (1, n))

        A.interpolate(_A_const)
        A.x.scatter_forward()

        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        gait_loader = _DummyGaitLoader(V)
        neumann_bcs = [
            (gait_loader.t_hip, 2),
            (gait_loader.t_glmed, 2),
            (gait_loader.t_glmax, 2),
        ]

        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, neumann_bcs)
        mech.setup()

        driver = GaitEnergyDriver(mech, gait_loader, cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()

        psi_form = fem.form(psi_expr * cfg.dx)
        psi_local = fem.assemble_scalar(psi_form)
        psi_global = comm.allreduce(psi_local, op=MPI.SUM)

        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        vol_global = comm.allreduce(vol_local, op=MPI.SUM)

        psi_avg = psi_global / vol_global
        assert psi_avg > 1e-8, f"Average energy density too small: {psi_avg:.3e}"
