"""Unified tests for remodeling drivers (Instant and Gait)."""

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitDriver

@pytest.fixture
def mech_with_dummy_gait(spaces, cfg, bc_mech, dummy_gait_loader):
    """Mechanics solver on unit cube with simple dummy gait loading."""
    u = fem.Function(spaces.V, name="u")
    rho = fem.Function(spaces.Q, name="rho"); rho.x.array[:] = 0.6; rho.x.scatter_forward()
    
    neumann_bcs = [
        (dummy_gait_loader.t_hip, dummy_gait_loader.tag),
        (dummy_gait_loader.t_glmed, dummy_gait_loader.tag),
        (dummy_gait_loader.t_glmax, dummy_gait_loader.tag),
    ]
    mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
    mech.setup()
    return mech, dummy_gait_loader


class TestGaitDriverUnitCube:
    def test_energy_does_not_scale_with_cpd(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        drv = GaitDriver(mech, gait, mech.cfg)
        drv.update_snapshots()
        psi_loc = fem.assemble_scalar(fem.form(drv.stimulus_expr() * mech.cfg.dx))
        psi = mech.comm.allreduce(psi_loc, op=MPI.SUM)
        assert psi > 0, f"Expected positive energy, got {psi:.3e}"

    def test_energy_scales_with_load(self, mech_with_dummy_gait):
        mech, gait = mech_with_dummy_gait
        # Set n_power to 2.0.
        # Stimulus psi = (Sum n_i * sigma_i^m)^(1/m).
        # Since sigma is linear in load, psi is linear in load.
        # So doubling load should double psi.
        mech.cfg.n_power = 2.0

        gait.load_scale = 1.0
        drv_base = GaitDriver(mech, gait, mech.cfg)
        drv_base.update_snapshots()
        psi_base_loc = fem.assemble_scalar(fem.form(drv_base.stimulus_expr() * mech.cfg.dx))
        psi_base = mech.comm.allreduce(psi_base_loc, op=MPI.SUM)

        gait.load_scale = 2.0
        drv_double = GaitDriver(mech, gait, mech.cfg)
        drv_double.update_snapshots()
        psi_double_loc = fem.assemble_scalar(fem.form(drv_double.stimulus_expr() * mech.cfg.dx))
        psi_double = mech.comm.allreduce(psi_double_loc, op=MPI.SUM)

        ratio = psi_double / max(psi_base, 1e-300)
        expected = 2.0
        assert 0.5 * expected < ratio < 1.5 * expected, (
            "Energy scaling should be linear with load; "
            f"expected≈{expected:.2f}, ratio={ratio:.2f}"
        )



class TestGaitDriverFemur:
    def test_driver_produces_nonzero_energy(self, tmp_path, spaces, cfg, bc_mech, dummy_gait_loader):
        """Check if GaitDriver produces non-zero energy expression."""
        comm = MPI.COMM_WORLD
        
        # Setup fields
        u = fem.Function(spaces.V, name="u")
        rho = fem.Function(spaces.Q, name="rho")

        rho.x.array[:] = cfg.rho0

        neumann_bcs = [
            (dummy_gait_loader.t_hip, dummy_gait_loader.tag),
            (dummy_gait_loader.t_glmed, dummy_gait_loader.tag),
            (dummy_gait_loader.t_glmax, dummy_gait_loader.tag),
        ]

        mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
        mech.setup()

        driver = GaitDriver(mech, dummy_gait_loader, cfg)
        driver.update_snapshots()
        psi_expr = driver.stimulus_expr()

        psi_form = fem.form(psi_expr * cfg.dx)
        psi_local = fem.assemble_scalar(psi_form)
        psi_global = comm.allreduce(psi_local, op=MPI.SUM)

        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        vol_global = comm.allreduce(vol_local, op=MPI.SUM)

        psi_avg = psi_global / vol_global
        assert psi_avg > 1e-8, f"Average energy density too small: {psi_avg:.3e}"
