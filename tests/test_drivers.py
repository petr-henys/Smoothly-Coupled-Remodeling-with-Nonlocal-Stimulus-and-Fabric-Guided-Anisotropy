"""Tests for GaitDriver."""

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitDriver


@pytest.fixture
def mech_with_dummy_gait(spaces, cfg, bc_mech, dummy_gait_loader):
    """Mechanics solver on unit cube with simple dummy gait loading."""
    u = fem.Function(spaces.V, name="u")
    rho = fem.Function(spaces.Q, name="rho")
    rho.x.array[:] = 0.6
    rho.x.scatter_forward()

    t_hip = dummy_gait_loader["t_hip"]
    t_glmed = dummy_gait_loader["t_glmed"]
    tag = dummy_gait_loader["tag"]

    neumann_bcs = [(t_hip, tag), (t_glmed, tag)]
    mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
    mech.setup()
    return mech, dummy_gait_loader


class TestGaitDriverUnitCube:
    """Tests for GaitDriver on unit cube mesh."""

    def test_stimulus_positive(self, mech_with_dummy_gait):
        """Stimulus should be positive under load."""
        mech, gait_data = mech_with_dummy_gait

        drv = GaitDriver(
            mech,
            gait_data["t_hip"],
            gait_data["t_glmed"],
            gait_data["stages"],
            mech.cfg,
            css_transformer=None,  # No coordinate transform needed
        )
        drv.update_snapshots()

        psi_loc = fem.assemble_scalar(fem.form(drv.stimulus_expr() * mech.cfg.dx))
        psi = mech.comm.allreduce(psi_loc, op=MPI.SUM)
        assert psi > 0, f"Expected positive stimulus, got {psi:.3e}"

    def test_stimulus_scales_with_load(self, mech_with_dummy_gait):
        """Stimulus should scale with load magnitude."""
        mech, gait_data = mech_with_dummy_gait
        mech.cfg.n_power = 2.0

        def run_driver(scale: float) -> float:
            scaled_stages = []
            for s in gait_data["stages"]:
                s_copy = s.copy()
                s_copy["hip_magnitude"] *= scale
                s_copy["gl_magnitude"] *= scale
                scaled_stages.append(s_copy)

            drv = GaitDriver(
                mech,
                gait_data["t_hip"],
                gait_data["t_glmed"],
                scaled_stages,
                mech.cfg,
                css_transformer=None,
            )
            drv.update_snapshots()
            psi_loc = fem.assemble_scalar(fem.form(drv.stimulus_expr() * mech.cfg.dx))
            return mech.comm.allreduce(psi_loc, op=MPI.SUM)

        psi_base = run_driver(1.0)
        psi_double = run_driver(2.0)

        ratio = psi_double / max(psi_base, 1e-300)
        expected = 2.0
        assert 0.5 * expected < ratio < 1.5 * expected, (
            f"Stimulus should scale linearly with load; expected≈{expected:.2f}, got {ratio:.2f}"
        )

    def test_get_stimulus_stats(self, mech_with_dummy_gait):
        """Stimulus statistics should be consistent."""
        mech, gait_data = mech_with_dummy_gait

        drv = GaitDriver(
            mech,
            gait_data["t_hip"],
            gait_data["t_glmed"],
            gait_data["stages"],
            mech.cfg,
            css_transformer=None,
        )
        drv.update_snapshots()

        stats = drv.get_stimulus_stats()
        assert stats["psi_min"] <= stats["psi_avg"] <= stats["psi_max"]
