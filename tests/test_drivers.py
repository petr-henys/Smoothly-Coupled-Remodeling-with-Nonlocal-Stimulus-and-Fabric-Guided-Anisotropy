"""Tests for GaitDriver."""

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitDriver


@pytest.fixture
def mech_with_traction(spaces, cfg, bc_mech, dummy_load):
    """Mechanics solver on unit cube with simple traction loading."""
    u = fem.Function(spaces.V, name="u")
    rho = fem.Function(spaces.Q, name="rho")
    rho.x.array[:] = 0.6
    rho.x.scatter_forward()

    loader = dummy_load["loader"]
    load_tag = dummy_load["load_tag"]

    neumann_bcs = [(loader.hip_fun, load_tag)]
    mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
    mech.setup()
    return mech


class TestGaitDriverUnitCube:
    """Tests for GaitDriver on unit cube mesh."""

    def test_stimulus_positive(self, mech_with_traction):
        """Stimulus should be positive under load."""
        mech = mech_with_traction

        drv = GaitDriver(mech, mech.cfg)
        drv.setup()
        drv.update_snapshots()

        stats = drv.get_stimulus_stats()
        assert stats["psi_max"] > 0, f"Expected positive stimulus, got {stats['psi_max']:.3e}"

    def test_stimulus_scales_with_load(self, spaces, cfg, bc_mech):
        """Stimulus should scale with load magnitude."""
        import numpy as np
        from dolfinx import fem
        
        def run_driver(scale: float) -> float:
            u = fem.Function(spaces.V, name="u")
            rho = fem.Function(spaces.Q, name="rho")
            rho.x.array[:] = 0.6
            rho.x.scatter_forward()
            
            # Create traction with scale
            t_hip = fem.Function(spaces.V, name="t_hip")
            traction_vec = np.array([0.0, -0.1 * scale, 0.0], dtype=np.float64)
            n_dofs = t_hip.x.array.size // 3
            t_hip.x.array[:] = np.tile(traction_vec, n_dofs)
            t_hip.x.scatter_forward()
            
            neumann_bcs = [(t_hip, 2)]
            mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
            mech.setup()
            
            drv = GaitDriver(mech, cfg)
            drv.setup()
            drv.update_snapshots()
            
            stats = drv.get_stimulus_stats()
            mech.destroy()
            return stats["psi_avg"]

        psi_base = run_driver(1.0)
        psi_double = run_driver(2.0)

        # SED scales with strain^2 ~ load^2 for linear elasticity
        ratio = psi_double / max(psi_base, 1e-300)
        expected = 4.0  # load^2 scaling
        assert 0.25 * expected < ratio < 4.0 * expected, (
            f"Stimulus should scale quadratically with load; expected≈{expected:.2f}, got {ratio:.2f}"
        )

    def test_get_stimulus_stats(self, mech_with_traction):
        """Stimulus statistics should be consistent."""
        mech = mech_with_traction

        drv = GaitDriver(mech, mech.cfg)
        drv.setup()
        drv.update_snapshots()

        stats = drv.get_stimulus_stats()
        assert stats["psi_min"] <= stats["psi_avg"] <= stats["psi_max"]
