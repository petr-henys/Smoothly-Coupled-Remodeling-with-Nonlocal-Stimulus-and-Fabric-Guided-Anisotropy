"""Tests for GaitDriver."""

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitDriver
from simulation.loader import LoadingCase


class MockLoader:
    """Mock Loader for testing GaitDriver."""
    
    def __init__(self, V, traction_value=0.1):
        self.V = V
        self.load_tag = 2
        self.traction = fem.Function(V, name="Traction")
        self._traction_value = traction_value
        self._cache = {}
    
    def precompute_loading_cases(self, cases):
        """Precompute and cache traction arrays for all loading cases."""
        for case in cases:
            traction_vec = np.array([0.0, -self._traction_value, 0.0], dtype=np.float64)
            n_dofs = self.traction.x.array.size // 3
            traction_array = np.tile(traction_vec, n_dofs)
            self._cache[case.name] = {"traction": traction_array.copy()}
    
    def set_loading_case(self, case_name: str) -> None:
        """Apply cached traction for named case."""
        cached = self._cache[case_name]
        self.traction.x.array[:] = cached["traction"]
        self.traction.x.scatter_forward()


@pytest.fixture
def driver_setup(spaces, cfg, bc_mech):
    """Setup mechanics solver and loader for GaitDriver tests."""
    u = fem.Function(spaces.V, name="u")
    rho = fem.Function(spaces.Q, name="rho")
    rho.x.array[:] = 0.6
    rho.x.scatter_forward()

    loader = MockLoader(spaces.V, traction_value=0.1)
    loading_cases = [LoadingCase(name="test", day_cycles=1.0, hip=None, muscles=[])]
    
    neumann_bcs = [(loader.traction, 2)]
    mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
    mech.setup()
    
    return {
        "mech": mech,
        "loader": loader,
        "loading_cases": loading_cases,
        "cfg": cfg,
    }


class TestGaitDriverUnitCube:
    """Tests for GaitDriver on unit cube mesh."""

    def test_stimulus_positive(self, driver_setup):
        """Stimulus should be positive under load."""
        from simulation.stats import SweepStats
        drv = GaitDriver(
            driver_setup["mech"],
            driver_setup["cfg"],
            driver_setup["loader"],
            driver_setup["loading_cases"],
        )
        drv.setup()
        stats = drv.sweep()

        assert isinstance(stats, SweepStats), "sweep() should return SweepStats"
        assert stats.label == "mech"
        assert stats.ksp_iters > 0

        psi_max = MPI.COMM_WORLD.allreduce(float(drv.psi.x.array.max()), op=MPI.MAX)
        assert psi_max > 0, f"Expected positive stimulus, got {psi_max:.3e}"

    def test_stimulus_scales_with_load(self, spaces, cfg, bc_mech):
        """Stimulus should scale with load magnitude."""
        comm = MPI.COMM_WORLD
        
        def run_driver(scale: float) -> float:
            u = fem.Function(spaces.V, name="u")
            rho = fem.Function(spaces.Q, name="rho")
            rho.x.array[:] = 0.6
            rho.x.scatter_forward()
            
            loader = MockLoader(spaces.V, traction_value=0.1 * scale)
            loading_cases = [LoadingCase(name="test", day_cycles=1.0, hip=None, muscles=[])]
            
            neumann_bcs = [(loader.traction, 2)]
            mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
            mech.setup()
            
            drv = GaitDriver(mech, cfg, loader, loading_cases)
            drv.setup()
            drv.sweep()

            psi_avg = comm.allreduce(float(drv.psi.x.array.mean()), op=MPI.SUM) / comm.size
            mech.destroy()
            return psi_avg

        psi_base = run_driver(1.0)
        psi_double = run_driver(2.0)

        # SED scales with strain^2 ~ load^2 for linear elasticity
        ratio = psi_double / max(psi_base, 1e-300)
        expected = 4.0  # load^2 scaling
        assert 0.25 * expected < ratio < 4.0 * expected, (
            f"Stimulus should scale quadratically with load; expected≈{expected:.2f}, got {ratio:.2f}"
        )

    def test_psi_field_valid(self, driver_setup):
        """Stimulus field should have valid values after update."""
        drv = GaitDriver(
            driver_setup["mech"],
            driver_setup["cfg"],
            driver_setup["loader"],
            driver_setup["loading_cases"],
        )
        drv.setup()
        drv.sweep()

        comm = MPI.COMM_WORLD
        psi_min = comm.allreduce(float(drv.psi.x.array.min()), op=MPI.MIN)
        psi_max = comm.allreduce(float(drv.psi.x.array.max()), op=MPI.MAX)
        psi_avg = comm.allreduce(float(drv.psi.x.array.mean()), op=MPI.SUM) / comm.size
        assert psi_min <= psi_avg <= psi_max

    def test_multiple_loading_cases_averaged(self, spaces, cfg, bc_mech):
        """Multiple loading cases should produce averaged stimulus."""
        u = fem.Function(spaces.V, name="u")
        rho = fem.Function(spaces.Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()
        
        loader = MockLoader(spaces.V, traction_value=0.1)
        
        # Two identical loading cases with equal day_cycles
        loading_cases = [
            LoadingCase(name="case1", day_cycles=1.0, hip=None, muscles=[]),
            LoadingCase(name="case2", day_cycles=1.0, hip=None, muscles=[]),
        ]
        
        neumann_bcs = [(loader.traction, 2)]
        mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
        mech.setup()
        
        drv = GaitDriver(mech, cfg, loader, loading_cases)
        drv.setup()
        drv.sweep()

        # Should have computed for both cases - verify via psi field
        psi_max = MPI.COMM_WORLD.allreduce(float(drv.psi.x.array.max()), op=MPI.MAX)
        assert psi_max > 0

        mech.destroy()
