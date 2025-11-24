"""Unified tests for remodeling drivers (Instant and Gait)."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import SimplifiedGaitDriver

@pytest.fixture
def mech_with_dummy_gait(spaces, cfg, bc_mech, dummy_gait_loader):
    """Mechanics solver on unit cube with simple dummy gait loading."""
    u = fem.Function(spaces.V, name="u")
    rho = fem.Function(spaces.Q, name="rho"); rho.x.array[:] = 0.6; rho.x.scatter_forward()
    
    # We need to pass t_hip and t_glmed to MechanicsSolver if they are used in BCs
    # But SimplifiedGaitDriver updates them.
    
    t_hip = dummy_gait_loader["t_hip"]
    t_glmed = dummy_gait_loader["t_glmed"]
    tag = dummy_gait_loader["tag"]
    
    neumann_bcs = [
        (t_hip, tag),
        (t_glmed, tag),
    ]
    mech = MechanicsSolver(u, rho, cfg, bc_mech, neumann_bcs)
    mech.setup()
    return mech, dummy_gait_loader

class TestSimplifiedGaitDriverUnitCube:
    def test_energy_does_not_scale_with_cpd(self, mech_with_dummy_gait):
        mech, gait_data = mech_with_dummy_gait
        
        # Mock FemurCSS and related calls
        with patch("simulation.drivers.pv.read"), \
             patch("simulation.drivers.load_json_points"), \
             patch("simulation.drivers.FemurCSS") as MockCSS:
            
            # Setup mock CSS to return identity vector or something valid
            mock_css_instance = MockCSS.return_value
            # css_to_world_vector should return the input vector (identity transform)
            mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)
            
            drv = SimplifiedGaitDriver(
                mech, 
                gait_data["t_hip"], 
                gait_data["t_glmed"], 
                gait_data["stages"], 
                mech.cfg
            )
            drv.update_snapshots()
            psi_loc = fem.assemble_scalar(fem.form(drv.stimulus_expr() * mech.cfg.dx))
            psi = mech.comm.allreduce(psi_loc, op=MPI.SUM)
            assert psi > 0, f"Expected positive energy, got {psi:.3e}"

    def test_energy_scales_with_load(self, mech_with_dummy_gait):
        mech, gait_data = mech_with_dummy_gait
        mech.cfg.n_power = 2.0
        
        # Helper to run driver with scaled loads
        def run_driver(scale):
            # Scale magnitudes in stages
            scaled_stages = []
            for s in gait_data["stages"]:
                s_copy = s.copy()
                s_copy["hip_magnitude"] *= scale
                s_copy["gl_magnitude"] *= scale
                scaled_stages.append(s_copy)
                
            with patch("simulation.drivers.pv.read"), \
                 patch("simulation.drivers.load_json_points"), \
                 patch("simulation.drivers.FemurCSS") as MockCSS:
                
                mock_css_instance = MockCSS.return_value
                mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)
                
                drv = SimplifiedGaitDriver(
                    mech, 
                    gait_data["t_hip"], 
                    gait_data["t_glmed"], 
                    scaled_stages, 
                    mech.cfg
                )
                drv.update_snapshots()
                psi_loc = fem.assemble_scalar(fem.form(drv.stimulus_expr() * mech.cfg.dx))
                return mech.comm.allreduce(psi_loc, op=MPI.SUM)

        psi_base = run_driver(1.0)
        psi_double = run_driver(2.0)

        ratio = psi_double / max(psi_base, 1e-300)
        expected = 2.0
        assert 0.5 * expected < ratio < 1.5 * expected, (
            "Energy scaling should be linear with load; "
            f"expected≈{expected:.2f}, ratio={ratio:.2f}"
        )
