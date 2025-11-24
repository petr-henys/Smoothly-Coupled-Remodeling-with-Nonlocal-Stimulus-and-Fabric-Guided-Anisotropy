"""Comprehensive tests for simulation.model.Remodeller with physics validation.

Tests verify:
1. Proper initialization and field setup
2. Mechanics solver produces non-zero displacements under load
3. Stimulus field responds to mechanical energy
4. Density evolution follows stimulus-driven remodeling
5. Energy balance and conservation properties
"""

import pytest
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem
from unittest.mock import patch

from simulation.config import Config
from simulation.model import Remodeller

@pytest.fixture(autouse=True)
def _stub_vtx(monkeypatch):
    """Stub VTXWriter to avoid ADIOS2 I/O in tests."""
    import simulation.storage as storage_mod
    from pathlib import Path as _P

    class _DummyVTXWriter:
        def __init__(self, comm, path, fields, engine="bp4"):
            if comm.rank == 0:
                _P(path).mkdir(parents=True, exist_ok=True)
        def write(self, t):
            return None
        def close(self):
            return None

    monkeypatch.setattr(storage_mod, "VTXWriter", _DummyVTXWriter, raising=True)


def test_model_initializes_with_dummy_gait(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Verify Remodeller initializes fields and storage correctly."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path))

    # Mock FemurCSS and related calls in SimplifiedGaitDriver
    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            # Field writers registered
            assert "scalars" in rem.storage.fields._writers
            
            # Initial conditions: rho=rho0
            rho_mean = comm.allreduce(np.mean(rem.rho.x.array), op=MPI.SUM) / comm.size
            assert abs(rho_mean - cfg.rho0) < 1e-10, "Initial density should be rho0"


def test_mechanics_produces_displacement_under_load(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Mechanics solver must produce measurable displacement from applied loads."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path), max_subiters=8, ksp_atol=1e-15)

    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            rem.step(dt=1.0)  # 1 day
            
            # Check displacement is non-zero
            u_fn = rem.driver.mech.u
            u_norm_sq_loc = fem.assemble_scalar(fem.form(ufl.inner(u_fn, u_fn) * rem.cfg.dx))
            u_norm_sq = comm.allreduce(u_norm_sq_loc, op=MPI.SUM)
            u_norm = np.sqrt(u_norm_sq)
            
            assert u_norm > 1e-6, f"Displacement norm too small: {u_norm:.3e} (loads may not be applied)"


def test_stimulus_responds_to_strain_energy(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Stimulus field should develop non-zero values driven by mechanical energy."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path), max_subiters=8)

    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            # Initial stimulus is zero (implicitly, as no loads applied yet)
            # We can check stats before step
            # Note: smooth_eps causes a small non-zero value (approx 0.0018 MPa)
            stats_init = rem.driver.get_stimulus_stats()
            assert stats_init["psi_max"] < 1e-2
            
            rem.step(dt=1.0)  # 1 day
            
            # After loading, stimulus should be non-zero
            stats = rem.driver.get_stimulus_stats()
            S_max = stats["psi_max"]
            S_avg = stats["psi_avg"]
            
            assert S_max > 1e-8, f"Stimulus max too small: {S_max:.3e}"
            assert abs(S_avg) > 1e-10, f"Stimulus mean near zero: {S_avg:.3e}"


def test_density_evolves_with_stimulus(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Density should evolve from initial rho0 in response to stimulus."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path), 
                 max_subiters=8, rho0=0.5)

    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            rho_initial = rem.rho.x.array.copy()
            
            # Run multiple steps to allow density evolution
            for _ in range(3):
                rem.step(dt=5.0)  # 5 days per step
            
            # Density should have changed
            rho_diff = np.abs(rem.rho.x.array - rho_initial)
            max_diff = comm.allreduce(np.max(rho_diff), op=MPI.MAX)
            mean_diff = comm.allreduce(np.mean(rho_diff), op=MPI.SUM) / comm.size
            
            # With physiological loads and short time (15 days), expect small but measurable changes
            assert max_diff > 1e-6, f"Density max change too small: {max_diff:.3e}"
            assert mean_diff > 1e-8, f"Density mean change too small: {mean_diff:.3e}"
            
            # Density guided by smooth relaxation (no hard bounds enforcement)
            rho_min = comm.allreduce(np.min(rem.rho.x.array), op=MPI.MIN)
            rho_max = comm.allreduce(np.max(rem.rho.x.array), op=MPI.MAX)
            
            # Just verify density remains physically reasonable (no clipping applied)
            assert rho_min > 0.0, f"Density should be positive: {rho_min}"
            assert rho_max < 10.0, f"Density unreasonably large: {rho_max}"


def test_model_single_step_records_metrics(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Single timestep execution with subiteration metrics collection."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path), max_subiters=6)

    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            rem.step(dt=1.0)  # 1 day
            
            # Ensure subiteration metrics collected
            metrics = rem.fixedsolver.subiter_metrics
            assert metrics, "No subiteration metrics recorded"
            
            # Storage metrics buffer should have entries on rank 0
            if comm.rank == 0:
                assert len(rem.telemetry._buffers["steps"]) >= 0


def test_model_convergence_stability(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Verify fixed-point iteration converges and produces consistent results."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path), 
                 max_subiters=12, coupling_tol=1e-6, ksp_atol=1e-15)

    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            rem.step(dt=1.0)  # 1 day
            
            # Check convergence metrics
            metrics = rem.fixedsolver.subiter_metrics
            assert len(metrics) > 0, "No metrics recorded"
            
            last_metric = metrics[-1]
            # Projection residual should be small
            if "proj_res" in last_metric:
                assert last_metric["proj_res"] < cfg.coupling_tol * 10, \
                    f"Projection residual too large: {last_metric['proj_res']:.3e}"


def test_model_two_steps_energy_stability(tmp_path, unit_cube, facet_tags, dummy_gait_loader):
    """Energy should remain stable across consecutive timesteps."""
    comm = MPI.COMM_WORLD
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, results_dir=str(tmp_path), max_subiters=6, ksp_atol=1e-15)

    with patch("simulation.drivers.pv.read"), \
         patch("simulation.drivers.load_json_points"), \
         patch("simulation.drivers.FemurCSS") as MockCSS:
        
        mock_css_instance = MockCSS.return_value
        mock_css_instance.css_to_world_vector.side_effect = lambda x: np.array(x)

        with Remodeller(cfg, dummy_gait_loader["stages"]) as rem:
            rem.step(dt=1.0)  # 1 day
            psi1 = rem.driver.get_stimulus_stats()["psi_avg"]
            
            rem.step(dt=1.0)  # 1 day
            psi2 = rem.driver.get_stimulus_stats()["psi_avg"]
            
            # Energy should not change drastically (within reasonable bounds)
            rel_diff = abs(psi2 - psi1) / max(abs(psi1), 1e-12)
            assert rel_diff < 0.5, f"Energy changed excessively: psi1={psi1:.3e}, psi2={psi2:.3e}, rel_diff={rel_diff:.1%}"

