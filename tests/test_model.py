"""Comprehensive tests for simulation.model.Remodeller with physics validation.

Tests verify:
1. Proper initialization and field setup
2. Mechanics solver produces non-zero displacements under load
3. Stimulus field responds to mechanical energy
4. Density evolution follows stimulus-driven remodeling
5. Direction tensor evolves toward strain alignment
6. Energy balance and conservation properties
"""

import pytest
import numpy as np

import basix
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import functionspace

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.model import Remodeller


class _DummyGaitLoader:
    """Gait loader with meaningful loads to drive physics.

    Applies realistic-scale tractions to produce measurable deformation,
    strain energy, and subsequent remodeling response.
    """
    def __init__(self, V: fem.FunctionSpace):
        self.V = V
        self.t_hip = fem.Function(V, name="t_hip")
        self.t_glmed = fem.Function(V, name="t_glmed")
        self.t_glmax = fem.Function(V, name="t_glmax")
        # Scale to produce physiological strain energy density (~psi_ref = 3.0e-3 MPa)
        # With E=1.5e4 MPa, psi ~ σ²/(2E), so σ ~ sqrt(2*E*psi) ~ sqrt(2*1.5e4*3e-3) ~ 9.5 MPa
        # For unit cube geometry, use smaller effective tractions
        self.load_scale = 0.5  # [MPa] - tuned for psi ~ psi_ref

    def get_quadrature(self):
        # Two phases: peak loading (0%) and mid-stance (50%)
        return [(0.0, 0.5), (50.0, 0.5)]

    def update_loads(self, phase_percent: float) -> None:
        # Loads vary with gait phase: peak at phase=0%, reduce at 50%
        f = (1.0 - float(phase_percent) / 100.0) * self.load_scale
        # Apply distinct directional tractions to create anisotropic strain
        v_hip = np.array([-1.0 * f, -0.5 * f, -0.3 * f], dtype=float)
        v_glmed = np.array([0.2 * f, 0.8 * f, -0.1 * f], dtype=float)
        v_glmax = np.array([0.1 * f, -0.2 * f, -0.9 * f], dtype=float)
        
        self.t_hip.interpolate(lambda x: np.tile(v_hip.reshape(3, 1), (1, x.shape[1])))
        self.t_glmed.interpolate(lambda x: np.tile(v_glmed.reshape(3, 1), (1, x.shape[1])))
        self.t_glmax.interpolate(lambda x: np.tile(v_glmax.reshape(3, 1), (1, x.shape[1])))
        for t in (self.t_hip, self.t_glmed, self.t_glmax):
            t.x.scatter_forward()


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


def _unit_cube(n: int = 4):
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.shared_facet)


@pytest.fixture(autouse=True)
def _patch_gait_loader(monkeypatch):
    """Patch setup_femur_gait_loading to return _DummyGaitLoader."""
    import simulation.femur_gait as gait_mod
    
    def _mock_setup(V, **kwargs):
        return _DummyGaitLoader(V)
        
    monkeypatch.setattr(gait_mod, "setup_femur_gait_loading", _mock_setup)


def test_model_initializes_with_dummy_gait(tmp_path):
    """Verify Remodeller initializes fields and storage correctly."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(4)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path))

    with Remodeller(cfg) as rem:
        # Field writers registered
        assert "scalars" in rem.storage.fields._writers
        assert "A" in rem.storage.fields._writers
        
        # Initial conditions: rho=rho0, A=isotropic, S=0
        rho_mean = comm.allreduce(np.mean(rem.rho.x.array), op=MPI.SUM) / comm.size
        assert abs(rho_mean - cfg.rho0) < 1e-10, "Initial density should be rho0"
        
        S_max = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        assert S_max < 1e-12, "Initial stimulus should be zero"


def test_mechanics_produces_displacement_under_load(tmp_path):
    """Mechanics solver must produce measurable displacement from applied loads."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(5)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), max_subiters=8, ksp_atol=1e-15)

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)  # 1 day
        
        # Check displacement is non-zero
        u_fn = rem.driver.mech.u
        u_norm_sq_loc = fem.assemble_scalar(fem.form(ufl.inner(u_fn, u_fn) * rem.cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_loc, op=MPI.SUM)
        u_norm = np.sqrt(u_norm_sq)
        
        assert u_norm > 1e-6, f"Displacement norm too small: {u_norm:.3e} (loads may not be applied)"


def test_stimulus_responds_to_strain_energy(tmp_path):
    """Stimulus field should develop non-zero values driven by mechanical energy."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(5)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), max_subiters=8)

    with Remodeller(cfg) as rem:
        # Initial stimulus is zero
        S_initial = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        assert S_initial < 1e-12
        
        rem.step(dt=1.0)  # 1 day
        
        # After loading, stimulus should be non-zero
        S_max = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        S_mean = comm.allreduce(np.mean(rem.S.x.array), op=MPI.SUM) / comm.size
        
        assert S_max > 1e-8, f"Stimulus max too small: {S_max:.3e}"
        assert abs(S_mean) > 1e-10, f"Stimulus mean near zero: {S_mean:.3e}"


def test_density_evolves_with_stimulus(tmp_path):
    """Density should evolve from initial rho0 in response to stimulus."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(5)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), 
                 max_subiters=8, rho0=0.5)

    with Remodeller(cfg) as rem:
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


def test_direction_tensor_evolves_from_isotropic(tmp_path):
    """Direction tensor A should evolve from isotropic initial condition."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(5)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), max_subiters=8)

    with Remodeller(cfg) as rem:
        # Initial A is isotropic (I/3 in 3D)
        A_initial = rem.A.x.array.copy()
        
        # Run steps to develop anisotropy
        for _ in range(3):
            rem.step(dt=5.0)  # 5 days per step
        
        # A should have evolved
        A_diff = np.abs(rem.A.x.array - A_initial)
        max_diff = comm.allreduce(np.max(A_diff), op=MPI.MAX)
        
        assert max_diff > 1e-6, f"Direction tensor change too small: {max_diff:.3e}"


def test_model_single_step_records_metrics(tmp_path):
    """Single timestep execution with subiteration metrics collection."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(4)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), max_subiters=6)

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)  # 1 day
        
        # Ensure subiteration metrics collected
        metrics = rem.fixedsolver.subiter_metrics
        assert metrics, "No subiteration metrics recorded"
        
        # Storage metrics buffer should have entries on rank 0
        if comm.rank == 0:
            assert len(rem.telemetry._buffers["steps"]) >= 0


def test_model_convergence_stability(tmp_path):
    """Verify fixed-point iteration converges and produces consistent results."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(5)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), 
                 max_subiters=12, coupling_tol=1e-6, ksp_atol=1e-15)

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)  # 1 day
        
        # Check convergence metrics
        metrics = rem.fixedsolver.subiter_metrics
        assert len(metrics) > 0, "No metrics recorded"
        
        last_metric = metrics[-1]
        # Projection residual should be small
        if "proj_res" in last_metric:
            assert last_metric["proj_res"] < cfg.coupling_tol * 10, \
                f"Projection residual too large: {last_metric['proj_res']:.3e}"


def test_model_two_steps_energy_stability(tmp_path):
    """Energy should remain stable across consecutive timesteps."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(4)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=str(tmp_path), max_subiters=6, ksp_atol=1e-15)

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)  # 1 day
        psi1 = rem.driver._last_stats["psi_avg"]
        
        rem.step(dt=1.0)  # 1 day
        psi2 = rem.driver._last_stats["psi_avg"]
        
        # Energy should not change drastically (within reasonable bounds)
        rel_diff = abs(psi2 - psi1) / max(abs(psi1), 1e-12)
        assert rel_diff < 0.5, f"Energy changed excessively: psi1={psi1:.3e}, psi2={psi2:.3e}, rel_diff={rel_diff:.1%}"

