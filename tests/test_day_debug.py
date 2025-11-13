"""Debug test for day-based time unit stimulus solver."""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.model import Remodeller


def _unit_cube(n: int = 4):
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.shared_facet)


@pytest.fixture(autouse=True)
def _stub_vtx(monkeypatch):
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


@pytest.fixture(autouse=True)
def _shim_gait_loader(monkeypatch):
    import simulation.model as model_mod
    import ufl

    class _DummyGaitLoader:
        def __init__(self, V: fem.FunctionSpace):
            self.V = V
            self.t_hip = fem.Function(V, name="t_hip")
            self.t_glmed = fem.Function(V, name="t_glmed")
            self.t_glmax = fem.Function(V, name="t_glmax")
            self.load_scale = 0.5  # [MPa] - tuned for psi ~ psi_ref

        def get_quadrature(self):
            return [(0.0, 0.5), (50.0, 0.5)]

        def update_loads(self, phase_percent: float) -> None:
            f = (1.0 - float(phase_percent) / 100.0) * self.load_scale
            v_hip = np.array([-1.0 * f, -0.5 * f, -0.3 * f], dtype=float)
            v_glmed = np.array([0.2 * f, 0.8 * f, -0.1 * f], dtype=float)
            v_glmax = np.array([0.1 * f, -0.2 * f, -0.9 * f], dtype=float)
            
            self.t_hip.interpolate(lambda x: np.tile(v_hip.reshape(3, 1), (1, x.shape[1])))
            self.t_glmed.interpolate(lambda x: np.tile(v_glmed.reshape(3, 1), (1, x.shape[1])))
            self.t_glmax.interpolate(lambda x: np.tile(v_glmax.reshape(3, 1), (1, x.shape[1])))
            for t in (self.t_hip, self.t_glmed, self.t_glmax):
                t.x.scatter_forward()

    def _factory(V, *args, **kwargs):
        return _DummyGaitLoader(V)

    monkeypatch.setattr(model_mod, "setup_femur_gait_loading", _factory, raising=True)


def test_debug_stimulus_with_days(tmp_path):
    """Debug: Check stimulus RHS assembly with day-based time."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(5)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True, results_dir=str(tmp_path), max_subiters=8)

    with Remodeller(cfg) as rem:
        # Check config parameters
        if comm.rank == 0:
            print(f"\n=== Config (day-based) ===")
            print(f"dt: {rem.cfg.dt} [days]")
            print(f"cS: {rem.cfg.cS:.3e} [MPa·day]")
            print(f"tauS: {rem.cfg.tauS:.3e} [1/day]")
            print(f"kappaS: {rem.cfg.kappaS:.3e} [mm²/day]")
            print(f"rS_gain: {rem.cfg.rS_gain:.3e} [1/(MPa·day)]")
            print(f"psi_ref: {rem.cfg.psi_ref:.3e} [MPa]")
        
        # Take one step
        rem.step(dt=1.0)
        
        # Check psi from driver
        psi_expr = rem.driver.energy_expr()
        psi_local = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        psi_integrated = comm.allreduce(psi_local, op=MPI.SUM)
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        vol = comm.allreduce(vol_local, op=MPI.SUM)
        psi_avg_driver = psi_integrated / vol
        
        # Check psi from mechanics solver
        psi_avg_mech = rem.mechsolver.average_strain_energy()
        
        if comm.rank == 0:
            print(f"\n=== After 1 step (dt=1 day) ===")
            print(f"psi_avg (from driver): {psi_avg_driver:.3e} [MPa]")
            print(f"psi_avg (from mechsolver): {psi_avg_mech:.3e} [MPa]")
            print(f"psi_driver/psi_ref: {psi_avg_driver/cfg.psi_ref:.3f}")
            print(f"psi_mech/psi_ref: {psi_avg_mech/cfg.psi_ref:.3f}")
        
        # Check stimulus
        S_max = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        S_mean = comm.allreduce(np.mean(rem.S.x.array), op=MPI.SUM) / comm.size
        
        if comm.rank == 0:
            print(f"S_max: {S_max:.3e}")
            print(f"S_mean: {S_mean:.3e}")
        
        # Explicit RHS check
        dt = 1.0
        source_expected = cfg.rS_gain * (psi_avg_driver - cfg.psi_ref) * dt
        if comm.rank == 0:
            print(f"\nExpected source term: {source_expected:.3e}")
            print(f"rS_gain * (psi_driver - psi_ref) * dt = {cfg.rS_gain:.3e} * ({psi_avg_driver:.3e} - {cfg.psi_ref:.3e}) * {dt}")
        
        assert psi_avg_driver > 1e-8, f"Driver strain energy near zero: {psi_avg_driver:.3e}"
