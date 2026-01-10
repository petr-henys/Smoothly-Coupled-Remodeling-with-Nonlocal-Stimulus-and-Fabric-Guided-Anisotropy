"""Integration tests for Remodeller: field setup, physics coupling, energy balance."""

import pytest
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.params import MaterialParams, DensityParams, SolverParams, OutputParams, FabricParams
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


def test_model_initializes_with_traction(tmp_path, unit_cube, facet_tags, dummy_load):
    """Verify Remodeller initializes fields and storage correctly."""
    comm = MPI.COMM_WORLD
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        assert "fields" in rem.storage.fields._writers
        rho = rem.state_fields["rho"]
        rho_mean = comm.allreduce(np.mean(rho.x.array), op=MPI.SUM) / comm.size
        assert abs(rho_mean - cfg.density.rho0) < 1e-10


def test_mechanics_produces_displacement_under_load(tmp_path, unit_cube, facet_tags, dummy_load):
    """Mechanics solver must produce measurable displacement from applied loads."""
    comm = MPI.COMM_WORLD
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=8, ksp_atol=1e-15),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        rem.step(1.0)

        u_fn = rem.driver.mech.u
        u_norm_sq_loc = fem.assemble_scalar(fem.form(ufl.inner(u_fn, u_fn) * rem.cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_loc, op=MPI.SUM)
        u_norm = np.sqrt(u_norm_sq)

        assert u_norm > 1e-6, f"Displacement norm too small: {u_norm:.3e}"


def test_stimulus_responds_to_strain_energy(tmp_path, unit_cube, facet_tags, dummy_load):
    """Stimulus field should develop non-zero values driven by mechanical energy."""
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=8),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        psi_max_init = MPI.COMM_WORLD.allreduce(float(rem.driver.psi.x.array.max()), op=MPI.MAX)
        assert psi_max_init < 1e-2

        rem.step(1.0)

        psi_max = MPI.COMM_WORLD.allreduce(float(rem.driver.psi.x.array.max()), op=MPI.MAX)
        assert psi_max > 1e-8, f"Stimulus max too small: {psi_max:.3e}"


def test_density_evolves_with_stimulus(tmp_path, unit_cube, facet_tags, dummy_load):
    """Density should evolve from initial rho0 in response to stimulus."""
    comm = MPI.COMM_WORLD
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=8),
        density=DensityParams(rho0=0.8),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        rho = rem.state_fields["rho"]
        rho_initial = rho.x.array.copy()

        for i in range(3):
            rem.step(5.0)

        rho_diff = np.abs(rho.x.array - rho_initial)
        max_diff = comm.allreduce(np.max(rho_diff), op=MPI.MAX)
        mean_diff = comm.allreduce(np.mean(rho_diff), op=MPI.SUM) / comm.size

        assert max_diff > 1e-6, f"Density max change too small: {max_diff:.3e}"
        assert mean_diff > 1e-8, f"Density mean change too small: {mean_diff:.3e}"

        rho_min = comm.allreduce(np.min(rho.x.array), op=MPI.MIN)
        rho_max = comm.allreduce(np.max(rho.x.array), op=MPI.MAX)

        assert rho_min > 0.0
        assert rho_max < 10.0


def test_model_single_step_records_metrics(tmp_path, unit_cube, facet_tags, dummy_load):
    """Single timestep execution with subiteration metrics collection."""
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=6),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        rem.step(1.0)
        metrics = rem.fixedsolver.subiter_metrics
        assert metrics, "No subiteration metrics recorded"


def test_model_convergence_stability(tmp_path, unit_cube, facet_tags, dummy_load):
    """Verify fixed-point iteration converges and produces consistent results."""
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=12, coupling_tol=1e-6, ksp_atol=1e-15),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        rem.step(1.0)

        metrics = rem.fixedsolver.subiter_metrics
        assert len(metrics) > 0

        last_metric = metrics[-1]
        if "proj_res" in last_metric:
            # Check that final residual is less than max_subiters reached (convergence progressing)
            assert last_metric["proj_res"] < 1.0, f"Residual too large: {last_metric['proj_res']}"


def test_model_two_steps_energy_stability(tmp_path, unit_cube, facet_tags, dummy_load):
    """Energy should remain stable across consecutive timesteps."""
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=6, ksp_atol=1e-15),
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        rem.step(1.0)
        psi1 = MPI.COMM_WORLD.allreduce(float(rem.driver.psi.x.array.mean()), op=MPI.SUM) / MPI.COMM_WORLD.size

        rem.step(1.0)
        psi2 = MPI.COMM_WORLD.allreduce(float(rem.driver.psi.x.array.mean()), op=MPI.SUM) / MPI.COMM_WORLD.size

        rel_diff = abs(psi2 - psi1) / max(abs(psi1), 1e-12)
        assert rel_diff < 0.5


def test_fabric_evolves_with_load(tmp_path, unit_cube, facet_tags, dummy_load):
    """Fabric tensor should evolve from isotropic state under anisotropic load."""
    comm = MPI.COMM_WORLD
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        output=OutputParams(results_dir=str(tmp_path)),
        solver=SolverParams(max_subiters=8),
        fabric=FabricParams(fabric_tau=5.0), # Fast evolution
    )

    with Remodeller(cfg, loader=dummy_load["loader"]) as rem:
        L = rem.state_fields["L"]
        
        # Initial state is isotropic (L=0)
        L_norm_init = comm.allreduce(np.linalg.norm(L.x.array), op=MPI.SUM)
        assert L_norm_init < 1e-10
        
        # Step forward
        rem.step(5.0)
        
        # Should develop anisotropy (L != 0)
        L_norm_final = comm.allreduce(np.linalg.norm(L.x.array), op=MPI.SUM)
        assert L_norm_final > 1e-6, f"Fabric did not evolve: {L_norm_final:.3e}"
