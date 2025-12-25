"""Run Anderson vs Picard acceleration comparison sweep (strict + reproducible).

This is a hardened variant of `run_anderson_sweep.py` that avoids the two most
common failure modes when you want a *clean* Anderson-vs-Picard story:

1) **Stale output contamination**: because output directories are hash-based and
   deterministic, rerunning after changing *base* (non-swept) parameters can
   silently mix old CSVs/checkpoints with new configs.
2) **Ambiguous plotting**: if the analysis script can't clearly tell whether a
   timestep converged (vs. merely hit `max_subiters`), you can end up “proving”
   the wrong thing.

This script fixes (1) by:
  - deleting the run directory before each run (rank 0 + barrier)
  - including all base parameters in the sweep hash (as single-valued custom
    parameters), so changing them creates *new* hashes automatically.

Usage:
    mpirun -n 4 python run_anderson_sweep_strict.py
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

from parametrizer import (
    ParameterSweep,
    Parametrizer,
    ParamValue,
    SimulationRunner,
)
from simulation.box_factory import BoxSolverFactory
from simulation.box_loader import BoxLoader
from simulation.box_mesh import BoxGeometry, BoxMeshBuilder
from simulation.box_scenarios import get_parabolic_pressure_case
from simulation.config import Config
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import (
    DensityParams,
    GeometryParams,
    OutputParams,
    SolverParams,
    StimulusParams,
    TimeParams,
)
from simulation.progress import SweepProgressReporter


@dataclass
class AndersonSweepConfig:
    """Non-swept parameters chosen to make Picard struggle."""

    # Box dimensions [mm]
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 20.0

    # Mesh resolution (fixed for acceleration comparison)
    N: int = 24

    # Loading - higher pressure for stronger coupling
    pressure: float = 5.0
    load_edge_factor: float = 0.0

    # Final time
    total_time: float = 100.0

    # Solver settings
    coupling_tol: float = 1e-6
    max_subiters: int = 100

    # Reaction kinetics - stronger coupling
    k_rho_form: float = 0.02
    k_rho_resorb: float = 0.02

    # Stimulus dynamics - faster response
    stimulus_tau: float = 5.0


def _clean_dir(comm: MPI.Comm, path: Path) -> None:
    """Remove `path` on rank 0 and synchronize."""
    if comm.rank == 0:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
    comm.barrier()


def create_anderson_runner(cfg: AndersonSweepConfig) -> SimulationRunner:
    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        # Extract swept knobs
        dt_days = float(param_point["dt_days"])
        accel_type = str(param_point["accel_type"])
        m = int(param_point.get("m", 5))
        beta = float(param_point.get("beta", 1.0))

        # Hard safety: never reuse an old run directory.
        _clean_dir(comm, output_path)

        # Update reporter with correct total_time
        if reporter is not None:
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=cfg.total_time)

        # Mesh
        N = int(cfg.N)
        geometry = BoxGeometry(
            Lx=cfg.Lx,
            Ly=cfg.Ly,
            Lz=cfg.Lz,
            nx=N,
            ny=N,
            nz=int(N * cfg.Lz / cfg.Lx),
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()

        # Config
        sim_cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            density=DensityParams(
                k_rho_form=float(cfg.k_rho_form),
                k_rho_resorb=float(cfg.k_rho_resorb),
            ),
            stimulus=StimulusParams(
                stimulus_tau=float(cfg.stimulus_tau),
            ),
            time=TimeParams(
                total_time=float(cfg.total_time),
                dt_initial=dt_days,
                adaptive_dt=False,
            ),
            solver=SolverParams(
                coupling_tol=float(cfg.coupling_tol),
                max_subiters=int(cfg.max_subiters),
                accel_type=accel_type,
                m=m,
                beta=beta,
                # IMPORTANT: enable safeguard for robustness on hard cases.
                # (Does nothing for accel_type="picard" because AA is not constructed.)
                safeguard=True,
            ),
            output=OutputParams(results_dir=str(output_path)),
            geometry=GeometryParams(
                fix_tag=BoxMeshBuilder.TAG_BOTTOM,
                load_tag=BoxMeshBuilder.TAG_TOP,
            ),
        )

        loader = BoxLoader(domain, facet_tags, load_tag=BoxMeshBuilder.TAG_TOP)
        loading_cases = [
            get_parabolic_pressure_case(
                pressure=float(cfg.pressure),
                gradient_axis=0,
                center_factor=2.0,
                edge_factor=float(cfg.load_edge_factor),
                box_extent=(0.0, cfg.Lx),
                name="parabolic_compression",
            )
        ]

        factory = BoxSolverFactory(sim_cfg)
        with Remodeller(sim_cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            remodeller.simulate(reporter=reporter)

    return runner


def main() -> None:
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AndersonSweepStrict")

    accel_types = ["picard", "anderson"]
    dt_values = [50, 25, 10]
    m_values = [5]
    beta_values = [1.0]

    cfg = AndersonSweepConfig()

    # Include *all* base parameters in the hash to avoid directory re-use when you tweak cfg.
    base_hash_params = {
        "base.N": [cfg.N],
        "base.pressure": [cfg.pressure],
        "base.k_rho_form": [cfg.k_rho_form],
        "base.k_rho_resorb": [cfg.k_rho_resorb],
        "base.stimulus_tau": [cfg.stimulus_tau],
        "base.coupling_tol": [cfg.coupling_tol],
        "base.max_subiters": [cfg.max_subiters],
        "base.total_time": [cfg.total_time],
    }

    sweep = ParameterSweep(
        params={
            **base_hash_params,
            "accel_type": accel_types,
            "dt_days": dt_values,
            "m": m_values,
            "beta": beta_values,
        },
        base_output_dir=Path("results/anderson_sweep_strict"),
        metadata={
            "description": "Anderson vs Picard (strict, no stale outputs)",
            "accel_types": accel_types,
            "dt_values": dt_values,
            "m_values": m_values,
            "beta_values": beta_values,
            "cfg": cfg.__dict__,
        },
        validate_config_params=False,  # we intentionally use custom keys like 'base.*'
    )

    runner = create_anderson_runner(cfg)

    if comm.rank == 0:
        logger.info("=" * 60)
        logger.info("ANDERSON VS PICARD (STRICT) SWEEP")
        logger.info("=" * 60)
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"dt values: {dt_values}")
        logger.info(f"cfg: {cfg}")
        logger.info("=" * 60)

    Parametrizer(sweep, runner, comm).run()

    if comm.rank == 0:
        logger.info("Sweep complete!")


if __name__ == "__main__":
    main()
