"""Run compact femur mechanostat sweep over lazy-zone and saturation parameters.

This script varies the two reviewer-requested stimulus-law parameters:

    - stimulus_delta0  (lazy-zone threshold scale)
    - stimulus_kappa   (tanh saturation width)

on the accelerated femur benchmark and stores each run in a hash-based output
directory for subsequent post-processing.

Usage:
    mpirun -n 4 python run_mechanostat_sweep.py

Outputs:
    results/mechanostat_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/
    │   ├── fields.bp/
    │   └── steps.csv
    └── ...
"""

from __future__ import annotations

import copy
from pathlib import Path

from mpi4py import MPI

from femur import FEBio2Dolfinx, FemurPaths, Loader, get_standard_gait_cases
from parametrizer import ParameterSweep, Parametrizer, ParamValue
from simulation.factory import DefaultSolverFactory
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import SweepProgressReporter
from sweep_utils import clean_output_dir, reset_reporter, write_standard_checkpoint


PARAMS_FILE = "stiff_params_femur.json"
OUTPUT_DIR = Path("results/mechanostat_sweep")

# Compact reviewer-facing sweep around the accelerated femur default.
DELTA0_VALUES = [0.05, 0.10, 0.20]
KAPPA_VALUES = [0.20, 0.40, 0.80]
SWEEP_TOTAL_TIME_DAYS = 150.0


def create_mechanostat_runner(
    base_params: dict,
    mdl: FEBio2Dolfinx,
):
    """Create runner for compact mechanostat parameter sweep."""

    base_cases = get_standard_gait_cases()

    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        params = copy.deepcopy(base_params)

        stimulus_delta0 = float(param_point["stimulus.stimulus_delta0"])
        stimulus_kappa = float(param_point["stimulus.stimulus_kappa"])

        params["stimulus"].stimulus_delta0 = stimulus_delta0
        params["stimulus"].stimulus_kappa = stimulus_kappa
        params["output"].results_dir = str(output_path)

        reset_reporter(reporter, params["time"].total_time)

        mesh = mdl.mesh_dolfinx
        sim_cfg = create_config(mesh, mdl.meshtags, params)

        loader = Loader(
            mesh,
            facet_tags=mdl.meshtags,
            load_tag=sim_cfg.geometry.load_tag,
            loading_cases=base_cases,
        )

        factory = DefaultSolverFactory(sim_cfg)

        with Remodeller(sim_cfg, loader=loader, factory=factory) as remodeller:
            remodeller.simulate(reporter=reporter)
            write_standard_checkpoint(sim_cfg, remodeller)

    return runner


def main() -> None:
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="MechanostatSweep")

    params = load_default_params(PARAMS_FILE)
    params["time"].total_time = SWEEP_TOTAL_TIME_DAYS
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    base_cases = get_standard_gait_cases()

    baseline_delta0 = float(params["stimulus"].stimulus_delta0)
    baseline_kappa = float(params["stimulus"].stimulus_kappa)

    sweep = ParameterSweep(
        params={
            "stimulus.stimulus_delta0": [float(v) for v in DELTA0_VALUES],
            "stimulus.stimulus_kappa": [float(v) for v in KAPPA_VALUES],
        },
        base_output_dir=OUTPUT_DIR,
        metadata={
            "description": "Compact femur mechanostat sweep: stimulus_delta0 × stimulus_kappa",
            "objective": "Quantify sensitivity of femur density predictions to lazy-zone and saturation parameters",
            "params_file": PARAMS_FILE,
            "delta0_values": [float(v) for v in DELTA0_VALUES],
            "kappa_values": [float(v) for v in KAPPA_VALUES],
            "baseline_delta0": baseline_delta0,
            "baseline_kappa": baseline_kappa,
            "base_cases": [case.name for case in base_cases],
            "total_time_days": params["time"].total_time,
            "sweep_note": "Shortened 150-day femur horizon for compact reviewer-facing mechanostat sensitivity study",
        },
    )

    clean_output_dir(sweep.base_output_dir, comm, logger)

    runner = create_mechanostat_runner(params, mdl)

    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("MECHANOSTAT PARAMETER SWEEP")
        logger.info("=" * 70)
        logger.info(f"Params: {PARAMS_FILE}")
        logger.info(
            f"Baseline stimulus_delta0={baseline_delta0:g}, stimulus_kappa={baseline_kappa:g}"
        )
        logger.info(f"delta0 values: {DELTA0_VALUES}")
        logger.info(f"kappa values: {KAPPA_VALUES}")
        logger.info(f"Loading cases: {[case.name for case in base_cases]}")
        logger.info(f"Total runs: {sweep.total_runs()} ({len(DELTA0_VALUES)} × {len(KAPPA_VALUES)})")
        logger.info(f"Simulation time: {params['time'].total_time} days")
        logger.info("=" * 70)

    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()

    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: mpirun -n 4 python analysis/mechanostat_sensitivity_analysis.py")


if __name__ == "__main__":
    main()
