"""Run psi_ref ratio sweep: varying psi_ref_cort/psi_ref_trab.

This script runs femur simulations over a range of ratios

    r = psi_ref_cort / psi_ref_trab

and writes results into hash-based subdirectories.

Usage:
    mpirun -n 4 python run_psi_ref_ratio_sweep.py

Outputs:
    results/psi_ref_ratio_sweep/
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
import math
from pathlib import Path
from typing import List

from mpi4py import MPI

from femur import FEBio2Dolfinx, FemurPaths, get_standard_gait_cases, Loader
from femur.loader import LoadingCase
from parametrizer import ParameterSweep, Parametrizer, ParamValue
from simulation.factory import DefaultSolverFactory
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import SweepProgressReporter
from sweep_utils import clean_output_dir, reset_reporter, write_standard_checkpoint


# =============================================================================
# Sweep configuration (edit here)
# =============================================================================

# Baseline params JSON
PARAMS_FILE = "stiff_params_femur.json" 

# Ratios r = psi_ref_cort / psi_ref_trab
PSI_REF_RATIOS = [0.33, 0.5, 0.85, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0]

# Output directory
OUTPUT_DIR = Path("results/psi_ref_ratio_sweep")


def create_psi_ref_ratio_runner(
    base_params: dict,
    base_cases: List[LoadingCase],
    mdl: FEBio2Dolfinx,
):
    """Create a runner for psi_ref ratio sweep.

    Uses a constant geometric mean psi0 and varies the ratio r:
        r = psi_ref_cort / psi_ref_trab
        psi_ref_trab = psi0 / sqrt(r)
        psi_ref_cort = psi0 * sqrt(r)

    Args:
        base_params: Loaded parameters from a params JSON.
        base_cases: Base loading cases (standard gait).
        mdl: Loaded femur mesh model.
    """

    psi_ref_trab_base = float(base_params["stimulus"].psi_ref_trab)
    psi_ref_cort_base = float(base_params["stimulus"].psi_ref_cort)
    if psi_ref_trab_base <= 0.0 or psi_ref_cort_base <= 0.0:
        raise ValueError("Baseline psi_ref values must be > 0")
    psi0 = math.sqrt(psi_ref_trab_base * psi_ref_cort_base)

    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        params = copy.deepcopy(base_params)

        ratio = float(param_point["psi_ref_ratio"])
        if ratio <= 0.0:
            raise ValueError("psi_ref_ratio must be > 0")

        s = math.sqrt(ratio)
        psi_ref_trab = psi0 / s
        psi_ref_cort = psi0 * s

        params["stimulus"].psi_ref_trab = float(psi_ref_trab)
        params["stimulus"].psi_ref_cort = float(psi_ref_cort)
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
    logger = get_logger(comm, name="PsiRefRatioSweep")

    ratios = [float(r) for r in PSI_REF_RATIOS]
    if any(r <= 0.0 for r in ratios):
        raise ValueError("All ratios must be > 0")

    base_output_dir = Path(OUTPUT_DIR)

    # Load femur mesh once (shared across all runs)
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)

    # Load baseline parameters
    params = load_default_params(PARAMS_FILE)
    base_cases = get_standard_gait_cases()

    psi_ref_trab_base = float(params["stimulus"].psi_ref_trab)
    psi_ref_cort_base = float(params["stimulus"].psi_ref_cort)
    ratio_base = psi_ref_cort_base / psi_ref_trab_base
    psi0 = math.sqrt(psi_ref_trab_base * psi_ref_cort_base)

    sweep = ParameterSweep(
        params={
            "psi_ref_ratio": ratios,
        },
        base_output_dir=base_output_dir,
        validate_config_params=False,
        metadata={
            "description": "psi_ref ratio sweep: psi_ref_cort/psi_ref_trab",
            "params_file": PARAMS_FILE,
            "ratios": ratios,
            "baseline_psi_ref_trab": psi_ref_trab_base,
            "baseline_psi_ref_cort": psi_ref_cort_base,
            "baseline_ratio": ratio_base,
            "psi0_geometric_mean": psi0,
            "base_cases": [c.name for c in base_cases],
            "total_time_days": params["time"].total_time,
        },
    )

    clean_output_dir(sweep.base_output_dir, comm, logger)

    runner = create_psi_ref_ratio_runner(params, base_cases, mdl)

    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("PSI_REF RATIO SWEEP")
        logger.info("=" * 70)
        logger.info(f"Params: {PARAMS_FILE}")
        logger.info(
            f"Baseline psi_ref_trab={psi_ref_trab_base:g}, psi_ref_cort={psi_ref_cort_base:g} (ratio={ratio_base:g})"
        )
        logger.info(f"Fixed geometric mean psi0 = sqrt(trab*cort) = {psi0:g}")
        logger.info(f"Ratios: {ratios}")
        logger.info(f"Loading cases: {[c.name for c in base_cases]}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"Simulation time: {params['time'].total_time} days")
        logger.info("=" * 70)

    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()

    if comm.rank == 0:
        logger.info("Sweep complete!")


if __name__ == "__main__":
    main()
