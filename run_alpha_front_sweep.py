"""Run alpha_front sweep: varying hip frontal-angle offset from -15° to +25°.

This script runs femur simulations over a range of hip frontal angle values,
applying the same offset to ALL loading cases (heel_strike, mid_stance, etc.).
Each simulation saves checkpoints for subsequent analysis.

Usage:
    mpirun -n 4 python run_alpha_front_sweep.py

Outputs:
    results/alpha_front_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/     <- For analysis (adios4dolfinx)
    │   ├── fields.bp/         <- For visualization (VTXWriter)
    │   └── steps.csv
    ├── <hash2>/
    │   └── ...

Post-processing:
    python analysis/alpha_front_postprocess.py
    
This creates a combined VTX file where pseudo-time corresponds to different
alpha_front values (each "time step" = last timestep of different simulation).
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np
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


def modify_loading_cases_alpha_front(
    base_cases: List[LoadingCase],
    alpha_front_offset: float,
) -> List[LoadingCase]:
    """Apply alpha_front offset to all hip loads in loading cases.
    
    Args:
        base_cases: Original loading cases.
        alpha_front_offset: Offset to add to each hip's alpha_front [deg].
    
    Returns:
        New list of LoadingCase with modified alpha_front values.
    """
    modified_cases = []
    for case in base_cases:
        if case.hip is not None:
            new_hip = replace(
                case.hip,
                alpha_front=case.hip.alpha_front + alpha_front_offset,
            )
        else:
            new_hip = None
        
        new_case = replace(case, hip=new_hip)
        modified_cases.append(new_case)
    
    return modified_cases


def create_alpha_front_runner(
    base_params: dict,
    base_cases: List[LoadingCase],
    mdl: FEBio2Dolfinx,
):
    """Create a runner for alpha_front sweep.
    
    Args:
        base_params: Loaded parameters from params file.
        base_cases: Base loading cases (standard gait).
        mdl: Loaded femur mesh model.
    
    Returns:
        Runner function for Parametrizer.
    """
    
    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        """Run single simulation with alpha_front offset."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract alpha_front offset from param_point
        alpha_front_offset = float(param_point["alpha_front_offset"])
        
        # Apply parameter overrides
        params["output"].results_dir = str(output_path)
        
        reset_reporter(reporter, params["time"].total_time)
        
        # Modify loading cases with alpha_front offset
        modified_cases = modify_loading_cases_alpha_front(base_cases, alpha_front_offset)
        
        # Create config (mesh is shared, reuse mdl)
        mesh = mdl.mesh_dolfinx
        sim_cfg = create_config(mesh, mdl.meshtags, params)
        
        # Create loader with modified loading cases
        loader = Loader(
            mesh,
            facet_tags=mdl.meshtags,
            load_tag=sim_cfg.geometry.load_tag,
            loading_cases=modified_cases,
        )
        
        # Create factory and run
        factory = DefaultSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, factory=factory) as remodeller:
            remodeller.simulate(reporter=reporter)
            write_standard_checkpoint(sim_cfg, remodeller)
    
    return runner


def main() -> None:
    """Run alpha_front sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AlphaFrontSweep")
    
    # Load femur mesh once (shared across all runs)
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    
    # Load stiff parameters as baseline
    params = load_default_params("stiff_params_femur.json")
    
    # Get standard gait cases as baseline
    base_cases = get_standard_gait_cases()
    
    # Define alpha_front offset range: -15° to +25°
    alpha_front_min = -15.0
    alpha_front_max = 25.0
    n_values = 10
    alpha_front_offsets = np.linspace(alpha_front_min, alpha_front_max, n_values).tolist()
    
    # Define sweep parameters
    sweep = ParameterSweep(
        params={
            "alpha_front_offset": alpha_front_offsets,
        },
        base_output_dir=Path("results/alpha_front_sweep"),
        validate_config_params=False,  # Custom parameter, not in Config
        metadata={
            "description": "Alpha_front sweep: hip frontal angle offset from -15° to +25°",
            "objective": "Study effect of loading direction on bone adaptation",
            "alpha_front_min_deg": alpha_front_min,
            "alpha_front_max_deg": alpha_front_max,
            "n_values": n_values,
            "alpha_front_offsets": alpha_front_offsets,
            "base_cases": [c.name for c in base_cases],
            "total_time_days": params["time"].total_time,
        },
    )
    
    clean_output_dir(sweep.base_output_dir, comm, logger)
    
    # Create runner
    runner = create_alpha_front_runner(params, base_cases, mdl)
    
    # Log sweep info
    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("ALPHA_FRONT SWEEP")
        logger.info("=" * 70)
        logger.info(f"Hip frontal angle offset range: {alpha_front_min}° to {alpha_front_max}°")
        logger.info(f"Number of values: {n_values}")
        logger.info(f"Offsets: {[f'{v:.1f}°' for v in alpha_front_offsets]}")
        logger.info(f"Loading cases: {[c.name for c in base_cases]}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"Simulation time: {params['time'].total_time} days")
        logger.info("=" * 70)
    
    # Run sweep
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run post-processing with: python analysis/alpha_front_postprocess.py")


if __name__ == "__main__":
    main()
