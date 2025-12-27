"""Run Anderson vs Picard acceleration comparison sweep.

This script compares Anderson acceleration vs Picard iteration by sweeping
over timestep values while keeping mesh resolution fixed.

Usage:
    mpirun -n 4 python run_anderson_sweep.py

Outputs:
    results/anderson_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   └── fields.bp/
    ├── <hash2>/
    │   └── ...
"""

from __future__ import annotations

import copy
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
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import SweepProgressReporter


def create_anderson_runner(
    base_params: dict,
    box: dict,
    fixed_N: int,
) -> SimulationRunner:
    """Create a runner that compares Anderson vs Picard at fixed mesh resolution.
    
    Args:
        base_params: Loaded parameters from default_params_box.json.
        box: Box geometry/loading parameters.
        fixed_N: Fixed mesh resolution for all runs.
    
    Returns:
        Runner function for Parametrizer.
    """
    
    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        """Run single simulation with given acceleration type."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract swept parameters
        dt_days = float(param_point["dt_days"])
        accel_type = str(param_point["accel_type"])
        
        # Modify params for this run
        params["time"].dt_initial = dt_days
        params["time"].adaptive_dt = False
        params["solver"].accel_type = accel_type
        params["output"].results_dir = str(output_path)
        params["geometry"].fix_tag = BoxMeshBuilder.TAG_BOTTOM
        params["geometry"].load_tag = BoxMeshBuilder.TAG_TOP
        
        total_time = params["time"].total_time
        
        # Update reporter with correct total_time for this run
        if reporter is not None:
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=total_time)
        
        # Create mesh with fixed resolution N (scale nz with aspect ratio)
        geometry = BoxGeometry(
            Lx=box["Lx"], Ly=box["Ly"], Lz=box["Lz"],
            nx=fixed_N, ny=fixed_N, nz=int(fixed_N * box["Lz"] / box["Lx"]),
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()
        
        # Create config
        sim_cfg = create_config(domain, facet_tags, params)
        
        # Create loader and loading cases
        loader = BoxLoader(domain, facet_tags, load_tag=BoxMeshBuilder.TAG_TOP)
        loading_cases = [get_parabolic_pressure_case(
            pressure=box["pressure"],
            gradient_axis=box["gradient_axis"],
            center_factor=box["center_factor"],
            edge_factor=box["edge_factor"],
            box_extent=(0.0, box["Lx"]),
            name="parabolic_compression",
        )]
        
        # Create factory and run
        factory = BoxSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            remodeller.simulate(reporter=reporter)
    
    return runner


def main() -> None:
    """Run Anderson vs Picard sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AndersonSweep")
    
    # Load parameters from JSON (same defaults as convergence sweep)
    params = load_default_params("default_params_box.json")
    box = params["box"]
    
    # Fixed mesh resolution for acceleration comparison
    FIXED_N = 24
    
    # Sweep parameters
    accel_types = ["picard", "anderson"]
    dt_values = [10.0]
    
    # Define sweep
    sweep = ParameterSweep(
        params={
            "accel_type": accel_types,
            "dt_days": dt_values,
        },
        base_output_dir=Path("results/anderson_sweep"),
        metadata={
            "description": "Anderson vs Picard acceleration comparison",
            "fixed_N": FIXED_N,
            "accel_types": accel_types,
            "dt_values": dt_values,
        },
    )
    
    # Create runner
    runner = create_anderson_runner(params, box, FIXED_N)
    
    # Run sweep
    if comm.rank == 0:
        logger.info("=" * 60)
        logger.info("ANDERSON VS PICARD SWEEP")
        logger.info("=" * 60)
        logger.info(f"Fixed N: {FIXED_N}")
        logger.info(f"Acceleration types: {accel_types}")
        logger.info(f"dt values: {dt_values}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("=" * 60)
    
    Parametrizer(sweep, runner, comm).run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")


if __name__ == "__main__":
    main()
