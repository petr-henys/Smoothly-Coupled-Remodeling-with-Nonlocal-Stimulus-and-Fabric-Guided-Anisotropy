"""Run convergence sweep: varying mesh resolution (N) and timestep (dt).

This script runs simulations over a grid of (N, dt) values and saves
checkpoints for subsequent convergence analysis.

Usage:
    mpirun -n 4 python run_convergence_sweep.py

Outputs:
    results/convergence_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/     <- For analysis (adios4dolfinx)
    │   └── fields.bp/         <- For visualization (VTXWriter)
    ├── <hash2>/
    │   └── ...
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from mpi4py import MPI

from parametrizer import (
    ParameterSweep,
    Parametrizer,
    ParamValue,
    SimulationRunner,
)
from box import (
    BoxSolverFactory,
    BoxLoader,
    BoxGeometry,
    BoxMeshBuilder,
    get_parabolic_pressure_case,
)
from simulation.checkpoint import CheckpointStorage
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import SweepProgressReporter


def create_convergence_runner(
    base_params: dict,
    box: dict,
) -> SimulationRunner:
    """Create a runner that uses N from param_point to set mesh resolution.
    
    Args:
        base_params: Loaded parameters from default_params.json.
        box: Box geometry/loading parameters.
    
    Returns:
        Runner function for Parametrizer.
    """
    
    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        """Run single convergence simulation with checkpoint output."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract N and dt from param_point
        N = int(param_point["N"])
        dt_days = float(param_point["dt_days"])
        
        # Modify params for this run
        params["time"].dt_initial = dt_days
        params["time"].adaptive_dt = False
        params["output"].results_dir = str(output_path)
        params["geometry"].fix_tag = BoxMeshBuilder.TAG_BOTTOM
        params["geometry"].load_tag = BoxMeshBuilder.TAG_TOP
        
        total_time = params["time"].total_time
        
        # Update reporter with correct total_time for this run
        if reporter is not None:
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=total_time)
        
        # Create mesh with resolution N (scale nz with aspect ratio)
        geometry = BoxGeometry(
            Lx=box["Lx"], Ly=box["Ly"], Lz=box["Lz"],
            nx=N, ny=N, nz=int(N * box["Lz"] / box["Lx"]),
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()
        
        # Create config
        sim_cfg = create_config(domain, facet_tags, params)
        
        # Create loader and loading cases
        loading_cases = [get_parabolic_pressure_case(
            pressure=box["pressure"],
            gradient_axis=box["gradient_axis"],
            center_factor=box["center_factor"],
            edge_factor=box["edge_factor"],
            box_extent=(0.0, box["Lx"]),
            name="parabolic_compression",
        )]
        loader = BoxLoader(domain, facet_tags, loading_cases=loading_cases)
        
        # Create factory and run
        factory = BoxSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, factory=factory) as remodeller:
            # Run simulation with unified sweep reporter
            remodeller.simulate(reporter=reporter)
            
            # Write final checkpoint for convergence analysis.
            checkpoint = CheckpointStorage(sim_cfg)
            final_time = sim_cfg.time.total_time

            # Mechanics: psi (cycle-weighted SED average over all loading cases)
            psi = remodeller.driver.stimulus_field()
            if psi is not None:
                checkpoint.write_function(psi, final_time)

            # Registry state fields (rho, S, L from density/stimulus/fabric solvers)
            state_fields = remodeller.registry.state_fields
            for name in ("rho", "S", "L"):
                f = state_fields.get(name)
                if f is not None:
                    checkpoint.write_function(f, final_time)
            checkpoint.close()
    
    return runner


def main() -> None:
    """Run convergence sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="ConvergenceSweep")
    
    # Load parameters from JSON
    params = load_default_params("stiff_params_box.json")
    box = params["box"]
    
    # Modify for convergence study
    params["time"].total_time = 100.0  # Shorter for convergence study
    
    # Define sweep parameters
    N_values = [12, 16, 24, 32]  # Mesh resolutions
    dt_values = [25.0, 10.0, 5.0, 2.5, 1.25]  # Timesteps [days]
    
    # Convert to sweep format
    sweep = ParameterSweep(
        params={
            "N": N_values,
            "dt_days": dt_values,
        },
        base_output_dir=Path("results/convergence_sweep"),
        metadata={
            "description": "Convergence sweep: spatial (N) and temporal (dt)",
            "N_values": N_values,
            "dt_values": dt_values,
        },
    )
    
    # Clean output directory before new computation
    if comm.rank == 0:
        if sweep.base_output_dir.exists():
            logger.info(f"Cleaning output directory: {sweep.base_output_dir}")
            shutil.rmtree(sweep.base_output_dir)
    comm.Barrier()
    
    # Create runner
    runner = create_convergence_runner(params, box)
    
    # Run sweep
    if comm.rank == 0:
        logger.info("=" * 60)
        logger.info("CONVERGENCE SWEEP")
        logger.info("=" * 60)
        logger.info(f"N values: {N_values}")
        logger.info(f"dt values: {dt_values}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("Checkpointing: adios4dolfinx")
        logger.info("=" * 60)
    
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: python analysis/convergence_errors.py")


if __name__ == "__main__":
    main()
