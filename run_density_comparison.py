"""Compare simulated bone density with CT-derived density.

This experiment for the manuscript:
1. Runs bone remodeling simulation on the femur mesh (if not already done)
2. Saves checkpoint for analysis

Analysis and plotting is performed separately by `analysis/density_comparison_plot.py`.

Configuration:
    Edit the switches in the Configuration section below:
    - FORCE_RERUN: Set True to rerun simulation even if checkpoint exists
    - OUTPUT_DIR: Output directory path
    - PARAMS_FILE: Parameter file name

Usage:
    mpirun -n 4 python run_density_comparison.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from mpi4py import MPI

from dolfinx import fem

from femur import FEBio2Dolfinx, FemurPaths, get_standard_gait_cases, Loader
from simulation.checkpoint import CheckpointStorage
from simulation.factory import DefaultSolverFactory
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import ProgressReporter

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh


# ===========================================================================
# Configuration - EDIT THESE SWITCHES
# ===========================================================================

# If True, rerun simulation even if checkpoint exists
FORCE_RERUN = False

# Output directory
OUTPUT_DIR = Path("results/density_comparison")

# Parameter file for simulation
PARAMS_FILE = "stiff_params_femur.json"


# ===========================================================================
# Main simulation runner
# ===========================================================================

def run_simulation(
    output_dir: Path,
    params_file: str = "stiff_params_femur.json",
) -> None:
    """Run femur remodeling simulation with checkpointing.
    
    Args:
        output_dir: Directory for all outputs.
        params_file: Parameter file name.
    """
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="DensityComparison")
    
    # Load femur mesh from FEBio format
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    mesh = mdl.mesh_dolfinx
    
    # Load parameters from JSON
    params = load_default_params(params_file)
    
    # Override output directory
    params["output"].results_dir = str(output_dir)
    
    # Create simulation configuration
    cfg = create_config(mesh, mdl.meshtags, params)
    
    # Reset log file on rank 0
    if comm.rank == 0:
        with open(cfg.log_file, "w") as f:
            f.write("")
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total simulation time: {cfg.time.total_time} days")
    
    # Load standard gait scenarios
    loading_cases = get_standard_gait_cases()
    
    if comm.rank == 0:
        logger.info(f"Defined {len(loading_cases)} loading case(s): {[c.name for c in loading_cases]}")
    
    # Create traction loader with precomputed loading cases
    loader = Loader(mesh, facet_tags=mdl.meshtags, load_tag=cfg.geometry.load_tag, loading_cases=loading_cases)
    
    # Save tractions for visualization
    loader.save_tractions_vtx(cfg.output.results_dir)
    if comm.rank == 0:
        logger.info(f"Saved tractions to {cfg.output.results_dir}/tractions.bp")
    
    # Create solver factory
    factory = DefaultSolverFactory(cfg)
    
    # Run simulation with progress reporting
    with Remodeller(cfg, loader=loader, factory=factory) as remodeller:
        with ProgressReporter(comm, cfg.time.total_time, cfg.solver.max_subiters) as reporter:
            remodeller.simulate(reporter=reporter)
        
        # Write final checkpoint for analysis
        checkpoint = CheckpointStorage(cfg)
        final_time = cfg.time.total_time
        
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
        
        if comm.rank == 0:
            logger.info(f"Checkpoint saved to {output_dir}/checkpoint.bp")


def main() -> None:
    """Run density comparison experiment."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="DensityComparison")
    
    output_dir = OUTPUT_DIR
    checkpoint_path = output_dir / "checkpoint.bp"
    
    # Check if checkpoint exists (simulation already done)
    checkpoint_exists = checkpoint_path.exists()
    run_simulation_flag = FORCE_RERUN or not checkpoint_exists
    
    if run_simulation_flag:
        # Clean output directory before new computation
        if comm.rank == 0:
            if output_dir.exists():
                logger.info(f"Cleaning output directory: {output_dir}")
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        
        if comm.rank == 0:
            logger.info("=" * 60)
            logger.info("DENSITY COMPARISON EXPERIMENT")
            logger.info("=" * 60)
            logger.info(f"Output: {output_dir}")
            logger.info("Mode: Running simulation")
            logger.info("=" * 60)
            logger.info("Step 1: Running bone remodeling simulation...")
        
        run_simulation(output_dir, PARAMS_FILE)
        
        if comm.rank == 0:
            logger.info("Simulation complete!")
            logger.info("Run 'python analysis/density_comparison_plot.py' for analysis.")
            
    else:
        if comm.rank == 0:
            logger.info("=" * 60)
            logger.info("DENSITY COMPARISON EXPERIMENT")
            logger.info("=" * 60)
            logger.info("Checkpoint exists. Skipping simulation.")
            logger.info("Run 'python analysis/density_comparison_plot.py' for analysis.")
            logger.info("Set FORCE_RERUN=True to rerun simulation.")
            logger.info("=" * 60)


if __name__ == "__main__":
    main()
