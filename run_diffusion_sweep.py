"""Run diffusion regularization sweep: varying D_rho, stimulus_D, and fabric_D.

This script runs simulations over a grid of diffusion parameter values and saves
checkpoints for subsequent analysis (checkerboarding, solver performance).

Baseline diffusion values are taken from stiff_params_box.json.

Usage:
    mpirun -n 4 python run_diffusion_sweep.py

Outputs:
    results/diffusion_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/     <- For analysis (adios4dolfinx)
    │   ├── fields.bp/         <- For visualization (VTXWriter)
    │   ├── steps.csv          <- Solver metrics per step
    │   └── subiterations.csv  <- Detailed iteration metrics
    ├── <hash2>/
    │   └── ...

Post-processing:
    python analysis/diffusion_analysis.py
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


def create_diffusion_runner(
    base_params: dict,
    box: dict,
) -> SimulationRunner:
    """Create a runner for diffusion regularization sweep.
    
    Args:
        base_params: Loaded parameters from stiff_params_box.json.
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
        """Run single simulation with diffusion parameter overrides."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract diffusion parameters from param_point
        D_rho = float(param_point["density.D_rho"])
        stimulus_D = float(param_point["stimulus.stimulus_D"])
        fabric_D = float(param_point["fabric.fabric_D"])
        
        # Apply parameter overrides
        params["density"].D_rho = D_rho
        params["stimulus"].stimulus_D = stimulus_D
        params["fabric"].fabric_D = fabric_D
        params["output"].results_dir = str(output_path)
        params["geometry"].fix_tag = BoxMeshBuilder.TAG_BOTTOM
        params["geometry"].load_tag = BoxMeshBuilder.TAG_TOP
        
        total_time = params["time"].total_time
        
        # Update reporter with correct total_time for this run
        if reporter is not None:
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=total_time)
        
        # Create mesh using default resolution from box params
        geometry = BoxGeometry(
            Lx=box["Lx"], Ly=box["Ly"], Lz=box["Lz"],
            nx=box["nx"], ny=box["ny"], nz=box["nz"],
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()
        
        # Create config
        sim_cfg = create_config(domain, facet_tags, params)
        
        # Create loading cases (load_tag goes here, not in BoxLoader)
        loading_cases = [get_parabolic_pressure_case(
            pressure=box["pressure"],
            load_tag=BoxMeshBuilder.TAG_TOP,
            gradient_axis=box["gradient_axis"],
            center_factor=box["center_factor"],
            edge_factor=box["edge_factor"],
            box_extent=(0.0, box["Lx"]),
            name="parabolic_compression",
        )]
        
        # Create pressure loader - tags are extracted from loading_cases
        loader = BoxLoader(domain, facet_tags, loading_cases=loading_cases)
        
        # Create factory and run
        factory = BoxSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, factory=factory) as remodeller:
            # Run simulation with unified sweep reporter
            remodeller.simulate(reporter=reporter)
            
            # Write final checkpoint for analysis
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
    """Run diffusion regularization sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="DiffusionSweep")
    
    # Load stiff parameters as baseline (more challenging for solver)
    params = load_default_params("stiff_params_box.json")
    box = params["box"]
    
    # Override for diffusion study:
    params["time"].adaptive_dt = False
    
    # Use baseline diffusivities from loaded params
    D_rho_base = float(params["density"].D_rho)
    stimulus_D_base = float(params["stimulus"].stimulus_D)
    fabric_D_base = float(params["fabric"].fabric_D)
    h = box["Lx"] / box["nx"]  # element size for logging
    
    # Define sweep parameters: baseline ± one value on each side (factor of 10)
    # This gives 3 values per parameter: [low, baseline, high]
    factor = 10.
    
    D_rho_values = [
        D_rho_base / factor,      # Under-regularized
        D_rho_base,               # Baseline (ℓ = 1.5h)
        D_rho_base * factor,      # Over-regularized
    ]
    
    stimulus_D_values = [
        stimulus_D_base / factor,
        stimulus_D_base,
        stimulus_D_base * factor,
    ]
    
    fabric_D_values = [
        fabric_D_base / factor,
        fabric_D_base,
        fabric_D_base * factor,
    ]
    
    # Convert to sweep format
    sweep = ParameterSweep(
        params={
            "density.D_rho": D_rho_values,
            "stimulus.stimulus_D": stimulus_D_values,
            "fabric.fabric_D": fabric_D_values,
        },
        base_output_dir=Path("results/diffusion_sweep"),
        metadata={
            "description": "Diffusion regularization sweep: D_rho × stimulus_D × fabric_D",
            "objective": "Study checkerboarding prevention vs solver performance",
            "element_size_mm": h,
            "baseline_D_rho": D_rho_base,
            "baseline_stimulus_D": stimulus_D_base,
            "baseline_fabric_D": fabric_D_base,
            "D_rho_values": D_rho_values,
            "stimulus_D_values": stimulus_D_values,
            "fabric_D_values": fabric_D_values,
            "mesh_nx": box["nx"],
            "mesh_ny": box["ny"],
            "mesh_nz": box["nz"],
            "total_time_days": params["time"].total_time,
            "dt_days": params["time"].dt_initial,
        },
    )
    
    # Clean output directory before new computation
    if comm.rank == 0:
        if sweep.base_output_dir.exists():
            logger.info(f"Cleaning output directory: {sweep.base_output_dir}")
            shutil.rmtree(sweep.base_output_dir)
    comm.Barrier()
    
    # Create runner
    runner = create_diffusion_runner(params, box)
    
    # Log sweep info
    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("DIFFUSION REGULARIZATION SWEEP")
        logger.info("=" * 70)
        logger.info(f"Baseline from stiff_params_box.json, h = {h:.3f} mm")
        logger.info(f"D_rho: {D_rho_base:.5f} → sweep {[f'{v:.5f}' for v in D_rho_values]}")
        logger.info(f"stimulus_D: {stimulus_D_base:.5f} → sweep {[f'{v:.5f}' for v in stimulus_D_values]}")
        logger.info(f"fabric_D: {fabric_D_base:.5f} → sweep {[f'{v:.5f}' for v in fabric_D_values]}")
        logger.info(f"Total runs: {sweep.total_runs()} (3 × 3 × 3)")
        logger.info(f"Mesh: {box['nx']}×{box['ny']}×{box['nz']}, time: {params['time'].total_time} days")
        logger.info("=" * 70)
    
    # Run sweep
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: python analysis/diffusion_analysis.py")


if __name__ == "__main__":
    main()
