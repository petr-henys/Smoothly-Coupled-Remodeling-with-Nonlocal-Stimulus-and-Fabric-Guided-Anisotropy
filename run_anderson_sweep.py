"""Run Anderson acceleration parameter sweep: varying m, beta, and lam.

This script runs simulations over a grid of Anderson acceleration parameters
and saves solver performance metrics for subsequent analysis.

Anderson parameters chosen for sweep (most impactful on convergence):
- m: History size - number of previous iterates for extrapolation
- beta: Mixing/relaxation parameter - step size damping
- lam: Tikhonov regularization - stabilization for ill-conditioned Gram matrix

Each parameter has 3 levels: min, baseline, max (3×3×3 = 27 runs).

Usage:
    mpirun -n 4 python run_anderson_sweep.py

Outputs:
    results/anderson_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/
    │   ├── fields.bp/
    │   ├── steps.csv          <- Key metrics: num_subiters, convergence
    │   └── subiterations.csv  <- Detailed per-iteration stats
    ├── <hash2>/
    │   └── ...

Post-processing:
    python analysis/anderson_analysis.py
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


def create_anderson_runner(
    base_params: dict,
    box: dict,
) -> SimulationRunner:
    """Create a runner for Anderson parameter sweep.
    
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
        """Run single simulation with Anderson parameter overrides."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract Anderson parameters from param_point
        m = int(param_point["solver.m"])
        beta = float(param_point["solver.beta"])
        lam = float(param_point["solver.lam"])
        
        # Apply parameter overrides
        params["solver"].m = m
        params["solver"].beta = beta
        params["solver"].lam = lam
        params["solver"].accel_type = "anderson"  # Ensure Anderson is used
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
        
        # Create loading cases
        loading_cases = [get_parabolic_pressure_case(
            pressure=box["pressure"],
            load_tag=BoxMeshBuilder.TAG_TOP,
            gradient_axis=box["gradient_axis"],
            center_factor=box["center_factor"],
            edge_factor=box["edge_factor"],
            box_extent=(0.0, box["Lx"]),
            name="parabolic_compression",
        )]
        
        # Create pressure loader
        loader = BoxLoader(domain, facet_tags, loading_cases=loading_cases)
        
        # Create factory and run
        factory = BoxSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, factory=factory) as remodeller:
            # Run simulation with unified sweep reporter
            remodeller.simulate(reporter=reporter)
            
            # Write final checkpoint for analysis
            checkpoint = CheckpointStorage(sim_cfg)
            final_time = sim_cfg.time.total_time

            # Mechanics fields
            psi = remodeller.driver.stimulus_field()
            if psi is not None:
                checkpoint.write_function(psi, final_time)

            sigma = remodeller.driver.sigma_field()
            if sigma is not None:
                checkpoint.write_function(sigma, final_time)

            # State fields
            state_fields = remodeller.registry.state_fields
            for name in ("rho", "S", "L"):
                f = state_fields.get(name)
                if f is not None:
                    checkpoint.write_function(f, final_time)
            checkpoint.close()
    
    return runner


def main() -> None:
    """Run Anderson parameter sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AndersonSweep")
    
    # Load stiff parameters as baseline (challenging for fixed-point)
    params = load_default_params("stiff_params_box.json")
    box = params["box"]
    
    # Override for Anderson study:
    # - Shorter simulation time (sufficient to see convergence behavior)
    params["time"].total_time = 100.0  # 100 days
    params["time"].adaptive_dt = False
    # Keep default dt_initial from stiff_params (25 days)
    
    # Baseline Anderson values from stiff_params_box.json
    m_base = int(params["solver"].m)          # 5
    beta_base = float(params["solver"].beta)  # 1.0
    lam_base = float(params["solver"].lam)    # 1e-3
    
    # Define sweep parameters: 3 levels per parameter
    #
    # m (history size):
    #   - Controls how many previous iterates are used for extrapolation
    #   - Too small: weak acceleration, slow convergence
    #   - Too large: ill-conditioned Gram matrix, instability
    #   Baseline: 5
    #   Min: 2 (minimal acceleration)
    #   Max: 10 (aggressive acceleration, may need more regularization)
    #
    # beta (mixing/relaxation):
    #   - Controls step size: x_{k+1} = x_k + beta * delta
    #   - Too small: slow convergence (over-damped)
    #   - Too large: oscillations, divergence (under-damped)
    #   Baseline: 1.0 (no damping)
    #   Min: 0.5 (conservative, more stable)
    #   Max: 1.5 (aggressive, faster but riskier)
    #
    # lam (Tikhonov regularization):
    #   - Stabilizes the least-squares solve for Anderson coefficients
    #   - Too small: ill-conditioning, numerical instability
    #   - Too large: degrades to Picard (no acceleration benefit)
    #   Baseline: 1e-3
    #   Min: 1e-4 (minimal regularization, aggressive)
    #   Max: 1e-2 (strong regularization, conservative)
    
    m_values = [2, m_base, 10]
    beta_values = [0.5, beta_base, 1.5]
    lam_values = [1e-4, lam_base, 1e-2]
    
    # Convert to sweep format
    sweep = ParameterSweep(
        params={
            "solver.m": m_values,
            "solver.beta": beta_values,
            "solver.lam": lam_values,
        },
        base_output_dir=Path("results/anderson_sweep"),
        metadata={
            "description": "Anderson acceleration sweep: m × beta × lam",
            "objective": "Study Anderson convergence sensitivity to key parameters",
            "baseline_m": m_base,
            "baseline_beta": beta_base,
            "baseline_lam": lam_base,
            "m_values": m_values,
            "beta_values": beta_values,
            "lam_values": lam_values,
            "mesh_nx": box["nx"],
            "mesh_ny": box["ny"],
            "mesh_nz": box["nz"],
            "total_time_days": params["time"].total_time,
            "dt_days": params["time"].dt_initial,
            "notes": {
                "m": "History size: number of previous iterates for extrapolation",
                "beta": "Mixing parameter: step size damping (1.0=no damping)",
                "lam": "Tikhonov regularization: stabilizes Gram matrix solve",
            },
        },
    )
    
    # Clean output directory before new computation
    if comm.rank == 0:
        if sweep.base_output_dir.exists():
            logger.info(f"Cleaning output directory: {sweep.base_output_dir}")
            shutil.rmtree(sweep.base_output_dir)
    comm.Barrier()
    
    # Create runner
    runner = create_anderson_runner(params, box)
    
    # Log sweep info
    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("ANDERSON ACCELERATION PARAMETER SWEEP")
        logger.info("=" * 70)
        logger.info(f"Baseline from stiff_params_box.json")
        logger.info("-" * 70)
        logger.info(f"Baseline m = {m_base} (history size)")
        logger.info(f"Baseline beta = {beta_base:.2f} (mixing parameter)")
        logger.info(f"Baseline lam = {lam_base:.1e} (Tikhonov regularization)")
        logger.info("-" * 70)
        logger.info(f"m sweep: {m_values}")
        logger.info(f"beta sweep: {[f'{v:.2f}' for v in beta_values]}")
        logger.info(f"lam sweep: {[f'{v:.1e}' for v in lam_values]}")
        logger.info(f"Total runs: {sweep.total_runs()} (3 × 3 × 3)")
        logger.info("-" * 70)
        logger.info(f"Mesh: {box['nx']}×{box['ny']}×{box['nz']} on {box['Lx']}×{box['Ly']}×{box['Lz']} mm")
        logger.info(f"Simulation time: {params['time'].total_time} days, dt={params['time'].dt_initial} days")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("Key outputs: steps.csv, subiterations.csv (solver performance)")
        logger.info("=" * 70)
    
    # Run sweep
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: python analysis/anderson_analysis.py")


if __name__ == "__main__":
    main()
