"""Run fabric parameter sweep: varying fabric_tau, fabric_cA, and fabric_gammaF.

This script runs simulations over a grid of fabric parameter values and saves
checkpoints (including sigma stress field) for subsequent analysis.

Fabric parameters chosen for sweep (most impactful on fabric evolution):
- fabric_tau: Time constant [days] - controls adaptation rate
- fabric_cA: Coupling strength - controls fabric-stress coupling magnitude
- fabric_gammaF: Power-law exponent - controls eigenvalue scaling

Each parameter has 3 levels: min, baseline, max (3×3×3 = 27 runs).

Usage:
    mpirun -n 4 python run_fabric_sweep.py

Outputs:
    results/fabric_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/     <- For analysis (adios4dolfinx), includes sigma
    │   ├── fields.bp/         <- For visualization (VTXWriter), includes sigma
    │   ├── steps.csv          <- Solver metrics per step
    │   └── subiterations.csv  <- Detailed iteration metrics
    ├── <hash2>/
    │   └── ...

Post-processing:
    python analysis/fabric_analysis.py
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
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


def create_fabric_runner(
    base_params: dict,
    box: dict,
) -> SimulationRunner:
    """Create a runner for fabric parameter sweep.
    
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
        """Run single simulation with fabric parameter overrides."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract fabric parameters from param_point
        fabric_tau = float(param_point["fabric.fabric_tau"])
        fabric_cA = float(param_point["fabric.fabric_cA"])
        fabric_gammaF = float(param_point["fabric.fabric_gammaF"])
        
        # Apply parameter overrides
        params["fabric"].fabric_tau = fabric_tau
        params["fabric"].fabric_cA = fabric_cA
        params["fabric"].fabric_gammaF = fabric_gammaF
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
            # Initialize L with random symmetric traceless tensor
            # This ensures L has a different direction than Qbar initially,
            # allowing us to observe alignment dynamics
            L = remodeller.L
            L_old = remodeller.L_old
            n_owned = L.function_space.dofmap.index_map.size_local * L.function_space.dofmap.index_map_bs
            
            # Generate random symmetric traceless 3x3 tensors
            # Use fixed seed based on rank for reproducibility across runs
            rng = np.random.default_rng(seed=42 + comm.rank)
            n_tensors = n_owned // 9
            random_L = rng.standard_normal((n_tensors, 3, 3)) * 0.5  # Scale factor
            # Symmetrize
            random_L = 0.5 * (random_L + np.swapaxes(random_L, 1, 2))
            # Make traceless (tr(L) = 0)
            trace = np.trace(random_L, axis1=1, axis2=2)
            for i in range(3):
                random_L[:, i, i] -= trace / 3.0
            
            L.x.array[:n_owned] = random_L.flatten()
            L.x.scatter_forward()
            L_old.x.array[:n_owned] = random_L.flatten()
            L_old.x.scatter_forward()
            
            # Run simulation with unified sweep reporter
            remodeller.simulate(reporter=reporter)
            
            # Write final checkpoint for analysis
            checkpoint = CheckpointStorage(sim_cfg)
            final_time = sim_cfg.time.total_time

            # Mechanics: psi (cycle-weighted SED average over all loading cases)
            psi = remodeller.driver.stimulus_field()
            if psi is not None:
                checkpoint.write_function(psi, final_time)

            # Stress tensor (sigma) - computed in post_step_update
            sigma = remodeller.driver.sigma_field()
            if sigma is not None:
                checkpoint.write_function(sigma, final_time)

            # Qbar (stress-stress product for fabric alignment)
            Qbar = remodeller.driver.Qbar_field()
            if Qbar is not None:
                checkpoint.write_function(Qbar, final_time)

            # Registry state fields (rho, S, L from density/stimulus/fabric solvers)
            state_fields = remodeller.registry.state_fields
            for name in ("rho", "S", "L"):
                f = state_fields.get(name)
                if f is not None:
                    checkpoint.write_function(f, final_time)
            checkpoint.close()
    
    return runner


def main() -> None:
    """Run fabric parameter sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="FabricSweep")
    
    # Load stiff parameters as baseline
    params = load_default_params("stiff_params_box.json")
    box = params["box"]
    
    # Override for fabric study:
    # - Shorter simulation time (sufficient to see fabric evolution effects)
    params["time"].total_time = 200.0  # 200 days
    params["time"].adaptive_dt = False
    # Keep default dt_initial from stiff_params (25 days)
    
    # Baseline fabric values from stiff_params_box.json
    fabric_tau_base = float(params["fabric"].fabric_tau)      # 120.0 days
    fabric_cA_base = float(params["fabric"].fabric_cA)        # 1.0
    fabric_gammaF_base = float(params["fabric"].fabric_gammaF)  # 1.0
    
    # Define sweep parameters: 3 levels per parameter
    # 
    # fabric_tau: Time constant - smaller = faster adaptation, larger = slower
    #   Baseline: 120 days
    #   Min: 30 days (4× faster adaptation - very responsive)
    #   Max: 480 days (4× slower adaptation - very sluggish)
    #
    # fabric_cA: Coupling strength - controls how strongly fabric aligns with stress
    #   Baseline: 1.0
    #   Min: 0.25 (weak coupling - fabric barely responds)
    #   Max: 4.0 (strong coupling - aggressive fabric alignment)
    #
    # fabric_gammaF: Power-law exponent for eigenvalue scaling
    #   Baseline: 1.0 (linear)
    #   Min: 0.5 (sublinear - softer eigenvalue contrast)
    #   Max: 2.0 (superlinear - sharper eigenvalue contrast)
    
    fabric_tau_values = [
        fabric_tau_base / 4.0,    # 30 days - fast adaptation
        fabric_tau_base,          # 120 days - baseline
        fabric_tau_base * 4.0,    # 480 days - slow adaptation
    ]
    
    fabric_cA_values = [
        fabric_cA_base / 4.0,     # 0.25 - weak coupling
        fabric_cA_base,           # 1.0 - baseline
        fabric_cA_base * 4.0,     # 4.0 - strong coupling
    ]
    
    fabric_gammaF_values = [
        0.5,                      # Sublinear - softer contrast
        fabric_gammaF_base,       # 1.0 - baseline (linear)
        2.0,                      # Superlinear - sharper contrast
    ]
    
    # Convert to sweep format
    sweep = ParameterSweep(
        params={
            "fabric.fabric_tau": fabric_tau_values,
            "fabric.fabric_cA": fabric_cA_values,
            "fabric.fabric_gammaF": fabric_gammaF_values,
        },
        base_output_dir=Path("results/fabric_sweep"),
        metadata={
            "description": "Fabric parameter sweep: fabric_tau × fabric_cA × fabric_gammaF",
            "objective": "Study fabric evolution sensitivity to key parameters",
            "baseline_fabric_tau": fabric_tau_base,
            "baseline_fabric_cA": fabric_cA_base,
            "baseline_fabric_gammaF": fabric_gammaF_base,
            "fabric_tau_values": fabric_tau_values,
            "fabric_cA_values": fabric_cA_values,
            "fabric_gammaF_values": fabric_gammaF_values,
            "mesh_nx": box["nx"],
            "mesh_ny": box["ny"],
            "mesh_nz": box["nz"],
            "total_time_days": params["time"].total_time,
            "dt_days": params["time"].dt_initial,
            "notes": {
                "fabric_tau": "Time constant [days]: smaller=faster adaptation",
                "fabric_cA": "Coupling strength: controls fabric-stress alignment magnitude",
                "fabric_gammaF": "Power-law exponent: controls eigenvalue contrast (0.5=soft, 2.0=sharp)",
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
    runner = create_fabric_runner(params, box)
    
    # Log sweep info
    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("FABRIC PARAMETER SWEEP")
        logger.info("=" * 70)
        logger.info(f"Baseline from stiff_params_box.json")
        logger.info("-" * 70)
        logger.info(f"Baseline fabric_tau = {fabric_tau_base:.1f} days")
        logger.info(f"Baseline fabric_cA = {fabric_cA_base:.2f}")
        logger.info(f"Baseline fabric_gammaF = {fabric_gammaF_base:.2f}")
        logger.info("-" * 70)
        logger.info(f"fabric_tau sweep: {[f'{v:.1f}' for v in fabric_tau_values]} days")
        logger.info(f"fabric_cA sweep: {[f'{v:.2f}' for v in fabric_cA_values]}")
        logger.info(f"fabric_gammaF sweep: {[f'{v:.2f}' for v in fabric_gammaF_values]}")
        logger.info(f"Total runs: {sweep.total_runs()} (3 × 3 × 3)")
        logger.info("-" * 70)
        logger.info(f"Mesh: {box['nx']}×{box['ny']}×{box['nz']} on {box['Lx']}×{box['Ly']}×{box['Lz']} mm")
        logger.info(f"Simulation time: {params['time'].total_time} days, dt={params['time'].dt_initial} days")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("Checkpointing: adios4dolfinx (includes sigma stress field)")
        logger.info("=" * 70)
    
    # Run sweep
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: python analysis/fabric_analysis.py")


if __name__ == "__main__":
    main()
