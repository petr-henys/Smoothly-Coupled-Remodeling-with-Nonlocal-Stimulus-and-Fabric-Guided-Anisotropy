"""Run stimulus/mechanostat parameter sweep: varying kappa, delta0, and psi_ref.

This script runs simulations over a grid of mechanostat parameters and saves
fields for subsequent analysis of remodeling response.

Stimulus parameters chosen for sweep (define mechanostat behavior):
- stimulus_kappa: Saturation width - controls sharpness of formation/resorption transition
- stimulus_delta0: Lazy zone half-width - dead zone where no remodeling occurs
- psi_ref_trab: Reference SED - homeostatic target (what is "normal" loading)

Each parameter has 3 levels: min, baseline, max (3×3×3 = 27 runs).

Usage:
    mpirun -n 4 python run_stimulus_sweep.py

Outputs:
    results/stimulus_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/     <- includes rho, S, sigma
    │   ├── fields.bp/
    │   ├── steps.csv
    │   └── subiterations.csv
    ├── <hash2>/
    │   └── ...

Post-processing:
    python analysis/stimulus_analysis.py
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


def create_stimulus_runner(
    base_params: dict,
    box: dict,
) -> SimulationRunner:
    """Create a runner for stimulus/mechanostat parameter sweep.
    
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
        """Run single simulation with stimulus parameter overrides."""
        
        # Deep copy params so each run is independent
        params = copy.deepcopy(base_params)
        
        # Extract stimulus parameters from param_point
        kappa = float(param_point["stimulus.stimulus_kappa"])
        delta0 = float(param_point["stimulus.stimulus_delta0"])
        psi_ref = float(param_point["stimulus.psi_ref_trab"])
        
        # Apply parameter overrides
        params["stimulus"].stimulus_kappa = kappa
        params["stimulus"].stimulus_delta0 = delta0
        params["stimulus"].psi_ref_trab = psi_ref
        params["stimulus"].psi_ref_cort = psi_ref  # Keep cortical = trabecular
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
    """Run stimulus/mechanostat parameter sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="StimulusSweep")
    
    # Load stiff parameters as baseline
    params = load_default_params("stiff_params_box.json")
    box = params["box"]
    
    # Override for stimulus study:
    # - Longer simulation time to see density adaptation
    params["time"].total_time = 200.0  # 200 days
    params["time"].adaptive_dt = False
    
    # Baseline stimulus values from stiff_params_box.json
    kappa_base = float(params["stimulus"].stimulus_kappa)      # 0.3
    delta0_base = float(params["stimulus"].stimulus_delta0)    # 0.1
    psi_ref_base = float(params["stimulus"].psi_ref_trab)      # 5e-5 MPa
    
    # Define sweep parameters: 3 levels per parameter
    #
    # stimulus_kappa (saturation width):
    #   - Controls how sharply S transitions from lazy zone to saturation
    #   - Small κ: sharp transition (bistable-like behavior)
    #   - Large κ: gradual transition (smooth response)
    #   Baseline: 0.3
    #   Min: 0.1 (sharp - aggressive switching)
    #   Max: 1.0 (gradual - smooth response)
    #
    # stimulus_delta0 (lazy zone half-width):
    #   - Dead zone around homeostasis where S ≈ 0 (no remodeling)
    #   - Small δ₀: narrow lazy zone (sensitive to small deviations)
    #   - Large δ₀: wide lazy zone (tolerant to loading variations)
    #   Baseline: 0.1 (10% tolerance)
    #   Min: 0.02 (very sensitive - 2% tolerance)
    #   Max: 0.3 (tolerant - 30% tolerance)
    #
    # psi_ref_trab (reference SED):
    #   - Homeostatic target for SED (where S=0)
    #   - Smaller ψ_ref: bone perceives itself as "overloaded" → formation
    #   - Larger ψ_ref: bone perceives itself as "underloaded" → resorption
    #   Baseline: 5e-5 MPa
    #   Min: 1e-5 MPa (perceives high load → formation-dominated)
    #   Max: 2.5e-4 MPa (perceives low load → resorption-dominated)
    
    kappa_values = [0.1, kappa_base, 1.0]
    delta0_values = [0.02, delta0_base, 0.3]
    psi_ref_values = [1e-5, psi_ref_base, 2.5e-4]
    
    # Convert to sweep format
    sweep = ParameterSweep(
        params={
            "stimulus.stimulus_kappa": kappa_values,
            "stimulus.stimulus_delta0": delta0_values,
            "stimulus.psi_ref_trab": psi_ref_values,
        },
        base_output_dir=Path("results/stimulus_sweep"),
        metadata={
            "description": "Stimulus/mechanostat sweep: kappa × delta0 × psi_ref",
            "objective": "Study mechanostat sensitivity and remodeling response",
            "baseline_kappa": kappa_base,
            "baseline_delta0": delta0_base,
            "baseline_psi_ref": psi_ref_base,
            "kappa_values": kappa_values,
            "delta0_values": delta0_values,
            "psi_ref_values": psi_ref_values,
            "mesh_nx": box["nx"],
            "mesh_ny": box["ny"],
            "mesh_nz": box["nz"],
            "total_time_days": params["time"].total_time,
            "dt_days": params["time"].dt_initial,
            "notes": {
                "kappa": "Saturation width: small=sharp transition, large=gradual",
                "delta0": "Lazy zone: small=sensitive, large=tolerant",
                "psi_ref": "Reference SED: small=formation bias, large=resorption bias",
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
    runner = create_stimulus_runner(params, box)
    
    # Log sweep info
    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("STIMULUS/MECHANOSTAT PARAMETER SWEEP")
        logger.info("=" * 70)
        logger.info(f"Baseline from stiff_params_box.json")
        logger.info("-" * 70)
        logger.info(f"Baseline kappa = {kappa_base:.2f} (saturation width)")
        logger.info(f"Baseline delta0 = {delta0_base:.2f} (lazy zone)")
        logger.info(f"Baseline psi_ref = {psi_ref_base:.2e} MPa (reference SED)")
        logger.info("-" * 70)
        logger.info(f"kappa sweep: {[f'{v:.2f}' for v in kappa_values]}")
        logger.info(f"delta0 sweep: {[f'{v:.2f}' for v in delta0_values]}")
        logger.info(f"psi_ref sweep: {[f'{v:.2e}' for v in psi_ref_values]} MPa")
        logger.info(f"Total runs: {sweep.total_runs()} (3 × 3 × 3)")
        logger.info("-" * 70)
        logger.info(f"Mesh: {box['nx']}×{box['ny']}×{box['nz']} on {box['Lx']}×{box['Ly']}×{box['Lz']} mm")
        logger.info(f"Simulation time: {params['time'].total_time} days, dt={params['time'].dt_initial} days")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("Key outputs: checkpoint.bp (rho, S, psi, sigma)")
        logger.info("=" * 70)
    
    # Run sweep
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: python analysis/stimulus_analysis.py")


if __name__ == "__main__":
    main()
