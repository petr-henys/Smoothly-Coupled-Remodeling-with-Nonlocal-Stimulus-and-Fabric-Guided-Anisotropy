"""Run diffusion regularization sweep: varying D_rho, stimulus_D, and fabric_D.

This script runs simulations over a grid of diffusion parameter values and saves
checkpoints for subsequent analysis (checkerboarding, solver performance).

Baseline diffusion values are derived from the regularization length formula:
    ℓ = α·h  with α = 1.5 (target 1.5× element size)
    D_S = ℓ²/τ_S
    D_A = ℓ²/τ_A  
    D_ρ = ℓ²·(Δt⁻¹ + r_char)  where r_char = max(k_form/ρ_max, k_res/ρ_min)

See manuscript/sections/parameter_governance.tex for derivation.

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


def compute_baseline_diffusivities(params: dict, box: dict) -> dict[str, float]:
    """Compute baseline diffusion coefficients from regularization length formula.
    
    Uses ℓ = α·h with α = 1.5 (target 1.5× element size).
    
    Formulas from manuscript (parameter_governance.tex):
        D_S = ℓ²/τ_S
        D_A = ℓ²/τ_A  
        D_ρ = ℓ²·(Δt⁻¹ + r_char)  where r_char = max(k_form/ρ_max, k_res/ρ_min)
    
    Args:
        params: Parameter dict with density, stimulus, fabric, time groups.
        box: Box geometry parameters (Lx, nx, etc.)
    
    Returns:
        Dict with baseline D_rho, stimulus_D, fabric_D values.
    """
    # Characteristic element size h (assuming uniform mesh)
    h = box["Lx"] / box["nx"]  # mm
    
    # Target regularization length: ℓ = 1.5·h
    alpha = 1.5
    ell = alpha * h  # mm
    ell_sq = ell ** 2  # mm²
    
    # Time constants
    tau_S = float(params["stimulus"].stimulus_tau)  # days
    tau_A = float(params["fabric"].fabric_tau)      # days
    dt = float(params["time"].dt_initial)           # days
    
    # Reaction characteristic rate
    k_form = float(params["density"].k_rho_form)
    k_res = float(params["density"].k_rho_resorb)
    rho_max = float(params["density"].rho_max)
    rho_min = float(params["density"].rho_min)
    r_char = max(k_form / rho_max, k_res / rho_min)  # day⁻¹
    
    # Compute baseline diffusivities
    D_S_baseline = ell_sq / tau_S                    # mm²/day
    D_A_baseline = ell_sq / tau_A                    # mm²/day
    D_rho_baseline = ell_sq * (1.0 / dt + r_char)   # mm²/day
    
    return {
        "D_rho": D_rho_baseline,
        "stimulus_D": D_S_baseline,
        "fabric_D": D_A_baseline,
        "ell": ell,
        "h": h,
        "r_char": r_char,
    }


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
    # - Shorter simulation time (sufficient to show checkerboarding effects)
    params["time"].total_time = 100.0  # 100 days
    params["time"].adaptive_dt = False
    # Keep default dt_initial from stiff_params (25 days)
    
    # Compute baseline diffusivities from regularization length formula
    baseline = compute_baseline_diffusivities(params, box)
    
    D_rho_base = baseline["D_rho"]
    stimulus_D_base = baseline["stimulus_D"]
    fabric_D_base = baseline["fabric_D"]
    
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
            "regularization_length_mm": baseline["ell"],
            "element_size_mm": baseline["h"],
            "alpha": 1.5,
            "r_char": baseline["r_char"],
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
        logger.info(f"Baseline from stiff_params_box.json with ℓ = 1.5·h")
        logger.info(f"Element size h = {baseline['h']:.3f} mm")
        logger.info(f"Regularization length ℓ = {baseline['ell']:.3f} mm")
        logger.info(f"Reaction rate r_char = {baseline['r_char']:.4f} day⁻¹")
        logger.info("-" * 70)
        logger.info(f"Baseline D_rho = {D_rho_base:.4f} mm²/day")
        logger.info(f"Baseline stimulus_D = {stimulus_D_base:.4f} mm²/day")
        logger.info(f"Baseline fabric_D = {fabric_D_base:.4f} mm²/day")
        logger.info("-" * 70)
        logger.info(f"D_rho sweep: {[f'{v:.4f}' for v in D_rho_values]}")
        logger.info(f"stimulus_D sweep: {[f'{v:.4f}' for v in stimulus_D_values]}")
        logger.info(f"fabric_D sweep: {[f'{v:.4f}' for v in fabric_D_values]}")
        logger.info(f"Total runs: {sweep.total_runs()} (3 × 3 × 3)")
        logger.info("-" * 70)
        logger.info(f"Mesh: {box['nx']}×{box['ny']}×{box['nz']} on {box['Lx']}×{box['Ly']}×{box['Lz']} mm")
        logger.info(f"Simulation time: {params['time'].total_time} days, dt={params['time'].dt_initial} days")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("Checkpointing: adios4dolfinx")
        logger.info("=" * 70)
    
    # Run sweep
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Run analysis with: python analysis/diffusion_analysis.py")


if __name__ == "__main__":
    main()
