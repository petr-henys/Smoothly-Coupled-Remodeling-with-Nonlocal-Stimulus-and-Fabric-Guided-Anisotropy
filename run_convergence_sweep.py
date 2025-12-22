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

from dataclasses import dataclass
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
from simulation.checkpoint import CheckpointStorage, HAS_ADIOS4DOLFINX
from simulation.config import Config
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import (
    GeometryParams,
    OutputParams,
    SolverParams,
    TimeParams,
)
from simulation.progress import SweepProgressReporter


@dataclass
class ConvergenceConfig:
    """Configuration for convergence sweep.
    
    Non-swept parameters for the box model.
    """
    # Box dimensions [mm]
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 20.0
    
    # Loading
    pressure: float = 1.0
    
    # Final time for convergence comparison
    # All runs simulate to same final time
    total_time: float = 50.0
    
    # Solver settings (relaxed for faster convergence)
    coupling_tol: float = 1e-3
    max_subiters: int = 30


def create_convergence_runner(
    cfg: ConvergenceConfig,
    N_list: list[int],
) -> SimulationRunner:
    """Create a runner that uses N from param_point to set mesh resolution.
    
    Args:
        cfg: Base configuration.
        N_list: List of N values to map to mesh resolution.
    
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
        
        # Extract N and dt from param_point
        N = int(param_point["N"])
        dt_days = float(param_point["dt_days"])
        
        # Update reporter with correct total_time for this run
        if reporter is not None:
            # Reset remodeling bar with correct total for this run
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=cfg.total_time)
        
        # Create mesh with resolution N
        geometry = BoxGeometry(
            Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
            nx=N, ny=N, nz=int(N * cfg.Lz / cfg.Lx),  # Scale nz with aspect ratio
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()
        
        # Create config with proper output path
        sim_cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            time=TimeParams(
                total_time=cfg.total_time,
                dt_initial=dt_days,
                adaptive_dt=False,  # Fixed dt for convergence study
            ),
            solver=SolverParams(
                coupling_tol=cfg.coupling_tol,
                max_subiters=cfg.max_subiters,
                accel_type="anderson",
            ),
            output=OutputParams(results_dir=str(output_path)),
            geometry=GeometryParams(
                fix_tag=BoxMeshBuilder.TAG_BOTTOM,
                load_tag=BoxMeshBuilder.TAG_TOP,
            ),
        )
        
        # Store N in config for later reference (e.g., in analysis)
        # This is a simple way to persist the mesh resolution
        sim_cfg._mesh_resolution = N
        
        # Create loader and loading cases
        loader = BoxLoader(domain, facet_tags, load_tag=BoxMeshBuilder.TAG_TOP)
        loading_cases = [get_parabolic_pressure_case(
            pressure=cfg.pressure,
            gradient_axis=0,
            center_factor=2.0,
            edge_factor=0.3,
            box_extent=(0.0, cfg.Lx),
            name="parabolic_compression",
        )]
        
        # Create factory and run
        factory = BoxSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            # Run simulation with unified sweep reporter
            remodeller.simulate(reporter=reporter)
            
            # Write final checkpoint for convergence analysis
            if HAS_ADIOS4DOLFINX:
                checkpoint = CheckpointStorage(sim_cfg)
                
                state_fields = remodeller.registry.state_fields
                rho = state_fields.get("rho")
                S = state_fields.get("S")
                u = remodeller.driver.mech.u
                
                final_time = sim_cfg.time.total_time
                if rho is not None:
                    checkpoint.write_function(rho, final_time)
                if S is not None:
                    checkpoint.write_function(S, final_time)
                if u is not None:
                    checkpoint.write_function(u, final_time)
                checkpoint.close()
    
    return runner


def main() -> None:
    """Run convergence sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="ConvergenceSweep")
    
    # Define sweep parameters
    N_values = [5, 8, 10, 15, 20]  # Mesh resolutions
    dt_values = [10.0, 5.0, 2.0, 1.0]  # Timesteps [days]
    
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
    
    # Configuration for non-swept parameters
    cfg = ConvergenceConfig(
        total_time=50.0,  # Short runs for convergence study
        coupling_tol=1e-3,
    )
    
    # Create runner
    runner = create_convergence_runner(cfg, N_values)
    
    # Run sweep
    if comm.rank == 0:
        logger.info("=" * 60)
        logger.info("CONVERGENCE SWEEP")
        logger.info("=" * 60)
        logger.info(f"N values: {N_values}")
        logger.info(f"dt values: {dt_values}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"Output: {sweep.base_output_dir}")
        if HAS_ADIOS4DOLFINX:
            logger.info("Checkpointing: adios4dolfinx (recommended)")
        else:
            logger.info("Checkpointing: DISABLED (install adios4dolfinx)")
        logger.info("=" * 60)
    
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info(f"Run analysis with: python analysis/convergence_errors.py")


if __name__ == "__main__":
    main()
