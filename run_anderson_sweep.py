"""Run Anderson vs Picard acceleration comparison sweep.

This script demonstrates the necessity of Anderson acceleration by comparing
convergence behavior between Anderson and Picard (no acceleration) across
different timesteps.

Usage:
    mpirun -n 4 python run_anderson_sweep.py

Outputs:
    results/anderson_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── steps.csv           <- Per-timestep metrics (subiters, errors, etc.)
    │   ├── subiterations.csv   <- Per-subiteration details
    │   └── fields.bp/          <- Visualization output
    ├── <hash2>/
    │   └── ...

Post-processing:
    python analysis/anderson_performance_plot.py
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
from simulation.config import Config
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import (
    DensityParams,
    GeometryParams,
    OutputParams,
    SolverParams,
    StimulusParams,
    TimeParams,
)
from simulation.progress import SweepProgressReporter


@dataclass
class AndersonSweepConfig:
    """Configuration for Anderson acceleration sweep.
    
    Non-swept parameters for the box model.
    Parameters chosen to create an ill-conditioned problem where
    Anderson acceleration provides clear benefits over Picard.
    """
    # Box dimensions [mm]
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 20.0
    
    # Mesh resolution (fixed for acceleration comparison)
    N: int = 24
    
    # Loading - higher pressure for stronger coupling
    pressure: float = 5.0
    
    # Final time - long enough to see acceleration benefits
    total_time: float = 200.0

    # Solver settings - tight tolerance to stress the solver
    coupling_tol: float = 1e-6
    max_subiters: int = 100  # High limit to observe convergence behavior

    # Loading smoothness
    load_edge_factor: float = 0.0
    
    # Reaction kinetics - higher rates for stronger coupling
    k_rho_form: float = 0.02      # 10x higher than default
    k_rho_resorb: float = 0.02    # 10x higher than default
    
    # Stimulus dynamics - faster response for tighter coupling
    stimulus_tau: float = 5.0     # 5x faster than default (25.0)


def create_anderson_runner(cfg: AndersonSweepConfig) -> SimulationRunner:
    """Create a runner for Anderson vs Picard comparison.
    
    Args:
        cfg: Base configuration.
    
    Returns:
        Runner function for Parametrizer.
    """
    
    def runner(
        param_point: dict[str, ParamValue],
        output_path: Path,
        comm: MPI.Comm,
        reporter: SweepProgressReporter | None = None,
    ) -> None:
        """Run single simulation with specified acceleration settings."""
        
        # Extract parameters
        dt_days = float(param_point["dt_days"])
        accel_type = str(param_point["accel_type"])
        m = int(param_point.get("m", 5))
        beta = float(param_point.get("beta", 1.0))
        
        # Update reporter with correct total_time
        if reporter is not None:
            if reporter.progress is not None and reporter.main_task_id is not None:
                reporter.progress.reset(reporter.main_task_id)
                reporter.progress.update(reporter.main_task_id, total=cfg.total_time)
        
        # Create mesh with fixed resolution
        N = cfg.N
        geometry = BoxGeometry(
            Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz,
            nx=N, ny=N, nz=int(N * cfg.Lz / cfg.Lx),
        )
        builder = BoxMeshBuilder(geometry, comm)
        domain, facet_tags = builder.build()
        
        # Create config with ill-conditioned parameters
        sim_cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            density=DensityParams(
                k_rho_form=cfg.k_rho_form,
                k_rho_resorb=cfg.k_rho_resorb,
            ),
            stimulus=StimulusParams(
                stimulus_tau=cfg.stimulus_tau,
            ),
            time=TimeParams(
                total_time=cfg.total_time,
                dt_initial=dt_days,
                adaptive_dt=False,
            ),
            solver=SolverParams(
                coupling_tol=cfg.coupling_tol,
                max_subiters=cfg.max_subiters,
                accel_type=accel_type,
                m=m,
                beta=beta,
                # No safeguard for fair comparison on well-conditioned problems.
                # Safeguard is useful for ill-conditioned problems where AA may diverge.
                safeguard=False,
            ),
            output=OutputParams(results_dir=str(output_path)),
            geometry=GeometryParams(
                fix_tag=BoxMeshBuilder.TAG_BOTTOM,
                load_tag=BoxMeshBuilder.TAG_TOP,
            ),
        )
        
        # Create loader and loading cases
        loader = BoxLoader(domain, facet_tags, load_tag=BoxMeshBuilder.TAG_TOP)
        loading_cases = [get_parabolic_pressure_case(
            pressure=cfg.pressure,
            gradient_axis=0,
            center_factor=2.0,
            edge_factor=cfg.load_edge_factor,
            box_extent=(0.0, cfg.Lx),
            name="parabolic_compression",
        )]
        
        # Create factory and run
        factory = BoxSolverFactory(sim_cfg)
        
        with Remodeller(sim_cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            remodeller.simulate(reporter=reporter)
    
    return runner


def main() -> None:
    """Run Anderson vs Picard comparison sweep."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AndersonSweep")
    
    # Define sweep parameters
    # Compare Anderson vs Picard across different timesteps
    accel_types = ["picard", "anderson"]
    
    # Timesteps [days] - larger steps for ill-conditioned problem
    # With fast kinetics (tau=5, k=0.02), larger dt creates more nonlinearity
    dt_values = [50, 25, 10]
    
    # Anderson history sizes (only used when accel_type="anderson")
    m_values = [5]  # Default history size
    
    # Mixing parameter
    beta_values = [1.0]  # Full step (no damping)
    
    sweep = ParameterSweep(
        params={
            "accel_type": accel_types,
            "dt_days": dt_values,
            "m": m_values,
            "beta": beta_values,
        },
        base_output_dir=Path("results/anderson_sweep"),
        metadata={
            "description": "Anderson vs Picard acceleration comparison",
            "accel_types": accel_types,
            "dt_values": dt_values,
            "m_values": m_values,
            "beta_values": beta_values,
        },
    )
    
    # Configuration for non-swept parameters
    cfg = AndersonSweepConfig(
        N=32,
        total_time=100.0,
        coupling_tol=1e-6,
        max_subiters=100,
    )
    
    # Create runner
    runner = create_anderson_runner(cfg)
    
    # Run sweep
    if comm.rank == 0:
        logger.info("=" * 60)
        logger.info("ANDERSON VS PICARD ACCELERATION SWEEP")
        logger.info("=" * 60)
        logger.info(f"Acceleration types: {accel_types}")
        logger.info(f"dt values: {dt_values}")
        logger.info(f"Anderson m values: {m_values}")
        logger.info(f"Mesh resolution: N={cfg.N}")
        logger.info(f"Total runs: {sweep.total_runs()}")
        logger.info(f"Output: {sweep.base_output_dir}")
        logger.info("=" * 60)
    
    parametrizer = Parametrizer(sweep, runner, comm)
    parametrizer.run()
    
    if comm.rank == 0:
        logger.info("Sweep complete!")
        logger.info("Analyze with: python analysis/anderson_performance_plot.py")


if __name__ == "__main__":
    main()
