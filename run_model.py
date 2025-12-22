"""Driver script for running bone remodeling simulation.

This is a thin entry point that:
1. Loads the mesh
2. Configures the simulation
3. Sets up loading scenarios
4. Runs the remodeling loop
"""

from mpi4py import MPI

from simulation.config import Config
from simulation.factory import DefaultSolverFactory
from simulation.febio_parser import FEBio2Dolfinx
from simulation.loader import Loader
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.paths import FemurPaths
from simulation.progress import ProgressReporter
from simulation.scenarios import get_standard_gait_cases


def main() -> None:
    """Run the bone remodeling simulation."""
    comm = MPI.COMM_WORLD

    # Load femur mesh from FEBio format
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    mesh = mdl.mesh_dolfinx

    # Create simulation configuration
    cfg = Config(domain=mesh, facet_tags=mdl.meshtags)

    # Reset log file on rank 0
    if comm.rank == 0:
        with open(cfg.log_file, "w") as f:
            f.write("")

    logger = get_logger(comm, name="Driver", log_file=cfg.log_file)

    # Create traction loader for the proximal surface (hip + muscles)
    loader = Loader(mesh, facet_tags=mdl.meshtags, load_tag=cfg.geometry.load_tag)

    # Load standard gait scenarios
    loading_cases = get_standard_gait_cases()

    if comm.rank == 0:
        logger.info(f"Defined {len(loading_cases)} loading case(s): {[c.name for c in loading_cases]}")

    # Create solver factory
    factory = DefaultSolverFactory(cfg)

    # Run simulation with progress reporting
    with Remodeller(cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
        with ProgressReporter(comm, cfg.time.total_time, cfg.solver.max_subiters) as reporter:
            remodeller.simulate(reporter=reporter)


if __name__ == "__main__":
    main()
