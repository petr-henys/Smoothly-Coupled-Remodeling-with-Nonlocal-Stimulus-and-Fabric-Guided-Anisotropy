"""Driver script for running bone remodeling simulation.

This is a thin entry point that:
1. Loads the mesh
2. Configures the simulation from default_params_femur.json
3. Sets up loading scenarios
4. Runs the remodeling loop
"""

from mpi4py import MPI

from femur import FEBio2Dolfinx, FemurPaths, get_standard_gait_cases, Loader
from simulation.factory import DefaultSolverFactory
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import ProgressReporter


def main() -> None:
    """Run the bone remodeling simulation."""
    comm = MPI.COMM_WORLD

    # Load femur mesh from FEBio format
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    mesh = mdl.mesh_dolfinx

    # Load parameters from JSON
    params = load_default_params("stiff_params_femur.json")
    
    # Modify params as needed (example):
    # params["time"].total_time = 200.0
    # params["density"].k_rho_form = 0.05

    # Create simulation configuration
    cfg = create_config(mesh, mdl.meshtags, params)

    # Reset log file on rank 0
    if comm.rank == 0:
        with open(cfg.log_file, "w") as f:
            f.write("")

    logger = get_logger(comm, name="Driver", log_file=cfg.log_file)

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


if __name__ == "__main__":
    main()
