"""Driver script for running bone remodeling simulation on a box mesh.

This is a simplified entry point for box-shaped specimens with:
- Bottom surface fixed (z=0)
- Top surface loaded with pressure (uniform or graded)

Useful for:
- Testing remodeling algorithms
- Parameter studies
- Trabecular bone specimen simulations
- Validation against analytical solutions

Usage:
    mpirun -n 4 python run_box_model.py
"""

from __future__ import annotations

from pathlib import Path

from mpi4py import MPI

from box import (
    BoxSolverFactory,
    BoxLoader,
    BoxGeometry,
    BoxMeshBuilder,
    get_parabolic_pressure_case,
)
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params
from simulation.progress import ProgressReporter
from simulation.storage import UnifiedStorage

def main() -> None:
    """Run the box model bone remodeling simulation."""
    comm = MPI.COMM_WORLD

    # Load parameters from JSON
    params = load_default_params("default_params_box.json")
    box = params["box"]
    
    # Modify params as needed
    params["output"].results_dir = ".results_box"
    params["output"].saving_interval = 1
    params["geometry"].fix_tag = BoxMeshBuilder.TAG_BOTTOM
    params["geometry"].load_tag = BoxMeshBuilder.TAG_TOP

    # Create box mesh with tagged boundaries
    geometry = BoxGeometry(
        Lx=box["Lx"], Ly=box["Ly"], Lz=box["Lz"],
        nx=box["nx"], ny=box["ny"], nz=box["nz"],
    )
    builder = BoxMeshBuilder(geometry, comm)
    domain, facet_tags = builder.build()

    # Create simulation configuration
    cfg = create_config(domain, facet_tags, params)

    # Reset log file on rank 0
    if comm.rank == 0:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.log_file, "w") as f:
            f.write("")

    logger = get_logger(comm, name="BoxModel", log_file=cfg.log_file)

    # Log geometry info
    if comm.rank == 0:
        logger.info(f"Box geometry: {box['Lx']} x {box['Ly']} x {box['Lz']} mm")
        logger.info(f"Mesh: {box['nx']} x {box['ny']} x {box['nz']} elements")
        logger.info(f"Total nodes: {domain.geometry.x.shape[0]}")
        logger.info(f"Adaptive time stepping: {cfg.time.adaptive_dt}")

    # Create parabolic loading case (non-uniform to drive interesting adaptation)
    loading_cases = [get_parabolic_pressure_case(
        pressure=box["pressure"],
        load_tag=BoxMeshBuilder.TAG_TOP,
        gradient_axis=box["gradient_axis"],
        center_factor=box["center_factor"],
        edge_factor=box["edge_factor"],
        box_extent=(0.0, box["Lx"]),
        name="parabolic_compression",
    )]
    if comm.rank == 0:
        logger.info(f"Loading: parabolic pressure, base = {box['pressure']} MPa")
        logger.info(f"  Along axis {box['gradient_axis']}: edge={box['edge_factor']*box['pressure']:.2f}, center={box['center_factor']*box['pressure']:.2f} MPa")

    # Create pressure loader - tags are automatically extracted from loading cases
    loader = BoxLoader(domain, facet_tags, loading_cases=loading_cases)

    # Create solver factory
    factory = BoxSolverFactory(cfg)

    # Save traction field for visualization
    loader.set_loading_case(loading_cases[0].name)
    traction_storage = UnifiedStorage(cfg)
    traction_storage.fields.register("traction", [loader.traction], filename="traction.bp")
    traction_storage.fields.write("traction", 0.0)
    traction_storage.close()
    if comm.rank == 0:
        logger.info(f"Saved traction field to {cfg.output.results_dir}/traction.bp")

    # Run simulation with progress reporting
    with Remodeller(cfg, loader=loader, factory=factory) as remodeller:
        with ProgressReporter(comm, cfg.time.total_time, cfg.solver.max_subiters) as reporter:
            remodeller.simulate(reporter=reporter)

    if comm.rank == 0:
        logger.info(f"Simulation completed. Results in: {cfg.output.results_dir}")

if __name__ == "__main__":
    main()
