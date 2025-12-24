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

from simulation.box_factory import BoxSolverFactory
from simulation.box_loader import BoxLoader, GradientType
from simulation.box_mesh import BoxGeometry, BoxMeshBuilder
from simulation.box_scenarios import get_parabolic_pressure_case
from simulation.config import Config
from simulation.logger import get_logger
from simulation.model import Remodeller
from simulation.params import (
    DensityParams,
    FabricParams,
    GeometryParams,
    MaterialParams,
    OutputParams,
    SolverParams,
    StimulusParams,
    TimeParams,
)
from simulation.progress import ProgressReporter
from simulation.storage import UnifiedStorage


# =============================================================================
# SIMULATION PARAMETERS - Edit these values to configure the simulation
# =============================================================================

# Geometry [mm]
LX = 10.0       # Box length in x
LY = 10.0       # Box length in y
LZ = 30.0       # Box height in z
NX = 10         # Number of elements in x
NY = 10         # Number of elements in y
NZ = 30         # Number of elements in z

# Loading [MPa]
# Using parabolic pressure: peak at center, low at edges
# This creates interesting spatial pattern - densification at center, resorption at edges
PRESSURE = 1.0              # Base pressure magnitude
GRADIENT_AXIS = 0           # Gradient along x-axis
GRADIENT_TYPE = "parabolic" # "linear", "parabolic", or "bending"
CENTER_FACTOR = 2.0         # Factor at center (peak for parabolic)
EDGE_FACTOR = 0.3           # Factor at edges (min for parabolic)

# Time stepping [days]
TOTAL_TIME = 10.    # Total simulation time
DT_INITIAL = 10    # Fixed timestep
ADAPTIVE_DT = False   # Disable adaptive (use fixed dt for speed)
DT_MIN = 1e-2          # Minimum timestep
DT_MAX = 50.0         # Maximum timestep

# Material properties [MPa, g/cm³]
RHO0 = 1.0      # Initial density [g/cm³]
E0 = 7500.0     # Reference Young's modulus [MPa]

# Stimulus parameters
# PSI_REF sets the reference specific energy (psi/rho); tune per load magnitude and material scaling.
PSI_REF = 5e-5             # Reference SED [MPa]
STIMULUS_TAU = 50.0          # Stimulus time constant [days]
STIMULUS_DELTA0 = 0.20       # Lazy zone half-width (dimensionless)
STIMULUS_D = 0.01            # Stimulus diffusion [mm²/day] - low for sharp gradients

# Density evolution parameters
K_RHO_FORM = 0.05       # Formation rate [g/cm³/day]
K_RHO_RESORB = 0.05     # Resorption rate [g/cm³/day]
D_RHO = 0.01             # Density diffusion [mm²/day] - low to preserve structure

# Fabric parameters
FABRIC_D = 0.01           # Fabric diffusion [mm²/day] - moderate

# Solver settings
ACCEL_TYPE = "anderson"  # Fixed-point acceleration: "anderson" or "picard"
COUPLING_TOL = 1e-4      # Coupling tolerance (relaxed for faster demos)
MAX_SUBITERS = 50        # Maximum sub-iterations per timestep

# Output
OUTPUT_DIR = ".results_box"  # Output directory
SAVE_INTERVAL = 1            # Save every N steps

# =============================================================================


def main() -> None:
    """Run the box model bone remodeling simulation."""
    comm = MPI.COMM_WORLD

    # Create box mesh with tagged boundaries
    geometry = BoxGeometry(
        Lx=LX, Ly=LY, Lz=LZ,
        nx=NX, ny=NY, nz=NZ,
    )
    builder = BoxMeshBuilder(geometry, comm)
    domain, facet_tags = builder.build()

    # Create simulation configuration
    cfg = Config(
        domain=domain,
        facet_tags=facet_tags,
        material=MaterialParams(
            E0=E0,
            nu0=0.3,
            n_trab=2.0,
            n_cort=1.3,
            stiff_pE=1.,
            stiff_pG=1.,
        ),
        density=DensityParams(
            rho0=RHO0,
            rho_min=0.1,
            rho_max=2.0,
            k_rho_form=K_RHO_FORM,
            k_rho_resorb=K_RHO_RESORB,
            D_rho=D_RHO,
        ),
        stimulus=StimulusParams(
            psi_ref=PSI_REF,
            stimulus_tau=STIMULUS_TAU,
            stimulus_D=STIMULUS_D,
            stimulus_delta0=STIMULUS_DELTA0,
        ),
        fabric=FabricParams(
            fabric_D=FABRIC_D
        ),
        solver=SolverParams(
            accel_type=ACCEL_TYPE,
            coupling_tol=COUPLING_TOL,
            max_subiters=MAX_SUBITERS,
        ),
        time=TimeParams(
            total_time=TOTAL_TIME,
            dt_initial=DT_INITIAL,
            adaptive_dt=ADAPTIVE_DT,
            dt_min=DT_MIN,
            dt_max=DT_MAX,
        ),
        output=OutputParams(
            results_dir=OUTPUT_DIR,
            saving_interval=SAVE_INTERVAL,
        ),
        geometry=GeometryParams(
            fix_tag=BoxMeshBuilder.TAG_BOTTOM,  # z=0 fixed
            load_tag=BoxMeshBuilder.TAG_TOP,    # z=Lz loaded
        ),
    )

    # Reset log file on rank 0
    if comm.rank == 0:
        log_path = Path(cfg.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.log_file, "w") as f:
            f.write("")

    logger = get_logger(comm, name="BoxModel", log_file=cfg.log_file)

    # Log geometry info
    if comm.rank == 0:
        logger.info(f"Box geometry: {LX} x {LY} x {LZ} mm")
        logger.info(f"Mesh: {NX} x {NY} x {NZ} elements")
        logger.info(f"Total nodes: {domain.geometry.x.shape[0]}")
        logger.info(f"Adaptive time stepping: {ADAPTIVE_DT}")

    # Create pressure loader
    loader = BoxLoader(domain, facet_tags, load_tag=BoxMeshBuilder.TAG_TOP)

    # Create parabolic loading case (non-uniform to drive interesting adaptation)
    # Parabolic: peak pressure at center, low at edges -> "dome" of densification
    loading_cases = [get_parabolic_pressure_case(
        pressure=PRESSURE,
        gradient_axis=GRADIENT_AXIS,
        center_factor=CENTER_FACTOR,
        edge_factor=EDGE_FACTOR,
        box_extent=(0.0, LX),
        name="parabolic_compression",
    )]
    if comm.rank == 0:
        logger.info(f"Loading: parabolic pressure, base = {PRESSURE} MPa")
        logger.info(f"  Along axis {GRADIENT_AXIS}: edge={EDGE_FACTOR*PRESSURE:.2f}, center={CENTER_FACTOR*PRESSURE:.2f} MPa")

    # Create solver factory
    factory = BoxSolverFactory(cfg)

    # Precompute loading cases and save traction field
    loader.precompute_loading_cases(loading_cases)
    loader.set_loading_case(loading_cases[0].name)
    
    # Save traction field for visualization
    traction_storage = UnifiedStorage(cfg)
    traction_storage.fields.register("traction", [loader.traction], filename="traction.bp")
    traction_storage.fields.write("traction", 0.0)
    traction_storage.close()
    if comm.rank == 0:
        logger.info(f"Saved traction field to {OUTPUT_DIR}/traction.bp")

    # Run simulation with progress reporting
    with Remodeller(cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
        with ProgressReporter(comm, cfg.time.total_time, cfg.solver.max_subiters) as reporter:
            remodeller.simulate(reporter=reporter)

    if comm.rank == 0:
        logger.info(f"Simulation completed. Results in: {cfg.output.results_dir}")


if __name__ == "__main__":
    main()
