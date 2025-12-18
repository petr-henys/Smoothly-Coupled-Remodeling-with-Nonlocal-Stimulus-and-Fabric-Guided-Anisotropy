"""Run full Remodeller convergence sweep (N × dt)."""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mpi4py import MPI
from dolfinx import mesh

from simulation.config import Config
from simulation.model import Remodeller
from simulation.utils import build_facetag
from parametrizer import Parametrizer, ParameterSweep
from analysis.analysis_utils import save_function_npz


def run_remodeller(
    param_point: Dict[str, Any],
    output_path: Path,
    comm: MPI.Comm
) -> None:
    """Execute full Remodeller simulation with given parameters.
    
    Remodeller has full telemetry enabled and saves all metrics per iteration/substep
    to telemetry CSV files. Field snapshots saved as NPZ for postprocessing.
    
    Args:
        param_point: Parameter dictionary (must contain all required params: 
                    'N', 'dt_days')
        output_path: Output directory for this run
        comm: MPI communicator
    """
    N = param_point["N"]
    dt_days = param_point["dt_days"]
    
    # Create mesh
    domain = mesh.create_unit_cube(
        comm, N, N, N,
        ghost_mode=mesh.GhostMode.shared_facet
    )
    facet_tags = build_facetag(domain)
    
    # Config
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            n_trab=2.0,
            n_cort=1.2,
            rho_trab_max=0.8,
            rho_cort_min=1.2,
            results_dir=str(output_path),
            verbose=False,
            coupling_tol=1e-8,
            max_subiters=100,
        )
    
    # Run simulation (telemetry saves all metrics automatically)
    with Remodeller(cfg) as remodeller:
        remodeller.simulate(dt=dt_days, total_time=1000.0)
        
        # Save final field states as NPZ for convergence analysis
        if comm.rank == 0:
            print(f"Saving NPZ field snapshots for N={N}, dt={dt_days}")
        
        save_function_npz(remodeller.u, output_path / "u.npz", comm)
        save_function_npz(remodeller.rho, output_path / "rho.npz", comm)
        save_function_npz(remodeller.S, output_path / "S.npz", comm)
        save_function_npz(remodeller.A, output_path / "A.npz", comm)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    
    base_dir = "results/convergence_sweep"
    
    sweep = ParameterSweep(
        params={
            "N": [16, 24, 36, 54],
            "dt_days": [6.25, 12.5, 25.0, 50.0]
        },
        base_output_dir=base_dir,
    )
    
    parametrizer = Parametrizer(sweep, run_remodeller, comm)
    parametrizer.run()
