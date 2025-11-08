"""Run Anderson acceleration parameter sweep (data generation).

Sweeps Anderson acceleration parameters (accel_type, m, beta) across
different temporal resolutions. Outputs telemetry CSVs and NPZ snapshots per run.

Usage:
  mpirun -np 8 python3 analysis/run_anderson.py
"""

from __future__ import annotations

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
from analysis.utils import save_function_npz


BASE_DIR = Path("results/anderson_sweep")


def run_anderson(param_point: Dict[str, Any], output_path: Path, comm: MPI.Comm) -> None:
    """Execute a single simulation for an Anderson parameter point.

    Expected keys in param_point:
      - N, dt_days, accel_type, m, beta, total_time_days, coupling_tol, max_subiters
    """
    N = 24
    total_time_days = 1000.0
    coupling_tol = 1e-8
    max_subiters = 100
    dt_days = float(param_point["dt_days"])
    accel_type = str(param_point["accel_type"])


    # Mesh
    domain = mesh.create_unit_cube(
        comm, N, N, N,
        ghost_mode=mesh.GhostMode.shared_facet
    )
    facet_tags = build_facetag(domain)

    # Config (tight tolerance for convergence comparison)
    cfg = Config(
        domain=domain,
        facet_tags=facet_tags,
        results_dir=str(output_path),
        enable_telemetry=True,
        verbose=False,
        coupling_tol=coupling_tol,
        max_subiters=max_subiters,
        accel_type=accel_type,
        restart_on_cond=1e12,
        saving_interval=5,
        coupling_each_iter=False,  # Track coupling strength
        coupling_eps=1e-3,
    )

    # Run simulation (telemetry saves all metrics automatically)
    with Remodeller(cfg) as remodeller:
        remodeller.simulate(dt=dt_days, total_time=total_time_days)
        
        # Save final field states as NPZ
        if comm.rank == 0:
            print(f"Saving NPZ for {accel_type} dt={dt_days}")
        
        save_function_npz(remodeller.u, output_path / "u.npz", comm)
        save_function_npz(remodeller.rho, output_path / "rho.npz", comm)
        save_function_npz(remodeller.S, output_path / "S.npz", comm)
        save_function_npz(remodeller.A, output_path / "A.npz", comm)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    sweep = ParameterSweep(
        params={
            "dt_days": [25.0, 50.0, 100.0],
            "accel_type": ["picard", "anderson"],
        },
        base_output_dir=BASE_DIR,
        metadata={"analysis": "anderson", "description": "Anderson vs Picard comparison"},
    )

    parametrizer = Parametrizer(sweep, run_anderson, comm)
    parametrizer.run()
