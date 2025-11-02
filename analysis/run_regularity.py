"""Run regularity parameter sweep (data generation).

Sweeps key smoothing parameters that impact solution regularity and
nonlinear convergence. Outputs telemetry CSVs and NPZ snapshots per run.

Usage:
  python run_regularity.py
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


BASE_DIR = Path("results/regularity_sweep")


def run_regularity(param_point: Dict[str, Any], output_path: Path, comm: MPI.Comm) -> None:
    """Execute a single simulation for a regularity parameter point.

    Expected keys in param_point:
      - N, dt_days, smooth_eps, ell_dim, kappaS_dim, beta_par_dim, beta_perp_dim, total_time_days
    """
    N = int(param_point["N"])
    dt_days = float(param_point["dt_days"])
    smooth_eps = float(param_point["smooth_eps"])
    ell_dim = float(param_point["ell_dim"])
    kappaS_dim = float(param_point["kappaS_dim"])
    beta_par_dim = float(param_point["beta_par_dim"])
    beta_perp_dim = float(param_point["beta_perp_dim"])
    total_time_days = float(param_point["total_time_days"])

    # Mesh
    domain = mesh.create_unit_cube(
        comm, N, N, N,
        ghost_mode=mesh.GhostMode.shared_facet
    )
    facet_tags = build_facetag(domain)

    # Config (Anderson default; tight tolerance to highlight convergence trends)
    cfg = Config(
        domain=domain,
        facet_tags=facet_tags,
        results_dir=str(output_path),
        enable_telemetry=True,
        verbose=False,
        coupling_tol=1e-8,
        max_subiters=100,
        smooth_eps=smooth_eps,
        accel_type="anderson",
        m=8,
        beta=1.0,
        restart_on_cond=1e12,
        saving_interval=5,
        coupling_each_iter=False,
        coupling_eps=1e-3,
        ell_dim=ell_dim,
        kappaS_dim=kappaS_dim,
        beta_par_dim=beta_par_dim,
        beta_perp_dim=beta_perp_dim,
    )

    with Remodeller(cfg) as remodeller:
        remodeller.simulate(dt=dt_days, total_time=total_time_days)
        # NPZ snapshots
        save_function_npz(remodeller.u, output_path / "u.npz", comm)
        save_function_npz(remodeller.rho, output_path / "rho.npz", comm)
        save_function_npz(remodeller.S, output_path / "S.npz", comm)
        save_function_npz(remodeller.A, output_path / "A.npz", comm)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    sweep = ParameterSweep(
        params={
            "N": [36],
            "dt_days": [25.0],
            "smooth_eps": [1e-8, 1e-5, 1e-3],
            "ell_dim": [0.1, 0.2, 0.3],
            "kappaS_dim": [1e-5, 1e-4],
            "beta_par_dim": [1e-6, 1e-5, 1e-4],
            "beta_perp_dim": [1e-6, 1e-5, 1e-4],
            "total_time_days": [500.0],
        },
        base_output_dir=BASE_DIR,
        metadata={"analysis": "regularity", "description": "smoothing + physics sweep"},
    )

    parametrizer = Parametrizer(sweep, run_regularity, comm)
    parametrizer.run()

