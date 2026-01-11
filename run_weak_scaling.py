"""MPI scaling benchmark for the Bone Remodeling Solver (Femur Stiff Configuration).

Measures performance (wall time per step) of the full coupled multiphysics solver
on the stiff femur configuration. Can be used for:
- Strong scaling: Increase ranks with fixed mesh (default).

Usage:
    mpirun -n 1 python run_weak_scaling.py
    mpirun -n 2 python run_weak_scaling.py
    mpirun -n 4 python run_weak_scaling.py

Outputs:
    results/scaling_stiff/scaling.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from femur import FEBio2Dolfinx, FemurPaths, get_standard_gait_cases, Loader
from simulation.factory import DefaultSolverFactory
from simulation.model import Remodeller
from simulation.params import create_config, load_default_params


@dataclass(frozen=True)
class BenchmarkResult:
    timestamp_utc: str
    config_file: str
    ranks: int
    refine_level: int
    cells_global: int
    dofs_dens: int
    dofs_mech: int
    dofs_stim: int
    dofs_fabric: int
    dofs_total: int
    steps_measured: int
    t_avg_step_s: float
    t_setup_s: float
    t_total_s: float
    min_step_s: float
    max_step_s: float


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_conda_toolchain_on_path() -> None:
    """Ensure FFCx JIT finds a C compiler."""
    conda_bin = Path(sys.executable).resolve().parent
    gcc = conda_bin / "gcc"
    if not gcc.exists():
        return
    path = os.environ.get("PATH", "")
    if str(conda_bin) not in path.split(os.pathsep):
        os.environ["PATH"] = str(conda_bin) + os.pathsep + path


def _mkdir_mpi(path: Path, comm: MPI.Comm) -> None:
    if comm.rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    comm.Barrier()


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())

    def _rewrite_with_new_header(existing_fieldnames: list[str]) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

        # Best-effort backfill of newly introduced DOF columns.
        for existing_row in existing_rows:
            if "dofs_stim" not in existing_row or existing_row["dofs_stim"] == "":
                existing_row["dofs_stim"] = existing_row.get("dofs_dens", "")

            try:
                dofs_dens = int(existing_row.get("dofs_dens", ""))
                dofs_mech = int(existing_row.get("dofs_mech", ""))
            except (TypeError, ValueError):
                dofs_dens = 0
                dofs_mech = 0

            gdim = 0
            if dofs_dens > 0 and dofs_mech % dofs_dens == 0:
                gdim = dofs_mech // dofs_dens

            if "dofs_fabric" not in existing_row or existing_row["dofs_fabric"] == "":
                existing_row["dofs_fabric"] = str(dofs_dens * gdim * gdim) if gdim else ""

            if "dofs_total" not in existing_row or existing_row["dofs_total"] == "":
                try:
                    dofs_stim = int(existing_row.get("dofs_stim", ""))
                    dofs_fabric = int(existing_row.get("dofs_fabric", ""))
                except (TypeError, ValueError):
                    dofs_stim = 0
                    dofs_fabric = 0
                existing_row["dofs_total"] = str(dofs_mech + dofs_dens + dofs_stim + dofs_fabric)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow({k: existing_row.get(k, "") for k in fieldnames})

    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_fieldnames = next(reader, [])

        if existing_fieldnames != fieldnames:
            if set(existing_fieldnames).issubset(fieldnames):
                _rewrite_with_new_header(existing_fieldnames)
            else:
                raise ValueError(
                    f"Existing CSV header does not match expected schema: {path}\n"
                    f"  existing: {existing_fieldnames}\n"
                    f"  expected: {fieldnames}"
                )

    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="stiff_params_femur.json", help="Parameter file.")
    p.add_argument("--steps", type=int, default=2, help="Number of time steps to measure.")
    p.add_argument("--outdir", type=str, default="results/scaling_stiff", help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_conda_toolchain_on_path()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    outdir = Path(args.outdir)
    _mkdir_mpi(outdir, comm)

    # 1. Load Mesh and Params
    comm.Barrier()
    t0_setup = time.perf_counter()

    # Load mesh (replicates run_femur_model.py logic)
    # Note: FEBio2Dolfinx handles mesh loading internally
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    mesh = mdl.mesh_dolfinx

    # Load parameters
    params = load_default_params(args.config)
    
    # Configure time steps for benchmark
    dt = params["time"].dt_initial
    params["time"].total_time = dt * args.steps
    # Disable heavy I/O
    params["output"].saving_interval = 999999
    
    # Create config object
    cfg = create_config(mesh, mdl.meshtags, params)
    
    # Setup Loader
    loading_cases = get_standard_gait_cases()
    loader = Loader(mesh, facet_tags=mdl.meshtags, load_tag=cfg.geometry.load_tag, loading_cases=loading_cases)
    
    # Setup Factory
    factory = DefaultSolverFactory(cfg)
    
    comm.Barrier()
    t1_setup = time.perf_counter()
    t_setup = t1_setup - t0_setup

    # Mesh stats
    tdim = mesh.topology.dim
    cells_global = mesh.topology.index_map(tdim).size_global
    
    # Estimate DOFs (approximated, as actual FunctionSpaces are inside Solver)
    # Density/stimulus are scalar P1, Mech is vector P1, Fabric is tensor P1.
    gdim = mesh.geometry.dim
    n_nodes = mesh.topology.index_map(0).size_global
    dofs_dens = n_nodes
    dofs_mech = n_nodes * gdim
    dofs_stim = n_nodes
    dofs_fabric = n_nodes * gdim * gdim
    dofs_total = dofs_mech + dofs_dens + dofs_stim + dofs_fabric

    if rank == 0:
        print(f"Setup complete in {t_setup:.2f}s")
        print(
            "Mesh:"
            f" {cells_global} cells,"
            f" ~{dofs_total} DOFs total"
            f" (u:{dofs_mech}, rho:{dofs_dens}, S:{dofs_stim}, L:{dofs_fabric})"
        )
        print(f"Running {args.steps} steps...")

    # 2. Run Benchmark
    remodeller = Remodeller(cfg, loader=loader, factory=factory)
    
    comm.Barrier()
    t0_solve = time.perf_counter()
    
    # We don't use reporter to minimize overhead, or pass a dummy if required?
    # Remodeller.simulate optional arg: reporter
    remodeller.simulate(reporter=None)
    
    comm.Barrier()
    t1_solve = time.perf_counter()
    
    total_solve_time = t1_solve - t0_solve
    avg_step_time = total_solve_time / args.steps if args.steps > 0 else 0.0

    # 3. Save Results
    result = BenchmarkResult(
        timestamp_utc=_utc_now_str(),
        config_file=args.config,
        ranks=size,
        refine_level=0,
        cells_global=cells_global,
        dofs_dens=dofs_dens,
        dofs_mech=dofs_mech,
        dofs_stim=dofs_stim,
        dofs_fabric=dofs_fabric,
        dofs_total=dofs_total,
        steps_measured=args.steps,
        t_avg_step_s=avg_step_time,
        t_setup_s=t_setup,
        t_total_s=total_solve_time,
        min_step_s=0.0,
        max_step_s=0.0
    )

    if rank == 0:
        csv_path = outdir / "scaling.csv"
        _write_csv_row(csv_path, asdict(result))
        
        print(f"Benchmark finished.")
        print(f"Ranks: {size}")
        print(f"Avg time/step: {avg_step_time:.4f} s")
        print(f"Total solve time: {total_solve_time:.4f} s")


if __name__ == "__main__":
    main()
