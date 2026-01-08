"""Post-process psi_ref ratio sweep: extract final timesteps into pseudo-time VTX.

Reads checkpoints from the psi_ref ratio sweep, extracts the last timestep from
each simulation, and writes them to a single VTX file where "pseudo-time"
corresponds to different ratio values:

    r = psi_ref_cort / psi_ref_trab

Usage:
    python analysis/ref_ratio_postprocess.py

Input:
    results/psi_ref_ratio_sweep/
    ├── sweep_summary.json
    ├── <hash1>/checkpoint.bp/
    ├── <hash2>/checkpoint.bp/
    └── ...

Output:
    results/psi_ref_ratio_sweep/combined_final.bp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI
from dolfinx import fem, io

from simulation.checkpoint import load_checkpoint_mesh, load_checkpoint_function
from simulation.logger import get_logger


def main() -> None:
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="PsiRefRatioPostprocess")

    sweep_dir = Path("results/psi_ref_ratio_sweep")
    summary_path = sweep_dir / "sweep_summary.json"
    output_path = sweep_dir / "combined_final.bp"

    if comm.rank == 0:
        if not summary_path.exists():
            raise FileNotFoundError(f"Sweep summary not found: {summary_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = None
    summary = comm.bcast(summary, root=0)

    runs = summary.get("runs", [])
    if not runs:
        if comm.rank == 0:
            logger.error("No runs found in sweep summary!")
        return

    # Sort by ratio
    runs_sorted = sorted(runs, key=lambda r: float(r["psi_ref_ratio"]))
    n_runs = len(runs_sorted)

    if comm.rank == 0:
        logger.info(f"Found {n_runs} runs in sweep")
        for i, run in enumerate(runs_sorted):
            ratio = float(run["psi_ref_ratio"])
            logger.info(f"  [{i}] psi_ref_ratio = {ratio:g}")

    first_checkpoint = sweep_dir / runs_sorted[0]["output_dir"] / "checkpoint.bp"
    if not first_checkpoint.exists():
        if comm.rank == 0:
            logger.error(f"First checkpoint not found: {first_checkpoint}")
        return

    mesh = load_checkpoint_mesh(first_checkpoint, comm)
    if comm.rank == 0:
        logger.info(f"Loaded mesh: {mesh.topology.index_map(3).size_global} cells")

    P1 = fem.functionspace(mesh, ("Lagrange", 1))

    rho_out = fem.Function(P1, name="rho")
    S_out = fem.Function(P1, name="S")
    psi_out = fem.Function(P1, name="psi")

    if comm.rank == 0:
        logger.info(f"Writing combined VTX to: {output_path}")

    with io.VTXWriter(comm, output_path, [rho_out, S_out, psi_out], engine="BP4") as vtx:
        for _, run in enumerate(runs_sorted):
            run_hash = run["output_dir"]
            ratio = float(run["psi_ref_ratio"])
            checkpoint_path = sweep_dir / run_hash / "checkpoint.bp"

            if not checkpoint_path.exists():
                if comm.rank == 0:
                    logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
                continue

            if comm.rank == 0:
                logger.info(f"Loading ratio={ratio:g} (latest checkpoint step)")

            try:
                rho_loaded = load_checkpoint_function(checkpoint_path, "rho", P1, time=None)
                rho_out.x.array[:] = rho_loaded.x.array[:]
                rho_out.x.scatter_forward()
            except Exception as e:
                if comm.rank == 0:
                    logger.warning(f"Error loading rho: {e}")
                rho_out.x.array[:] = 0.0

            try:
                S_loaded = load_checkpoint_function(checkpoint_path, "S", P1, time=None)
                S_out.x.array[:] = S_loaded.x.array[:]
                S_out.x.scatter_forward()
            except Exception as e:
                if comm.rank == 0:
                    logger.warning(f"Error loading S: {e}")
                S_out.x.array[:] = 0.0

            try:
                psi_loaded = load_checkpoint_function(checkpoint_path, "psi", P1, time=None)
                psi_out.x.array[:] = psi_loaded.x.array[:]
                psi_out.x.scatter_forward()
            except Exception as e:
                if comm.rank == 0:
                    logger.debug(f"psi not available: {e}")
                psi_out.x.array[:] = 0.0

            # Pseudo-time = ratio for ParaView
            vtx.write(float(ratio))

    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("POST-PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output: {output_path}")
        logger.info("Pseudo-time values = psi_ref_ratio (psi_ref_cort/psi_ref_trab)")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
