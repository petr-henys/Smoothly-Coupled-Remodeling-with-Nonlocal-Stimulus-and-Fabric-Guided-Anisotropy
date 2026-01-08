"""Post-process alpha_front sweep: extract final timesteps into pseudo-time VTX.

This script reads checkpoints from the alpha_front sweep, extracts the last
timestep from each simulation, and writes them to a single VTX file where
"pseudo-time" corresponds to different alpha_front offset values.

When opened in ParaView:
- Each "time step" shows the final state of a different simulation
- Stepping through time shows how fields vary with hip frontal angle

Usage:
    python analysis/alpha_front_postprocess.py

Input:
    results/alpha_front_sweep/
    ├── sweep_summary.json
    ├── <hash1>/checkpoint.bp/
    ├── <hash2>/checkpoint.bp/
    └── ...

Output:
    results/alpha_front_sweep/combined_final.bp/
    - VTX file with pseudo-time = alpha_front_offset index
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io

from simulation.checkpoint import (
    load_checkpoint_mesh,
    load_checkpoint_function,
)
from simulation.logger import get_logger


def main() -> None:
    """Post-process alpha_front sweep results."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AlphaFrontPostprocess")
    
    # Paths
    sweep_dir = Path("results/alpha_front_sweep")
    summary_path = sweep_dir / "sweep_summary.json"
    output_path = sweep_dir / "combined_final.bp"
    
    # Load sweep summary
    if comm.rank == 0:
        if not summary_path.exists():
            raise FileNotFoundError(f"Sweep summary not found: {summary_path}")
        with open(summary_path, "r") as f:
            summary = json.load(f)
    else:
        summary = None
    summary = comm.bcast(summary, root=0)
    
    # Get sorted list of runs by alpha_front_offset
    runs = summary.get("runs", [])
    if not runs:
        if comm.rank == 0:
            logger.error("No runs found in sweep summary!")
        return
    
    # Sort runs by alpha_front_offset
    # Note: param values are stored directly in run dict (not nested under "params")
    runs_sorted = sorted(runs, key=lambda r: float(r["alpha_front_offset"]))
    n_runs = len(runs_sorted)
    
    if comm.rank == 0:
        logger.info(f"Found {n_runs} runs in sweep")
        for i, run in enumerate(runs_sorted):
            offset = run["alpha_front_offset"]
            logger.info(f"  [{i}] alpha_front_offset = {offset:.1f}°")
    
    # Load mesh from first checkpoint (all should have same mesh)
    # "output_dir" contains the hash subdirectory name
    first_checkpoint = sweep_dir / runs_sorted[0]["output_dir"] / "checkpoint.bp"
    if not first_checkpoint.exists():
        if comm.rank == 0:
            logger.error(f"First checkpoint not found: {first_checkpoint}")
        return
    
    mesh = load_checkpoint_mesh(first_checkpoint, comm)
    if comm.rank == 0:
        logger.info(f"Loaded mesh: {mesh.topology.index_map(3).size_global} cells")
    
    # Create function spaces
    P1 = fem.functionspace(mesh, ("Lagrange", 1))
    P1_tensor = fem.functionspace(mesh, ("Lagrange", 1, (3, 3)))  # For fabric L
    
    # Create output functions
    rho_out = fem.Function(P1, name="rho")
    S_out = fem.Function(P1, name="S")
    psi_out = fem.Function(P1, name="psi")
    L_out = fem.Function(P1_tensor, name="L")
    
    # Create VTXWriter for combined output
    # We'll write each simulation's final state as a pseudo-timestep
    if comm.rank == 0:
        logger.info(f"Writing combined VTX to: {output_path}")
    
    with io.VTXWriter(comm, output_path, [rho_out, S_out, psi_out], engine="BP4") as vtx:
        for pseudo_t, run in enumerate(runs_sorted):
            run_hash = run["output_dir"]
            alpha_offset = float(run["alpha_front_offset"])
            checkpoint_path = sweep_dir / run_hash / "checkpoint.bp"
            
            if not checkpoint_path.exists():
                if comm.rank == 0:
                    logger.warning(f"Checkpoint not found: {checkpoint_path}, skipping")
                continue
            
            if comm.rank == 0:
                logger.info(f"Loading [{pseudo_t}] alpha={alpha_offset:.1f}° (latest checkpoint step)")
            
            # Load fields at final time
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
            
            # Write pseudo-timestep (use index as pseudo-time for ParaView)
            # We use alpha_offset as the pseudo-time so it's meaningful in ParaView
            vtx.write(float(alpha_offset))
    
    if comm.rank == 0:
        logger.info("=" * 70)
        logger.info("POST-PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output: {output_path}")
        logger.info(f"Pseudo-time values = alpha_front_offset [deg]")
        logger.info("Open in ParaView and step through 'time' to see different simulations")
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
