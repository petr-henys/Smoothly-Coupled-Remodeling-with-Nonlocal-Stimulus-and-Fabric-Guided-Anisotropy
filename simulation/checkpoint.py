"""Checkpoint storage using adios4dolfinx for MPI-independent restart/analysis.

This module provides checkpointing functionality separate from VTXWriter:
- VTXWriter: Visualization output for ParaView (cannot be read back)
- Checkpoint: Analysis/restart files that can be read by any MPI configuration

Usage in simulation:
    from simulation.checkpoint import CheckpointStorage
    
    with CheckpointStorage(cfg) as ckpt:
        # Write mesh once (required before any function writes)
        ckpt.write_mesh()
        
        # Write functions at each timestep
        ckpt.write_function(rho, t)
        ckpt.write_function(u, t)
        ckpt.write_function(S, t)

Usage in analysis:
    from simulation.checkpoint import load_checkpoint
    
    mesh, facet_tags = load_checkpoint_mesh(checkpoint_path, comm)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    rho = load_checkpoint_function(checkpoint_path, "rho", V, t_final, comm)

Note: Requires adios4dolfinx >= 0.10.0 (pip install adios4dolfinx)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mpi4py import MPI
from dolfinx import fem, mesh

try:
    import adios4dolfinx as adx
    HAS_ADIOS4DOLFINX = True
except ImportError:
    HAS_ADIOS4DOLFINX = False

if TYPE_CHECKING:
    from simulation.config import Config

from simulation.logger import get_logger


def _check_adios4dolfinx() -> None:
    """Raise ImportError if adios4dolfinx is not available."""
    if not HAS_ADIOS4DOLFINX:
        raise ImportError(
            "adios4dolfinx is required for checkpointing. "
            "Install with: pip install adios4dolfinx"
        )


class CheckpointStorage:
    """Checkpoint storage using adios4dolfinx for N-to-M restarts.
    
    Writes mesh + functions to a single BP file that can be read
    by any MPI configuration (different number of processes).
    
    Args:
        cfg: Simulation configuration.
        filename: Checkpoint filename (default: "checkpoint.bp").
    """
    
    __slots__ = ("comm", "logger", "checkpoint_path", "_mesh_written", "_domain", "_facet_tags")
    
    def __init__(self, cfg: "Config", filename: str = "checkpoint.bp") -> None:
        _check_adios4dolfinx()
        
        self.comm = cfg.domain.comm
        self.logger = get_logger(self.comm, name="Checkpoint")
        self._domain = cfg.domain
        self._facet_tags = cfg.facet_tags
        
        output_dir = Path(cfg.output.results_dir)
        self.checkpoint_path = output_dir / filename
        self._mesh_written = False
        
        # Ensure output directory exists
        if self.comm.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()
    
    def write_mesh(self) -> None:
        """Write mesh to checkpoint file (must be called before write_function)."""
        if self._mesh_written:
            self.logger.debug(lambda: "Mesh already written, skipping")
            return
        
        # adios4dolfinx API: write_mesh(filename, mesh, ...)
        adx.write_mesh(self.checkpoint_path, self._domain)
        
        if self._facet_tags is not None:
            # adios4dolfinx API: write_meshtags(filename, mesh, meshtags, ...)
            adx.write_meshtags(self.checkpoint_path, self._domain, self._facet_tags)
        
        self._mesh_written = True
        self.logger.debug(lambda: f"Wrote mesh to {self.checkpoint_path}")
    
    def write_function(self, func: fem.Function, t: float) -> None:
        """Write function checkpoint at time t.
        
        Args:
            func: DOLFINx Function to checkpoint.
            t: Time value for this checkpoint.
        """
        if not self._mesh_written:
            self.write_mesh()
        
        # adios4dolfinx API: write_function(filename, u, ..., time=...)
        adx.write_function(self.checkpoint_path, func, time=t)
        self.logger.debug(lambda: f"Wrote {func.name} at t={t:.4f}")
    
    def write_functions(self, funcs: Sequence[fem.Function], t: float) -> None:
        """Write multiple function checkpoints at time t."""
        for func in funcs:
            self.write_function(func, t)
    
    def close(self) -> None:
        """Close checkpoint (no-op, but maintains context manager pattern)."""
        self.comm.Barrier()
        self.logger.debug(lambda: f"Checkpoint closed: {self.checkpoint_path}")
    
    def __enter__(self) -> "CheckpointStorage":
        return self
    
    def __exit__(self, *_) -> None:
        self.close()


# =============================================================================
# Loading functions for analysis
# =============================================================================

def load_checkpoint_mesh(
    checkpoint_path: Path | str,
    comm: MPI.Comm,
) -> tuple[mesh.Mesh, mesh.MeshTags | None]:
    """Load mesh and optional meshtags from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp file.
        comm: MPI communicator.
    
    Returns:
        Tuple of (mesh, meshtags) where meshtags may be None.
    """
    _check_adios4dolfinx()
    
    checkpoint_path = Path(checkpoint_path)
    # adios4dolfinx API: read_mesh(filename, comm, ...)
    domain = adx.read_mesh(checkpoint_path, comm)
    
    # Try to read meshtags (may not exist)
    try:
        # adios4dolfinx API: read_meshtags(filename, mesh, meshtag_name=...)
        facet_tags = adx.read_meshtags(checkpoint_path, domain, meshtag_name="meshtags")
    except (KeyError, RuntimeError):
        facet_tags = None
    
    return domain, facet_tags


def load_checkpoint_function(
    checkpoint_path: Path | str,
    name: str,
    function_space: fem.FunctionSpace,
    time: float,
    comm: MPI.Comm,
) -> fem.Function:
    """Load a function from checkpoint at a specific time.
    
    Args:
        checkpoint_path: Path to checkpoint.bp file.
        name: Function name (as registered during write).
        function_space: Target function space (must match saved function).
        time: Time value to load.
        comm: MPI communicator (unused, kept for API compatibility).
    
    Returns:
        fem.Function with loaded data.
    """
    _check_adios4dolfinx()
    
    checkpoint_path = Path(checkpoint_path)
    func = fem.Function(function_space, name=name)
    # adios4dolfinx API: read_function(filename, u, ..., time=..., name=...)
    adx.read_function(checkpoint_path, func, time=time, name=name)
    return func


def get_checkpoint_times(
    checkpoint_path: Path | str,
    name: str,
    comm: MPI.Comm,
) -> list[float]:
    """Get all available time values for a function in checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp file.
        name: Function name.
        comm: MPI communicator.
    
    Returns:
        List of time values.
    """
    _check_adios4dolfinx()
    
    # adios4dolfinx stores time as a variable attribute
    # This requires reading the ADIOS2 file directly
    import adios2
    
    checkpoint_path = Path(checkpoint_path)
    
    times = []
    if comm.rank == 0:
        with adios2.open(str(checkpoint_path), "r", comm=MPI.COMM_SELF) as fh:
            for step in fh:
                if f"{name}/values" in fh.available_variables():
                    t = step.read_attribute(f"{name}/time")
                    times.append(float(t))
    
    times = comm.bcast(times, root=0)
    return times
