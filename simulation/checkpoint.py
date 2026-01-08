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
    from simulation.checkpoint import (
        load_checkpoint_mesh,
        load_checkpoint_meshtags,
        load_checkpoint_function,
    )
    
    mesh = load_checkpoint_mesh(checkpoint_path, comm)
    facet_tags = load_checkpoint_meshtags(checkpoint_path, mesh)
    V = fem.functionspace(mesh, ("Lagrange", 1))
    rho = load_checkpoint_function(checkpoint_path, "rho", V, t_final)

Note: Requires adios4dolfinx >= 0.10.0 (pip install adios4dolfinx)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from mpi4py import MPI
from dolfinx import fem, mesh
import adios4dolfinx as adx

if TYPE_CHECKING:
    from simulation.config import Config

from simulation.logger import get_logger


class CheckpointStorage:
    """Manages MPI-independent checkpoints (mesh + functions) using adios4dolfinx.
    
    Writes mesh + functions to a single BP file that can be read
    by any MPI configuration (different number of processes).
    
    Args:
        cfg: Simulation configuration.
        filename: Checkpoint filename (default: "checkpoint.bp").
    """
    
    __slots__ = ("comm", "logger", "checkpoint_path", "_mesh_written", "_domain", "_facet_tags")
    
    def __init__(self, cfg: "Config", filename: str = "checkpoint.bp") -> None:
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
) -> mesh.Mesh:
    """Load mesh from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp file.
        comm: MPI communicator.
    
    Returns:
        Loaded mesh.
    """
    checkpoint_path = Path(checkpoint_path)
    return adx.read_mesh(checkpoint_path, comm)


def load_checkpoint_meshtags(
    checkpoint_path: Path | str,
    domain: mesh.Mesh,
    meshtag_name: str = "meshtags",
) -> mesh.MeshTags:
    """Load meshtags from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp file.
        domain: Mesh the meshtags belong to.
        meshtag_name: Name of the meshtag in the checkpoint.
    
    Returns:
        Loaded meshtags.
        
    Raises:
        RuntimeError: If meshtags not found in checkpoint.
    """
    checkpoint_path = Path(checkpoint_path)
    return adx.read_meshtags(checkpoint_path, domain, meshtag_name=meshtag_name)


def load_checkpoint_function(
    checkpoint_path: Path | str,
    name: str,
    function_space: fem.FunctionSpace,
    time: float | None,
) -> fem.Function:
    """Load a function from checkpoint at a specific time.
    
    Args:
        checkpoint_path: Path to checkpoint.bp file.
        name: Function name (as registered during write).
        function_space: Target function space (must match saved function).
        time: Time value to load.
    
    Returns:
        fem.Function with loaded data.
    """
    checkpoint_path = Path(checkpoint_path)
    func = fem.Function(function_space, name=name)

    # adios4dolfinx defaults `time=0.0` if omitted, which is NOT the same as
    # "latest" and often fails. Therefore we:
    # - If `time` is None: resolve the latest available time from the BP stream.
    # - If `time` is provided but missing: fall back to latest.
    # - If no time information exists at all: fall back to legacy checkpoints.
    comm = function_space.mesh.comm

    resolved_time: float | None = time
    if resolved_time is None:
        times = get_checkpoint_times(checkpoint_path, name, comm)
        resolved_time = max(times) if times else None

    if resolved_time is not None:
        try:
            # adios4dolfinx API: read_function(filename, u, ..., time=..., name=...)
            adx.read_function(checkpoint_path, func, time=resolved_time, name=name)
            return func
        except Exception:
            # Fall back to latest discoverable time
            times = get_checkpoint_times(checkpoint_path, name, comm)
            if times:
                adx.read_function(checkpoint_path, func, time=max(times), name=name)
                return func

    # Final fallback: legacy checkpoints (pre-time-dependent writing)
    adx.read_function(checkpoint_path, func, time=0.0, legacy=True, name=name)
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
    # NOTE:
    # - Not all checkpoints store explicit time metadata.
    # - Newer/packaged adios2 Python bindings in some environments do not expose
    #   `adios2.open` and instead use `adios2.Stream`.
    #
    # We therefore try to extract per-step times from variables like
    # `{name}_time` when available, and otherwise return an empty list.
    import adios2

    checkpoint_path = Path(checkpoint_path)

    times: list[float] = []
    if comm.rank == 0:
        try:
            if hasattr(adios2, "Stream"):
                time_var = f"{name}_time"
                values_var = f"{name}_values"
                with adios2.Stream(str(checkpoint_path), "r", MPI.COMM_SELF) as s:
                    for step in s:
                        available = step.available_variables() if hasattr(step, "available_variables") else s.available_variables()
                        if values_var not in available:
                            continue
                        if time_var in available:
                            t = step.read(time_var)
                            # Usually scalar ndarray; coerce robustly
                            times.append(float(getattr(t, "item", lambda: t)()))
            elif hasattr(adios2, "open"):
                # Legacy API (kept for completeness)
                with adios2.open(str(checkpoint_path), "r", comm=MPI.COMM_SELF) as fh:
                    time_attr = f"{name}/time"
                    values_var = f"{name}/values"
                    for step in fh:
                        if values_var in fh.available_variables():
                            try:
                                t = step.read_attribute(time_attr)
                                times.append(float(t))
                            except Exception:
                                # No time metadata
                                pass
        except Exception:
            # Treat time extraction as best-effort
            times = []

    times = comm.bcast(times, root=0)
    return times
