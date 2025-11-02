"""
Unified storage system for field I/O and metrics tracking.

The refactored module provides:
- A shared base class handling MPI-aware directory creation and logger setup.
- Lazy VTX writer management for field outputs.
- Root-only CSV writers with buffered flushes for metrics.

Physics-facing code is untouched; the data schemas exposed to the rest of the
codebase remain identical so downstream consumers do not need to change.
"""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import VTXWriter

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    from simulation.config import Config

from simulation.logger import get_logger


class _BaseStorage:
    """Common helpers shared by field and metrics storage backends."""

    __slots__ = ("cfg", "comm", "logger", "_root")

    def __init__(self, cfg: "Config", comm: MPI.Comm, *, name: str) -> None:
        self.cfg = cfg
        self.comm = comm
        self.logger = get_logger(comm, verbose=getattr(cfg, "verbose", True), name=name)
        self._root = self._ensure_dir(cfg.results_dir)

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    @property
    def is_root(self) -> bool:
        return self.comm.rank == 0

    @property
    def root_dir(self) -> Path:
        return self._root

    def _ensure_dir(self, target: Union[str, Path]) -> Path:
        """Create directory on rank 0 and synchronize across all ranks.
        
        COLLECTIVE OPERATION: All ranks must call this simultaneously.
        """
        path = Path(target)
        success = True
        error_msg = ""
        
        if self.is_root:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                success = False
                error_msg = f"Failed to create directory {path}: {e}"
        
        success = self.comm.bcast(success, root=0)
        self.comm.Barrier()
        
        if not success:
            if self.is_root:
                raise RuntimeError(error_msg)
            else:
                raise RuntimeError(f"Directory creation failed on rank 0: {path}")
        
        return path


class FieldStorage(_BaseStorage):
    """Manages VTX field output (displacement, scalar, tensor fields).
    
    IMPORTANT: All write operations are COLLECTIVE - all MPI ranks must call
    them simultaneously. Do NOT call write methods inside rank-specific blocks
    (e.g., 'if rank == 0:').
    """

    __slots__ = ("_writers", "_writer_paths", "_write_counts")

    def __init__(self, cfg: "Config", comm: MPI.Comm) -> None:
        super().__init__(cfg, comm, name="Storage.Fields")
        self._writers: Dict[str, VTXWriter] = {}
        self._writer_paths: Dict[str, Path] = {}
        self._write_counts: Dict[str, int] = defaultdict(int)  # Count writes, not store all times

    # ------------------------------------------------------------------
    # Writer management
    # ------------------------------------------------------------------
    def register_group(
        self,
        key: str,
        fields: Sequence[fem.Function],
        filename: Optional[str] = None,
        *,
        engine: str = "bp4",
    ) -> None:
        """Eagerly register a VTX writer group.
        
        COLLECTIVE OPERATION: All ranks must call this simultaneously.
        """
        if key in self._writers:
            self.logger.warning("Field writer '{0}' already registered; skipping", key)
            return
        self._create_writer(key, fields, filename=filename, engine=engine)

    def _create_writer(
        self,
        key: str,
        fields: Sequence[fem.Function],
        *,
        filename: Optional[str] = None,
        engine: str = "bp4",
    ) -> VTXWriter:
        """Create VTX writer. COLLECTIVE OPERATION."""
        path = self.root_dir / (filename or f"{key}.bp")
        writer = VTXWriter(self.comm, str(path), list(fields), engine=engine)
        self._writers[key] = writer
        self._writer_paths[key] = path
        self._write_counts[key] = 0
        self.logger.debug(lambda: f"Registered VTX writer '{key}': {path}")
        return writer

    def _get_writer(self, key: str, fields: Sequence[fem.Function]) -> VTXWriter:
        """Get or create writer. COLLECTIVE OPERATION if creating new writer.
        
        WARNING: Lazy creation is dangerous for MPI - ensure all ranks call
        write methods simultaneously to avoid deadlock during writer creation.
        """
        writer = self._writers.get(key)
        if writer is None:
            writer = self._create_writer(key, fields)
        return writer

    # ------------------------------------------------------------------
    # Public API (ALL METHODS ARE COLLECTIVE OPERATIONS)
    # ------------------------------------------------------------------
    def write_displacement(self, u: fem.Function, t: float) -> None:
        """Write displacement field. COLLECTIVE OPERATION."""
        writer = self._get_writer("u", [u])
        writer.write(t)
        self._write_counts["u"] += 1

    def write_scalars(self, rho: fem.Function, S: fem.Function, t: float) -> None:
        """Write scalar fields (density, stimulus). COLLECTIVE OPERATION."""
        writer = self._get_writer("scalars", [rho, S])
        writer.write(t)
        self._write_counts["scalars"] += 1

    def write_tensor(self, A: fem.Function, t: float) -> None:
        """Write tensor field (orientation). COLLECTIVE OPERATION."""
        writer = self._get_writer("A", [A])
        writer.write(t)
        self._write_counts["A"] += 1

    def write_all(
        self,
        u: fem.Function,
        rho: fem.Function,
        S: fem.Function,
        A: fem.Function,
        t: float,
    ) -> None:
        """Write all fields. COLLECTIVE OPERATION."""
        self.write_displacement(u, t)
        self.write_scalars(rho, S, t)
        self.write_tensor(A, t)

    def close(self) -> None:
        """Close all VTX writers. COLLECTIVE OPERATION.
        
        All ranks must call this simultaneously. VTXWriter.close() is a
        collective operation and will deadlock if not called by all ranks.
        """
        self.comm.Barrier()
        
        # Close all writers (VTXWriter.close() is COLLECTIVE)
        for key, writer in list(self._writers.items()):
            writer.close()
        
        # Log statistics
        for key, path in self._writer_paths.items():
            count = self._write_counts.get(key, 0)
            self.logger.debug(lambda k=key, p=path, c=count: f"Closed {p} ({c} timesteps)")
        
        # Clear state
        self._writers.clear()
        self._writer_paths.clear()
        self._write_counts.clear()
        
        self.comm.Barrier()

    def __enter__(self) -> "FieldStorage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class MetricsStorage(_BaseStorage):
    """Root-only buffered CSV writer for per-step metrics.
    
    NOTE: All write operations are rank-0 only (non-collective).
    Non-root ranks return immediately from record() and flush_all().
    """

    __slots__ = ("_flush_interval", "_buffers", "_files", "_writers")

    def __init__(self, cfg: "Config", comm: MPI.Comm, *, flush_interval: int = 10) -> None:
        super().__init__(cfg, comm, name="Storage.Metrics")
        self._flush_interval = max(1, int(flush_interval))
        self._buffers: Dict[str, List[Dict]] = defaultdict(list)
        self._files: Dict[str, object] = {}
        self._writers: Dict[str, csv.DictWriter] = {}

    # ------------------------------------------------------------------
    # CSV lifecycle
    # ------------------------------------------------------------------
    def register_csv(
        self,
        name: str,
        columns: Sequence[str],
        *,
        gz: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Register CSV file for metrics. Rank-0 only (non-collective)."""
        if not self.is_root:
            return

        resolved_name = filename or f"{name}.csv" + (".gz" if gz else "")
        path = self.root_dir / resolved_name

        handle = gzip.open(path, "wt", newline="") if gz else open(path, "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()

        self._files[name] = handle
        self._writers[name] = writer
        self._buffers[name] = []
        self.logger.debug(lambda: f"Registered metrics CSV '{name}': {path}")

    def record(self, name: str, data: Dict) -> None:
        """Record metrics data. Rank-0 only (non-collective).

        Raises KeyError if the CSV stream ``name`` was not registered.
        """
        if not self.is_root:
            return

        buffer = self._buffers.get(name)
        if buffer is None:
            raise KeyError(f"Metrics CSV '{name}' not registered")

        buffer.append(dict(data))
        if len(buffer) >= self._flush_interval:
            self._flush(name)

    def _flush(self, name: str) -> None:
        """Flush buffer to disk. Rank-0 only (non-collective)."""
        if not self.is_root:
            return
        buffer = self._buffers.get(name)
        writer = self._writers.get(name)
        handle = self._files.get(name)
        if buffer is None or writer is None or handle is None:
            return
        for row in buffer:
            writer.writerow(row)
        buffer.clear()
        handle.flush()

    def flush_all(self) -> None:
        """Flush all buffers. Rank-0 only (non-collective)."""
        for name in list(self._buffers.keys()):
            self._flush(name)

    def close(self) -> None:
        """Close all CSV files. Rank-0 only (non-collective)."""
        self.flush_all()
        if self.is_root:
            for name, handle in list(self._files.items()):
                handle.close()
                self.logger.debug(lambda n=name: f"Closed metrics CSV '{n}'")
        self._files.clear()
        self._writers.clear()
        self._buffers.clear()

    def __enter__(self) -> "MetricsStorage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class UnifiedStorage:
    """Unified manager combining field I/O and metrics tracking.
    
    IMPORTANT MPI USAGE:
    - All field write operations (write_step, fields.write_*) are COLLECTIVE
    - Metrics operations (metrics.record) are rank-0 only (non-collective)
    - close() is COLLECTIVE (must be called by all ranks)
    """

    __slots__ = ("cfg", "comm", "fields", "metrics", "telemetry", "_steps_metrics_enabled")

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.comm = cfg.domain.comm
        self.fields = FieldStorage(cfg, self.comm)
        self.metrics = MetricsStorage(cfg, self.comm)
        self.telemetry = getattr(cfg, "telemetry", None)
        # Avoid duplicating per-step CSVs when telemetry already collects them
        self._steps_metrics_enabled = self.telemetry is None
        self._register_standard_metrics()

    def _register_standard_metrics(self) -> None:
        if not self._steps_metrics_enabled:
            return
        self.metrics.register_csv(
            "steps",
            columns=[
                "step",
                "time_days",
                "dt_days",
                "num_dofs_total",
                "rss_mem_mb",
                "mech_iters",
                "stim_iters",
                "dens_iters",
                "dir_iters",
                "coupling_iters",
                "coupling_time",
            ],
            gz=False,
        )

    def write_step(
        self,
        *,
        step: int,
        time_days: float,
        dt_days: float,
        u: fem.Function,
        rho: fem.Function,
        S: fem.Function,
        A: fem.Function,
        num_dofs_total: int,
        rss_mem_mb: float,
        solver_stats: Dict[str, int],
        coupling_stats: Dict[str, float],
    ) -> None:
        """Write simulation step data. COLLECTIVE OPERATION (due to field writes)."""
        self.fields.write_all(u, rho, S, A, time_days)
        record = {
            "step": step,
            "time_days": time_days,
            "dt_days": dt_days,
            "num_dofs_total": int(num_dofs_total),
            "rss_mem_mb": float(rss_mem_mb),
            "mech_iters": solver_stats.get("mech", 0),
            "stim_iters": solver_stats.get("stim", 0),
            "dens_iters": solver_stats.get("dens", 0),
            "dir_iters": solver_stats.get("dir", 0),
            "coupling_iters": coupling_stats.get("iters", 0),
            "coupling_time": coupling_stats.get("time", 0.0),
        }
        if self._steps_metrics_enabled:
            self.metrics.record("steps", record)

    def close(self) -> None:
        """Close all storage backends. COLLECTIVE OPERATION."""
        self.fields.close()  # COLLECTIVE
        self.metrics.close()  # Rank-0 only, but safe to call from all ranks

    def __enter__(self) -> "UnifiedStorage":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # No snapshot helpers; field output is handled via VTXWriter only.
