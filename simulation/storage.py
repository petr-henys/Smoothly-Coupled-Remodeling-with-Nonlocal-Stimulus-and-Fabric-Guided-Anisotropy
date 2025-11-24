"""Universal storage system for field I/O and metrics tracking."""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import VTXWriter

if TYPE_CHECKING:
    from simulation.config import Config

from simulation.logger import get_logger


class FieldStorage:
    """VTX field output. COLLECTIVE operations only."""

    __slots__ = ("comm", "logger", "output_dir", "_writers", "_write_counts")

    def __init__(self, cfg: "Config", comm: MPI.Comm) -> None:
        self.comm = comm
        self.logger = get_logger(comm, name="Storage.Fields")
        self.output_dir = Path(cfg.results_dir)
        self._writers: Dict[str, VTXWriter] = {}
        self._write_counts: Dict[str, int] = defaultdict(int)
        
        # Create output directory (rank 0 only, then barrier)
        if comm.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    def register(
        self,
        key: str,
        fields: Sequence[fem.Function],
        filename: Optional[str] = None,
        engine: str = "bp4",
    ) -> None:
        """Register VTX writer. COLLECTIVE operation."""
        path = self.output_dir / (filename or f"{key}.bp")
        writer = VTXWriter(self.comm, str(path), list(fields), engine=engine)
        self._writers[key] = writer
        self._write_counts[key] = 0
        self.logger.debug(lambda: f"Registered '{key}': {path}")

    def write(self, key: str, t: float) -> None:
        """Write field group. COLLECTIVE operation."""
        self._writers[key].write(t)
        self._write_counts[key] += 1

    def close(self) -> None:
        """Close all writers. COLLECTIVE operation."""
        self.comm.Barrier()
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        self._write_counts.clear()
        self.comm.Barrier()

    def __enter__(self) -> "FieldStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class MetricsStorage:
    """CSV metrics writer. Rank-0 only operations."""

    __slots__ = ("comm", "logger", "output_dir", "_flush_interval", "_buffers", "_files", "_writers")

    def __init__(self, cfg: "Config", comm: MPI.Comm, flush_interval: int = 10) -> None:
        self.comm = comm
        self.logger = get_logger(comm, name="Storage.Metrics")
        self.output_dir = Path(cfg.results_dir)
        self._flush_interval = flush_interval
        self._buffers: Dict[str, List[Dict]] = defaultdict(list)
        self._files: Dict[str, object] = {}
        self._writers: Dict[str, csv.DictWriter] = {}
        
        # Create output directory (rank 0 only, then barrier)
        if comm.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    def register_csv(
        self,
        name: str,
        columns: Sequence[str],
        gz: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Register CSV file. Rank-0 only."""
        if self.comm.rank != 0:
            return

        path = self.output_dir / (filename or f"{name}.csv{'.gz' if gz else ''}")
        handle = gzip.open(path, "wt", newline="") if gz else open(path, "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        
        self._files[name] = handle
        self._writers[name] = writer
        self._buffers[name] = []
        self.logger.debug(lambda: f"Registered CSV '{name}': {path}")

    def record(self, name: str, data: Dict) -> None:
        """Record data row. Rank-0 only."""
        if self.comm.rank != 0:
            return

        self._buffers[name].append(dict(data))
        if len(self._buffers[name]) >= self._flush_interval:
            self._flush(name)

    def _flush(self, name: str) -> None:
        """Flush buffer to disk. Rank-0 only."""
        if self.comm.rank != 0:
            return
        
        buffer = self._buffers[name]
        writer = self._writers[name]
        handle = self._files[name]
        
        for row in buffer:
            writer.writerow(row)
        buffer.clear()
        handle.flush()

    def flush_all(self) -> None:
        """Flush all buffers. Rank-0 only."""
        if self.comm.rank != 0:
            return
        for name in list(self._buffers.keys()):
            self._flush(name)

    def close(self) -> None:
        """Close all CSV files. Rank-0 only."""
        if self.comm.rank != 0:
            return
        
        self.flush_all()
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._writers.clear()
        self._buffers.clear()

    def __enter__(self) -> "MetricsStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class UnifiedStorage:
    """Unified field + metrics storage."""

    __slots__ = ("comm", "fields", "metrics", "_metrics_enabled")

    def __init__(self, cfg: "Config") -> None:
        self.comm = cfg.domain.comm
        self.fields = FieldStorage(cfg, self.comm)
        self.metrics = MetricsStorage(cfg, self.comm)
        
        # Register standard metrics CSV if telemetry not already doing it
        self._metrics_enabled = not hasattr(cfg, "telemetry") or cfg.telemetry is None
        if self._metrics_enabled:
            self.metrics.register_csv(
                "steps",
                columns=[
                    "step", "time_days", "dt_days", "num_dofs_total", "rss_mem_mb",
                    "mech_iters", "stim_iters", "dens_iters", "dir_iters",
                    "coupling_iters", "coupling_time",
                ],
            )

    def write_step(
        self,
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
        """Write timestep data. COLLECTIVE operation (field writes)."""
        # Field writes (COLLECTIVE)
        self.fields.write("u", time_days)
        self.fields.write("scalars", time_days)
        self.fields.write("A", time_days)
        
        # Metrics (rank-0 only)
        if self._metrics_enabled:
            self.metrics.record("steps", {
                "step": step,
                "time_days": time_days,
                "dt_days": dt_days,
                "num_dofs_total": num_dofs_total,
                "rss_mem_mb": rss_mem_mb,
                "mech_iters": solver_stats.get("mech", 0),
                "stim_iters": solver_stats.get("stim", 0),
                "dens_iters": solver_stats.get("dens", 0),
                "dir_iters": solver_stats.get("dir", 0),
                "coupling_iters": coupling_stats.get("iters", 0),
                "coupling_time": coupling_stats.get("time", 0.0),
            })

    def close(self) -> None:
        """Close storage. COLLECTIVE operation."""
        self.fields.close()
        self.metrics.close()

    def __enter__(self) -> "UnifiedStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()
