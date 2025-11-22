"""MPI-safe VTX field and CSV metrics storage."""

from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import VTXWriter

if TYPE_CHECKING:
    from simulation.config import Config

from simulation.logger import get_logger


FLUSH_INTERVAL: int = 10


class FieldStorage:
    """VTX field output manager (MPI-collective)."""

    __slots__ = ("comm", "logger", "output_dir", "_writers", "_write_counts")

    def __init__(self, cfg: "Config", comm: MPI.Comm) -> None:
        self.comm = comm
        self.logger = get_logger(comm, verbose=cfg.verbose, name="Storage.Fields")
        self.output_dir = Path(cfg.results_dir)
        self._writers: Dict[str, VTXWriter] = {}
        self._write_counts: Dict[str, int] = defaultdict(int)
        
        # Disable ADIOS2 profiling to avoid profiling.json creation in temp dirs
        # Must be set before any writers are constructed
        if os.environ.get("ADIOS2_PROFILE", "").lower() not in ("off", "0", "false"):
            os.environ["ADIOS2_PROFILE"] = "OFF"
            os.environ["ADIOS2_PROFILE_LEVEL"] = "0"
        
        # Create output directory (rank 0 only, then barrier)
        if comm.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    def register(
        self,
        key: str,
        fields: Sequence[fem.Function],
        filename: str | None = None,
        engine: str = "bp4",
    ) -> None:
        """Register VTX writer (MPI-collective)."""
        path = self.output_dir / (filename or f"{key}.bp")
        writer = VTXWriter(self.comm, str(path), list(fields), engine=engine)
        self._writers[key] = writer
        self._write_counts[key] = 0
        self.logger.debug(lambda: f"Registered '{key}': {path}")

    def write(self, key: str, t: float) -> None:
        """Write timestep (MPI-collective)."""
        self._writers[key].write(t)
        self._write_counts[key] += 1

    def close(self) -> None:
        """Close all VTX writers (COLLECTIVE)."""
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


class UnifiedStorage:
    """Field (VTX) storage."""

    __slots__ = ("comm", "fields")

    def __init__(self, cfg: "Config") -> None:
        self.comm = cfg.domain.comm
        self.fields = FieldStorage(cfg, self.comm)

    def write_fields(self, key: str, t: float) -> None:
        """Write fields (COLLECTIVE)."""
        self.fields.write(key, t)

    def close(self) -> None:
        """Close all storage (COLLECTIVE)."""
        self.fields.close()

    def __enter__(self) -> "UnifiedStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()
