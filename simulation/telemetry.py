"""Experiment tracking: metadata JSON and buffered CSV streams (rank-0 only)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from mpi4py import MPI

from simulation.logger import get_logger


def _iso_utc(dt: datetime) -> str:
    """Format datetime as ISO-8601 Zulu (UTC) string."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


FLUSH_INTERVAL: int = 20


class Telemetry:
    """Buffered CSV event streams and JSON metadata (rank-0 I/O)."""

    def __init__(
        self,
        comm: MPI.Comm,
        outdir: str,
        verbose: bool = True,
    ):
        self.comm = comm
        self.outdir = Path(outdir)
        self.logger = get_logger(comm, verbose=(verbose is True), name="Telemetry")

        if comm.rank == 0:
            self.outdir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        self._csv_files: Dict[str, object] = {}
        self._csv_writers: Dict[str, object] = {}
        self._buffers: Dict[str, List[Dict]] = {}
        self._start_time = datetime.now(timezone.utc)

        self.logger.info("Telemetry initialized")

    @property
    def is_root(self) -> bool:
        return self.comm.rank == 0

    def register_csv(
        self,
        stream_name: str,
        columns: List[str],
        filename: str | None = None,
    ) -> None:
        """Register CSV stream with header (rank-0 only)."""
        if self.comm.rank != 0:
            return

        if filename is None:
            filename = f"{stream_name}.csv"

        path = self.outdir / filename
        f = open(path, "w", newline="")

        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        self._csv_files[stream_name] = f
        self._csv_writers[stream_name] = writer
        self._buffers[stream_name] = []
        self.logger.debug(lambda: f"Registered CSV stream '{stream_name}' at {path}")

    def record(self, stream_name: str, data: Dict[str, Any], csv_event: bool = True) -> None:
        """Buffer event row (rank-0 only)."""
        if self.comm.rank != 0 or not csv_event:
            return
        if stream_name not in self._csv_writers:
            raise KeyError(f"Telemetry stream '{stream_name}' not registered")

        self._buffers[stream_name].append(dict(data))
        if len(self._buffers[stream_name]) >= FLUSH_INTERVAL:
            self._flush(stream_name)

    def _flush(self, stream_name: str) -> None:
        """Write buffered events to CSV (rank-0 only)."""
        if self.comm.rank != 0 or stream_name not in self._buffers:
            return

        writer = self._csv_writers[stream_name]
        for record in self._buffers[stream_name]:
            writer.writerow(record)

        self._buffers[stream_name].clear()
        self._csv_files[stream_name].flush()

    def flush_all(self) -> None:
        """Flush all streams (rank-0 only)."""
        for stream_name in list(self._buffers.keys()):
            self._flush(stream_name)

    def write_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "metadata.json",
        overwrite: bool = True,
    ) -> None:
        """Write/merge JSON metadata with standard fields (rank-0 only)."""
        if self.comm.rank != 0:
            return

        path = self.outdir / filename

        # Always inject standard fields
        metadata.setdefault("start_time", _iso_utc(self._start_time))
        metadata.setdefault("mpi_size", self.comm.size)

        if not overwrite and path.exists():
            with open(path, "r") as f:
                existing = json.load(f)
            existing.update(metadata)
            metadata = existing

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.debug(lambda: f"Wrote telemetry metadata JSON to {path}")

    def close(self) -> None:
        """Flush and close all streams (rank-0 only)."""
        self.flush_all()

        if self.comm.rank == 0:
            for f in self._csv_files.values():
                f.close()

        self._csv_files.clear()
        self._csv_writers.clear()
        self._buffers.clear()

        self.logger.info("Telemetry closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
