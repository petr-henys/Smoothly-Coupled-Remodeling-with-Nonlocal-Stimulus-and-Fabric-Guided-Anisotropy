"""Telemetry system for experiment tracking and reproducibility."""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from mpi4py import MPI

from simulation.logger import get_logger


def _iso_utc(dt: datetime) -> str:
    """Format datetime as ISO-8601 Zulu (UTC) string."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class Telemetry:
    """Experiment tracking and telemetry system."""

    def __init__(
        self,
        comm: MPI.Comm,
        outdir: str,
        flush_interval: Optional[int] = None,
    ):
        self.comm = comm
        self.outdir = Path(outdir)
        self.logger = get_logger(comm, name="Telemetry")

        if comm.rank == 0:
            self.outdir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

        self._csv_files: Dict[str, object] = {}
        self._csv_writers: Dict[str, object] = {}
        self._buffers: Dict[str, List[Dict]] = {}
        self._flush_interval = max(1, int(20 if flush_interval is None else flush_interval))
        self._start_time = datetime.now(timezone.utc)

        self.logger.info("Telemetry initialized")

    @property
    def is_root(self) -> bool:
        return self.comm.rank == 0

    def register_csv(
        self,
        stream_name: str,
        columns: List[str],
        gz: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Register a CSV event stream."""
        if self.comm.rank != 0:
            return

        if filename is None:
            filename = f"{stream_name}.csv" + (".gz" if gz else "")

        path = self.outdir / filename
        if gz:
            f = gzip.open(path, "wt", newline="")
        else:
            f = open(path, "w", newline="")

        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        self._csv_files[stream_name] = f
        self._csv_writers[stream_name] = writer
        self._buffers[stream_name] = []
        self.logger.debug(lambda: f"Registered CSV stream '{stream_name}' at {path}")

    def record(self, stream_name: str, data: Dict[str, Any], csv_event: bool = True) -> None:
        """Record an event to a registered stream."""
        if self.comm.rank != 0 or not csv_event:
            return
        if stream_name not in self._csv_writers:
            raise KeyError(f"Telemetry stream '{stream_name}' not registered")

        self._buffers[stream_name].append(dict(data))
        if len(self._buffers[stream_name]) >= self._flush_interval:
            self._flush(stream_name)

    def _flush(self, stream_name: str) -> None:
        """Flush buffered events to disk."""
        if self.comm.rank != 0 or stream_name not in self._buffers:
            return

        writer = self._csv_writers[stream_name]
        for record in self._buffers[stream_name]:
            writer.writerow(record)

        self._buffers[stream_name].clear()
        self._csv_files[stream_name].flush()

    def flush_all(self) -> None:
        """Flush all buffered events."""
        for stream_name in list(self._buffers.keys()):
            self._flush(stream_name)

    def write_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "metadata.json",
        overwrite: bool = True,
        *,
        inject_standard_fields: bool = True,
    ) -> None:
        """Write run metadata to JSON file."""
        if self.comm.rank != 0:
            return

        path = self.outdir / filename

        if inject_standard_fields:
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
        """Flush and close all streams."""
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
