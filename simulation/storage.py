"""VTX field output and CSV metrics (MPI-collective)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import VTXWriter

if TYPE_CHECKING:
    from simulation.config import Config
    from simulation.stats import SweepStats

from simulation.logger import get_logger


class FieldStorage:
    """Manages VTX writers for field output."""

    __slots__ = ("comm", "logger", "output_dir", "_writers", "_write_counts")

    def __init__(self, cfg: "Config", comm: MPI.Comm) -> None:
        self.comm = comm
        self.logger = get_logger(comm, name="Storage.Fields")
        self.output_dir = Path(cfg.output.results_dir)
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
        """Register VTX writer (collective)."""
        path = self.output_dir / (filename or f"{key}.bp")
        writer = VTXWriter(self.comm, str(path), list(fields), engine=engine)
        self._writers[key] = writer
        self._write_counts[key] = 0
        self.logger.debug(lambda: f"Registered '{key}': {path}")

    def write(self, key: str, t: float) -> None:
        """Write field group (collective)."""
        self._writers[key].write(t)
        self._write_counts[key] += 1

    def close(self) -> None:
        """Close all writers (collective)."""
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
    """CSV writer for solver performance telemetry (rank 0 only).
    
    Writes two CSV files:
    - steps.csv: One row per accepted timestep (aggregated stats)
    - subiterations.csv: One row per Picard/coupling iteration (detailed stats)
    """

    __slots__ = (
        "comm", "logger", "output_dir",
        "_steps_file", "_steps_writer", "_steps_header_written",
        "_subiters_file", "_subiters_writer", "_subiters_header_written",
    )

    # Column definitions (order matters for CSV)
    STEPS_COLUMNS = [
        "step", "time_days", "dt_days", "num_subiters", "converged",
        "error_norm", "mech_iters", "fab_iters", "stim_iters", "dens_iters",
        "mech_time", "fab_time", "stim_time", "dens_time",
        "max_condH", "aa_rejections", "aa_restarts", "memory_mb",
    ]

    SUBITERS_COLUMNS = [
        "step", "iter", "time_days", "proj_res", "aa_step_res",
        "mech_iters", "fab_iters", "stim_iters", "dens_iters",
        "mech_time", "fab_time", "stim_time", "dens_time",
        "restart", "restart_reason", "limited",
        "condH", "aa_hist", "memory_mb",
    ]

    def __init__(self, cfg: "Config", comm: MPI.Comm) -> None:
        self.comm = comm
        self.logger = get_logger(comm, name="Storage.Metrics")
        self.output_dir = Path(cfg.output.results_dir)

        self._steps_file = None
        self._steps_writer = None
        self._steps_header_written = False

        self._subiters_file = None
        self._subiters_writer = None
        self._subiters_header_written = False

        # Create output directory (rank 0 only, then barrier)
        if comm.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    def _ensure_files_open(self) -> None:
        """Open CSV files lazily on first write (rank 0 only)."""
        if self.comm.rank != 0:
            return

        if self._steps_file is None:
            steps_path = self.output_dir / "steps.csv"
            self._steps_file = open(steps_path, "w", newline="", encoding="utf-8")
            self._steps_writer = csv.DictWriter(
                self._steps_file, fieldnames=self.STEPS_COLUMNS, extrasaction="ignore"
            )

        if self._subiters_file is None:
            subiters_path = self.output_dir / "subiterations.csv"
            self._subiters_file = open(subiters_path, "w", newline="", encoding="utf-8")
            self._subiters_writer = csv.DictWriter(
                self._subiters_file, fieldnames=self.SUBITERS_COLUMNS, extrasaction="ignore"
            )

    def write_step(
        self,
        step: int,
        time_days: float,
        dt_days: float,
        converged: bool,
        error_norm: float,
        subiter_metrics: List[Dict[str, Any]],
        memory_mb: float = 0.0,
    ) -> None:
        """Write one timestep's metrics to CSV (rank 0 only).
        
        Args:
            step: Timestep index (1-based).
            time_days: Simulation time after this step [days].
            dt_days: Timestep size [days].
            converged: Whether coupling converged.
            error_norm: WRMS error norm for adaptive timestepping.
            subiter_metrics: List of per-iteration metric dicts from FixedPointSolver.
            memory_mb: Peak memory usage [MB].
        """
        if self.comm.rank != 0:
            return

        self._ensure_files_open()

        # Aggregate per-block stats from subiterations
        block_iters: Dict[str, int] = defaultdict(int)
        block_times: Dict[str, float] = defaultdict(float)
        max_condH = 0.0
        aa_rejections = 0
        aa_restarts = 0

        for rec in subiter_metrics:
            max_condH = max(max_condH, rec.get("condH", 0.0))
            if not rec.get("aa_accepted", True):
                aa_rejections += 1
            if rec.get("aa_restart"):
                aa_restarts += 1

            # Extract block stats (list of SweepStats)
            for stats in rec.get("block_stats", []):
                label = stats.label
                block_iters[label] += stats.ksp_iters
                block_times[label] += stats.solve_time

        # Write step row
        step_row = {
            "step": step,
            "time_days": time_days,
            "dt_days": dt_days,
            "num_subiters": len(subiter_metrics),
            "converged": int(converged),
            "error_norm": error_norm,
            "mech_iters": block_iters.get("mech", 0),
            "fab_iters": block_iters.get("fab", 0),
            "stim_iters": block_iters.get("stim", 0),
            "dens_iters": block_iters.get("dens", 0),
            "mech_time": block_times.get("mech", 0.0),
            "fab_time": block_times.get("fab", 0.0),
            "stim_time": block_times.get("stim", 0.0),
            "dens_time": block_times.get("dens", 0.0),
            "max_condH": max_condH,
            "aa_rejections": aa_rejections,
            "aa_restarts": aa_restarts,
            "memory_mb": memory_mb,
        }

        if not self._steps_header_written:
            self._steps_writer.writeheader()
            self._steps_header_written = True
        self._steps_writer.writerow(step_row)
        self._steps_file.flush()

        # Write subiteration rows
        if not self._subiters_header_written:
            self._subiters_writer.writeheader()
            self._subiters_header_written = True

        for rec in subiter_metrics:
            # Extract per-block stats for this iteration
            iter_block_iters: Dict[str, int] = {}
            iter_block_times: Dict[str, float] = {}
            for stats in rec.get("block_stats", []):
                iter_block_iters[stats.label] = stats.ksp_iters
                iter_block_times[stats.label] = stats.solve_time

            subiter_row = {
                "step": step,
                "iter": rec.get("iter", 0),
                "time_days": time_days,
                "proj_res": rec.get("proj_res", 0.0),
                "aa_step_res": rec.get("aa_step_res", 0.0),
                "mech_iters": iter_block_iters.get("mech", 0),
                "fab_iters": iter_block_iters.get("fab", 0),
                "stim_iters": iter_block_iters.get("stim", 0),
                "dens_iters": iter_block_iters.get("dens", 0),
                "mech_time": iter_block_times.get("mech", 0.0),
                "fab_time": iter_block_times.get("fab", 0.0),
                "stim_time": iter_block_times.get("stim", 0.0),
                "dens_time": iter_block_times.get("dens", 0.0),
                "restart": 1 if rec.get("aa_restart") else 0,
                "restart_reason": str(rec.get("aa_restart", "")),
                "limited": 1 if rec.get("aa_limited") else 0,
                "condH": rec.get("condH", 0.0),
                "aa_hist": rec.get("aa_hist", 0),
                "memory_mb": rec.get("mem_mb", 0.0),
            }
            self._subiters_writer.writerow(subiter_row)

        self._subiters_file.flush()

        self.logger.debug(lambda: f"Wrote step {step} metrics ({len(subiter_metrics)} subiters)")

    def close(self) -> None:
        """Close CSV files (rank 0 only)."""
        if self.comm.rank == 0:
            if self._steps_file is not None:
                self._steps_file.close()
                self._steps_file = None
            if self._subiters_file is not None:
                self._subiters_file.close()
                self._subiters_file = None
        self.comm.Barrier()

    def __enter__(self) -> "MetricsStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class UnifiedStorage:
    """Top-level storage wrapper."""

    __slots__ = ("comm", "fields", "metrics")

    def __init__(self, cfg: "Config") -> None:
        self.comm = cfg.domain.comm
        self.fields = FieldStorage(cfg, self.comm)
        self.metrics = MetricsStorage(cfg, self.comm)

    def close(self) -> None:
        """Close storage (collective)."""
        self.fields.close()
        self.metrics.close()

    def __enter__(self) -> "UnifiedStorage":
        return self

    def __exit__(self, *_) -> None:
        self.close()
