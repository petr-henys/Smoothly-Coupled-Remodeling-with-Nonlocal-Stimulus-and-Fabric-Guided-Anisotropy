"""Utility functions for parameter sweep scripts.

Reduces boilerplate in run_*_sweep.py scripts by providing common operations:
- Reporter reset for progress tracking
- Standard checkpoint writing
- MPI-safe output directory cleanup
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from mpi4py import MPI
    from dolfinx import fem
    from simulation.config import Config
    from simulation.model import Remodeller
    from simulation.progress import SweepProgressReporter

from simulation.checkpoint import CheckpointStorage


def reset_reporter(reporter: SweepProgressReporter | None, total_time: float) -> None:
    """Reset sweep progress reporter with new total_time for current run."""
    if reporter is None:
        return
    if reporter.progress is not None and reporter.main_task_id is not None:
        reporter.progress.reset(reporter.main_task_id)
        reporter.progress.update(reporter.main_task_id, total=total_time)


def write_standard_checkpoint(
    cfg: Config,
    remodeller: Remodeller,
    *,
    include_sigma: bool = False,
    include_Qbar: bool = False,
    extra_fields: Sequence[fem.Function] | None = None,
) -> None:
    """Write standard checkpoint with core fields at final time.

    Always writes: psi, rho, S, L (if available).
    Optionally writes: sigma, Qbar, and any extra_fields.

    Args:
        cfg: Simulation configuration.
        remodeller: Remodeller instance after simulation.
        include_sigma: Include stress tensor field.
        include_Qbar: Include stress-stress product field.
        extra_fields: Additional fields to checkpoint.
    """
    checkpoint = CheckpointStorage(cfg)
    final_time = cfg.time.total_time

    # Mechanics fields
    psi = remodeller.driver.stimulus_field()
    if psi is not None:
        checkpoint.write_function(psi, final_time)

    if include_sigma:
        sigma = remodeller.driver.sigma_field()
        if sigma is not None:
            checkpoint.write_function(sigma, final_time)

    if include_Qbar:
        Qbar = remodeller.driver.Qbar_field()
        if Qbar is not None:
            checkpoint.write_function(Qbar, final_time)

    # Core state fields
    state_fields = remodeller.registry.state_fields
    for name in ("rho", "S", "L"):
        f = state_fields.get(name)
        if f is not None:
            checkpoint.write_function(f, final_time)

    # Extra fields
    for f in extra_fields or []:
        checkpoint.write_function(f, final_time)

    checkpoint.close()


def clean_output_dir(path: Path, comm: MPI.Comm, logger=None) -> None:
    """MPI-safe cleanup of output directory before sweep.

    Only rank 0 performs the deletion. All ranks synchronize via Barrier.

    Args:
        path: Directory to remove.
        comm: MPI communicator.
        logger: Optional logger for info message.
    """
    if comm.rank == 0:
        if path.exists():
            if logger is not None:
                logger.info(f"Cleaning output directory: {path}")
            shutil.rmtree(path)
    comm.Barrier()
