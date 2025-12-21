"""Rich progress bar for simulation (rank-0 only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py import MPI


class ProgressReporter:
    """Rich progress bar: main task (simulation time) + subtask (coupling iters)."""

    def __init__(self, comm: "MPI.Comm", total_time: float, max_subiters: int):
        """Initialize progress reporter.

        Args:
            comm: MPI communicator.
            total_time: Total simulation time [days].
            max_subiters: Maximum coupling subiterations per step.
        """
        self.rank = comm.rank
        self.progress = None
        self.main_task_id = None
        self.sub_task_id = None
        self._total_time = total_time
        self._max_subiters = max_subiters

        if self.rank == 0:
            self._setup()

    def _setup(self) -> None:
        """Set up Rich progress bar (rank 0 only)."""
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        console = Console(stderr=True, force_terminal=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn("{task.fields[info]}"),
            console=console,
            transient=False,
        )
        self.main_task_id = self.progress.add_task(
            "Remodeling", total=self._total_time, info=" " * 35
        )
        self.sub_task_id = self.progress.add_task(
            "  Coupling", total=self._max_subiters, info=" " * 35
        )

    def start(self) -> None:
        """Start the progress display."""
        if self.progress is not None:
            self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self.progress is not None:
            self.progress.stop()
            self.progress = None

    def update_main(self, t: float, dt: float, error: float, done: bool = False) -> None:
        """Update main progress bar.

        Args:
            t: Current simulation time [days].
            dt: Current timestep [days].
            error: WRMS error estimate.
            done: Whether simulation is complete.
        """
        if self.progress is None or self.main_task_id is None:
            return

        if done:
            info_str = f"t={t:5.1f}d dt={dt:5.1f} done"
        else:
            info_str = f"t={t:5.1f}d dt={dt:5.1f} err={error:.1e}"

        self.progress.update(self.main_task_id, completed=t, info=f"{info_str:<35}")

    def update_subiter(self, current: int, total: int | None = None, info: str = "") -> None:
        """Update subiteration progress bar.

        Args:
            current: Current subiteration number.
            total: Total subiterations (updates task total if provided).
            info: Additional info string.
        """
        if self.progress is None or self.sub_task_id is None:
            return

        if total is not None:
            self.progress.update(self.sub_task_id, total=total)

        self.progress.update(self.sub_task_id, completed=current, info=f"{info:<35}")

    def reset_subiter(self) -> None:
        """Reset subiteration progress for a new timestep."""
        if self.progress is None or self.sub_task_id is None:
            return
        self.progress.reset(self.sub_task_id)

    def __enter__(self) -> "ProgressReporter":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
