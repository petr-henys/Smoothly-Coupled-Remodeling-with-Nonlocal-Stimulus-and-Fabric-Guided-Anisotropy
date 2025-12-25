"""Rich progress bar for simulation (rank-0 only).

Provides:
- ProgressReporter: For single simulation runs (2 bars: remodeling + coupling)
- SweepProgressReporter: For parameter sweeps (3 bars: sweep + remodeling + coupling)

Both use a single Rich Progress instance to avoid line jumping.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mpi4py import MPI
    from rich.progress import Progress, TaskID


def _is_interactive() -> bool:
    """Check if we're in an interactive terminal (not pytest/CI)."""
    # Pytest sets this
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False
    # CI environments
    if os.environ.get("CI"):
        return False
    # Check if stderr is a TTY
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class ProgressReporter:
    """Rich progress bar: main task (simulation time) + subtask (coupling iters).
    
    For standalone simulation runs. For sweeps, use SweepProgressReporter instead.
    """

    def __init__(self, comm: "MPI.Comm", total_time: float, max_subiters: int):
        """Initialize progress reporter.

        Args:
            comm: MPI communicator.
            total_time: Total simulation time [days].
            max_subiters: Maximum coupling subiterations per step.
        """
        self.rank = comm.rank
        self.progress: "Progress | None" = None
        self.main_task_id: "TaskID | None" = None
        self.sub_task_id: "TaskID | None" = None
        self._total_time = total_time
        self._max_subiters = max_subiters
        self._owns_progress = True  # We own the Progress instance

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

        # Only force terminal in interactive mode to avoid hanging in pytest
        console = Console(stderr=True, force_terminal=_is_interactive())
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn("{task.fields[info]}"),
            console=console,
            transient=False,
            refresh_per_second=4,
        )
        self.main_task_id = self.progress.add_task(
            "Remodeling", total=self._total_time, info=""
        )
        self.sub_task_id = self.progress.add_task(
            "  Coupling", total=self._max_subiters, info=""
        )

    def start(self) -> None:
        """Start the progress display."""
        if self.progress is not None and self._owns_progress:
            self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self.progress is not None and self._owns_progress:
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
            info_str = f"t={t:5.1f}d dt={dt:.2f}d done"
        else:
            info_str = f"t={t:5.1f}d dt={dt:.2f}d err={error:.1e}"

        self.progress.update(self.main_task_id, completed=t, info=info_str)

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

        self.progress.update(self.sub_task_id, completed=current, info=info)

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


class SweepProgressReporter:
    """Three-level progress bar for parameter sweeps.
    
    Displays:
    - Sweep progress (run M of N)
    - Remodeling progress (simulation time)
    - Coupling progress (iterations)
    
    All in a single Rich Progress instance to prevent line jumping.
    """

    def __init__(self, comm: "MPI.Comm", total_runs: int, max_subiters: int = 30):
        """Initialize sweep progress reporter.

        Args:
            comm: MPI communicator.
            total_runs: Total number of parameter sweep runs.
            max_subiters: Default maximum coupling subiterations.
        """
        self.rank = comm.rank
        self.progress: "Progress | None" = None
        self.sweep_task_id: "TaskID | None" = None
        self.main_task_id: "TaskID | None" = None
        self.sub_task_id: "TaskID | None" = None
        self._total_runs = total_runs
        self._max_subiters = max_subiters
        self._current_total_time = 100.0  # Will be updated per run

        if self.rank == 0:
            self._setup()

    def _setup(self) -> None:
        """Set up Rich progress bar (rank 0 only)."""
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        # Only force terminal in interactive mode to avoid hanging in pytest
        console = Console(stderr=True, force_terminal=_is_interactive())
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[info]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True),
            console=console,
            transient=False,
            refresh_per_second=4,
        )
        # Sweep level (outermost)
        self.sweep_task_id = self.progress.add_task(
            "[bold]Sweep", total=self._total_runs, info=""
        )
        # Remodeling level (per-run simulation time)
        self.main_task_id = self.progress.add_task(
            "  Remodel", total=self._current_total_time, info=""
        )
        # Coupling level (innermost)
        self.sub_task_id = self.progress.add_task(
            "  Coupling", total=self._max_subiters, info=""
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

    # -------------------------------------------------------------------------
    # Sweep-level updates
    # -------------------------------------------------------------------------

    def start_run(self, run_idx: int, total_time: float, params_info: str = "") -> None:
        """Signal start of a new parameter run.
        
        Args:
            run_idx: Current run index (0-based).
            total_time: Total simulation time for this run [days].
            params_info: String describing current parameters.
        """
        if self.progress is None:
            return
        
        self._current_total_time = total_time
        
        # Update sweep bar info
        if self.sweep_task_id is not None:
            run_info = f"[{run_idx+1}/{self._total_runs}] {params_info}"
            self.progress.update(self.sweep_task_id, info=run_info)
        
        # Reset remodeling bar for new run
        if self.main_task_id is not None:
            self.progress.reset(self.main_task_id)
            self.progress.update(self.main_task_id, total=total_time, info="")
        
        # Reset coupling bar
        if self.sub_task_id is not None:
            self.progress.reset(self.sub_task_id)

    def finish_run(self) -> None:
        """Signal completion of current parameter run."""
        if self.progress is None or self.sweep_task_id is None:
            return
        self.progress.update(self.sweep_task_id, advance=1)

    # -------------------------------------------------------------------------
    # Simulation-level updates (compatible with ProgressReporter interface)
    # -------------------------------------------------------------------------

    def update_main(self, t: float, dt: float, error: float, done: bool = False) -> None:
        """Update remodeling progress bar."""
        if self.progress is None or self.main_task_id is None:
            return

        if done:
            info_str = f"t={t:.1f}d done"
        else:
            info_str = f"t={t:.1f}d dt={dt:.2f}d"

        self.progress.update(self.main_task_id, completed=t, info=info_str)

    def update_subiter(self, current: int, total: int | None = None, info: str = "") -> None:
        """Update coupling iteration progress bar."""
        if self.progress is None or self.sub_task_id is None:
            return

        if total is not None:
            self.progress.update(self.sub_task_id, total=total)

        self.progress.update(self.sub_task_id, completed=current, info=info)

    def reset_subiter(self) -> None:
        """Reset coupling progress for a new timestep."""
        if self.progress is None or self.sub_task_id is None:
            return
        self.progress.reset(self.sub_task_id)

    def __enter__(self) -> "SweepProgressReporter":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
