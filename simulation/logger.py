"""
MPI-aware logging - rank 0 only output via PETSc.Sys.Print.

Simple, explicit design:
- Only rank 0 prints (MPI-safe by default)
- Lazy message evaluation via callables
- Standard levels: DEBUG < INFO < WARNING < ERROR
- No environment variables, no fallbacks

Usage:
    logger = get_logger(comm, verbose=True, name="Solver")
    logger.info("Iteration {0}", iter_count)
    logger.debug(lambda: f"Expensive: {compute_stats()}")
"""

from enum import IntEnum
from typing import Any, Callable, Union

from mpi4py import MPI
from petsc4py import PETSc


class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class Logger:
    """Rank-0 only logger with lazy evaluation."""

    __slots__ = ("comm", "level", "name", "prefix")

    def __init__(self, comm: MPI.Comm, level: Level, name: str):
        self.comm = comm
        self.level = level
        self.name = name
        self.prefix = f"[{name}] " if name else ""

    def is_enabled_for(self, lvl: Level) -> bool:
        """Check if level is enabled."""
        return lvl >= self.level

    def _format(self, msg: Union[str, Callable[[], str]], args: tuple) -> str:
        """Evaluate message (lazy if callable) and format with args."""
        text = msg() if callable(msg) else str(msg)
        return self.prefix + (text.format(*args) if args else text)

    def log(self, lvl: Level, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        """Log message if level enabled. Rank-0 only output."""
        if self.is_enabled_for(lvl):
            PETSc.Sys.Print(self._format(msg, args), comm=self.comm)

    def debug(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.DEBUG, msg, *args)

    def info(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.INFO, msg, *args)

    def warning(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.WARNING, msg, *args)

    def error(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.ERROR, msg, *args)


def get_logger(comm: MPI.Comm, verbose: bool, name: str = "") -> Logger:
    """Create logger. verbose=True → INFO, verbose=False → WARNING."""
    level = Level.INFO if verbose else Level.WARNING
    return Logger(comm, level, name)

