"""MPI-safe logging: rank-0 only output via PETSc.Sys.Print with lazy evaluation."""

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
    """Rank-0 logger with lazy string evaluation."""

    __slots__ = ("comm", "level", "name", "prefix")

    def __init__(self, comm: MPI.Comm, level: Level, name: str):
        self.comm = comm
        self.level = level
        self.name = name
        self.prefix = f"[{name}] " if name else ""

    def is_enabled_for(self, lvl: Level) -> bool:
        """Check if level enabled."""
        return lvl >= self.level

    def _format(self, msg: Union[str, Callable[[], str]], args: tuple) -> str:
        """Evaluate lazy message and format args."""
        text = msg() if callable(msg) else str(msg)
        return self.prefix + (text.format(*args) if args else text)

    def log(self, lvl: Level, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        """Log if level enabled (rank-0 only output)."""
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
    """Create logger with level INFO (verbose=True) or WARNING (verbose=False)."""
    level = Level.INFO if verbose else Level.WARNING
    return Logger(comm, level, name)

