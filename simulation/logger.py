"""MPI-safe logging with rank-0 output via `PETSc.Sys.Print`."""

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
    """Rank-0 logger with optional lazy message evaluation."""

    __slots__ = ("comm", "console_level", "file_level", "name", "prefix", "log_file")

    def __init__(self, comm: MPI.Comm, console_level: Level, file_level: Level, name: str, log_file: str = None):
        self.comm = comm
        self.console_level = console_level
        self.file_level = file_level
        self.name = name
        self.prefix = f"[{name}] " if name else ""
        self.log_file = log_file

    def is_enabled_for(self, lvl: Level) -> bool:
        """Return True if `lvl` is enabled for console or file output."""
        return lvl >= self.console_level or (self.log_file is not None and lvl >= self.file_level)

    def _format(self, msg: Union[str, Callable[[], str]], args: tuple) -> str:
        """Format a message, evaluating callables lazily."""
        text = msg() if callable(msg) else str(msg)
        return self.prefix + (text.format(*args) if args else text)

    def log(self, lvl: Level, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        """Log a message to console and optionally to a file (rank 0 only)."""
        formatted = None
        
        # Console output
        if lvl >= self.console_level:
            if formatted is None: formatted = self._format(msg, args)
            PETSc.Sys.Print(formatted, comm=self.comm)
            
        # File output (Rank 0 only)
        if self.comm.rank == 0 and self.log_file and lvl >= self.file_level:
            if formatted is None: formatted = self._format(msg, args)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(formatted + "\n")

    def debug(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.DEBUG, msg, *args)

    def info(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.INFO, msg, *args)

    def warning(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.WARNING, msg, *args)

    def error(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.ERROR, msg, *args)


def get_logger(comm: MPI.Comm, name: str = "", log_file: str = None) -> Logger:
    """Create a logger with default levels (console: WARNING, file: DEBUG)."""
    console_level = Level.WARNING
    file_level = Level.DEBUG
    return Logger(comm, console_level, file_level, name, log_file)
