"""MPI-safe logging: rank-0 only output via PETSc.Sys.Print with lazy evaluation."""

from enum import IntEnum
from typing import Any, Callable, Union

from mpi4py import MPI
from petsc4py import PETSc

# Global configuration for log file
_GLOBAL_LOG_FILE: Union[str, None] = None


def configure_logging(filepath: str) -> None:
    """Set the global log file path for all loggers."""
    global _GLOBAL_LOG_FILE
    _GLOBAL_LOG_FILE = filepath


class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class Logger:
    """Rank-0 logger with lazy string evaluation."""

    __slots__ = ("comm", "console_level", "file_level", "name", "prefix", "filepath")

    def __init__(self, comm: MPI.Comm, console_level: Level, file_level: Level, name: str, filepath: str = None):
        self.comm = comm
        self.console_level = console_level
        self.file_level = file_level
        self.name = name
        self.prefix = f"[{name}] " if name else ""
        
        # Use provided filepath or fall back to global configuration
        global _GLOBAL_LOG_FILE
        self.filepath = filepath if filepath is not None else _GLOBAL_LOG_FILE

    def is_enabled_for(self, lvl: Level) -> bool:
        """Check if level enabled."""
        if self.filepath:
            return lvl >= self.console_level or lvl >= self.file_level
        return lvl >= self.console_level

    def _format(self, msg: Union[str, Callable[[], str]], args: tuple) -> str:
        """Evaluate lazy message and format args."""
        text = msg() if callable(msg) else str(msg)
        return self.prefix + (text.format(*args) if args else text)

    def log(self, lvl: Level, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        """Log if level enabled (rank-0 only output)."""
        if self.is_enabled_for(lvl):
            text = self._format(msg, args)
            
            if lvl >= self.console_level:
                PETSc.Sys.Print(text, comm=self.comm)
            
            if self.filepath and self.comm.rank == 0 and lvl >= self.file_level:
                try:
                    with open(self.filepath, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
                except IOError:
                    pass  # Fail silently if cannot write to log file

    def debug(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.DEBUG, msg, *args)

    def info(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.INFO, msg, *args)

    def warning(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.WARNING, msg, *args)

    def error(self, msg: Union[str, Callable[[], str]], *args: Any) -> None:
        self.log(Level.ERROR, msg, *args)


def get_logger(comm: MPI.Comm, verbose: bool, name: str = "", filepath: str = None) -> Logger:
    """Create logger with level INFO (verbose=True) or WARNING (verbose=False)."""
    console_level = Level.INFO if verbose else Level.WARNING
    file_level = Level.INFO
    return Logger(comm, console_level, file_level, name, filepath)

