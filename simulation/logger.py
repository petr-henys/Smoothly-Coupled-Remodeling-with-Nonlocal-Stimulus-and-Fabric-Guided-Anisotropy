"""MPI-safe logging via PETSc.Sys.Print (rank-0 only)."""

import logging
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from mpi4py import MPI
from petsc4py import PETSc


# ---------------------------------------------------------------------------
# MPI-safe PETSc-based Logger (for parallel simulation code)
# ---------------------------------------------------------------------------

class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


class Logger:
    """Rank-0 logger with lazy message evaluation."""

    __slots__ = ("comm", "console_level", "file_level", "name", "prefix", "log_file")

    def __init__(self, comm: MPI.Comm, console_level: Level, file_level: Level, name: str, log_file: str | None = None):
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


def get_logger(comm: MPI.Comm, name: str = "", log_file: str | None = None) -> Logger:
    """Create an MPI-safe logger with default levels (console: WARNING, file: DEBUG)."""
    console_level = Level.WARNING
    file_level = Level.DEBUG
    return Logger(comm, console_level, file_level, name, log_file)


# ---------------------------------------------------------------------------
# Standard Python logging utilities (for non-MPI scripts)
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "WARNING",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """Configure the root logger with console and optional file handlers."""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    level_upper = level.upper()
    level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    if level_upper not in level_map:
        raise ValueError(f"Invalid log level: {level!r}. Must be one of {list(level_map.keys())}")
    numeric_level = level_map[level_upper]
    
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)
    
    logging.getLogger('mpi4py.MPI').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def get_std_logger(name: str) -> logging.Logger:
    """Get a standard Python logger by name (for non-MPI scripts)."""
    if name == "__main__":
        import __main__
        if hasattr(__main__, '__file__') and __main__.__file__:
            return logging.getLogger(f"main.{Path(__main__.__file__).stem}")
        return logging.getLogger("main")
    if name.startswith("src."):
        return logging.getLogger(name[4:])
    return logging.getLogger(name)


def get_class_logger(cls) -> logging.Logger:
    """Get a standard Python logger for a class instance."""
    module_name = cls.__class__.__module__
    class_name = cls.__class__.__name__
    if module_name == "__main__":
        return logging.getLogger(class_name)
    if module_name.startswith("src."):
        module_name = module_name[4:]
    return logging.getLogger(f"{module_name}.{class_name}")
