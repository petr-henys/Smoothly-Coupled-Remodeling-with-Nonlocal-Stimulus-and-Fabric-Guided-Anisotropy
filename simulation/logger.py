from __future__ import annotations

"""
Lightweight MPI-aware logging utilities.

This module centralises rank-aware logging so user code does not need to call
``PETSc.Sys.Print`` directly. Loggers expose a small API compatible with the
common ``logging.Logger`` subset that is currently used across the project.

Key features compared to the previous helper:
  * Lazy message evaluation: pass a callable to defer expensive string formatting.
  * Structured levels with ``is_enabled_for`` to gate heavy computations.
  * Cheap child logger creation to keep consistent prefixes per submodule.

The default ``get_logger`` uses the provided ``verbose`` flag; no environment
variables are consulted.
"""

from enum import IntEnum
from typing import Any, Callable, Union

from mpi4py import MPI
from petsc4py import PETSc


MessageFactory = Union[str, Callable[[], str]]


class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40


 


class Logger:
    """Minimal rank-aware logger."""

    __slots__ = ("_comm", "_level", "_name", "_prefix")

    def __init__(self, comm: MPI.Comm, level: Level = Level.INFO, name: str = ""):
        self._comm = comm
        self._level = Level(int(level))
        self._name = name.strip()
        self._prefix = f"[{self._name}] " if self._name else ""

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def level(self) -> Level:
        return self._level

    @level.setter
    def level(self, lvl: Level) -> None:
        self._level = Level(int(lvl))

    @property
    def name(self) -> str:
        return self._name

    def is_enabled_for(self, lvl: Level) -> bool:
        return lvl >= self._level

    def child(self, suffix: str) -> "Logger":
        name = f"{self._name}.{suffix}" if self._name else suffix
        return Logger(self._comm, level=self._level, name=name)

    # ------------------------------------------------------------------
    # Core emission logic
    # ------------------------------------------------------------------
    def _coerce_message(self, msg: MessageFactory, args: tuple[Any, ...]) -> str:
        if callable(msg):
            msg = msg()
        text = str(msg)
        if args:
            text = text.format(*args)
        return self._prefix + text

    def log(self, lvl: Level, msg: MessageFactory, *args: Any) -> None:
        if not self.is_enabled_for(lvl):
            return
        PETSc.Sys.Print(self._coerce_message(msg, args), comm=self._comm)

    def debug(self, msg: MessageFactory, *args: Any) -> None:
        self.log(Level.DEBUG, msg, *args)

    def info(self, msg: MessageFactory, *args: Any) -> None:
        self.log(Level.INFO, msg, *args)

    def warning(self, msg: MessageFactory, *args: Any) -> None:
        self.log(Level.WARNING, msg, *args)

    def error(self, msg: MessageFactory, *args: Any) -> None:
        self.log(Level.ERROR, msg, *args)


def get_logger(comm: MPI.Comm, verbose: bool = True, name: str = "") -> Logger:
    """Factory using only the verbose flag (no env overrides)."""
    level = Level.INFO if verbose else Level.WARNING
    return Logger(comm=comm, level=level, name=name)
