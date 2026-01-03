"""Protocol definitions for solver blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from dolfinx import fem
    from simulation.stats import SweepStats


@runtime_checkable
class CouplingBlock(Protocol):
    """Protocol defining the interface for solvers in the coupling loop.

    Required properties:
    - state_fields / state_fields_old: fields for coupling and time integration
    - output_fields: fields for VTX storage

    Required methods:
    - setup(), assemble_lhs(), sweep(), post_step_update(), destroy()
    """

    @property
    def state_fields(self) -> Tuple["fem.Function", ...]:
        """Fields for fixed-point state vector (empty if none)."""
        ...

    @property
    def state_fields_old(self) -> Tuple["fem.Function", ...]:
        """Old-step counterparts for state_fields (same order)."""
        ...

    @property
    def output_fields(self) -> Tuple["fem.Function", ...]:
        """Fields for VTX storage (may include state and derived quantities)."""
        ...

    def setup(self) -> None:
        """One-time initialization: compile forms, create KSP."""
        ...

    def assemble_lhs(self) -> None:
        """Reassemble LHS (e.g., when dt or coefficients change)."""
        ...

    def sweep(self) -> "SweepStats":
        """One Gauss-Seidel block update; returns SweepStats."""
        ...

    def post_step_update(self) -> None:
        """Hook after accepted timestep (e.g., compute derived fields)."""
        ...

    def destroy(self) -> None:
        """Release PETSc resources."""
        ...
