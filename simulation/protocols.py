"""Protocols defining interfaces for simulation components.

These protocols enable static type checking and document the expected
interface contracts for solver blocks and other pluggable components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from dolfinx import fem
    from simulation.stats import SweepStats


@runtime_checkable
class CouplingBlock(Protocol):
    """Protocol for blocks participating in the fixed-point coupling loop.

    Each block must provide:

    State fields (for fixed-point coupling and time integration):
    - `state_fields`: Tuple of fields that form the coupled state vector.
      Blocks with empty state_fields still run each sweep but do not
      contribute entries to the Anderson-mixed state.
    - `state_fields_old`: Tuple of old-step counterparts (same order as state_fields).

    Output fields (for storage/visualization):
    - `output_fields`: Tuple of fields to write to VTX storage.
      Can include state_fields and/or derived quantities.

    Lifecycle methods:
    - `setup()`: One-time initialization (create matrices, KSP, etc.)
    - `assemble_lhs()`: Reassemble dt-dependent or field-dependent LHS
    - `sweep()`: One block update in the Gauss-Seidel iteration (returns SweepStats)
    - `post_step_update()`: Hook called after each accepted timestep
      (e.g., compute derived fields like eigenvectors)
    - `destroy()`: Release PETSc/solver resources

    Example usage:
        >>> def register_block(block: CouplingBlock) -> None:
        ...     block.setup()
        ...     for _ in range(max_iters):
        ...         stats = block.sweep()  # Returns SweepStats
        ...     block.post_step_update()
    """

    @property
    def state_fields(self) -> Tuple["fem.Function", ...]:
        """Fields that participate in the coupled fixed-point state vector.

        Returns an empty tuple if the block does not contribute state
        (e.g., mechanics solver that only produces derived quantities).
        """
        ...

    @property
    def state_fields_old(self) -> Tuple["fem.Function", ...]:
        """Old-step counterparts for state_fields (same order).

        Used by the time integrator for AB2 prediction and by the
        orchestrator for rollback on rejected steps.

        Returns an empty tuple if state_fields is empty.
        """
        ...

    @property
    def output_fields(self) -> Tuple["fem.Function", ...]:
        """Fields to write to VTX storage.

        May include state_fields and/or derived quantities (e.g., eigenvectors).
        The orchestrator collects output_fields from all blocks for auto-registration.

        Returns an empty tuple if the block produces no output.
        """
        ...

    def setup(self) -> None:
        """Initialize solver: compile forms, allocate matrices, create KSP."""
        ...

    def assemble_lhs(self) -> None:
        """Reassemble the left-hand-side operator (e.g., stiffness matrix).

        Called when dt changes or when field-dependent coefficients update.
        """
        ...

    def sweep(self) -> "SweepStats":
        """Perform one Gauss-Seidel block update.

        Returns:
            SweepStats with mandatory fields (label, ksp_iters, ksp_reason,
            solve_time) and optional physics-specific data in `extra`.
        """
        ...

    def post_step_update(self) -> None:
        """Hook called after each accepted timestep.

        Use this to compute derived fields (e.g., eigenvectors from tensors)
        that are needed for output but not for the coupling iteration.

        Default implementation does nothing.
        """
        ...

    def destroy(self) -> None:
        """Release PETSc resources (KSP, matrices, vectors)."""
        ...
