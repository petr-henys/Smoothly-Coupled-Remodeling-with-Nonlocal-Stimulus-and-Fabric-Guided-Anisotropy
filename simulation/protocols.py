"""Protocols defining interfaces for simulation components.

These protocols enable static type checking and document the expected
interface contracts for solver blocks and other pluggable components.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Protocol, Tuple, runtime_checkable

if TYPE_CHECKING:
    from dolfinx import fem


@runtime_checkable
class CouplingBlock(Protocol):
    """Protocol for blocks participating in the fixed-point coupling loop.

    Each block must provide:
    - `state_fields`: Tuple of fields that form the coupled state vector.
      Blocks with empty state_fields still run each sweep but do not
      contribute entries to the Anderson-mixed state.
    - `setup()`: One-time initialization (create matrices, KSP, etc.)
    - `assemble_lhs()`: Reassemble dt-dependent or field-dependent LHS
    - `sweep()`: One block update in the Gauss-Seidel iteration
    - `destroy()`: Release PETSc/solver resources

    Example usage:
        >>> def register_block(block: CouplingBlock) -> None:
        ...     block.setup()
        ...     for _ in range(max_iters):
        ...         info = block.sweep()
    """

    @property
    def state_fields(self) -> Tuple["fem.Function", ...]:
        """Fields that participate in the coupled fixed-point state vector.

        Returns an empty tuple if the block does not contribute state
        (e.g., mechanics solver that only produces derived quantities).
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

    def sweep(self) -> Dict:
        """Perform one Gauss-Seidel block update.

        Returns:
            Dict with at least 'label' (str) and 'reason' (int, KSP converged reason).
        """
        ...

    def destroy(self) -> None:
        """Release PETSc resources (KSP, matrices, vectors)."""
        ...
