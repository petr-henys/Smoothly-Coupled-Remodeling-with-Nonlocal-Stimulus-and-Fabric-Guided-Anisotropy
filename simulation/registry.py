"""Block registry for automatic state/output field discovery."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

from mpi4py import MPI
from dolfinx import fem

from simulation.protocols import CouplingBlock
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config


class BlockRegistry:
    """Collects blocks and auto-discovers their state/output fields."""

    def __init__(self, comm: MPI.Comm, cfg: "Config"):
        self.comm = comm
        self.cfg = cfg
        self.logger = get_logger(comm, name="BlockRegistry", log_file=cfg.log_file)

        self._blocks: List[CouplingBlock] = []
        self._state_fields: Dict[str, fem.Function] = {}
        self._state_fields_old: Dict[str, fem.Function] = {}
        self._output_fields: List[fem.Function] = []

        # Track field IDs to avoid duplicates
        self._state_field_ids: set = set()
        self._output_field_ids: set = set()

    def register(self, block: CouplingBlock) -> None:
        """Register a coupling block and discover its fields.

        Args:
            block: A CouplingBlock instance implementing the protocol.

        Raises:
            TypeError: If block does not implement CouplingBlock protocol.
            ValueError: If state_fields and state_fields_old have different lengths.
        """
        # Runtime protocol check
        if not isinstance(block, CouplingBlock):
            raise TypeError(
                f"Block {type(block).__name__} does not implement CouplingBlock protocol."
            )

        self._blocks.append(block)

        # Discover state fields
        state = tuple(block.state_fields)
        state_old = tuple(block.state_fields_old)

        if len(state) != len(state_old):
            raise ValueError(
                f"Block {type(block).__name__} has mismatched state_fields ({len(state)}) "
                f"and state_fields_old ({len(state_old)}). They must have equal length."
            )

        for f, f_old in zip(state, state_old):
            fid = id(f)
            if fid not in self._state_field_ids:
                self._state_field_ids.add(fid)
                name = getattr(f, "name", f"field_{len(self._state_fields)}")
                self._state_fields[name] = f
                self._state_fields_old[name] = f_old
                self.logger.debug(f"Registered state field: {name}")

        # Discover output fields
        for f in tuple(block.output_fields):
            fid = id(f)
            if fid not in self._output_field_ids:
                self._output_field_ids.add(fid)
                self._output_fields.append(f)
                name = getattr(f, "name", "unnamed")
                self.logger.debug(f"Registered output field: {name}")

    @property
    def blocks(self) -> Tuple[CouplingBlock, ...]:
        """Return all registered blocks in registration order."""
        return tuple(self._blocks)

    @property
    def state_fields(self) -> Dict[str, fem.Function]:
        """Return mapping of state field names to Functions.

        Used by TimeIntegrator for AB2 prediction and error estimation.
        """
        return self._state_fields.copy()

    @property
    def state_fields_old(self) -> Dict[str, fem.Function]:
        """Return mapping of state field names to old-step counterparts.

        Used for rollback on rejected timesteps.
        """
        return self._state_fields_old.copy()

    @property
    def output_fields(self) -> List[fem.Function]:
        """Return list of all output fields for VTX storage.

        Fields are deduplicated and sorted: CG (Lagrange) first, then DG.
        This ensures VTXWriter initializes with the CG mesh topology, preventing
        "Point array size mismatch" errors when mixing Point (CG) and Cell (DG) data.
        """
        def _sort_key(f):
            # Use ufl_element() for robust family detection
            # basix.ufl elements have 'discontinuous' property
            return 0 if not f.function_space.ufl_element().discontinuous else 1

        return sorted(self._output_fields, key=_sort_key)

    def setup_all(self) -> None:
        """Call setup() on all registered blocks."""
        for block in self._blocks:
            block.setup()

    def assemble_lhs_all(self) -> None:
        """Call assemble_lhs() on all registered blocks."""
        for block in self._blocks:
            block.assemble_lhs()

    def post_step_update_all(self) -> None:
        """Call post_step_update() on all registered blocks.

        Use after each accepted timestep to compute derived fields.
        """
        for block in self._blocks:
            block.post_step_update()

    def destroy_all(self) -> None:
        """Call destroy() on all registered blocks."""
        for block in self._blocks:
            block.destroy()

    def __len__(self) -> int:
        """Return number of registered blocks."""
        return len(self._blocks)

    def __iter__(self):
        """Iterate over registered blocks."""
        return iter(self._blocks)
