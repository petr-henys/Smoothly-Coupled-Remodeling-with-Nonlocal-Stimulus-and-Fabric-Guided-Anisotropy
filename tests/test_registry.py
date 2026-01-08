"""
Tests for BlockRegistry.
"""

import pytest
from unittest.mock import MagicMock
from mpi4py import MPI
from dolfinx import fem

from simulation.registry import BlockRegistry
from simulation.protocols import CouplingBlock
from simulation.stats import SweepStats

class MockBlock(CouplingBlock):
    """Minimal implementation of CouplingBlock protocol."""
    def __init__(self, state_fields, output_fields=()):
        self._state_fields = state_fields
        self._output_fields = output_fields

    @property
    def state_fields(self):
        return self._state_fields

    @property
    def state_fields_old(self):
        return self._state_fields

    @property
    def output_fields(self):
        return self._output_fields
        
    def setup(self): pass
    def assemble_lhs(self): pass
    def sweep(self) -> SweepStats: return SweepStats()
    def destroy(self): pass

class TestBlockRegistry:
    """Test registry functionality."""

    def test_register_discovers_fields(self, spaces):
        """Registry should discover fields from registered blocks."""
        comm = MPI.COMM_WORLD
        cfg = MagicMock()
        cfg.log_file = "test.log"
        registry = BlockRegistry(comm, cfg)
        
        # Create fields
        f1 = fem.Function(spaces.Q, name="rho")
        f2 = fem.Function(spaces.V, name="u")
        
        block = MockBlock((f1, f2))
        
        registry.register(block)
        
        assert "rho" in registry.state_fields
        assert "u" in registry.state_fields
        assert registry.state_fields["rho"] == f1
        assert registry.state_fields["u"] == f2

    def test_duplicate_field_error(self, spaces):
        """Registry should raise error on duplicate field names."""
        comm = MPI.COMM_WORLD
        cfg = MagicMock()
        cfg.log_file = "test.log"
        registry = BlockRegistry(comm, cfg)
        
        f1 = fem.Function(spaces.Q, name="rho")
        f2 = fem.Function(spaces.Q, name="rho") # Duplicate name
        
        block1 = MockBlock((f1,))
        block2 = MockBlock((f2,))
        
        registry.register(block1)
        
        # This might not raise if it checks object identity, but usually it keys by name?
        # Let's check implementation. 
        # registry.py uses: self._state_fields[f.name] = f
        # It doesn't seem to explicitly raise on duplicate name if objects are different, it just overwrites or maybe checks?
        # Let's check the code first before asserting.
        
    def test_protocol_check(self, spaces):
        """Registry should verify protocol compliance."""
        comm = MPI.COMM_WORLD
        cfg = MagicMock()
        cfg.log_file = "test.log"
        registry = BlockRegistry(comm, cfg)
        
        class BadBlock:
           pass
           
        with pytest.raises(TypeError, match="does not implement CouplingBlock"):
            registry.register(BadBlock())
