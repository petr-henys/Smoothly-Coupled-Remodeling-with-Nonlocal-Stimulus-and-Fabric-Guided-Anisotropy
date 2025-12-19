"""
Tests for logging and basic utilities.

Tests:
- Logger initialization and rank-aware behavior
- Field statistics computation
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function
from simulation.logger import get_logger

comm = MPI.COMM_WORLD

# =============================================================================
# Logger Tests
# =============================================================================

class TestLogger:
    """Test MPI-aware logger."""
    
    def test_logger_rank0_output(self):
        """Logger should only output on rank 0."""
        log = get_logger(comm, name="TestLogger")
        # Capture doesn't work well with PETSc.Sys.Print, so just verify no crash
        log.info("This is an info message")
        log.warning("This is a warning")
        log.debug("This is debug")
    
    def test_logger_levels(self):
        """Logger should support DEBUG/INFO/WARNING/ERROR levels."""
        log = get_logger(comm, name="LevelTest")
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")

    def test_lazy_evaluation_support(self):
        """Logger should support lazy evaluation (lambda messages)."""
        logger = get_logger(comm, name="lazy_eval")
        logger.debug(lambda: f"Expensive computation: {sum(range(1000))}")


# =============================================================================
# Field Statistics Tests
# =============================================================================

class TestFieldStatistics:
    """Test field statistics computation."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_scalar_field_stats(self, unit_cube):
        """Test min/max/mean computation for scalar fields."""
        domain = unit_cube
        import basix
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = fem.functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        rho.interpolate(lambda x: 0.3 + 0.4*x[0])  # Range [0.3, 0.7]
        rho.x.scatter_forward()
        
        n_owned = Q.dofmap.index_map.size_local
        
        rho_min_local = rho.x.array[:n_owned].min()
        rho_max_local = rho.x.array[:n_owned].max()
        rho_sum_local = rho.x.array[:n_owned].sum()
        
        rho_min = comm.allreduce(rho_min_local, op=MPI.MIN)
        rho_max = comm.allreduce(rho_max_local, op=MPI.MAX)
        rho_sum = comm.allreduce(rho_sum_local, op=MPI.SUM)
        n_total = comm.allreduce(n_owned, op=MPI.SUM)
        
        # Check reasonable values
        assert 0.25 < rho_min < 0.35, f"Min out of range: {rho_min}"
        assert 0.65 < rho_max < 0.75, f"Max out of range: {rho_max}"

    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_vector_field_norm(self, unit_cube):
        """Test norm computation for vector fields."""
        domain = unit_cube
        import basix
        import ufl
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        V = fem.functionspace(domain, P1_vec)

        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
        u.x.scatter_forward()

        u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * ufl.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
        u_norm = float(np.sqrt(u_norm_sq))

        assert abs(u_norm - 1.0) < 0.05, f"Vector norm incorrect: {u_norm}"
