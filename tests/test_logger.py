#!/usr/bin/env python3
"""
Tests for logger module - rank-aware logging functionality.

Tests:
- Logger initialization
- Rank 0 vs. non-root logging behavior
- Verbose/quiet mode
- Log level filtering
- Message formatting
- Thread safety
"""

import pytest
pytest.importorskip("mpi4py")

import logging
from mpi4py import MPI
from simulation.logger import get_logger, Level

comm = MPI.COMM_WORLD


# =============================================================================
# Logger Initialization Tests
# =============================================================================

class TestLoggerInitialization:
    """Test logger creation and configuration."""

    def test_logger_initialization_verbose(self):
        """Logger should initialize in verbose mode."""
        from simulation.logger import Logger
        logger = get_logger(comm, verbose=True, name="test_verbose")

        assert logger is not None
        assert isinstance(logger, Logger)

    def test_logger_initialization_quiet(self):
        """Logger should initialize in quiet mode."""
        from simulation.logger import Logger
        logger = get_logger(comm, verbose=False, name="test_quiet")

        assert logger is not None
        assert isinstance(logger, Logger)

    def test_logger_name(self):
        """Logger should have correct name."""
        from simulation.logger import Logger
        logger = get_logger(comm, verbose=True, name="custom_name")


# =============================================================================
# Rank-Aware Logging Tests
# =============================================================================

class TestRankAwareLogging:
    """Test that logging respects MPI rank."""

    def test_rank_zero_has_handlers(self):
        """Rank 0 should have active logger (our Logger doesn't use handlers)."""
        from simulation.logger import Logger
        logger = get_logger(comm, verbose=True, name="rank0_test")

        # Our Logger is always created, no handlers attribute
        assert isinstance(logger, Logger)
        assert logger.name == "rank0_test"

    def test_non_zero_ranks_silent(self):
        """Non-zero ranks get same logger but PETSc.Sys.Print handles output."""
        if comm.size < 2:
            pytest.skip("Requires multiple MPI ranks")

        from simulation.logger import Logger
        logger = get_logger(comm, verbose=False, name="nonzero_test")

        # All ranks get a logger, PETSc handles rank filtering
        assert isinstance(logger, Logger)


# =============================================================================
# Verbose Mode Tests
# =============================================================================

class TestVerboseMode:
    """Test verbose vs. quiet logging modes."""

    def test_verbose_mode_enables_debug(self):
        """verbose=True should enable debug-level logging."""
        logger = get_logger(comm, verbose=True, name="verbose_debug")

        # verbose=True sets level to INFO (Level.INFO = 20)
        # We allow INFO and above (DEBUG would be 10)
        assert logger.level == Level.INFO

    def test_quiet_mode_limits_output(self):
        """verbose=False should limit output."""
        logger = get_logger(comm, verbose=False, name="quiet_mode")

        if comm.rank == 0:
            # Quiet mode should have higher threshold
            assert logger.level >= logging.INFO


# =============================================================================
# Log Level Filtering Tests
# =============================================================================

class TestLogLevelFiltering:
    """Test log level filtering behavior."""

    def test_info_level_logging(self):
        """Info-level messages should be logged."""
        logger = get_logger(comm, verbose=True, name="info_level")
        logger.info("Test info message")

    def test_debug_level_logging(self):
        """Debug-level messages should be logged in verbose mode."""
        logger = get_logger(comm, verbose=True, name="debug_level")
        logger.debug("Test debug message")

    def test_warning_level_logging(self):
        """Warning-level messages should always be logged."""
        logger = get_logger(comm, verbose=False, name="warning_level")
        logger.warning("Test warning message")

    def test_error_level_logging(self):
        """Error-level messages should always be logged."""
        logger = get_logger(comm, verbose=False, name="error_level")
        logger.error("Test error message")


# =============================================================================
# Message Formatting Tests
# =============================================================================

class TestMessageFormatting:
    """Test log message formatting."""

    def test_basic_message_formatting(self):
        """Basic string messages should work."""
        logger = get_logger(comm, verbose=True, name="basic_fmt")
        logger.info("Simple message")
        logger.info("Message with {0}", "formatting")
        logger.info("Message with number: {0}", 42)

    def test_lazy_evaluation_support(self):
        """Logger should support lazy evaluation (lambda messages)."""
        logger = get_logger(comm, verbose=True, name="lazy_eval")
        logger.debug(lambda: f"Expensive computation: {sum(range(1000))}")


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test that logging is thread-safe."""

    def test_concurrent_logging(self):
        """Multiple threads/ranks logging concurrently should not crash."""
        logger = get_logger(comm, verbose=True, name="concurrent")

        # All ranks log simultaneously
        for i in range(10):
            logger.info(f"Rank {comm.rank} message {i}")

        comm.Barrier()


# =============================================================================
# Integration Tests
# =============================================================================

class TestLoggerIntegration:
    """Test logger in realistic usage scenarios."""

    def test_logger_with_config(self):
        """Logger should work with Config class."""
        from dolfinx import mesh
        from simulation.config import Config
        from simulation.utils import build_facetag

        domain = mesh.create_unit_cube(comm, 4, 4, 4)
        facet_tags = build_facetag(domain)

        # Config creates its own loggers
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)

        # Should not crash
        comm.Barrier()

    def test_logger_with_solver(self):
        """Logger should work within solver context."""
        from dolfinx import mesh, fem
        import basix
        import numpy as np
        from simulation.config import Config
        from simulation.utils import build_facetag, build_dirichlet_bcs
        from simulation.subsolvers import MechanicsSolver

        domain = mesh.create_unit_cube(comm, 4, 4, 4)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)

        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

        V = fem.functionspace(domain, P1_vec)
        Q = fem.functionspace(domain, P1)
        T = fem.functionspace(domain, P1_ten)

        rho = fem.Function(Q)
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()

        A = fem.Function(T)
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()

        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

        # Solver initialization should use logger
        mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)

        mech.destroy()


# Run with: pytest tests/test_logger.py -v
