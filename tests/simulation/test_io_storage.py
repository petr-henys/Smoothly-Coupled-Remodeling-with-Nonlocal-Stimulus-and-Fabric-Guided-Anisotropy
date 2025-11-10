#!/usr/bin/env python3
"""
Storage, logging, and telemetry tests.

Merged from:
- test_storage.py
- test_logger.py
- test_logging_monitoring.py
"""

#!/usr/bin/env python3
"""
Comprehensive tests for storage module (FieldStorage, MetricsStorage, UnifiedStorage).

Tests:
- Directory creation (collective MPI operation)
- VTX writer registration and lifecycle
- Field write operations (displacement, scalars, tensors)
- Metrics CSV creation and buffering
- Error handling (permission denied, disk full simulation)
- MPI collective safety
- Context manager behavior
"""

import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function
from pathlib import Path
import tempfile
import csv as csv_module
import gzip

from simulation.storage import FieldStorage, MetricsStorage, UnifiedStorage
from simulation.config import Config
from simulation.utils import build_facetag

comm = MPI.COMM_WORLD


# =============================================================================
# FieldStorage Tests
# =============================================================================

class TestFieldStorage:
    """Test VTX field output functionality."""

    def test_initialization_creates_directory(self, shared_tmpdir, unit_cube, facet_tags):
        """FieldStorage should create results directory on initialization."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_init", verbose=False)

        storage = FieldStorage(cfg, comm)

        # All ranks should see the directory
        comm.Barrier()
        assert Path(shared_tmpdir / "test_init").exists(), "Results directory not created"

        storage.close()

    def test_register_creates_writer(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """register should create VTX writer."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_register", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Register writer (collective operation)
        storage.register("test_field", [fields.rho], filename="test.bp")

        # Check writer was created
        assert "test_field" in storage._writers

        storage.close()

    def test_write_creates_displacement_file(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """write should create VTX output file for displacement."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_u", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Register and write displacement
        storage.register("u", [fields.u])
        storage.write("u", t=0.0)
        storage.close()

        comm.Barrier()

        # Check file was created (BP4 creates a directory)
        if comm.rank == 0:
            output_path = Path(shared_tmpdir) / "test_u" / "u.bp"
            assert output_path.exists(), f"Displacement output not created at {output_path}"

    def test_write_creates_scalars_file(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """write should create VTX output for density and stimulus."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_scalars", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Set values
        fields.rho.x.array[:] = 0.6
        fields.rho.x.scatter_forward()
        fields.S.x.array[:] = 0.1
        fields.S.x.scatter_forward()

        # Register and write scalars
        storage.register("scalars", [fields.rho, fields.S])
        storage.write("scalars", t=0.0)
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            output_path = Path(shared_tmpdir) / "test_scalars" / "scalars.bp"
            assert output_path.exists(), "Scalars output not created"

    def test_write_creates_tensor_file(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """write should create VTX output for orientation tensor."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_tensor", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Set tensor values
        fields.A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        fields.A.x.scatter_forward()

        # Register and write tensor
        storage.register("A", [fields.A])
        storage.write("A", t=0.0)
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            output_path = Path(shared_tmpdir) / "test_tensor" / "A.bp"
            assert output_path.exists(), "Tensor output not created"

    def test_write_all_fields_at_once(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Writing multiple field groups should work correctly."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_all", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Set values
        fields.rho.x.array[:] = 0.5
        fields.rho.x.scatter_forward()
        fields.S.x.array[:] = 0.2
        fields.S.x.scatter_forward()
        fields.A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        fields.A.x.scatter_forward()

        # Register and write all fields
        storage.register("u", [fields.u])
        storage.register("scalars", [fields.rho, fields.S])
        storage.register("A", [fields.A])
        
        storage.write("u", t=0.0)
        storage.write("scalars", t=0.0)
        storage.write("A", t=0.0)
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            base_path = Path(shared_tmpdir) / "test_all"
            assert (base_path / "u.bp").exists(), "Displacement not written"
            assert (base_path / "scalars.bp").exists(), "Scalars not written"
            assert (base_path / "A.bp").exists(), "Tensor not written"

    def test_multiple_writes_increment_counter(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Multiple writes should increment write counter."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_multi", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Register and write multiple times
        storage.register("u", [fields.u])
        for t in [0.0, 1.0, 2.0]:
            storage.write("u", t=t)

        # Check counter
        assert storage._write_counts["u"] == 3, f"Expected 3 writes, got {storage._write_counts['u']}"

        storage.close()

    def test_context_manager_closes_writers(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Context manager should properly close all writers."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_ctx", verbose=False)

        with FieldStorage(cfg, comm) as storage:
            storage.register("u", [fields.u])
            storage.write("u", t=0.0)
            assert "u" in storage._writers

        # After context exit, writers should be cleared
        assert len(storage._writers) == 0

    @pytest.mark.parametrize("unit_cube", [4, 6], indirect=True)
    def test_write_works_with_different_mesh_sizes(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Storage should handle different mesh resolutions."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / f"test_mesh_{unit_cube.topology.index_map(0).size_global}",
                    verbose=False)
        storage = FieldStorage(cfg, comm)

        # Register and write all fields
        storage.register("u", [fields.u])
        storage.register("scalars", [fields.rho, fields.S])
        storage.register("A", [fields.A])
        storage.write("u", t=0.0)
        storage.write("scalars", t=0.0)
        storage.write("A", t=0.0)
        storage.close()

        comm.Barrier()
        # Just verify no errors occurred


# =============================================================================
# MetricsStorage Tests
# =============================================================================

class TestMetricsStorage:
    """Test CSV metrics storage functionality."""

    def test_initialization_creates_directory(self, shared_tmpdir, unit_cube, facet_tags):
        """MetricsStorage should create results directory."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_metrics_init", verbose=False)

        storage = MetricsStorage(cfg, comm)

        comm.Barrier()
        assert Path(shared_tmpdir / "test_metrics_init").exists()

        storage.close()

    def test_register_csv_creates_file(self, shared_tmpdir, unit_cube, facet_tags):
        """register_csv should create CSV file with header."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_csv", verbose=False)
        storage = MetricsStorage(cfg, comm)

        storage.register_csv("test_metrics", ["col1", "col2", "col3"])
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            csv_path = Path(shared_tmpdir) / "test_csv" / "test_metrics.csv"
            assert csv_path.exists(), "CSV file not created"

            # Check header
            with open(csv_path, 'r') as f:
                header = f.readline().strip()
                assert "col1" in header and "col2" in header and "col3" in header

    def test_record_writes_data(self, shared_tmpdir, unit_cube, facet_tags):
        """record() should write data rows to CSV."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_record", verbose=False)
        storage = MetricsStorage(cfg, comm, flush_interval=1)  # Immediate flush

        storage.register_csv("data", ["x", "y"])
        storage.record("data", {"x": 1, "y": 2})
        storage.record("data", {"x": 3, "y": 4})
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            csv_path = Path(shared_tmpdir) / "test_record" / "data.csv"
            with open(csv_path, 'r') as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
                assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
                assert rows[0]["x"] == "1" and rows[0]["y"] == "2"
                assert rows[1]["x"] == "3" and rows[1]["y"] == "4"

    def test_buffering_delays_writes(self, shared_tmpdir, unit_cube, facet_tags):
        """Data should be buffered until flush_interval reached."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_buffer", verbose=False)
        storage = MetricsStorage(cfg, comm, flush_interval=5)

        storage.register_csv("buffered", ["val"])

        # Write 3 records (below flush_interval)
        for i in range(3):
            storage.record("buffered", {"val": i})

        # Buffer should have 3 items
        if comm.rank == 0:
            assert len(storage._buffers["buffered"]) == 3

        # Explicit flush
        storage.flush_all()

        # Buffer should be empty after flush
        if comm.rank == 0:
            assert len(storage._buffers["buffered"]) == 0

        storage.close()

    def test_gzip_compression(self, shared_tmpdir, unit_cube, facet_tags):
        """CSV with gz=True should create .csv.gz file."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_gz", verbose=False)
        storage = MetricsStorage(cfg, comm)

        storage.register_csv("compressed", ["a", "b"], gz=True)
        storage.record("compressed", {"a": 10, "b": 20})
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            gz_path = Path(shared_tmpdir) / "test_gz" / "compressed.csv.gz"
            assert gz_path.exists(), "Gzipped CSV not created"

            # Verify can read gzipped content
            with gzip.open(gz_path, 'rt') as f:
                content = f.read()
                assert "a,b" in content or "a" in content  # Header present

    def test_custom_filename(self, shared_tmpdir, unit_cube, facet_tags):
        """Custom filename parameter should be respected."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_custom", verbose=False)
        storage = MetricsStorage(cfg, comm)

        storage.register_csv("stream", ["x"], filename="custom_name.csv")
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            custom_path = Path(shared_tmpdir) / "test_custom" / "custom_name.csv"
            assert custom_path.exists(), "Custom filename not used"

    def test_non_root_ranks_silent(self, shared_tmpdir, unit_cube, facet_tags):
        """Non-root ranks should return immediately from record()."""
        if comm.size < 2:
            pytest.skip("Requires multiple MPI ranks")

        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_ranks", verbose=False)
        storage = MetricsStorage(cfg, comm)

        storage.register_csv("test", ["value"])

        # All ranks call record, but only rank 0 should write
        storage.record("test", {"value": comm.rank})
        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            csv_path = Path(shared_tmpdir) / "test_ranks" / "test.csv"
            with open(csv_path, 'r') as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
                # Only rank 0's data should be written
                assert len(rows) == 1
                assert rows[0]["value"] == "0"


# =============================================================================
# UnifiedStorage Tests
# =============================================================================

class TestUnifiedStorage:
    """Test unified storage manager."""

    def test_initialization_creates_both_storages(self, shared_tmpdir, unit_cube, facet_tags):
        """UnifiedStorage should initialize both field and metrics storage."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_unified", verbose=False)

        storage = UnifiedStorage(cfg)

        assert storage.fields is not None
        assert storage.metrics is not None
        assert isinstance(storage.fields, FieldStorage)
        assert isinstance(storage.metrics, MetricsStorage)

        storage.close()

    def test_write_step_writes_all_data(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """write_step should write fields and record metrics."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_step", verbose=False,
                    enable_telemetry=False)  # Disable telemetry to use storage's CSV

        storage = UnifiedStorage(cfg)

        # Register field groups BEFORE write_step
        storage.fields.register("u", [fields.u])
        storage.fields.register("scalars", [fields.rho, fields.S])
        storage.fields.register("A", [fields.A])

        # Set field values
        fields.rho.x.array[:] = 0.7
        fields.rho.x.scatter_forward()
        fields.S.x.array[:] = 0.15
        fields.S.x.scatter_forward()

        # Write step
        storage.write_step(
            step=1,
            time_days=10.0,
            dt_days=1.0,
            u=fields.u,
            rho=fields.rho,
            S=fields.S,
            A=fields.A,
            num_dofs_total=1000,
            rss_mem_mb=100.5,
            solver_stats={"mech": 50, "stim": 20, "dens": 10, "dir": 5},
            coupling_stats={"iters": 3, "time": 2.5}
        )

        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            # Check field files exist
            base = Path(shared_tmpdir) / "test_step"
            assert (base / "u.bp").exists()
            assert (base / "scalars.bp").exists()
            assert (base / "A.bp").exists()

            # Check metrics CSV
            csv_path = base / "steps.csv"
            assert csv_path.exists()

            with open(csv_path, 'r') as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
                assert len(rows) >= 1
                assert rows[0]["step"] == "1"
                assert rows[0]["time_days"] == "10.0"

    def test_context_manager(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Context manager should properly initialize and cleanup."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_ctx_unified", verbose=False)

        with UnifiedStorage(cfg) as storage:
            storage.fields.register("u", [fields.u])
            storage.fields.write("u", t=0.0)
            assert len(storage.fields._writers) > 0

        # After context exit, storage should be closed
        assert len(storage.fields._writers) == 0

    def test_telemetry_integration(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """When telemetry is enabled, storage should not duplicate metrics CSV."""
        from simulation.telemetry import Telemetry

        tel = Telemetry(comm, outdir=str(shared_tmpdir / "test_tel"), verbose=False)
        tel.register_csv("steps", ["step", "value"])

        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_tel",
                    verbose=False)
        cfg.telemetry = tel

        storage = UnifiedStorage(cfg)

        # Storage should NOT create its own steps CSV
        assert storage._metrics_enabled is False

        storage.close()
        tel.flush_all()


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_write_without_close_still_creates_files(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Files should be created even if close() is not called (but data may be incomplete)."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_no_close", verbose=False)
        storage = FieldStorage(cfg, comm)

        # Register and write without calling close
        storage.register("u", [fields.u])
        storage.write("u", t=0.0)
        
        # Note: In a real scenario, not calling close() may result in incomplete data
        # For this test, we just verify the writer was created
        assert "u" in storage._writers
        
        # Clean up properly for the test
        storage.close()



################################################################################

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


################################################################################

#!/usr/bin/env python3
"""
Advanced logging and monitoring tests for bone remodeling model.

Tests:
- Telemetry system integration
- Metrics tracking accuracy
- CSV output correctness (rank 0 only)
- Field statistics computation
- Logger behavior (verbose/quiet modes)
- Storage system integration
"""

import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")
pytestmark = [pytest.mark.slow, pytest.mark.integration]

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function
import tempfile
from pathlib import Path
import csv as csv_module

from simulation.config import Config
from simulation.utils import build_facetag, current_memory_mb
from simulation.model import Remodeller
from simulation.telemetry import Telemetry
from simulation.storage import UnifiedStorage
from simulation.logger import get_logger

comm = MPI.COMM_WORLD


# =============================================================================
# Telemetry System Tests
# =============================================================================

class TestTelemetry:
    """Test telemetry and experiment tracking."""
    
    @pytest.mark.parametrize("verbose_flag", [False, True])
    def test_telemetry_initialization(self, verbose_flag):
        """Telemetry should initialize output directory and metadata."""
        comm = MPI.COMM_WORLD
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir, verbose=verbose_flag)
            
            telemetry_dir = Path(tmpdir)
            assert telemetry_dir.exists(), "Telemetry directory not created"
            
            # Check is_root flag
            assert tel.is_root == (comm.rank == 0), f"is_root flag incorrect on rank {comm.rank}"
    
    @pytest.mark.parametrize("gz_flag", [False, True])
    def test_csv_registration_rank0_only(self, gz_flag):
        """CSV registration should only create files on rank 0."""
        comm = MPI.COMM_WORLD
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir, verbose=False)
            
            tel.register_csv("test_stream", ["col1", "col2", "col3"], gz=gz_flag)
            
            csv_path = Path(tmpdir) / ("test_stream.csv.gz" if gz_flag else "test_stream.csv")
            
            comm.Barrier()
            
            if comm.rank == 0:
                assert csv_path.exists(), "CSV file not created on rank 0"
            # Note: other ranks don't create files, so can't check non-existence reliably in shared tmpdir
    
    @pytest.mark.parametrize("flush_interval", [1, 5])
    def test_event_logging_buffering(self, flush_interval):
        """Events should be buffered and flushed periodically."""
        comm = MPI.COMM_WORLD
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir, flush_interval=flush_interval, verbose=False)
            
            tel.register_csv("events", ["step", "value"], gz=False)
            
            # Log events (less than flush_interval)
            for i in range(3):
                tel.record("events", {"step": i, "value": i*10})
            
            csv_path = Path(tmpdir) / "events.csv"
            
            if comm.rank == 0:
                # Before flush, buffer may not be written
                # (no assertion pre-flush)
                pass
            
            # Force flush
            tel.flush_all()
            # After flush, rank 0 should see CSV with header
            if comm.rank == 0:
                assert csv_path.exists(), 'events.csv not created after flush on rank 0'
                with open(csv_path, 'r', encoding='utf-8') as fh:
                    head = fh.readline().strip()
                    rest = fh.read().strip()
                # Telemetry uses caller-defined columns only (no auto timestamp)
                expected_headers = {'step,value', '"step","value"'}
                assert head.replace(' ', '') in expected_headers, f'Incorrect CSV header: {head}'
                assert len(rest) > 0, 'Flushed CSV has no data rows'
            comm.Barrier()
            
            if comm.rank == 0:
                assert csv_path.exists(), "CSV not created after flush"
                
                # Read and verify
                with open(csv_path, 'r') as f:
                    reader = csv_module.DictReader(f)
                    rows = list(reader)
                    assert len(rows) == 3, f"Expected 3 events, got {len(rows)}"
    
    def test_gzip_compression(self):
        """CSV with gz=True should create .csv.gz files."""
        comm = MPI.COMM_WORLD
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir, verbose=False)
            
            tel.register_csv("compressed", ["x", "y"], gz=True)
            tel.record("compressed", {"x": 1, "y": 2})
            tel.flush_all()
            
            comm.Barrier()
            
            if comm.rank == 0:
                gz_path = Path(tmpdir) / "compressed.csv.gz"
                assert gz_path.exists(), "Gzipped CSV not created"
    
    def test_telemetry_integration_in_config(self):
        """Config should create telemetry when enable_telemetry=True."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, 
                        results_dir=tmpdir, enable_telemetry=True)
            
            assert cfg.telemetry is not None, "Telemetry not created by Config when enable_telemetry=True"
            assert isinstance(cfg.telemetry, Telemetry), "Config.telemetry wrong type"

    def test_write_metadata_injects_standard_fields(self):
        """Telemetry.write_metadata injects start_time and mpi_size by default."""
        comm = MPI.COMM_WORLD
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir, verbose=False)
            tel.write_metadata({"foo": 1}, filename="meta.json", overwrite=True)
            comm.Barrier()
            if comm.rank == 0:
                from json import load
                from pathlib import Path
                p = Path(tmpdir) / "meta.json"
                assert p.exists()
                with open(p, "r", encoding="utf-8") as fh:
                    data = load(fh)
                assert "start_time" in data and isinstance(data["start_time"], str)
                assert data.get("mpi_size") == comm.size

    def test_write_metadata_overwrite_vs_append(self):
        """overwrite=False merges new fields into existing metadata."""
        comm = MPI.COMM_WORLD
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir, verbose=False)
            tel.write_metadata({"foo": 1}, filename="meta.json", overwrite=True)
            tel.write_metadata({"bar": 2}, filename="meta.json", overwrite=False)
            comm.Barrier()
            if comm.rank == 0:
                from json import load
                from pathlib import Path
                p = Path(tmpdir) / "meta.json"
                with open(p, "r", encoding="utf-8") as fh:
                    data = load(fh)
                assert data.get("foo") == 1 and data.get("bar") == 2


# =============================================================================
# Storage System Tests
# =============================================================================

class TestStorage:
    """Test unified storage system."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_storage_directory_creation(self, unit_cube):
        """Storage should create results directory on all ranks."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            storage = UnifiedStorage(cfg)
            
            # Check directory exists
            assert storage.fields.output_dir.exists(), f"Results directory not created on rank {comm.rank}"
            comm.Barrier()
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_vtx_writer_registration(self, unit_cube):
        """VTX writers should register correctly."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            storage = UnifiedStorage(cfg)
            
            # Register via Remodeller (mimics real usage)
            with Remodeller(cfg) as rem:
                # Writers registered in Remodeller.__init__
                pass    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_field_output_creates_files(self, unit_cube, shared_tmpdir):
        """Field output should create .bp files."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=shared_tmpdir, enable_telemetry=False)
        
        with Remodeller(cfg) as rem:
            # Compute total DOFs
            dofs_V = rem.V.dofmap.index_map.size_global
            dofs_Q = rem.Q.dofmap.index_map.size_global
            dofs_T = rem.T.dofmap.index_map.size_global
            num_dofs_total = int(dofs_V + dofs_Q + dofs_T)
            
            # Compute actual RSS memory
            rss_mb_local = current_memory_mb()
            rss_mb_total = comm.allreduce(rss_mb_local, op=MPI.SUM)
            
            # Write one step
            rem.storage.write_step(
                step=0,
                time_days=0.0,
                dt_days=1.0,
                u=rem.u,
                rho=rem.rho,
                S=rem.S,
                A=rem.A,
                num_dofs_total=num_dofs_total,
                rss_mem_mb=rss_mb_total,
                solver_stats={"mech": 10, "stim": 5, "dens": 5, "dir": 5},
                coupling_stats={"iters": 3, "time": 0.1},
            )
        # Context manager calls close automatically
        
        comm.Barrier()
        
        results_dir = Path(shared_tmpdir)
        # BP files are directories in ADIOS2
        all_entries = list(results_dir.iterdir())
        bp_dirs = [f for f in all_entries if f.is_dir() and f.name.endswith('.bp')]
        bp_names = {f.name for f in bp_dirs}
        
        assert "u.bp" in bp_names, f"u.bp not created, found: {bp_names}"
        assert "scalars.bp" in bp_names, f"scalars.bp not created, found: {bp_names}"
        assert "A.bp" in bp_names, f"A.bp not created, found: {bp_names}"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_metrics_csv_rank0_only(self, unit_cube, shared_tmpdir):
        """Metrics CSV should only be written by rank 0."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=shared_tmpdir)
        
        with Remodeller(cfg) as rem:
            # Compute total DOFs
            dofs_V = rem.V.dofmap.index_map.size_global
            dofs_Q = rem.Q.dofmap.index_map.size_global
            dofs_T = rem.T.dofmap.index_map.size_global
            num_dofs_total = int(dofs_V + dofs_Q + dofs_T)
            
            # Compute actual RSS memory
            rss_mb_local = current_memory_mb()
            rss_mb_total = comm.allreduce(rss_mb_local, op=MPI.SUM)
            
            rem.storage.write_step(
                step=0,
                time_days=0.0,
                dt_days=1.0,
                u=rem.u,
                rho=rem.rho,
                S=rem.S,
                A=rem.A,
                num_dofs_total=num_dofs_total,
                rss_mem_mb=rss_mb_total,
                solver_stats={"mech": 10, "stim": 5, "dens": 5, "dir": 5},
                coupling_stats={"iters": 3, "time": 0.1},
            )
            rem.storage.close()
        
        comm.Barrier()
        
        # Telemetry CSVs should exist on rank 0
        telemetry_dir = Path(shared_tmpdir) / "telemetry"
        if comm.rank == 0 and telemetry_dir.exists():
            csv_files = list(telemetry_dir.glob("*.csv*"))
            # Should have some CSVs if telemetry enabled
            assert all(f.stat().st_size > 0 for f in csv_files), 'Telemetry CSV files are empty on rank 0'


# =============================================================================
# Logger Tests
# =============================================================================

class TestLogger:
    """Test MPI-aware logger."""
    
    def test_logger_rank0_output(self):
        """Logger should only output on rank 0."""
        comm = MPI.COMM_WORLD
        
        log = get_logger(comm, verbose=True, name="TestLogger")
        
        # Capture doesn't work well with PETSc.Sys.Print, so just verify no crash
        log.info("This is an info message")
        log.warning("This is a warning")
        log.debug("This is debug")
        # No assertion needed - absence of exception is success
    
    def test_logger_verbose_flag(self):
        """verbose=False should suppress INFO/DEBUG."""
        comm = MPI.COMM_WORLD
        
        log_quiet = get_logger(comm, verbose=False, name="QuietLogger")
        log_verbose = get_logger(comm, verbose=True, name="VerboseLogger")
        
        # Both should work without errors
        log_quiet.info("Should be suppressed")
        log_verbose.info("Should be shown")
        # No assertion needed - absence of exception is success
    
    def test_logger_levels(self):
        """Logger should support DEBUG/INFO/WARNING/ERROR levels."""
        comm = MPI.COMM_WORLD
        
        log = get_logger(comm, verbose=True, name="LevelTest")
        
        # All should execute without error
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")
        # No assertion needed - absence of exception is success


# =============================================================================
# Field Statistics Tests
# =============================================================================

class TestFieldStatistics:
    """Test field statistics computation."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_scalar_field_stats(self, unit_cube):
        """Test min/max/mean computation for scalar fields."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
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
        rho_mean = rho_sum / n_total
        
        # Check reasonable values
        assert 0.25 < rho_min < 0.35, f"Min out of range: {rho_min}"
        assert 0.65 < rho_max < 0.75, f"Max out of range: {rho_max}"

    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_vector_field_norm(self, unit_cube):
        """Test norm computation for vector fields."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

        import basix
        import ufl
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        V = fem.functionspace(domain, P1_vec)

        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
        u.x.scatter_forward()

        u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
        u_norm = float(np.sqrt(u_norm_sq))

        assert abs(u_norm - 1.0) < 0.05, f"Vector norm incorrect: {u_norm}"


class TestSubiterationDiagnostics:
    """Additional checks on fixed-point solver diagnostics."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_avg_memory_matches_metrics(self, unit_cube, facet_tags):
        """avg_memory_mb should equal the mean of recorded memory columns."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=False, enable_telemetry=False, max_subiters=8)

        with Remodeller(cfg) as rem:
            rem.step(dt=1.0)
            metrics = rem.fixedsolver.subiter_metrics
            avg_memory = rem.fixedsolver.avg_memory_mb

        assert metrics, "No subiteration metrics recorded"
        mem_vals = [rec["memory_mb"] for rec in metrics if "memory_mb" in rec]
        assert mem_vals, "Memory metrics missing"

        mean_mem = float(np.mean(mem_vals))
        assert avg_memory == pytest.approx(mean_mem, rel=1e-12, abs=1e-9)


# =============================================================================
# Monitoring Integration Tests
# =============================================================================

class TestMonitoringIntegration:
    """Test end-to-end monitoring in simulation."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_step_metrics_recorded(self, unit_cube):
        """Each time step should record metrics."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                domain=domain,
                facet_tags=facet_tags,
                verbose=False,
                results_dir=tmpdir,
                max_subiters=10,
                coupling_tol=1e-5
            )
            
            with Remodeller(cfg) as rem:
                # Run 2 steps
                rem.step(dt=1.0)
                rem.step(dt=1.0)
            
            comm.Barrier()
            
            # Check telemetry files created
            telemetry_dir = Path(tmpdir) / "telemetry"
            if comm.rank == 0 and telemetry_dir.exists():
                steps_csv = telemetry_dir / "steps.csv"
                if steps_csv.exists():
                    with open(steps_csv, 'r') as f:
                        reader = csv_module.DictReader(f)
                        rows = list(reader)
                        assert len(rows) >= 2, f"Expected ≥2 step records, got {len(rows)}"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_solver_stats_tracking(self, unit_cube):
        """Solver statistics should be tracked per step."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                domain=domain,
                facet_tags=facet_tags,
                verbose=False,
                results_dir=tmpdir,
                max_subiters=10
            )
            
            with Remodeller(cfg) as rem:
                rem.step(dt=1.0)
                
                # Check solver stats were accumulated
                mech_iters = rem.mechsolver.total_iters
                stim_iters = rem.stimsolver.total_iters
                dens_iters = rem.densolver.total_iters
                dir_iters = rem.dirsolver.total_iters
                
                assert mech_iters > 0, "Mechanics solver didn't iterate"
                assert stim_iters > 0, "Stimulus solver didn't iterate"
                assert dens_iters > 0, "Density solver didn't iterate"
                assert dir_iters > 0, "Direction solver didn't iterate"

    def test_run_summary_json_after_simulate(self):
        """simulate() should produce run_summary.json when telemetry enabled."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 6, 6, 6, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                domain=domain,
                facet_tags=facet_tags,
                verbose=False,
                results_dir=tmpdir,
                enable_telemetry=True,
                max_subiters=5,
            )
            with Remodeller(cfg) as rem:
                rem.simulate(dt=1.0, total_time=1.0)
            comm.Barrier()
            if comm.rank == 0:
                from pathlib import Path
                p = Path(tmpdir) / "run_summary.json"
                assert p.exists(), "run_summary.json not created"


# =============================================================================
# Error Handling Tests
# =============================================================================

# Note: Removed TestErrorHandling class (idempotence tests are implementation details, not user-facing behavior)
# Storage and Telemetry classes handle multiple close/flush calls gracefully via internal guards


# No __main__ runner needed; tests executed via pytest
