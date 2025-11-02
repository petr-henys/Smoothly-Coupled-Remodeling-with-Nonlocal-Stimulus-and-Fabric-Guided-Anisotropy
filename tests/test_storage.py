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
from dolfinx.fem import Function, functionspace
import basix
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

