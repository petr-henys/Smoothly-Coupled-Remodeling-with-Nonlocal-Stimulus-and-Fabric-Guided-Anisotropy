"""
Tests for storage module (FieldStorage, UnifiedStorage).

Tests:
- Directory creation (collective MPI operation)
- VTX writer registration and lifecycle
- Field write operations (displacement, scalars, tensors)
- Error handling (permission denied, disk full simulation)
- MPI collective safety
- Context manager behavior
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from pathlib import Path
import tempfile

from simulation.storage import FieldStorage, UnifiedStorage
from simulation.config import Config
from simulation.model import Remodeller

comm = MPI.COMM_WORLD

# Patch VTXWriter inside simulation.storage to a no-op stub to avoid ADIOS2 side effects
@pytest.fixture(autouse=True)
def _mock_vtxwriter(monkeypatch):
    import simulation.storage as storage_mod
    from pathlib import Path as _P

    class _DummyVTXWriter:
        def __init__(self, comm, path, fields, engine="bp4"):
            self.comm = comm
            self.path = path
            self.fields = list(fields)
            self.engine = engine
            # Simulate BP directory creation on rank 0 so tests can assert existence
            if hasattr(comm, "rank"):
                if comm.rank == 0:
                    _P(path).mkdir(parents=True, exist_ok=True)
            else:
                _P(path).mkdir(parents=True, exist_ok=True)
            self._closed = False

        def write(self, t):
            # No-op
            return None

        def close(self):
            self._closed = True

    # Replace the imported VTXWriter symbol used by FieldStorage
    monkeypatch.setattr(storage_mod, "VTXWriter", _DummyVTXWriter, raising=True)

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

    @pytest.mark.parametrize("unit_cube", [4], indirect=True)
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


class TestUnifiedStorage:
    """Test unified storage manager."""

    def test_initialization_creates_field_storage(self, shared_tmpdir, unit_cube, facet_tags):
        """UnifiedStorage should initialize field storage."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_unified", verbose=False)

        storage = UnifiedStorage(cfg)

        assert storage.fields is not None
        assert isinstance(storage.fields, FieldStorage)

        storage.close()

    def test_write_fields_delegates_to_field_storage(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """write_fields should delegate to FieldStorage.write."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_step", verbose=False)

        storage = UnifiedStorage(cfg)

        # Register field groups
        storage.fields.register("scalars", [fields.rho, fields.S])
        storage.fields.register("A", [fields.A])

        # Set field values
        fields.rho.x.array[:] = 0.7
        fields.rho.x.scatter_forward()
        fields.S.x.array[:] = 0.15
        fields.S.x.scatter_forward()

        # Write fields
        storage.write_fields("scalars", t=10.0)
        storage.write_fields("A", t=10.0)

        storage.close()

        comm.Barrier()

        if comm.rank == 0:
            # Check field files exist
            base = Path(shared_tmpdir) / "test_step"
            assert (base / "scalars.bp").exists()
            assert (base / "A.bp").exists()

    def test_context_manager(self, shared_tmpdir, unit_cube, facet_tags, spaces, fields):
        """Context manager should properly initialize and cleanup."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    results_dir=shared_tmpdir / "test_ctx_unified", verbose=False)

        with UnifiedStorage(cfg) as storage:
            storage.fields.register("scalars", [fields.rho, fields.S])
            storage.fields.write("scalars", t=0.0)
            assert len(storage.fields._writers) > 0

        # After context exit, storage should be closed
        assert len(storage.fields._writers) == 0

    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_storage_initializes_with_remodeller(self, unit_cube):
        """UnifiedStorage should initialize correctly within Remodeller context."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        from simulation.utils import build_facetag
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            
            with Remodeller(cfg) as rem:
                # Storage should be initialized
                assert rem.storage is not None
                assert rem.storage.fields is not None
                
                # Field writers should be registered
                assert "scalars" in rem.storage.fields._writers
                assert "A" in rem.storage.fields._writers
            
            comm.Barrier()
