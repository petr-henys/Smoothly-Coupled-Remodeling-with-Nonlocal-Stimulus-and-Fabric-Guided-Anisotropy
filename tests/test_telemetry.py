"""
Tests for telemetry and logging modules.

Tests:
- Logger initialization and rank-aware behavior
- Telemetry system integration
- Metrics tracking accuracy
- CSV output correctness (rank 0 only)
- Field statistics computation
"""

import pytest
import logging
import numpy as np
import csv as csv_module
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function
import basix.ufl
import tempfile
from pathlib import Path

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.model import Remodeller
from simulation.telemetry import Telemetry
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
# Telemetry System Tests
# =============================================================================

class TestTelemetry:
    """Test telemetry and experiment tracking."""
    
    def test_telemetry_initialization(self):
        """Telemetry should initialize output directory and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir)
            
            telemetry_dir = Path(tmpdir)
            assert telemetry_dir.exists(), "Telemetry directory not created"
            assert tel.is_root == (comm.rank == 0), f"is_root flag incorrect on rank {comm.rank}"
    
    def test_csv_registration_rank0_only(self):
        """CSV registration should only create files on rank 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir)
            tel.register_csv("test_stream", ["col1", "col2", "col3"])
            
            csv_path = Path(tmpdir) / "test_stream.csv"
            comm.Barrier()
            
            if comm.rank == 0:
                assert csv_path.exists(), "CSV file not created on rank 0"
    
    def test_event_logging_buffering(self):
        """Events should be buffered and flushed periodically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir)
            tel.register_csv("events", ["step", "value"])
            
            # Log events
            for i in range(3):
                tel.record("events", {"step": i, "value": i*10})
            
            csv_path = Path(tmpdir) / "events.csv"
            
            # Force flush
            tel.flush_all()
            comm.Barrier()
            
            if comm.rank == 0:
                assert csv_path.exists(), "CSV not created after flush"
                with open(csv_path, 'r') as f:
                    reader = csv_module.DictReader(f)
                    rows = list(reader)
                    assert len(rows) == 3, f"Expected 3 events, got {len(rows)}"

    def test_write_metadata_injects_standard_fields(self):
        """Telemetry.write_metadata injects start_time and mpi_size by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tel = Telemetry(comm, outdir=tmpdir)
            tel.write_metadata({"foo": 1}, filename="meta.json", overwrite=True)
            comm.Barrier()
            if comm.rank == 0:
                from json import load
                p = Path(tmpdir) / "meta.json"
                assert p.exists()
                with open(p, "r", encoding="utf-8") as fh:
                    data = load(fh)
                assert "start_time" in data
                assert data.get("mpi_size") == comm.size

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

# =============================================================================
# Integration Tests
# =============================================================================

class TestMonitoringIntegration:
    """Test end-to-end monitoring in simulation."""
    
    def _create_mock_loader(self, domain):
        """Create a mock loader for testing."""
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
        V = fem.functionspace(domain, P1_vec)
        
        class MockLoader:
            def __init__(self):
                self.V = V
                self.load_tag = 1
                self.cut_tag = 1
                self.traction = fem.Function(V, name="Traction")
                self.traction_cut = fem.Function(V, name="TractionCut")
                self.traction.x.array[:] = 0.01  # Small non-zero load
                
            def apply_loading_case(self, case):
                pass
        
        return MockLoader()
    
    def _create_loading_cases(self):
        """Create loading cases for testing."""
        from simulation.loader import LoadingCase
        return [LoadingCase(name="test")]
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_telemetry_records_steps(self, unit_cube):
        """Telemetry should record step data correctly."""
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                domain=domain,
                facet_tags=facet_tags,
                results_dir=tmpdir,
                max_subiters=10,
                coupling_tol=1e-5
            )
            
            loader = self._create_mock_loader(domain)
            loading_cases = self._create_loading_cases()
            
            with Remodeller(cfg, loader=loader, loading_cases=loading_cases) as rem:
                # Run 2 steps
                rem.step(1.0, 0, 1.0)
                rem.step(1.0, 1, 2.0)
                
                # Verify telemetry has recorded data (rank 0 only)
                if comm.rank == 0:
                    assert rem.telemetry is not None
                    # Check that steps buffer has data
                    assert "steps" in rem.telemetry._buffers
            
            comm.Barrier()
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_solver_stats_tracking(self, unit_cube):
        """Solver statistics should be tracked per step."""
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                domain=domain,
                facet_tags=facet_tags,
                results_dir=tmpdir,
                max_subiters=10
            )
            
            loader = self._create_mock_loader(domain)
            loading_cases = self._create_loading_cases()
            
            with Remodeller(cfg, loader=loader, loading_cases=loading_cases) as rem:
                rem.step(1.0, 0, 1.0)
                
                # Check solver stats were accumulated
                mech_iters = rem.fixedsolver.mech_iters_total
                assert mech_iters >= 0, f"Mechanics solver iteration count invalid: {mech_iters}"

    def test_run_summary_json_after_simulate(self):
        """simulate() should produce run_summary.json when telemetry enabled."""
        domain = mesh.create_unit_cube(comm, 6, 6, 6, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(
                domain=domain,
                facet_tags=facet_tags,
                results_dir=tmpdir,
                max_subiters=5,
            )
            
            loader = self._create_mock_loader(domain)
            loading_cases = self._create_loading_cases()
            
            with Remodeller(cfg, loader=loader, loading_cases=loading_cases) as rem:
                rem.simulate(dt_initial=1.0, total_time=1.0)
            comm.Barrier()
            if comm.rank == 0:
                from pathlib import Path
                p = Path(tmpdir) / "run_summary.json"
                assert p.exists(), "run_summary.json not created"
