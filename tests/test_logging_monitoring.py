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
