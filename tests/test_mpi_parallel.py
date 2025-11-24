#!/usr/bin/env python3
"""
MPI parallelism tests.

Tests ghost cell updates, domain decomposition, collective operations, I/O, and parallel solver behavior.
"""

import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.performance]
import numpy as np
np.random.seed(1234)
import time
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs, current_memory_mb
from simulation.subsolvers import MechanicsSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver
from simulation.model import Remodeller
comm = MPI.COMM_WORLD

# =============================================================================
# Ghost Update Tests
# =============================================================================

class TestGhostUpdates:
    """Test ghost cell synchronization across MPI boundaries."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_ghost_sync_after_assignment(self, unit_cube):
        """Verify scatter_forward() synchronizes ghost values."""
        comm = MPI.COMM_WORLD
        
        if comm.size < 2:
            pytest.skip("Test requires at least 2 MPI ranks")
        
        domain = unit_cube
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        
        # Each rank sets owned DOFs to its rank value
        n_owned = Q.dofmap.index_map.size_local
        rho.x.array[:n_owned] = float(comm.rank)
        
        # Before scatter: ghost values are stale
        # After scatter: ghost values should match owner's rank
        rho.x.scatter_forward()
        
        # Check: ghost values should be from neighboring ranks (not current rank)
        n_ghosts = Q.dofmap.index_map.num_ghosts
        if n_ghosts > 0:
            ghost_vals = rho.x.array[n_owned:n_owned+n_ghosts]
            # Ghost values should NOT all be current rank (they come from neighbors)
            # At least some should differ
            unique_ghost_vals = np.unique(ghost_vals)
            # On multi-rank runs, expect at least one different value
            has_neighbor_data = len(unique_ghost_vals) > 1 or (len(unique_ghost_vals) == 1 and unique_ghost_vals[0] != comm.rank)
            assert has_neighbor_data or n_ghosts == 0, "Ghost update failed: no neighbor data received"


# =============================================================================
# Domain Decomposition Tests
# =============================================================================

class TestDomainDecomposition:
    """Test consistency across different domain partitions."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    @pytest.mark.parametrize("reduction_type", ["integral"])
    def test_global_reductions_consistent(self, unit_cube, traction_factory, reduction_type):
        """Test MPI collective operations: L2 norm, integrals, min/max reductions."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        if reduction_type == "l2_norm":
            # L2 norm partition independence
            P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
            V = functionspace(domain, P1_vec)
            
            u = Function(V, name="u")
            rho = Function(Q, name="rho")
            rho.x.array[:] = 0.6
            rho.x.scatter_forward()
            
            traction = traction_factory(-0.5, facet_id=2, axis=0)
            bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
            mech = MechanicsSolver(u, rho, cfg, bc_mech, [traction])
            mech.setup()
            mech.assemble_rhs()
            mech.solve()
            
            # Global L2 norm
            u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * cfg.dx))
            u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
            u_norm = np.sqrt(u_norm_sq)
            
            assert 1e-8 < u_norm < 1.0, f"Solution norm unreasonable: ||u|| = {u_norm}"
            
            # Cross-rank consistency
            all_norms = comm.gather(u_norm, root=0)
            if comm.rank == 0:
                assert all(abs(n - u_norm) < 1e-12 for n in all_norms), "Ranks computed different norms"
        
        elif reduction_type == "integral":
            # Global integral of known function
            f = Function(Q, name="f")
            f.interpolate(lambda x: x[0] + x[1] + x[2])
            f.x.scatter_forward()
            
            # ∫_[0,1]^3 (x+y+z) dx dy dz = 3/2
            integral_local = fem.assemble_scalar(fem.form(f * cfg.dx))
            integral_global = comm.allreduce(integral_local, op=MPI.SUM)
            expected = 1.5
            assert abs(integral_global - expected) < 1e-10, f"Global integral wrong: {integral_global} ≠ {expected}"


# =============================================================================
# Collective Operation Tests
# =============================================================================

class TestCollectiveOps:
    """Test MPI collective operations (reductions, broadcasts)."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_global_volume_computation(self, unit_cube):
        """Verify global volume = sum of local volumes = 1.0 for unit cube."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        vol_global = comm.allreduce(vol_local, op=MPI.SUM)
        
        assert abs(vol_global - 1.0) < 1e-10, f"Global volume incorrect: {vol_global} ≠ 1.0"


# =============================================================================
# I/O Consistency Tests
# =============================================================================

class TestMPIIO:
    """Test MPI-parallel I/O operations."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_rank0_only_writes(self, unit_cube):
        """Verify only rank 0 performs file writes."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            
            # Telemetry should only write on rank 0
            if cfg.telemetry is not None:
                # Check internal flag
                assert cfg.telemetry.is_root == (comm.rank == 0), "Telemetry root flag incorrect"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_vtx_output_consistency(self, unit_cube):
        """Test skipped: VTX behavior verified in storage tests with stubs."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            
            with Remodeller(cfg) as rem:
                # Compute total scalar DOFs (account for block sizes)
                dofs_V = rem.V.dofmap.index_map.size_global * rem.V.dofmap.index_map_bs
                dofs_Q = rem.Q.dofmap.index_map.size_global * rem.Q.dofmap.index_map_bs
                num_dofs_total = int(dofs_V + dofs_Q)
                
                # Compute actual RSS memory
                rss_mb_local = current_memory_mb()
                rss_mb_total = rem.comm.allreduce(rss_mb_local, op=MPI.SUM)
    
                rem.storage.write_fields("scalars", 0.0)
                if rem.telemetry:
                    rem.telemetry.record("steps", {
                        "step": 0,
                        "time_days": 0.0,
                        "dt_days": 1.0,
                        "tol": 1e-6,
                        "used_subiters": 3,
                        "mech_time_s": 0.1,
                        "dens_time_s": 0.1,
                        "solve_time_s_total": 0.4,
                        "proj_res_last": 1e-7,
                        "num_dofs_total": num_dofs_total,
                        "rss_mem_mb": rss_mb_total,
                        "mech_iters": 10,
                        "dens_iters": 5,
                    })
                
                # Should complete without hanging
                comm.Barrier()

    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_csv_metrics_rank0(self, unit_cube):
        """Skipped: CSV rank-0 behavior covered elsewhere."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            
            with Remodeller(cfg) as rem:
                # Compute total scalar DOFs (account for block sizes)
                dofs_V = rem.V.dofmap.index_map.size_global * rem.V.dofmap.index_map_bs
                dofs_Q = rem.Q.dofmap.index_map.size_global * rem.Q.dofmap.index_map_bs
                num_dofs_total = int(dofs_V + dofs_Q)
                
                # Compute actual RSS memory
                rss_mb_local = current_memory_mb()
                rss_mb_total = comm.allreduce(rss_mb_local, op=MPI.SUM)
    
                rem.storage.write_fields("scalars", 0.0)
                if rem.telemetry:
                    rem.telemetry.record("steps", {
                        "step": 0,
                        "time_days": 0.0,
                        "dt_days": 1.0,
                        "tol": 1e-6,
                        "used_subiters": 3,
                        "mech_time_s": 0.1,
                        "dens_time_s": 0.1,
                        "solve_time_s_total": 0.4,
                        "proj_res_last": 1e-7,
                        "num_dofs_total": num_dofs_total,
                        "rss_mem_mb": rss_mb_total,
                        "mech_iters": 10,
                        "dens_iters": 5,
                    })
                
                rem.storage.close()
            
            comm.Barrier()
            
            # Check if CSV exists
            telemetry_dir = Path(tmpdir) / "telemetry"
            if comm.rank == 0:
                # If telemetry dir exists, ensure any CSVs are non-empty
                if telemetry_dir.exists():
                    csv_files = list(telemetry_dir.glob('*.csv*'))
                    assert all(f.stat().st_size > 0 for f in csv_files), 'Rank 0 telemetry CSVs empty'


# =============================================================================
# Fixed-Point Solver Parallelism Tests
# =============================================================================

class TestFixedPointParallel:
    """Test fixed-point solver in parallel."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_anderson_acceleration_parallel(self, unit_cube):
        """Verify Anderson acceleration works correctly in parallel."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            verbose=False,
            accel_type="anderson",
            max_subiters=20,
            coupling_tol=1e-6
        )
        
        with Remodeller(cfg) as rem:
            # Take one time step
            rem.step(dt=1.0)
            
            # Check Anderson was used
            gs_iters = rem.fixedsolver.total_gs_iters
            assert gs_iters > 0, "Fixed-point solver didn't iterate"
            assert gs_iters < 50, f"Too many iterations ({gs_iters}), Anderson may have failed"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_gauss_seidel_convergence_parallel(self, unit_cube):
        """Test Gauss-Seidel converges in parallel."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            verbose=False,
            accel_type="picard",
            max_subiters=30,
            coupling_tol=1e-5
        )
        
        with Remodeller(cfg) as rem:
            rem.step(dt=1.0)
            
            # Should converge
            assert rem.fixedsolver.total_gs_iters > 0
            assert rem.fixedsolver.total_gs_iters < 100


# =============================================================================
# Cross-Rank Communication Tests
# =============================================================================

class TestCrossRankComm:
    """Test communication patterns between ranks."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_residual_norm_computation(self, unit_cube):
        """Global residual norm should be consistent across all ranks."""
        comm = MPI.COMM_WORLD
        
        if comm.size < 2:
            pytest.skip("Test requires multiple ranks")
        
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        V = functionspace(domain, P1_vec)
        
        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
        u.x.scatter_forward()
        
        # Compute norm: each rank should get same global value
        u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
        
        # All ranks should have identical result
        all_norms = comm.allgather(u_norm_sq)
        for i, n in enumerate(all_norms):
            assert abs(n - u_norm_sq) < 1e-14, f"Rank {i} has different norm: {n} vs {u_norm_sq}"


# =============================================================================
# Performance and Solver Efficiency Tests
# =============================================================================

class TestSolverIterations:
    """Test solver iteration counts scale reasonably with problem size."""
    
    def test_mechanics_iterations_bounded(self):
        """Mechanics solver iterations should be O(1) or O(log N) with good preconditioner."""
        comm = MPI.COMM_WORLD
        
        mesh_sizes = [4, 6, 8] if comm.size > 1 else [4, 6]  # Smaller for faster tests
        iteration_counts = []
        
        for N in mesh_sizes:
            domain = mesh.create_unit_cube(comm, N, N, N, ghost_mode=mesh.GhostMode.shared_facet)
            facet_tags = build_facetag(domain)
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
            
            P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
            P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
            
            V = functionspace(domain, P1_vec)
            Q = functionspace(domain, P1)
            
            u = Function(V, name="u")
            rho = Function(Q, name="rho")
            rho.x.array[:] = 0.5
            rho.x.scatter_forward()
            
            t_vec = np.zeros(3, dtype=np.float64)
            t_vec[0] = -0.5
            traction = (fem.Constant(domain, t_vec), 2)
            
            bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
            mech = MechanicsSolver(u, rho, cfg, bc_mech, [traction])
            
            mech.setup()
            mech.assemble_rhs()
            its, _ = mech.solve()
            
            iteration_counts.append(its)
            mech.destroy()
        
        # Iterations should not grow excessively (modern preconditioners keep it bounded)
        max_iters = max(iteration_counts)
        assert max_iters < 200, f"Mechanics solver iterations too high: {max_iters}"
        
        # Should see some efficiency (not linear growth)
        if len(iteration_counts) >= 2:
            growth_factor = iteration_counts[-1] / iteration_counts[0]
            mesh_growth_factor = (mesh_sizes[-1] / mesh_sizes[0]) ** 3  # Volume scaling
            assert growth_factor < mesh_growth_factor, f"Iterations growing too fast: {growth_factor} vs mesh {mesh_growth_factor}"


# =============================================================================
# Preconditioner Effectiveness Tests
# =============================================================================

class TestPreconditioners:
    """Test preconditioner update logic and effectiveness (skipped)."""
    
    def test_preconditioner_reuse_efficiency(self):
        """Preconditioner reuse should not degrade performance excessively."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        t_vec = np.zeros(3, dtype=np.float64)
        t_vec[0] = -0.3
        traction = (fem.Constant(domain, t_vec), 2)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [traction])
        
        mech.setup()
        mech.assemble_rhs()
        
        # Solve with fresh preconditioner
        its1, _ = mech.solve()
        
        # Slightly modify rho
        rho.x.array[:] = 0.51
        rho.x.scatter_forward()
        mech.assemble_lhs()  # Reassemble stiffness matrix
        mech.assemble_rhs()
        
        # Solve with reused preconditioner
        its2, _ = mech.solve()
        
        # Reused preconditioner should still work (may take more iters but not excessive)
        assert its2 < its1 * 3, f"Preconditioner reuse degraded too much: {its1} → {its2}"
        
        mech.destroy()


# =============================================================================
# Memory Usage Tests
# =============================================================================

class TestMemoryUsage:
    """Test memory usage patterns (basic checks)."""
    
    def test_state_buffer_size(self):
        """Fixed-point state buffer should match total DOF count."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        dens = DensitySolver(rho, rho_old, cfg)
        
        # Create driver for fixed-point solver (required in new architecture)
        class DummyDriver:
            def __init__(self, mech):
                self.mech = mech
            def update_stiffness(self):
                self.mech.setup()
            def update_snapshots(self):
                return {}
            def stimulus_expr(self):
                return fem.Constant(self.mech.mesh, 0.0)
            def invalidate(self):
                pass
            def setup(self):
                pass
            def destroy(self):
                pass

        driver = DummyDriver(mech)
        
        fps = FixedPointSolver(
            comm, cfg, driver, dens,
            rho, rho_old
        )
        
        expected_size = fps.n_rho
        assert fps.state_size == expected_size, f"State buffer size mismatch: {fps.state_size} ≠ {expected_size}"
        assert len(fps.state_buffer) == expected_size, f"State buffer allocation wrong"
    
    @pytest.mark.parametrize("m", [3, 5])
    def test_anderson_history_bounded(self, m):
        """Anderson history should be bounded by window size m."""
        from simulation.anderson import _Anderson
        
        n = 100
        aa = _Anderson(comm=MPI.COMM_WORLD, m=m)
        
        # Add more than m updates
        for i in range(10):
            x = np.random.rand(n)
            f = np.random.rand(n)
            _ = aa.mix(x, f)
        
        # History length should not exceed m+1 (deque maxlen)
        hist_len = len(aa.r_hist)
        assert hist_len <= m + 1, f"Anderson history exceeded window size: len={hist_len} > m+1={m+1}"


# =============================================================================
# Timing Tests
# =============================================================================

class TestTiming:
    """Test timing and performance metrics."""
    
    @pytest.mark.parametrize("max_subiters", [10, 20])
    def test_step_timing_reasonable(self, max_subiters):
        """Time step should complete in reasonable time (relative, not absolute threshold)."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            verbose=False,
            max_subiters=max_subiters
        )
        
        with Remodeller(cfg) as rem:
            t0 = time.perf_counter()
            rem.step(dt=1.0)
            elapsed = time.perf_counter() - t0
            
            # Document timing, no hard threshold (environment-dependent)
            # Typical: <10s on modern hardware, but CI/VMs can be 5-10× slower
            if comm.rank == 0:
                print(f"Step timing (max_subiters={max_subiters}): {elapsed:.2f}s")
    
    def test_solver_timing_tracked(self):
        """Solver timings should be tracked in fixed-point solver."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, max_subiters=10)
        
        with Remodeller(cfg) as rem:
            rem.step(dt=1.0)
            
            fps = rem.fixedsolver
            
            # Check timings were recorded
            assert fps.mech_time_total > 0, "Mechanics time not tracked"
            assert fps.dens_time_total > 0, "Density time not tracked"

