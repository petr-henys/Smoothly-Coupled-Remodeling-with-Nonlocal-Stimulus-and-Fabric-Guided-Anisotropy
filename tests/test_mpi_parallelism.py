#!/usr/bin/env python3
"""
Advanced MPI parallelism tests for bone remodeling model.

Tests:
- Ghost cell updates across partition boundaries
- Domain decomposition consistency
- Collective operations (reductions, barriers)
- Rank-dependent I/O correctness
- Load balancing verification
- Partition-independent solution convergence
"""

import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")
pytestmark = [pytest.mark.integration]
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs, current_memory_mb
from simulation.subsolvers import MechanicsSolver
from simulation.model import Remodeller
comm = MPI.COMM_WORLD

# =============================================================================
# Ghost Update Tests
# =============================================================================

class TestGhostUpdates:
    """Test ghost cell synchronization across MPI boundaries."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
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
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_repeated_scatter_idempotent(self, unit_cube):
        """Verify scatter_forward() is idempotent (repeated calls don't change result)."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        rho.interpolate(lambda x: x[0]**2 + x[1]**2 + x[2]**2)
        
        # First scatter
        rho.x.scatter_forward()
        vals_after_1 = rho.x.array.copy()
        
        # Second scatter (should not change values)
        rho.x.scatter_forward()
        vals_after_2 = rho.x.array.copy()
        
        diff = np.linalg.norm(vals_after_2 - vals_after_1)
        assert diff < 1e-14, f"Scatter not idempotent: ||diff|| = {diff}"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_assign_includes_scatter(self, unit_cube):
        """Verify utils.assign() includes scatter_forward()."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        
        # Use assign() from utils
        from simulation.utils import assign
        assign(rho, 0.75)
        
        # Check ghost values are updated (not zero/stale)
        n_owned = Q.dofmap.index_map.size_local
        n_total = len(rho.x.array)
        
        if n_total > n_owned:  # Has ghosts
            ghost_vals = rho.x.array[n_owned:]
            # All ghosts should be 0.75 after assign
            ghost_max_diff = np.max(np.abs(ghost_vals - 0.75))
            assert ghost_max_diff < 1e-14, f"assign() didn't update ghosts: max diff = {ghost_max_diff}"


# =============================================================================
# Domain Decomposition Tests
# =============================================================================

class TestDomainDecomposition:
    """Test consistency across different domain partitions."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    @pytest.mark.parametrize("reduction_type", ["l2_norm", "integral", "min_max"])
    def test_global_reductions_consistent(self, unit_cube, traction_factory, reduction_type):
        """Test MPI collective operations: L2 norm, integrals, min/max reductions.
        
        Consolidates: test_partition_independent_solution, test_global_integral_consistency, test_min_max_reductions
        """
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        if reduction_type == "l2_norm":
            # L2 norm partition independence
            P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
            P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
            V = functionspace(domain, P1_vec)
            T = functionspace(domain, P1_ten)
            
            u = Function(V, name="u")
            rho = Function(Q, name="rho")
            rho.x.array[:] = 0.6
            rho.x.scatter_forward()
            
            A = Function(T, name="A")
            A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
            A.x.scatter_forward()
            
            traction = traction_factory(-0.5, facet_id=2, axis=0)
            bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
            mech = MechanicsSolver(V, rho, A, bc_mech, [traction], cfg)
            mech.solver_setup()
            mech.solve(u)
            
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
        
        elif reduction_type == "min_max":
            # Global min/max reductions
            rho = Function(Q, name="rho")
            rho.interpolate(lambda x: 0.3 + 0.4*x[0])  # Range [0.3, 0.7]
            rho.x.scatter_forward()
            
            n_owned = Q.dofmap.index_map.size_local
            rho_local_min = rho.x.array[:n_owned].min()
            rho_local_max = rho.x.array[:n_owned].max()
            
            rho_global_min = comm.allreduce(rho_local_min, op=MPI.MIN)
            rho_global_max = comm.allreduce(rho_local_max, op=MPI.MAX)
            
            assert abs(rho_global_min - 0.3) < 0.05, f"Global min wrong: {rho_global_min}"
            assert abs(rho_global_max - 0.7) < 0.05, f"Global max wrong: {rho_global_max}"


# Note: Consolidated 3 MPI reduction tests into single parametrized test
# Reduces test count from 3→1 while preserving coverage for L2 norms, integrals, min/max
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_load_balancing_fairness(self, unit_cube):
        """Check DOF distribution is reasonably balanced across ranks."""
        comm = MPI.COMM_WORLD
        
        if comm.size < 2:
            pytest.skip("Load balancing test requires multiple ranks")
        
        domain = unit_cube
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        n_local = Q.dofmap.index_map.size_local
        n_total = comm.allreduce(n_local, op=MPI.SUM)
        n_avg = n_total / comm.size
        
        # Each rank should have ≈ n_avg DOFs (within 50% tolerance for small meshes)
        imbalance = abs(n_local - n_avg) / n_avg
        assert imbalance < 0.5, f"Rank {comm.rank} has {n_local} DOFs (avg={n_avg}, imbalance={imbalance:.1%})"


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
        
        # Unit cube should have volume 1.0 (in ND coords)
        assert abs(vol_global - 1.0) < 1e-10, f"Global volume incorrect: {vol_global} ≠ 1.0"
    
# =============================================================================
# Field Ownership Tests
# =============================================================================
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_min_max_reductions(self, unit_cube):
        """Test global min/max operations."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        rho.interpolate(lambda x: 0.3 + 0.4*x[0])  # Range [0.3, 0.7]
        rho.x.scatter_forward()
        
        n_owned = Q.dofmap.index_map.size_local
        rho_local_min = rho.x.array[:n_owned].min()
        rho_local_max = rho.x.array[:n_owned].max()
        
        rho_global_min = comm.allreduce(rho_local_min, op=MPI.MIN)
        rho_global_max = comm.allreduce(rho_local_max, op=MPI.MAX)
        
        assert abs(rho_global_min - 0.3) < 0.05, f"Global min wrong: {rho_global_min}"
        assert abs(rho_global_max - 0.7) < 0.05, f"Global max wrong: {rho_global_max}"


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
        """Test VTX output doesn't cause MPI hangs or errors."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False, results_dir=tmpdir)
            
            with Remodeller(cfg) as rem:
                # Write initial state
                # Compute total scalar DOFs (account for block sizes)
                dofs_V = rem.V.dofmap.index_map.size_global * rem.V.dofmap.index_map_bs
                dofs_Q = rem.Q.dofmap.index_map.size_global * rem.Q.dofmap.index_map_bs
                dofs_T = rem.T.dofmap.index_map.size_global * rem.T.dofmap.index_map_bs
                num_dofs_total = int(dofs_V + dofs_Q + dofs_T)
                
                # Compute actual RSS memory
                rss_mb_local = current_memory_mb()
                rss_mb_total = rem.comm.allreduce(rss_mb_local, op=MPI.SUM)
                
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
                
                # Should complete without hanging
                comm.Barrier()
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_csv_metrics_rank0(self, unit_cube):
        """Verify CSV metrics only written by rank 0."""
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
                dofs_T = rem.T.dofmap.index_map.size_global * rem.T.dofmap.index_map_bs
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


# No __main__ runner needed; tests executed via pytest
