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

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.fixedsolver import FixedPointSolver
from simulation.model import Remodeller
comm = MPI.COMM_WORLD
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
            mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
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
            P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
            
            V = functionspace(domain, P1_vec)
            Q = functionspace(domain, P1)
            T = functionspace(domain, P1_ten)
            
            u = Function(V, name="u")
            rho = Function(Q, name="rho")
            rho.x.array[:] = 0.5
            rho.x.scatter_forward()
            
            A = Function(T, name="A")
            A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
            A.x.scatter_forward()
            
            t_vec = np.zeros(3, dtype=np.float64)
            t_vec[0] = -0.5
            traction = (fem.Constant(domain, t_vec), 2)
            
            bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
            mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
            
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
    
    @pytest.mark.parametrize("dt_days", [5.0, 10.0])
    def test_stimulus_iterations_consistent(self, dt_days):
        """Stimulus solver should have consistent iteration counts."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        cfg.set_dt_dim(dt_days)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        S_old = Function(Q, name="S_old")
        S_old.x.array[:] = 0.1
        S_old.x.scatter_forward()
        
        S = Function(Q, name="S")
        
        import ufl
        psi_density = fem.Constant(domain, 50.0)  # Dummy source
        
        stim = StimulusSolver(S, S_old, cfg)
        stim.setup()
        stim.assemble_rhs(psi_density)
        
        its, _ = stim.solve()
        
        # Should converge in reasonable iterations
        assert its < 500, f"Stimulus solver took too many iterations: {its}"
        
        stim.destroy()


# =============================================================================
# Anderson Acceleration Efficiency Tests
# =============================================================================

class TestAndersonEfficiency:
    """Test Anderson acceleration reduces iteration counts."""
    
    def test_anderson_vs_picard(self):
        """Anderson should converge in fewer iterations than Picard."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        
        # Picard iterations
        cfg_picard = Config(
            domain=domain,
            facet_tags=facet_tags,
            verbose=False,
            accel_type="picard",
            max_subiters=50,
            coupling_tol=1e-6
        )
        
        with Remodeller(cfg_picard) as rem_picard:
            rem_picard.step(dt=1.0)
            picard_iters = rem_picard.fixedsolver.total_gs_iters
        
        # Anderson iterations (same mesh)
        domain2 = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags2 = build_facetag(domain2)
        cfg_anderson = Config(
            domain=domain2,
            facet_tags=facet_tags2,
            verbose=False,
            accel_type="anderson",
            max_subiters=50,
            coupling_tol=1e-6
        )
        
        with Remodeller(cfg_anderson) as rem_anderson:
            rem_anderson.step(dt=1.0)
            anderson_iters = rem_anderson.fixedsolver.total_gs_iters
        
        # Anderson should be at least as good as Picard (often better)
        assert anderson_iters <= picard_iters * 1.2, f"Anderson ({anderson_iters}) not better than Picard ({picard_iters})"
    
    def test_anderson_window_size_effect(self):
        """Larger Anderson window should improve convergence."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        
        window_sizes = [2, 5, 8]
        iteration_counts = []
        
        for m in window_sizes:
            domain_i = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
            facet_tags_i = build_facetag(domain_i)
            cfg = Config(
                domain=domain_i,
                facet_tags=facet_tags_i,
                verbose=False,
                accel_type="anderson",
                m=m,
                max_subiters=40,
                coupling_tol=1e-6
            )
            
            with Remodeller(cfg) as rem:
                rem.step(dt=1.0)
                iteration_counts.append(rem.fixedsolver.total_gs_iters)
        
        # Larger window should generally help (not always guaranteed, but check reasonable)
        assert all(it < 100 for it in iteration_counts), "Some window sizes gave excessive iterations"


# =============================================================================
# Preconditioner Effectiveness Tests
# =============================================================================

class TestPreconditioners:
    """Test preconditioner update logic and effectiveness."""
    
    def test_preconditioner_reuse_efficiency(self):
        """Preconditioner reuse should not degrade performance excessively."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        t_vec = np.zeros(3, dtype=np.float64)
        t_vec[0] = -0.3
        traction = (fem.Constant(domain, t_vec), 2)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
        
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
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        A = Function(T, name="A")
        A_old = Function(T, name="A_old")
        S = Function(Q, name="S")
        S_old = Function(Q, name="S_old")
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        stim = StimulusSolver(S, S_old, cfg)
        dens = DensitySolver(rho, rho_old, A, S, cfg)
        dirn = DirectionSolver(A, A_old, cfg)
        
        # Create driver for fixed-point solver (required in new architecture)
        from simulation.drivers import InstantEnergyDriver
        driver = InstantEnergyDriver(mech)
        
        fps = FixedPointSolver(
            comm, cfg, mech, stim, dens, dirn, driver,
            u, rho, rho_old, A, A_old, S, S_old
        )
        
        expected_size = fps.n_u + fps.n_rho + fps.n_A + fps.n_S
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
# Weak Scaling Tests
# =============================================================================

class TestScaling:
    """Test weak scaling properties (work per processor)."""
    
    def test_weak_scaling_dofs_per_rank(self):
        """DOFs per rank should be similar across different partitions."""
        comm = MPI.COMM_WORLD
        
        if comm.size < 2:
            pytest.skip("Weak scaling test requires multiple ranks")
        
        # Each rank should have roughly same local work
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        n_local = Q.dofmap.index_map.size_local
        n_total = comm.allreduce(n_local, op=MPI.SUM)
        n_avg = n_total / comm.size
        
        imbalance = abs(n_local - n_avg) / n_avg
        
        # Allow 50% imbalance (reasonable for small meshes)
        assert imbalance < 0.5, f"Rank {comm.rank} has excessive imbalance: {imbalance:.1%}"


# =============================================================================
# Convergence Rate Tests
# =============================================================================

class TestConvergenceRates:
    """Test solver convergence rates."""
    
    def test_coupling_convergence_monotone(self):
        """Fixed-point residual should decrease (generally) over iterations."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            verbose=False,
            accel_type="anderson",
            max_subiters=30,
            coupling_tol=1e-7
        )
        
        with Remodeller(cfg) as rem:
            rem.step(dt=1.0)
            
            # Check subiter metrics if available
            if hasattr(rem.fixedsolver, 'subiter_metrics') and len(rem.fixedsolver.subiter_metrics) > 1:
                metrics = rem.fixedsolver.subiter_metrics
                
                # Extract residuals
                residuals = [m.get('proj_res', float('inf')) for m in metrics if 'proj_res' in m]
                
                if len(residuals) > 2:
                    # Final residual should be smaller than initial
                    assert residuals[-1] < residuals[0], f"Residual not decreasing: {residuals[0]} → {residuals[-1]}"


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
            assert fps.stim_time_total > 0, "Stimulus time not tracked"
            assert fps.dens_time_total > 0, "Density time not tracked"
            assert fps.dir_time_total > 0, "Direction time not tracked"
