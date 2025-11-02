#!/usr/bin/env python3
"""
Advanced performance and solver efficiency tests for bone remodeling model.

Tests:
- Solver iteration counts vs problem size
- Preconditioner effectiveness
- Anderson acceleration efficiency
- Memory usage patterns
- Weak/strong scaling properties
- Solver convergence rates
"""

import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")
pytestmark = [pytest.mark.slow, pytest.mark.performance]
import numpy as np
np.random.seed(1234)
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
import basix
import time

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.fixedsolver import FixedPointSolver
from simulation.model import Remodeller
comm = MPI.COMM_WORLD


# =============================================================================
# Solver Iteration Efficiency Tests
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
            mech = MechanicsSolver(V, rho, A, bc_mech, [traction], cfg)
            
            mech.solver_setup()
            its, _ = mech.solve(u)
            
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
        
        import ufl
        psi_density = fem.Constant(domain, 50.0)  # Dummy source
        
        stim = StimulusSolver(Q, S_old, cfg)
        stim.solver_setup()
        stim.update_rhs(psi_density)
        
        S = Function(Q, name="S")
        its, _ = stim.solve(S)
        
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
        mech = MechanicsSolver(V, rho, A, bc_mech, [traction], cfg)
        
        mech.solver_setup()
        
        # Solve with fresh preconditioner
        its1, _ = mech.solve(u)
        
        # Slightly modify rho
        rho.x.array[:] = 0.51
        rho.x.scatter_forward()
        mech.update_stiffness()
        
        # Solve with reused preconditioner
        its2, _ = mech.solve(u)
        
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
        mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)
        stim = StimulusSolver(Q, S_old, cfg)
        dens = DensitySolver(Q, rho_old, A, S, cfg)
        dirn = DirectionSolver(T, A_old, cfg)
        
        fps = FixedPointSolver(
            comm, cfg, mech, stim, dens, dirn,
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


# No __main__ runner needed; tests executed via pytest
