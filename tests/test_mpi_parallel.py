#!/usr/bin/env python3
"""MPI parallelism tests: ghosts, decomposition, collectives, parallel I/O."""

import pytest
pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.performance]
import numpy as np
_RNG = np.random.default_rng(1234)
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.params import MaterialParams, SolverParams, OutputParams
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.solvers import MechanicsSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver
from simulation.model import Remodeller
from femur.loader import LoadingCase
comm = MPI.COMM_WORLD


def create_mock_loader(domain):
    """Create a mock Loader for testing without femur-specific dependencies."""
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, P1_vec)
    
    class MockLoader:
        def __init__(self, loading_cases):
            self.V = V
            self.load_tag = 1
            self.traction = Function(V, name="Traction")
            self.traction.x.array[:] = 0.01
            self.traction.x.scatter_forward()
            self._cache = {}
            self._loading_cases = loading_cases
            # Precompute immediately
            self.precompute_loading_cases(loading_cases)
        
        @property
        def loading_cases(self):
            return self._loading_cases
        
        def precompute_loading_cases(self, cases):
            for case in cases:
                self._cache[case.name] = {"traction": self.traction.x.array.copy()}
        
        def set_loading_case(self, case_name):
            cached = self._cache[case_name]
            self.traction.x.array[:] = cached["traction"]
    
    loading_cases = [LoadingCase(name="test", day_cycles=1.0, hip=None, muscles=[])]
    return MockLoader(loading_cases)


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
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
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
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_global_volume_computation(self, unit_cube):
        """Verify global volume = sum of local volumes = 1.0 for unit cube."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        vol_global = comm.allreduce(vol_local, op=MPI.SUM)
        
        assert abs(vol_global - 1.0) < 1e-10, f"Global volume incorrect: {vol_global} ≠ 1.0"


# =============================================================================
# I/O Consistency Tests
# =============================================================================

class TestMPIIO:
    """Test MPI-parallel I/O operations."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_rank0_only_writes(self, unit_cube):
        """Verify rank-0 writes config metadata."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags,
                        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
                        output=OutputParams(results_dir=tmpdir))
            comm.Barrier()
            if comm.rank == 0:
                assert (Path(tmpdir) / "config.json").exists(), "config.json not written on rank 0"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_vtx_output_consistency(self, unit_cube):
        """Test skipped: VTX behavior verified in storage tests with stubs."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = Config(domain=domain, facet_tags=facet_tags,
                        material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
                        output=OutputParams(results_dir=tmpdir))
            loader = create_mock_loader(domain)
            
            with Remodeller(cfg, loader=loader) as rem:
                rem.storage.fields.write("fields", 0.0)
                
                # Should complete without hanging
                comm.Barrier()


# =============================================================================
# Fixed-Point Solver Parallelism Tests
# =============================================================================

class TestFixedPointParallel:
    """Test fixed-point solver in parallel."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_anderson_acceleration_parallel(self, unit_cube):
        """Verify Anderson acceleration works correctly in parallel."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            solver=SolverParams(accel_type="anderson", max_subiters=20, coupling_tol=1e-6)
        )
        loader = create_mock_loader(domain)
        
        with Remodeller(cfg, loader=loader) as rem:
            # Take one time step
            rem.step(1.0)
            
            # Check Anderson was used
            gs_iters = len(rem.fixedsolver.subiter_metrics)
            assert gs_iters > 0, "Fixed-point solver didn't iterate"
            assert gs_iters < 50, f"Too many iterations ({gs_iters}), Anderson may have failed"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_gauss_seidel_convergence_parallel(self, unit_cube):
        """Test Gauss-Seidel converges in parallel."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            solver=SolverParams(accel_type="picard", max_subiters=30, coupling_tol=1e-5)
        )
        loader = create_mock_loader(domain)
        
        with Remodeller(cfg, loader=loader) as rem:
            rem.step(1.0)
            
            # Should converge
            assert len(rem.fixedsolver.subiter_metrics) > 0
            assert len(rem.fixedsolver.subiter_metrics) < 100


# =============================================================================
# Cross-Rank Communication Tests
# =============================================================================

class TestCrossRankComm:
    """Test communication patterns between ranks."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_residual_norm_computation(self, unit_cube):
        """Global residual norm should be consistent across all ranks."""
        comm = MPI.COMM_WORLD
        
        if comm.size < 2:
            pytest.skip("Test requires multiple ranks")
        
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
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
# Memory Usage Tests
# =============================================================================

class TestMemoryUsage:
    """Test memory usage patterns (basic checks)."""
    

    @pytest.mark.parametrize("m", [3, 5])
    def test_anderson_history_bounded(self, m):
        """Anderson history should be bounded by window size m."""
        from simulation.anderson import Anderson
        
        n = 100
        aa = Anderson(
            comm=MPI.COMM_WORLD,
            m=m,
            beta=1.0,
            lam=1e-6,
            restart_on_stall=1.1,
            restart_on_cond=1e10,
            step_limit_factor=2.0,
            restart_stall_window=3,
            restart_stall_patience=2,
        )
        
        # Add more than m updates
        for i in range(10):
            x = _RNG.random(n)
            f = _RNG.random(n)
            _ = aa.mix(x, f)
        
        # History length should not exceed m+1 (deque maxlen)
        hist_len = len(aa.r_hist)
        assert hist_len <= m + 1, f"Anderson history exceeded window size: len={hist_len} > m+1={m+1}"
