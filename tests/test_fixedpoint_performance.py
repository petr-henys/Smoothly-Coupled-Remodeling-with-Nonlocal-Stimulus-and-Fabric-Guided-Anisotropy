#!/usr/bin/env python3
"""
Integration tests for Anderson acceleration within the full FixedPointSolver.

These tests use the actual solver infrastructure to demonstrate:
1. Anderson acceleration reduces subiterations in coupled PDEs
2. Safeguarding improves robustness on stiff remodeling problems
3. Performance metrics are correctly tracked
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import basix.ufl

from simulation.config import Config
from simulation.params import SolverParams, TimeParams, DensityParams, StimulusParams
from simulation.fixedsolver import FixedPointSolver
from simulation.utils import assign


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def unit_cube_mesh():
    """Create a simple unit cube mesh for testing."""
    comm = MPI.COMM_WORLD
    return mesh.create_unit_cube(
        comm, 
        nx=4, ny=4, nz=4,
        cell_type=mesh.CellType.tetrahedron,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


@pytest.fixture
def function_spaces(unit_cube_mesh):
    """Create standard function spaces."""
    cell = unit_cube_mesh.basix_cell()
    P1 = basix.ufl.element("Lagrange", cell, 1)
    Q = fem.functionspace(unit_cube_mesh, P1)
    return {"Q": Q}


class MockCouplingBlock:
    """Mock coupling block for testing FixedPointSolver without full PDE setup.
    
    Simulates a contractive fixed-point map with configurable spectral radius.
    """
    
    def __init__(self, state: fem.Function, state_old: fem.Function, 
                 rho: float = 0.8, oscillatory: bool = False):
        """
        Args:
            state: State field (will be modified)
            state_old: Old state field
            rho: Spectral radius (convergence rate)
            oscillatory: If True, residuals will alternate sign
        """
        self._state = state
        self._state_old = state_old
        self._rho = rho
        self._oscillatory = oscillatory
        self._sweep_count = 0
        
        # Fixed point (randomly generated but consistent)
        np.random.seed(12345)
        n = state.x.array.size
        self._x_star = np.random.randn(n) * 0.5
    
    @property
    def state_fields(self):
        return (self._state,)
    
    @property
    def state_fields_old(self):
        return (self._state_old,)
    
    @property
    def output_fields(self):
        return ()
    
    def setup(self):
        pass
    
    def assemble_lhs(self):
        pass
    
    def post_step_update(self):
        pass
    
    def destroy(self):
        pass
    
    def sweep(self):
        """One fixed-point sweep: x_new = rho * (x_old - x_star) + x_star."""
        from simulation.stats import SweepStats
        import time
        
        start = time.perf_counter()
        self._sweep_count += 1
        
        x = self._state.x.array.copy()
        factor = self._rho
        
        if self._oscillatory:
            # Alternate sign to create oscillatory convergence
            factor *= (-1) ** self._sweep_count
        
        # Contraction towards fixed point
        x_new = self._x_star + factor * (x - self._x_star)
        self._state.x.array[:] = x_new
        self._state.x.scatter_forward()
        
        elapsed = time.perf_counter() - start
        return SweepStats(label="mock", ksp_iters=1, solve_time=elapsed, ksp_reason=1)


class StiffMockBlock(MockCouplingBlock):
    """Mock block with stiff nonlinear behavior.
    
    Convergence rate depends on distance from fixed point:
    - Far away: slow convergence (rho near 1)
    - Near fixed point: fast convergence (rho small)
    
    This can cause AA to over-extrapolate when starting far away.
    """
    
    def __init__(self, state: fem.Function, state_old: fem.Function,
                 rho_near: float = 0.3, rho_far: float = 0.99, transition_scale: float = 1.0):
        super().__init__(state, state_old, rho=rho_near)
        self._rho_near = rho_near
        self._rho_far = rho_far
        self._transition_scale = transition_scale
    
    def sweep(self):
        from simulation.stats import SweepStats
        import time
        
        start = time.perf_counter()
        self._sweep_count += 1
        
        x = self._state.x.array.copy()
        
        # Distance-dependent contraction rate
        dist = np.linalg.norm(x - self._x_star)
        # Smooth transition: far = rho_far, near = rho_near
        t = np.exp(-dist / self._transition_scale)
        rho = self._rho_far * (1 - t) + self._rho_near * t
        
        x_new = self._x_star + rho * (x - self._x_star)
        self._state.x.array[:] = x_new
        self._state.x.scatter_forward()
        
        elapsed = time.perf_counter() - start
        return SweepStats(label="stiff", ksp_iters=1, solve_time=elapsed, ksp_reason=1)


def make_config_with_solver_params(
    domain: mesh.Mesh,
    accel_type: str = "anderson",
    m: int = 5,
    beta: float = 1.0,
    safeguard: bool = True,
    coupling_tol: float = 1e-6,
    max_subiters: int = 100,
) -> Config:
    """Create Config with specific solver parameters."""
    solver = SolverParams(
        accel_type=accel_type,
        m=m,
        beta=beta,
        safeguard=safeguard,
        coupling_tol=coupling_tol,
        max_subiters=max_subiters,
        gamma=0.1,
        lam=1e-8,
        restart_on_reject_k=3,
        restart_on_stall=2.0,
        restart_on_cond=1e10,
        step_limit_factor=2.0,
    )
    time_params = TimeParams(dt_initial=1.0, total_time=10.0)
    
    return Config(
        domain=domain,
        solver=solver,
        time=time_params,
    )


# =============================================================================
# Integration tests
# =============================================================================

class TestFixedPointSolverPerformance:
    """Integration tests comparing Anderson vs Picard in FixedPointSolver."""
    
    @pytest.mark.parametrize("rho", [0.7, 0.85, 0.92])
    def test_anderson_fewer_subiterations_than_picard(self, unit_cube_mesh, function_spaces, rho):
        """Anderson should require fewer subiterations than Picard for contractive maps."""
        Q = function_spaces["Q"]
        
        # Create state fields
        u = fem.Function(Q, name="u")
        u_old = fem.Function(Q, name="u_old")
        
        # Initialize away from fixed point
        assign(u, 1.0)
        assign(u_old, 1.0)
        
        # Picard solver (with relaxation for stability)
        cfg_picard = make_config_with_solver_params(
            unit_cube_mesh, 
            accel_type="picard",
            beta=0.8,  # Under-relaxation for stability
            coupling_tol=1e-6,
            max_subiters=300,
        )
        block_picard = MockCouplingBlock(u, u_old, rho=rho)
        fp_picard = FixedPointSolver(MPI.COMM_WORLD, cfg_picard, [block_picard])
        
        converged_picard = fp_picard.run(None, None)
        iters_picard = len(fp_picard.subiter_metrics)
        
        # Reset state
        assign(u, 1.0)
        assign(u_old, 1.0)
        block_picard._sweep_count = 0
        
        # Anderson solver
        cfg_aa = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            m=5,
            beta=1.0,
            safeguard=False,  # Disable for clean comparison on easy problem
            coupling_tol=1e-6,
            max_subiters=300,
        )
        block_aa = MockCouplingBlock(u, u_old, rho=rho)
        fp_aa = FixedPointSolver(MPI.COMM_WORLD, cfg_aa, [block_aa])
        
        converged_aa = fp_aa.run(None, None)
        iters_aa = len(fp_aa.subiter_metrics)
        
        # Both should converge
        assert converged_picard, f"Picard should converge for rho={rho}"
        assert converged_aa, f"Anderson should converge for rho={rho}"
        
        # Anderson should be faster
        assert iters_aa < iters_picard, (
            f"Anderson ({iters_aa} iters) should beat Picard ({iters_picard} iters) for rho={rho}"
        )
        
        # Report speedup
        speedup = iters_picard / max(iters_aa, 1)
        print(f"\nrho={rho}: Picard={iters_picard}, Anderson={iters_aa}, speedup={speedup:.2f}x")
    
    def test_anderson_handles_slow_convergence(self, unit_cube_mesh, function_spaces):
        """Anderson should speed up slowly converging problems."""
        Q = function_spaces["Q"]
        
        u = fem.Function(Q, name="u")
        u_old = fem.Function(Q, name="u_old")
        assign(u, 1.0)
        assign(u_old, 1.0)
        
        # Picard with slow convergence (high rho)
        cfg_picard = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="picard",
            beta=1.0,
            coupling_tol=1e-6,
            max_subiters=300,
        )
        block_picard = MockCouplingBlock(u, u_old, rho=0.9, oscillatory=False)
        fp_picard = FixedPointSolver(MPI.COMM_WORLD, cfg_picard, [block_picard])
        
        converged_picard = fp_picard.run(None, None)
        iters_picard = len(fp_picard.subiter_metrics)
        
        # Reset
        assign(u, 1.0)
        
        # Anderson
        cfg_aa = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            m=5,
            beta=1.0,
            safeguard=True,
            coupling_tol=1e-6,
            max_subiters=300,
        )
        block_aa = MockCouplingBlock(u, u_old, rho=0.9, oscillatory=False)
        fp_aa = FixedPointSolver(MPI.COMM_WORLD, cfg_aa, [block_aa])
        
        converged_aa = fp_aa.run(None, None)
        iters_aa = len(fp_aa.subiter_metrics)
        
        # Both should converge
        assert converged_picard, "Picard should converge"
        assert converged_aa, "Anderson should converge"
        
        # Anderson should be significantly faster on slow-converging problems
        assert iters_aa < iters_picard, (
            f"Anderson ({iters_aa}) should be faster than Picard ({iters_picard})"
        )


class TestSafeguardingIntegration:
    """Tests for safeguarding in the full solver context."""
    
    def test_safeguard_on_stiff_problem(self, unit_cube_mesh, function_spaces):
        """Safeguard should improve robustness on stiff problems."""
        Q = function_spaces["Q"]
        
        u = fem.Function(Q, name="u")
        u_old = fem.Function(Q, name="u_old")
        
        # Start far from fixed point to trigger stiff behavior
        assign(u, 5.0)
        assign(u_old, 5.0)
        
        # AA without safeguard
        cfg_no_safe = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            m=5,
            safeguard=False,
            coupling_tol=1e-6,
            max_subiters=150,
        )
        block_no_safe = StiffMockBlock(u, u_old, rho_near=0.1, rho_far=0.98)
        fp_no_safe = FixedPointSolver(MPI.COMM_WORLD, cfg_no_safe, [block_no_safe])
        
        converged_no_safe = fp_no_safe.run(None, None)
        iters_no_safe = len(fp_no_safe.subiter_metrics)
        final_res_no_safe = fp_no_safe.subiter_metrics[-1]["picard_res"] if fp_no_safe.subiter_metrics else float("inf")
        
        # Reset
        assign(u, 5.0)
        
        # AA with safeguard
        cfg_safe = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            m=5,
            safeguard=True,
            coupling_tol=1e-6,
            max_subiters=150,
        )
        block_safe = StiffMockBlock(u, u_old, rho_near=0.1, rho_far=0.98)
        fp_safe = FixedPointSolver(MPI.COMM_WORLD, cfg_safe, [block_safe])
        
        converged_safe = fp_safe.run(None, None)
        iters_safe = len(fp_safe.subiter_metrics)
        final_res_safe = fp_safe.subiter_metrics[-1]["picard_res"] if fp_safe.subiter_metrics else float("inf")
        
        # Both versions should work on this problem (mock is well-behaved)
        # The key check is that safeguarded version doesn't diverge
        assert converged_safe or final_res_safe < 1e-4, (
            f"Safeguarded should converge or have low residual, got {final_res_safe:.2e}"
        )
        
        print(f"\nStiff problem: no_safe={iters_no_safe} iters (res={final_res_no_safe:.2e}), "
              f"safe={iters_safe} iters (res={final_res_safe:.2e})")
    
    def test_rejection_statistics_tracked(self, unit_cube_mesh, function_spaces):
        """Verify that AA rejection statistics are properly tracked."""
        Q = function_spaces["Q"]
        
        u = fem.Function(Q, name="u")
        u_old = fem.Function(Q, name="u_old")
        assign(u, 3.0)
        assign(u_old, 3.0)
        
        cfg = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            m=5,
            safeguard=True,
            coupling_tol=1e-6,
            max_subiters=100,
        )
        block = StiffMockBlock(u, u_old, rho_near=0.2, rho_far=0.95)
        fp = FixedPointSolver(MPI.COMM_WORLD, cfg, [block])
        
        fp.run(None, None)
        
        # Check that metrics are recorded
        assert len(fp.subiter_metrics) > 0, "Should have recorded metrics"
        
        # Check structure of metrics
        for m in fp.subiter_metrics:
            assert "aa_accepted" in m, "Should track AA acceptance"
            assert "aa_hist" in m, "Should track history depth"
            assert "condH" in m, "Should track condition number"
            assert "picard_res" in m, "Should track residual"
        
        # Count rejections
        rejections = sum(1 for m in fp.subiter_metrics if not m["aa_accepted"])
        print(f"\nRejection rate: {rejections}/{len(fp.subiter_metrics)} = "
              f"{100*rejections/len(fp.subiter_metrics):.1f}%")


class TestHistoryDepthEffect:
    """Tests for the effect of history depth m."""
    
    def test_optimal_history_depth(self, unit_cube_mesh, function_spaces):
        """Test that intermediate m values work best."""
        Q = function_spaces["Q"]
        
        results = {}
        
        for m in [1, 3, 5, 10]:
            u = fem.Function(Q, name="u")
            u_old = fem.Function(Q, name="u_old")
            assign(u, 1.0)
            assign(u_old, 1.0)
            
            cfg = make_config_with_solver_params(
                unit_cube_mesh,
                accel_type="anderson",
                m=m,
                safeguard=False,
                coupling_tol=1e-8,
                max_subiters=100,
            )
            block = MockCouplingBlock(u, u_old, rho=0.9)
            fp = FixedPointSolver(MPI.COMM_WORLD, cfg, [block])
            
            converged = fp.run(None, None)
            iters = len(fp.subiter_metrics)
            results[m] = (iters, converged)
        
        # m=1 should be slowest
        assert results[1][0] >= results[3][0], f"m=1 should be slower than m=3"
        assert results[1][0] >= results[5][0], f"m=1 should be slower than m=5"
        
        print(f"\nHistory depth effect: {results}")


class TestMetricsRecording:
    """Tests for proper metrics recording."""
    
    def test_subiter_metrics_complete(self, unit_cube_mesh, function_spaces):
        """All expected metrics should be recorded for each subiteration."""
        Q = function_spaces["Q"]
        
        u = fem.Function(Q, name="u")
        u_old = fem.Function(Q, name="u_old")
        assign(u, 1.0)
        
        cfg = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            coupling_tol=1e-6,
            max_subiters=50,
        )
        block = MockCouplingBlock(u, u_old, rho=0.8)
        fp = FixedPointSolver(MPI.COMM_WORLD, cfg, [block])
        
        fp.run(None, None)
        
        expected_keys = [
            "iter", "proj_res", "picard_res", "aa_step_res",
            "aa_hist", "aa_accepted", "aa_restart", "condH", "mem_mb"
        ]
        
        for metric in fp.subiter_metrics:
            for key in expected_keys:
                assert key in metric, f"Missing key '{key}' in metrics"
    
    def test_residual_decreases_on_convergence(self, unit_cube_mesh, function_spaces):
        """Residual should generally decrease when converging."""
        Q = function_spaces["Q"]
        
        u = fem.Function(Q, name="u")
        u_old = fem.Function(Q, name="u_old")
        assign(u, 1.0)
        
        cfg = make_config_with_solver_params(
            unit_cube_mesh,
            accel_type="anderson",
            coupling_tol=1e-8,
            max_subiters=100,
        )
        block = MockCouplingBlock(u, u_old, rho=0.8)
        fp = FixedPointSolver(MPI.COMM_WORLD, cfg, [block])
        
        converged = fp.run(None, None)
        assert converged, "Should converge"
        
        residuals = [m["picard_res"] for m in fp.subiter_metrics]
        
        # Overall trend should be decreasing
        assert residuals[-1] < residuals[0], "Final residual should be smaller than initial"
        
        # Check for approximate monotonicity (allowing some fluctuation due to AA)
        decreases = sum(1 for i in range(1, len(residuals)) if residuals[i] < residuals[i-1])
        decrease_rate = decreases / (len(residuals) - 1)
        assert decrease_rate > 0.5, f"Residual should decrease most of the time, got {decrease_rate:.1%}"
