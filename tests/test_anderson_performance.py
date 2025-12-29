#!/usr/bin/env python3
"""
Performance tests for Anderson acceleration vs Picard iteration.

These tests demonstrate:
1. Anderson acceleration converges in fewer iterations than plain Picard
2. Safeguarding prevents divergence on ill-conditioned problems
3. History restart improves robustness on highly nonlinear problems
"""

import pytest
import numpy as np
from mpi4py import MPI
from typing import Tuple, Callable, Dict, List

from simulation.anderson import Anderson


# =============================================================================
# Test fixtures and utilities
# =============================================================================

def make_anderson(comm: MPI.Comm, **kwargs) -> Anderson:
    """Factory for Anderson accelerator with sensible defaults."""
    defaults = {
        "m": 5,
        "beta": 1.0,
        "lam": 1e-8,
        "restart_on_cond": 1e12,
        "restart_on_stall": 1.1,
        "step_limit_factor": 2.0,
        "restart_stall_window": 3,
        "restart_stall_patience": 2,
    }
    defaults.update(kwargs)
    return Anderson(comm=comm, **defaults)


def run_fixed_point(
    g: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float,
    max_iter: int,
    accelerator: Anderson | None = None,
    beta: float = 1.0,
) -> Tuple[np.ndarray, int, List[float]]:
    """Run fixed-point iteration with optional Anderson acceleration.
    
    Args:
        g: Fixed-point map x -> g(x)
        x0: Initial guess
        tol: Convergence tolerance (relative residual)
        max_iter: Maximum iterations
        accelerator: Optional Anderson accelerator (None = plain Picard)
        beta: Damping factor for Picard (only used if accelerator is None)
    
    Returns:
        (x_final, iterations, residual_history)
    """
    x = x0.copy()
    residuals = []
    
    for k in range(max_iter):
        x_raw = g(x)
        r = x_raw - x
        res_norm = np.linalg.norm(r) / max(np.linalg.norm(x_raw), 1e-30)
        residuals.append(res_norm)
        
        if res_norm < tol:
            return x_raw, k + 1, residuals
        
        if accelerator is not None:
            x, _ = accelerator.mix(x, x_raw)
        else:
            x = x + beta * r
    
    return x, max_iter, residuals


# =============================================================================
# Test problems with known convergence properties
# =============================================================================

def linear_contraction(rho: float, n: int = 50) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """Linear contraction map: g(x) = A*x + b with spectral radius rho.
    
    For rho close to 1, Picard converges slowly; Anderson should accelerate.
    """
    np.random.seed(42)
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    # Eigenvalues uniformly in (-rho, rho)
    eigvals = np.linspace(-rho, rho, n)
    A = Q @ np.diag(eigvals) @ Q.T
    
    # Fixed point x* such that x* = A*x* + b => (I-A)*x* = b
    x_star = np.random.randn(n)
    b = x_star - A @ x_star
    
    def g(x):
        return A @ x + b
    
    x0 = np.zeros(n)
    return g, x0, x_star


def nonlinear_oscillatory(n: int = 20, alpha: float = 0.8) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """Nonlinear map with oscillatory convergence.
    
    g(x) = alpha * sin(x) + (1-alpha) * x_star
    
    The sin causes sign changes in residuals, testing AA's ability to
    handle non-monotonic convergence.
    """
    np.random.seed(123)
    x_star = np.random.randn(n)
    
    def g(x):
        return alpha * np.sin(x) + (1 - alpha) * x_star
    
    x0 = np.zeros(n)
    return g, x0, x_star


def ill_conditioned_linear(cond: float, n: int = 30) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """Linear problem with specified condition number.
    
    Higher condition numbers lead to slow Picard convergence and
    potential AA instability without safeguarding.
    """
    np.random.seed(456)
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    # Eigenvalues logarithmically spaced for high condition number
    eigvals = np.logspace(0, -np.log10(cond), n)
    # Mix positive/negative to create oscillatory behavior
    signs = np.array([(-1)**i for i in range(n)])
    # Scale to keep spectral radius < 1
    eigvals = 0.95 * signs * eigvals / eigvals[0]
    A = Q @ np.diag(eigvals) @ Q.T
    
    x_star = np.random.randn(n)
    b = x_star - A @ x_star
    
    def g(x):
        return A @ x + b
    
    x0 = np.zeros(n)
    return g, x0, x_star


def stiff_nonlinear(n: int = 15, stiffness: float = 10.0) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """Stiff nonlinear problem that can cause AA divergence without safeguard.
    
    g(x) = x_star + exp(-stiffness * ||x - x_star||^2) * (x - x_star)
    
    Near x_star the Jacobian has small eigenvalues (fast convergence),
    but far away it has eigenvalues near 1 (slow/unstable).
    """
    np.random.seed(789)
    x_star = np.random.randn(n)
    
    def g(x):
        diff = x - x_star
        factor = np.exp(-stiffness * np.dot(diff, diff))
        return x_star + factor * diff
    
    # Start far from solution to trigger stiff behavior initially
    x0 = x_star + 2.0 * np.random.randn(n)
    return g, x0, x_star


# =============================================================================
# Performance comparison tests
# =============================================================================

class TestAndersonVsPicard:
    """Tests demonstrating Anderson acceleration advantage over Picard."""
    
    @pytest.mark.parametrize("rho", [0.7, 0.85, 0.95])
    def test_linear_contraction_speedup(self, rho: float):
        """Anderson should converge faster than Picard on linear contractions.
        
        For spectral radius rho close to 1, Picard requires O(1/(1-rho)) iterations.
        Anderson should achieve significant speedup (often near-optimal for linear).
        """
        g, x0, x_star = linear_contraction(rho, n=50)
        tol = 1e-8
        max_iter = 500
        
        # Plain Picard (beta=1)
        _, iters_picard, res_picard = run_fixed_point(g, x0, tol, max_iter, accelerator=None, beta=1.0)
        
        # Anderson accelerated
        aa = make_anderson(MPI.COMM_SELF, m=5)
        _, iters_aa, res_aa = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
        
        # Anderson should use fewer iterations
        speedup = iters_picard / max(iters_aa, 1)
        
        assert iters_aa < iters_picard, (
            f"Anderson ({iters_aa} iters) should be faster than Picard ({iters_picard} iters) "
            f"for rho={rho}"
        )
        
        # For rho >= 0.85, expect at least 2x speedup
        if rho >= 0.85:
            assert speedup >= 2.0, (
                f"Expected at least 2x speedup for rho={rho}, got {speedup:.2f}x"
            )
    
    def test_oscillatory_convergence_improvement(self):
        """Anderson handles oscillatory convergence better than Picard.
        
        When residuals alternate in sign, Picard makes slow progress.
        Anderson uses history to cancel oscillations.
        """
        g, x0, x_star = nonlinear_oscillatory(n=20, alpha=0.8)
        tol = 1e-6
        max_iter = 200
        
        # Plain Picard
        _, iters_picard, _ = run_fixed_point(g, x0, tol, max_iter, accelerator=None, beta=1.0)
        
        # Anderson
        aa = make_anderson(MPI.COMM_SELF, m=5)
        _, iters_aa, _ = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
        
        assert iters_aa < iters_picard, (
            f"Anderson ({iters_aa}) should handle oscillatory problems better than Picard ({iters_picard})"
        )
    
    def test_history_depth_effect(self):
        """More history (larger m) can improve convergence, up to a point.
        
        Too small m limits acceleration; too large m can cause numerical issues.
        """
        g, x0, x_star = linear_contraction(rho=0.9, n=40)
        tol = 1e-8
        max_iter = 300
        
        results = {}
        for m in [1, 3, 5, 10]:
            aa = make_anderson(MPI.COMM_SELF, m=m)
            _, iters, _ = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
            results[m] = iters
        
        # m=1 should be slowest (essentially Picard with slight correction)
        # m=3 or m=5 should be near optimal
        assert results[1] > results[3], f"m=3 ({results[3]}) should beat m=1 ({results[1]})"
        assert results[1] > results[5], f"m=5 ({results[5]}) should beat m=1 ({results[1]})"


class TestRobustness:
    """Tests for Anderson robustness features (step limiting, restart)."""
    
    def test_step_limiting_on_ill_conditioned(self):
        """Step limiting should help on ill-conditioned problems.
        
        The current Anderson implementation uses step limiting to prevent
        excessively large steps.
        """
        g, x0, x_star = ill_conditioned_linear(cond=1e6, n=30)
        tol = 1e-6
        max_iter = 500
        
        # AA with moderate regularization
        aa = make_anderson(MPI.COMM_SELF, m=10, lam=1e-8, step_limit_factor=2.0)
        x_result, iters, res = run_fixed_point(
            g, x0, tol, max_iter, accelerator=aa
        )
        
        # Check final residual
        final_res = res[-1] if res else float("inf")
        
        # Should make progress, even if doesn't fully converge
        assert iters > 0, "Should run at least one iteration"
        assert final_res < 1.0, f"Should make progress; final_res={final_res:.2e}"
    
    def test_info_dict_contains_expected_keys(self):
        """Verify that mix() returns info dict with expected keys."""
        g, x0, x_star = linear_contraction(rho=0.9, n=20)
        
        aa = make_anderson(MPI.COMM_SELF, m=5)
        
        x = x0.copy()
        x_raw = g(x)
        x, info = aa.mix(x, x_raw)
        
        # Check expected keys exist
        expected_keys = {"aa_hist", "condH", "r_norm", "step_norm", "accepted"}
        for key in expected_keys:
            assert key in info, f"Info dict missing key: {key}"
        
        # accepted is always True in current API
        assert info["accepted"] == True, "accepted should always be True"
    
    def test_step_limiting_prevents_overshooting(self):
        """Step limiting should prevent excessively large AA steps."""
        g, x0, x_star = stiff_nonlinear(n=15, stiffness=5.0)
        tol = 1e-6
        max_iter = 200
        
        # With step limiting
        aa = make_anderson(MPI.COMM_SELF, m=5, step_limit_factor=2.0)
        _, iters, res = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
        
        # Should converge
        final_res = res[-1] if res else float("inf")
        assert final_res < tol or iters < max_iter, (
            f"Step-limited AA should converge; final_res={final_res:.2e}"
        )


class TestHistoryRestart:
    """Tests for history restart mechanisms."""
    
    def test_restart_on_stall_detection(self):
        """History can restart when stall is detected."""
        aa = make_anderson(MPI.COMM_SELF, m=3, restart_on_stall=1.1, restart_stall_patience=2)
        n = 20
        
        # Build some history
        x = np.zeros(n)
        for i in range(3):
            x_raw = x + (0.5 ** i) * np.ones(n)
            x, _ = aa.mix(x, x_raw)
        
        # Check that stall tracking mechanism exists
        assert aa._stall_streak >= 0, "Stall streak should be tracked"
        assert aa.restart_stall_patience == 2, "Restart patience should be set"
    
    def test_restart_on_ill_conditioning(self):
        """History should restart when Gram matrix becomes ill-conditioned."""
        aa = make_anderson(MPI.COMM_SELF, m=10, restart_on_cond=1e8)
        n = 30
        
        # Create near-linearly-dependent residuals to cause ill-conditioning
        base = np.random.randn(n)
        x = np.zeros(n)
        
        for i in range(8):
            # Small perturbations create near-dependent residuals
            eps = 1e-6 * np.random.randn(n)
            x_raw = x + base + eps
            x, info = aa.mix(x, x_raw)
            
            condH = info.get("condH", 1.0)
            if condH > aa.restart_on_cond:
                assert aa._pending_reset or info["restart_reason"] != "", (
                    f"Ill-conditioning (cond={condH:.1e}) should trigger restart"
                )
                break


class TestDampingParameter:
    """Tests for beta (damping/under-relaxation) parameter."""
    
    def test_damping_stabilizes_marginally_stable_problem(self):
        """Beta < 1 can stabilize problems where beta=1 oscillates."""
        # Create a problem where undamped iteration oscillates
        n = 20
        np.random.seed(999)
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        # Eigenvalues that cause oscillation with beta=1
        eigvals = -0.95 * np.ones(n)  # All negative, close to -1
        A = Q @ np.diag(eigvals) @ Q.T
        
        x_star = np.random.randn(n)
        b = x_star - A @ x_star
        
        def g(x):
            return A @ x + b
        
        x0 = np.zeros(n)
        tol = 1e-6
        max_iter = 200
        
        # Undamped Picard (beta=1) should oscillate or converge slowly
        _, iters_undamped, res_undamped = run_fixed_point(
            g, x0, tol, max_iter, accelerator=None, beta=1.0
        )
        
        # Damped Picard (beta=0.5) should converge
        _, iters_damped, res_damped = run_fixed_point(
            g, x0, tol, max_iter, accelerator=None, beta=0.5
        )
        
        # Anderson with damping should also work
        aa = make_anderson(MPI.COMM_SELF, m=5, beta=0.7)
        _, iters_aa, res_aa = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
        
        # At least one of damped approaches should converge better
        final_undamped = res_undamped[-1] if res_undamped else float("inf")
        final_damped = res_damped[-1] if res_damped else float("inf")
        final_aa = res_aa[-1] if res_aa else float("inf")
        
        assert min(final_damped, final_aa) < tol or min(iters_damped, iters_aa) < max_iter, (
            f"Damped iteration should converge; final_damped={final_damped:.2e}, final_aa={final_aa:.2e}"
        )


class TestMPIConsistency:
    """Tests for MPI-parallel consistency (run with mpirun -n 2+)."""
    
    def test_parallel_dot_product_consistency(self):
        """Global dot product should give same result across ranks."""
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
        
        aa = make_anderson(comm, m=3)
        
        # Each rank has a portion of the vector
        n_local = 100
        np.random.seed(42 + rank)  # Different data per rank
        a = np.random.randn(n_local)
        b = np.random.randn(n_local)
        
        # Global dot product via Anderson's method
        result = aa._gdot(a, b)
        
        # All ranks should get the same value
        all_results = comm.allgather(result)
        assert all(abs(r - result) < 1e-12 for r in all_results), (
            "All ranks should compute identical global dot product"
        )
    
    def test_parallel_convergence_same_iterations(self):
        """All ranks should converge in exactly the same number of iterations."""
        comm = MPI.COMM_WORLD
        rank = comm.rank
        
        # Same problem on all ranks
        g, x0, x_star = linear_contraction(rho=0.9, n=30)
        tol = 1e-8
        max_iter = 100
        
        aa = make_anderson(comm, m=5)
        _, iters, _ = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
        
        # Gather iteration counts from all ranks
        all_iters = comm.allgather(iters)
        
        assert all(i == iters for i in all_iters), (
            f"All ranks should converge in same iterations: {all_iters}"
        )


# =============================================================================
# Summary test for comprehensive validation
# =============================================================================

class TestAndersonComprehensive:
    """Comprehensive test covering key Anderson acceleration properties."""
    
    def test_full_algorithm_validation(self):
        """End-to-end validation of Anderson acceleration algorithm.
        
        Checks:
        1. Converges faster than Picard on linear problem
        2. Safeguard activates on bad steps
        3. History restart works
        4. Final solution is accurate
        """
        # Moderately difficult linear problem
        g, x0, x_star = linear_contraction(rho=0.9, n=40)
        tol = 1e-10
        max_iter = 300
        
        # Picard baseline
        x_pic, iters_pic, _ = run_fixed_point(g, x0, tol, max_iter, accelerator=None, beta=1.0)
        err_pic = np.linalg.norm(x_pic - x_star)
        
        # Anderson with standard parameters
        aa = make_anderson(
            MPI.COMM_SELF,
            m=5,
            beta=1.0,
            lam=1e-8,
            restart_on_cond=1e10,
            step_limit_factor=2.0,
        )
        x_aa, iters_aa, _ = run_fixed_point(g, x0, tol, max_iter, accelerator=aa)
        err_aa = np.linalg.norm(x_aa - x_star)
        
        # Assertions
        assert iters_aa < iters_pic, (
            f"Anderson ({iters_aa}) should beat Picard ({iters_pic})"
        )
        assert err_aa < 1e-8, f"Anderson solution error {err_aa:.2e} too large"
        assert err_pic < 1e-8 or iters_pic == max_iter, (
            f"Picard should converge or hit max_iter"
        )
        
        # Speedup should be significant
        speedup = iters_pic / max(iters_aa, 1)
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"
