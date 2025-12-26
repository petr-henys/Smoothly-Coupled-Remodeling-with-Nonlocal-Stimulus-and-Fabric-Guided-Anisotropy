#!/usr/bin/env python3
"""
Basic unit tests for Anderson acceleration implementation.

NOTE: Comprehensive performance tests are in test_anderson_performance.py.
This file contains only fundamental unit tests for Anderson class API.
"""

import pytest
import numpy as np
from mpi4py import MPI
from simulation.anderson import Anderson


def make_anderson(comm, m=5, beta=1.0, lam=1e-8, gamma=0.05, safeguard=True,
                  backtrack_max=5, restart_on_reject_k=2,
                  restart_on_cond=1e12, step_limit_factor=2.0, verbose=False):
    """Factory for Anderson accelerator with explicit parameters."""
    return Anderson(
        comm=comm,
        m=m,
        beta=beta,
        lam=lam,
        gamma=gamma,
        safeguard=safeguard,
        backtrack_max=backtrack_max,
        restart_on_reject_k=restart_on_reject_k,
        restart_on_cond=restart_on_cond,
        step_limit_factor=step_limit_factor,
        verbose=verbose,
    )


class TestAndersonBasicAPI:
    """Basic unit tests for Anderson acceleration API."""
    
    def test_initialization(self):
        """Test Anderson initializes with correct parameters."""
        m, beta, lam = 5, 0.9, 1e-6
        aa = make_anderson(MPI.COMM_WORLD, m=m, beta=beta, lam=lam)
        
        assert aa.m == m, f"m mismatch: {aa.m} != {m}"
        assert aa.beta == beta, f"beta mismatch: {aa.beta} != {beta}"
        assert aa.lam == lam, f"lam mismatch: {aa.lam} != {lam}"
        assert len(aa.x_hist) == 0, "History should be empty on init"
        assert len(aa.r_hist) == 0, "Residual history should be empty on init"

    def test_reset_clears_history(self):
        """Reset should clear all accumulated history."""
        aa = make_anderson(MPI.COMM_WORLD, m=3)
        
        # Accumulate some history via mix
        x = np.zeros(50)
        for i in range(5):
            x_raw = x + 0.1 * np.random.rand(50)
            x, _ = aa.mix(x, x_raw)
        
        assert len(aa.x_hist) > 0, "History should accumulate"
        
        aa.reset()
        
        assert len(aa.x_hist) == 0, "x_hist should be empty after reset"
        assert len(aa.r_hist) == 0, "r_hist should be empty after reset"
        assert aa.pending_reset is False, "pending_reset should be False"
        assert aa.reject_streak == 0, "reject_streak should reset"

    def test_mix_returns_correct_shape(self):
        """mix() should return array of same shape as input."""
        aa = make_anderson(MPI.COMM_SELF, m=5)
        n = 100
        x_old = np.random.rand(n)
        x_raw = x_old + 0.1 * np.random.rand(n)
        
        x_new, info = aa.mix(x_old, x_raw)
        
        assert x_new.shape == x_old.shape, f"Shape mismatch: {x_new.shape} != {x_old.shape}"
        assert isinstance(info, dict), "info should be a dict"
        assert "aa_hist" in info, "info missing 'aa_hist' key"
        assert "accepted" in info, "info missing 'accepted' key"

    def test_mix_info_contains_required_keys(self):
        """mix() info dict should contain all required diagnostic keys."""
        aa = make_anderson(MPI.COMM_SELF, m=3, safeguard=True)
        x = np.zeros(20)
        x_raw = np.ones(20)
        
        _, info = aa.mix(x, x_raw)
        
        required_keys = ["aa_hist", "accepted", "condH", "restart_reason"]
        for key in required_keys:
            assert key in info, f"info missing required key '{key}'"

    def test_history_accumulates_correctly(self):
        """History should accumulate up to m+1 entries (implementation detail), then slide."""
        m = 3
        aa = make_anderson(MPI.COMM_SELF, m=m, safeguard=False)
        
        x = np.zeros(20)
        for i in range(m + 3):
            x_raw = x + (0.5 ** i) * np.ones(20)
            x, _ = aa.mix(x, x_raw)
        
        # History is capped at m+1 (implementation uses maxlen=m+1)
        assert len(aa.x_hist) <= m + 1, f"x_hist should not exceed m+1={m+1}"
        assert len(aa.r_hist) <= m + 1, f"r_hist should not exceed m+1={m+1}"

    def test_first_iteration_uses_damping(self):
        """First mix() call should use beta damping."""
        aa = make_anderson(MPI.COMM_SELF, m=5, beta=0.8)
        
        x_old = np.zeros(10)
        x_raw = np.ones(10)
        
        x_new, info = aa.mix(x_old, x_raw)
        
        # First iteration: x_new = x_old + beta * (x_raw - x_old) = beta * x_raw
        expected = 0.8 * np.ones(10)
        assert np.allclose(x_new, expected), "First iteration should be damped Picard"
        # aa_hist is 1 after first iteration (one item in history)
        assert info["aa_hist"] >= 0, "aa_hist should be non-negative"


class TestAndersonNumericalStability:
    """Tests for numerical stability and edge cases."""
    
    def test_handles_zero_residual(self):
        """Should handle case when already at fixed point (zero residual)."""
        aa = make_anderson(MPI.COMM_SELF, m=3)
        
        x = np.ones(10)
        x_raw = np.ones(10)  # Same as x -> zero residual
        
        x_new, info = aa.mix(x, x_raw)
        
        # Should return x_raw without NaN
        assert np.allclose(x_new, x_raw), "Zero residual case should return x_raw"
        assert np.all(np.isfinite(x_new)), "Result should not contain NaN/Inf"

    def test_no_nan_inf_on_small_residuals(self):
        """Very small residuals should not cause NaN/Inf."""
        aa = make_anderson(MPI.COMM_SELF, m=3, lam=1e-12)
        
        x = np.ones(20)
        # Build some history first
        for i in range(3):
            x_raw = x + 1e-8 * np.random.rand(20)
            x, _ = aa.mix(x, x_raw)
        
        # Very small residual
        x_raw = x + 1e-14 * np.ones(20)
        x_new, _ = aa.mix(x, x_raw)
        
        assert np.all(np.isfinite(x_new)), "Small residuals should not produce NaN/Inf"

    def test_lambda_regularization_prevents_singular_gram(self):
        """Lambda regularization should prevent singular Gram matrix issues."""
        aa = make_anderson(MPI.COMM_SELF, m=5, lam=1e-6)
        
        # Create near-linearly-dependent residuals
        base = np.ones(30)
        x = np.zeros(30)
        
        for i in range(5):
            # Residuals are almost identical
            noise = 1e-10 * np.random.rand(30)
            x_raw = x + base + noise
            x, info = aa.mix(x, x_raw)
        
        # Should not crash due to singular matrix
        assert np.all(np.isfinite(x)), "Lambda regularization should prevent NaN"
        # Condition number should be tracked
        assert info["condH"] > 0, "condH should be positive"
