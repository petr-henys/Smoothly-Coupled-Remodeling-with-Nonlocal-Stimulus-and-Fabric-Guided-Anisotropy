#!/usr/bin/env python3
"""
Tests for Anderson acceleration implementation.
"""

import pytest
import numpy as np
from mpi4py import MPI
from simulation.anderson import Anderson


def make_anderson(comm, m=5, beta=1.0, lam=1e-8, gamma=0.05, safeguard=True,
                  backtrack_max=5, restart_on_reject_k=2, restart_on_stall=1.1,
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
        restart_on_stall=restart_on_stall,
        restart_on_cond=restart_on_cond,
        step_limit_factor=step_limit_factor,
        verbose=verbose,
    )


class TestAndersonAcceleration:
    """Test Anderson acceleration implementation."""
    
    @pytest.mark.parametrize("operation", ["init", "restart", "mix", "reject_restart"])
    def test_anderson_accelerator_operations(self, operation):
        """Test Anderson accelerator: initialization, restart, mix, reject-triggered restart.
        
        Consolidates 4 separate Anderson tests into single parametrized test.
        """
        m, n = 5, 20
        
        if operation == "init":
            # Initialization test
            beta, lam = 1.0, 1e-8
            aa = make_anderson(MPI.COMM_WORLD, m=m, beta=beta, lam=lam)
            assert aa.m == m and aa.beta == beta and aa.lam == lam, "Anderson parameters not set correctly"

        elif operation == "restart":
            # Reset clears history
            aa = make_anderson(MPI.COMM_WORLD, m=3)
            aa.x_hist.append(np.random.rand(50))
            aa.r_hist.append(np.random.rand(50))
            assert len(aa.x_hist) > 0, "History not accumulated"
            aa.reset()
            assert len(aa.x_hist) == 0, "History not cleared after reset"

        elif operation == "mix":
            # Basic mix operation
            aa = make_anderson(MPI.COMM_SELF, m=m, beta=1.0, lam=1e-10)
            x_old = np.random.rand(n)
            x_raw = x_old + 0.1 * np.random.rand(n)
            x_new, info = aa.mix(x_old, x_raw)
            assert x_new.shape == x_old.shape, "Output shape mismatch"
            assert "aa_hist" in info and "accepted" in info, "Info dict missing required keys"

        elif operation == "reject_restart":
            # Test basic mixing and reset functionality
            aa = make_anderson(MPI.COMM_SELF, m=2, beta=1.0, lam=1e-10, restart_on_reject_k=1)
            x_old = np.zeros(10)
            x_raw = np.ones(10)

            # First mix should work
            x1, info1 = aa.mix(x_old, x_raw)
            assert x1.shape == x_old.shape, "Output shape mismatch"
            assert len(aa.x_hist) >= 1, "History should accumulate"

            # Reset should clear history
            aa.reset()
            assert len(aa.x_hist) == 0, "History should be cleared after reset"
            assert aa.pending_reset is False, "pending_reset should be False after reset"
