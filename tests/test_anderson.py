#!/usr/bin/env python3
"""
Tests for Anderson acceleration implementation.
"""

import pytest
import numpy as np
from mpi4py import MPI
from simulation.anderson import _Anderson

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
            aa = _Anderson(MPI.COMM_WORLD, m=m, beta=beta, lam=lam)
            assert aa.m == m and aa.beta == beta and aa.lam == lam, "Anderson parameters not set correctly"

        elif operation == "restart":
            # Reset clears history
            aa = _Anderson(MPI.COMM_WORLD, m=3)
            aa.x_hist.append(np.random.rand(50))
            aa.r_hist.append(np.random.rand(50))
            assert len(aa.x_hist) > 0, "History not accumulated"
            aa.reset()
            assert len(aa.x_hist) == 0, "History not cleared after reset"

        elif operation == "mix":
            # Basic mix operation
            aa = _Anderson(MPI.COMM_SELF, m=m, beta=1.0, lam=1e-10)
            x_old = np.random.rand(n)
            x_raw = x_old + 0.1 * np.random.rand(n)
            x_new, info = aa.mix(x_old, x_raw)
            assert x_new.shape == x_old.shape, "Output shape mismatch"
            assert "aa_hist" in info and "accepted" in info, "Info dict missing required keys"

        elif operation == "reject_restart":
            # Restart triggered by rejection streak
            aa = _Anderson(MPI.COMM_SELF, m=2, beta=1.0, lam=1e-10, restart_on_reject_k=1)
            x_old, x_raw = np.zeros(10), np.ones(10)

            # Proxy residual larger than reference triggers rejection
            def prn(a, b, c):
                return 2.0

            # First rejection
            x1, info1 = aa.mix(x_old, x_raw, proj_residual_norm=prn)
            assert info1.get("accepted") is False, "First call should reject"

            # Second rejection triggers restart
            x2, info2 = aa.mix(x_old, x_raw, proj_residual_norm=prn)
            assert isinstance(info2.get("restart_reason", ""), str), "Restart reason missing"
            assert "reject_streak" in info2.get("restart_reason", ""), "Restart not scheduled on reject streak"

            # Third call honors pending reset
            _ = aa.mix(x_old, x_raw, proj_residual_norm=prn)
            assert len(aa.x_hist) <= 1, "History not cleared after scheduled reset"
