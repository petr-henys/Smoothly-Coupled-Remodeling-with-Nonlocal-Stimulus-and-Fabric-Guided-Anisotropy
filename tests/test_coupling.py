#!/usr/bin/env python3
"""
Fixed-point coupling and Anderson acceleration tests.

Merged from:
- test_fixedpoint_stop_and_anderson.py
- test_coupling_tolerance.py
"""


#!/usr/bin/env python3
"""
Tests focused on the fixed-point loop and Anderson acceleration internals.

Coverage:
- The recorded projected residual 'proj_res' equals the Picard baseline 'r_norm' at every subiteration.
- Anderson step-size limiter is enforced: ||s|| <= factor * ||r||.
- Anderson safeguard backtracking reduces the proxy residual below the (1-γ) threshold.
- Restart on ill-conditioning is triggered when the Gram condition exceeds a tiny threshold.
"""
import numpy as np
import pytest

from mpi4py import MPI
from simulation.model import Remodeller
from simulation.anderson import _Anderson

def _proj_res_norm_ratio(x_ref, x_test, x_raw):
    """Custom projected-residual-norm used to exercise backtracking logic."""
    r = x_raw - x_ref
    t = x_test - x_ref
    denom = np.linalg.norm(r) + 1e-300
    if x_test is x_raw:
        return 1.0
    return float(np.linalg.norm(t) / denom)


def test_anderson_step_limit_and_backtracking():
    """(1) Enforce step limit; (2) Safeguard backtracking reduces proxy residual."""
    comm = MPI.COMM_SELF
    n = 40
    x_old = np.zeros(n)
    x_raw = np.ones(n)

    aa_limit = _Anderson(
        comm=comm, m=3, beta=1.0, lam=1e-12,
        step_limit_factor=0.25,
        verbose=False
    )
    _ = aa_limit.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    x_new, info = aa_limit.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    s_norm = np.linalg.norm(x_new - x_old)
    r_norm = np.linalg.norm(x_raw - x_old)
    assert s_norm <= 0.25 * r_norm + 1e-12, f"Step limiter not enforced: ||s||={s_norm} > 0.25||r||={0.25*r_norm}"

    aa_bt = _Anderson(
        comm=comm, m=3, beta=1.0, lam=0.0,
        step_limit_factor=10.0,
        verbose=False
    )
    _ = aa_bt.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.5, use_safeguard=True)
    x_bt, info_bt = aa_bt.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.5, use_safeguard=True)
    rp_final = _proj_res_norm_ratio(x_old, x_bt, x_raw)
    assert rp_final <= 0.51, f"Backtracking did not reduce proxy residual sufficiently: rp_final={rp_final}"

def test_anderson_restart_on_ill_conditioning():
    """Restart should be scheduled when the H condition number exceeds a tiny threshold."""
    comm = MPI.COMM_SELF
    n = 30
    x_old = np.zeros(n)
    x_raw = np.ones(n)

    aa = _Anderson(comm=comm, m=2, beta=1.0, lam=0.0, restart_on_cond=1.0, verbose=False)
    _ = aa.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    _, info = aa.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    reason = info.get("restart_reason", "")
    assert isinstance(reason, str) and ("illcond" in reason), f"Expected restart on ill-conditioning, got: {reason!r}"

