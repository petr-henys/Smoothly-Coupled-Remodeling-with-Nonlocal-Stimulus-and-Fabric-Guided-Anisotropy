
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
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")

from mpi4py import MPI

from simulation.config import Config
from simulation.model import Remodeller
from simulation.anderson import _Anderson

def _proj_res_norm_ratio(x_ref, x_test, x_raw):
    """Custom projected-residual-norm used to exercise backtracking logic.

    r_norm = ||x_raw - x_ref|| / ||x_raw - x_ref|| = 1
    proxy  = ||x_test - x_ref|| / ||x_raw - x_ref||  (scales with step length)
    """
    import numpy as _np
    r = x_raw - x_ref
    t = x_test - x_ref
    denom = _np.linalg.norm(r) + 1e-300
    if x_test is x_raw:
        return 1.0
    return float(_np.linalg.norm(t) / denom)

@pytest.mark.integration
def test_proj_res_equals_picard_baseline():
    """The loop should always report 'proj_res' equal to the Picard baseline 'r_norm'."""
    comm = MPI.COMM_WORLD

    # Small problem; we don't need telemetry or per-iter coupling diagnostics here
    from dolfinx import mesh
    from simulation.utils import build_facetag

    domain = mesh.create_unit_cube(comm, 6, 6, 6, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(
        domain=domain,
        facet_tags=facet_tags,
        verbose=False,
        accel_type="anderson",
        max_subiters=12,
        coupling_tol=1e-6,
        enable_telemetry=False,
        coupling_each_iter=False,
    )

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)
        metrics = getattr(rem.fixedsolver, "subiter_metrics", [])

    # On all ranks assert equality of recorded norms
    assert metrics, "No subiteration metrics recorded"
    for rec in metrics:
        r_norm = rec.get("r_norm", float("nan"))
        proj = rec.get("proj_res", float("nan"))
        assert np.isfinite(r_norm) and np.isfinite(proj), "Non-finite norms in metrics"
        denom = max(1.0, abs(r_norm), abs(proj))
        assert abs(proj - r_norm) / denom < 1e-12, f"proj_res != r_norm at iter {rec.get('iter')}: {proj} vs {r_norm}"

@pytest.mark.unit
def test_anderson_step_limit_and_backtracking():
    """(1) Enforce step limit; (2) Safeguard backtracking reduces proxy residual."""
    comm = MPI.COMM_SELF  # isolate from MPI reductions
    import numpy as _np

    n = 40
    x_old = _np.zeros(n)
    x_raw = _np.ones(n)

    # --- (1) Step-size limiter path (requires p >= 2 so that AA branch is used)
    aa_limit = _Anderson(
        comm=comm, m=3, beta=1.0, lam=1e-12,
        step_limit_factor=0.25,  # very strict → will clip
        verbose=False
    )
    _ = aa_limit.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    x_new, info = aa_limit.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    s_norm = _np.linalg.norm(x_new - x_old)
    r_norm = _np.linalg.norm(x_raw - x_old)
    assert s_norm <= 0.25 * r_norm + 1e-12, f"Step limiter not enforced: ||s||={s_norm} > 0.25||r||={0.25*r_norm}"

    # --- (2) Safeguard backtracking (AA branch with γ=0.5; should backtrack to ≲ 0.5 of baseline)
    aa_bt = _Anderson(
        comm=comm, m=3, beta=1.0, lam=0.0,
        step_limit_factor=10.0,  # do not clip; let safeguard act
        verbose=False
    )
    _ = aa_bt.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.5, use_safeguard=True)
    x_bt, info_bt = aa_bt.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.5, use_safeguard=True)
    rp_final = _proj_res_norm_ratio(x_old, x_bt, x_raw)  # ≤ (1-γ_eff) * 1.0 ≈ 0.5
    assert rp_final <= 0.51, f"Backtracking did not reduce proxy residual sufficiently: rp_final={rp_final}"

@pytest.mark.unit
def test_anderson_restart_on_ill_conditioning():
    """Restart should be scheduled when the H condition number exceeds a tiny threshold."""
    comm = MPI.COMM_SELF
    import numpy as _np
    n = 30
    x_old = _np.zeros(n)
    x_raw = _np.ones(n)

    aa = _Anderson(comm=comm, m=2, beta=1.0, lam=0.0, restart_on_cond=1.0, verbose=False)
    _ = aa.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    _, info = aa.mix(x_old, x_raw, proj_residual_norm=_proj_res_norm_ratio, gamma=0.0, use_safeguard=False)
    reason = info.get("restart_reason", "")
    assert isinstance(reason, str) and ("illcond" in reason), f"Expected restart on ill-conditioning, got: {reason!r}"
