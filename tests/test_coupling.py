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
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")

from mpi4py import MPI

from simulation.config import Config
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

@pytest.mark.unit
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


################################################################################


# test_coupling_tolerance.py
# Verifies the influence of coupling tolerance (FixedPointSolver GS loop)
# on solution error relative to a tight-tolerance reference.

import math
import numpy as np
import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from simulation.config import Config
from simulation.model import Remodeller
from simulation.utils import build_facetag


def build_interior_cell_tags(domain, margin: float):
    """Tag interior cells in [margin, 1-margin]^d."""
    from dolfinx import mesh as _mesh
    import numpy as _np
    gdim = domain.geometry.dim
    def is_interior(x):
        cond = _np.ones(x.shape[1], dtype=bool)
        for i in range(gdim):
            cond &= (x[i] >= margin - 1e-12) & (x[i] <= 1.0 - margin + 1e-12)
        return cond
    dim = domain.topology.dim
    entities = _mesh.locate_entities(domain, dim, is_interior)
    owned = entities[entities < domain.topology.index_map(dim).size_local].astype(_np.int32)
    tags = _np.full_like(owned, 1, dtype=_np.int32)
    return _mesh.meshtags(domain, dim, owned, tags)


COMM = MPI.COMM_WORLD
RANK = COMM.rank
INTERIOR_MARGIN = 0.1


def _run_once(coupling_tol: float, max_subiters: int = 200, accel: str = "anderson", *, domain=None):
    """Build a small problem and perform a single step with given coupling tolerance."""
    if domain is None:
        domain = mesh.create_unit_cube(COMM, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                 accel_type=accel, m=5, beta=1.0, lam=1e-8,
                 max_subiters=max_subiters, min_subiters=1,
                 coupling_tol=coupling_tol, coupling_each_iter=False,
                 verbose=False, enable_telemetry=False)

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)
        
        tags = build_interior_cell_tags(domain, INTERIOR_MARGIN)
        dx_int = ufl.Measure("dx", domain=domain, subdomain_data=tags, subdomain_id=1)
        
        energy_mean = rem.mechsolver.average_strain_energy(rem.u)
        S_loc = fem.assemble_scalar(fem.form(rem.S * dx_int))
        rho_loc = fem.assemble_scalar(fem.form(rem.rho * dx_int))
        A2_expr = ufl.inner(rem.A, rem.A)
        anis_loc = fem.assemble_scalar(fem.form(A2_expr * dx_int))
        vol_loc = fem.assemble_scalar(fem.form(1.0 * dx_int))
        
        S_sum = rem.comm.allreduce(S_loc, op=MPI.SUM)
        rho_sum = rem.comm.allreduce(rho_loc, op=MPI.SUM)
        anis_sum = rem.comm.allreduce(anis_loc, op=MPI.SUM)
        vol = rem.comm.allreduce(vol_loc, op=MPI.SUM)
        
        S_mean = S_sum / max(vol, 1e-16)
        rho_mean = rho_sum / max(vol, 1e-16)
        anis_mean = anis_sum / max(vol, 1e-16)
        
        proj_res = [rec["proj_res"] for rec in getattr(rem.fixedsolver, "subiter_metrics", [])]
    
    return (energy_mean, S_mean, rho_mean, anis_mean), proj_res


@pytest.mark.parametrize("accel", ["anderson", "picard"])
def test_coupling_tol_vs_qoi_error(accel):
    """Tighter coupling tolerance should decrease QoI error monotonically."""
    ref_qois, ref_res = _run_once(coupling_tol=1e-10, max_subiters=200, accel=accel)

    tols = [1e-2, 1e-3, 1e-4]
    series = []
    for tol in tols:
        qois, res = _run_once(coupling_tol=tol, max_subiters=200, accel=accel)
        err = sum(abs(a-b) for a,b in zip(qois, ref_qois))
        series.append((tol, err))

    COMM.Barrier()
    if RANK != 0:
        return

    errs = [e for _, e in series]
    for i in range(1, len(errs)):
        assert errs[i] <= errs[i-1] * 1.05, f"Error did not decrease: {errs[i]} vs {errs[i-1]}"

    if errs[-1] > 1e-12 and errs[0] > 0:
        rel_drop = (errs[0] - errs[-1]) / max(errs[0], 1e-300)
        if rel_drop > 0.05:
            slope = (math.log(errs[-1]) - math.log(errs[0])) / (math.log(tols[-1]) - math.log(tols[0]))
            assert 0.3 <= slope <= 2.5, f"Scaling slope {slope:.2f} outside expected range"


def _run_once_full(coupling_tol: float, *, max_subiters: int = 200, min_subiters: int = 1,
                   accel: str = "anderson", coupling_each_iter: bool = False, domain=None,
                   return_gains: bool = False):
    """Build a small problem and perform a single step with given coupling tolerance."""
    if domain is None:
        domain = mesh.create_unit_cube(COMM, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                 accel_type=accel, m=5, beta=1.0, lam=1e-8,
                 max_subiters=max_subiters, min_subiters=min_subiters,
                 coupling_tol=coupling_tol, coupling_each_iter=coupling_each_iter,
                 verbose=False, enable_telemetry=False)

    diag_payload = None

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)
        
        tags = build_interior_cell_tags(domain, INTERIOR_MARGIN)
        dx_int = ufl.Measure("dx", domain=domain, subdomain_data=tags, subdomain_id=1)
        
        energy_mean = rem.mechsolver.average_strain_energy(rem.u)
        S_loc = fem.assemble_scalar(fem.form(rem.S * dx_int))
        rho_loc = fem.assemble_scalar(fem.form(rem.rho * dx_int))
        A2_expr = ufl.inner(rem.A, rem.A)
        anis_loc = fem.assemble_scalar(fem.form(A2_expr * dx_int))
        vol_loc = fem.assemble_scalar(fem.form(1.0 * dx_int))
        
        S_sum = rem.comm.allreduce(S_loc, op=MPI.SUM)
        rho_sum = rem.comm.allreduce(rho_loc, op=MPI.SUM)
        anis_sum = rem.comm.allreduce(anis_loc, op=MPI.SUM)
        vol = rem.comm.allreduce(vol_loc, op=MPI.SUM)
        
        S_mean = S_sum / max(vol, 1e-16)
        rho_mean = rho_sum / max(vol, 1e-16)
        anis_mean = anis_sum / max(vol, 1e-16)
        
        subm = list(getattr(rem.fixedsolver, "subiter_metrics", []))

        if return_gains:
            J_raw, rho_val = rem.fixedsolver.compute_interaction_gains(
                eps=getattr(cfg, "coupling_eps", 1e-3), project_u_dirichlet=True
            )
            last = subm[-1] if subm else {}
            J_gs = last.get("J_gs")
            rho_metric = last.get("rhoJ")
            diag_payload = {
                "J_raw": J_raw,
                "rho": rho_val,
                "J_gs": np.asarray(J_gs) if J_gs is not None else None,
                "rho_metric": rho_metric,
            }

    if return_gains:
        return (energy_mean, S_mean, rho_mean, anis_mean), subm, diag_payload
    return (energy_mean, S_mean, rho_mean, anis_mean), subm


@pytest.mark.parametrize("unit_cube", [6], indirect=True)
def test_spectral_radius_bounds_contraction(unit_cube):
    """Check that observed contraction is bounded by rho(J) and verify MPI consistency."""
    tol = 1e-4
    qois, metrics = _run_once_full(coupling_tol=tol, max_subiters=200, min_subiters=3,
                                   accel="anderson", coupling_each_iter=True, domain=unit_cube)

    COMM.Barrier()
    
    assert len(metrics) >= 3, "Not enough subiterations to estimate contraction."

    proj = [m["proj_res"] for m in metrics if np.isfinite(m.get("proj_res", np.nan))]
    ratios = []
    for i in range(1, len(proj)):
        if proj[i-1] > 0:
            ratios.append(proj[i] / proj[i-1])
    assert ratios, "No valid residual ratios found."

    last_rhoJ = metrics[-1].get("rhoJ", float("nan"))
    assert np.isfinite(last_rhoJ), "rhoJ not recorded."
    assert last_rhoJ < 1.0 + 1e-6, f"rho(J) >= 1: rhoJ={last_rhoJ:.3e}"

    tail = ratios[-min(5, len(ratios)):]
    median_ratio = float(np.median(tail))
    
    if RANK == 0:
        assert median_ratio <= last_rhoJ + 0.05, f"Contraction {median_ratio:.3e} exceeds rho(J) {last_rhoJ:.3e}"
    
    vals = COMM.allgather(float(last_rhoJ))
    vals_finite = [v for v in vals if np.isfinite(v)]
    if RANK == 0 and vals_finite:
        spread = max(vals_finite) - min(vals_finite)
        assert spread <= 1e-10, f"rho(J) spread: {spread:.3e} from {vals_finite}"


@pytest.mark.parametrize("unit_cube", [6], indirect=True)
def test_max_subiters_sufficient_and_insufficient(unit_cube):
    """Fix tolerance and show that too-small max_subiters fails to meet it."""
    tol = 1e-6
    q_suf, m_suf = _run_once_full(coupling_tol=tol, max_subiters=64, min_subiters=1, coupling_each_iter=False, domain=unit_cube)
    q_ins, m_ins = _run_once_full(coupling_tol=tol, max_subiters=2,  min_subiters=1, coupling_each_iter=False, domain=unit_cube)

    COMM.Barrier()
    if RANK != 0:
        return

    assert m_suf, "No subiteration metrics recorded for sufficient run."
    assert m_ins, "No subiteration metrics recorded for insufficient run."

    assert m_suf[-1]["proj_res"] < tol, f"Sufficient run did not meet tolerance: {m_suf[-1]['proj_res']} >= {tol}"
    assert m_ins[-1]["proj_res"] >= tol, f"Insufficient run unexpectedly met tolerance: {m_ins[-1]['proj_res']} < {tol}"

    err = sum(abs(a - b) for a, b in zip(q_ins, q_suf))
    assert err >= 1e-9, "QoIs identical despite failing to meet coupling tolerance."


@pytest.mark.integration
def test_rhoJ_record_matches_recompute():
    """When coupling_each_iter=True, the last recorded rhoJ should match a fresh recomputation."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 6, 6, 6, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(
        domain=domain,
        facet_tags=facet_tags,
        verbose=False,
        accel_type="anderson",
        coupling_each_iter=True,
        max_subiters=10,
        coupling_tol=1e-6,
        enable_telemetry=False
    )

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)
        metrics = getattr(rem.fixedsolver, "subiter_metrics", [])
        assert metrics, "No subiteration metrics recorded"
        last = metrics[-1]
        assert "rhoJ" in last, "rhoJ not present in subiteration metrics"
        rho_recorded = last["rhoJ"]

        _, rho_recomp = rem.fixedsolver.compute_interaction_gains(project_u_dirichlet=True)
        if np.isfinite(rho_recorded) and np.isfinite(rho_recomp):
            denom = max(1.0, abs(rho_recomp))
            assert abs(rho_recorded - rho_recomp) / denom < 0.10, f"rho(J) mismatch: recorded={rho_recorded}, recomputed={rho_recomp}"
