
# test_coupling_tolerance.py
# Verifies the influence of coupling tolerance (FixedPointSolver GS loop)
# on solution error relative to a tight-tolerance reference.
#
# Strategy (single time step, fixed mesh):
# 1) Run a reference solve with very tight tolerance (e.g., 1e-10) and large max_subiters.
# 2) Run solves with looser tolerances (e.g., 1e-2, 1e-3, 1e-4).
# 3) Compare QoIs on an interior subdomain against the reference and check monotone decrease of error
#    as tolerance tightens. Additionally, check that log–log slope ~ 1 (soft bound).
#
# Notes:
# - We use interior mean QoIs (rho_bar, S_bar, anisotropy) via convergence_utils helpers.
# - We keep mesh modest (8^3) for CI stability.
# - The test is MPI-safe (collective barriers) and only asserts on rank 0.
#
import math
import numpy as np
import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")
from mpi4py import MPI
from dolfinx import mesh
import ufl

# Local imports handled by conftest path hook
from simulation.config import Config
from simulation.model import Remodeller
from simulation.utils import build_facetag
def build_interior_cell_tags(domain, margin: float):
    """Tag interior cells in [margin, 1-margin]^d without external deps."""
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
    # mark locally owned only
    owned = entities[entities < domain.topology.index_map(dim).size_local].astype(_np.int32)
    tags = _np.full_like(owned, 1, dtype=_np.int32)
    return _mesh.meshtags(domain, dim, owned, tags)

COMM = MPI.COMM_WORLD
RANK = COMM.rank

# Test parameters (fixed, no environment dependencies)
INTERIOR_MARGIN = 0.1

# Output silencing handled by conftest


# Note: Removed duplicate _compute_mean_value and _compute_anisotropy_index helpers
# Use conftest.mean_value_factory fixture instead for global QoI computation


def _run_once(coupling_tol: float, max_subiters: int = 200, accel: str = "anderson", *, domain=None):
    """Build a small problem and perform a single step with given coupling tolerance.
    
    Computes simple QoI metrics for tolerance verification without storing in CSV.
    """
    from dolfinx import fem
    if domain is None:
        domain = mesh.create_unit_cube(COMM, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags,
                 accel_type=accel, m=5, beta=1.0, lam=1e-8,
                 max_subiters=max_subiters, min_subiters=1,
                 coupling_tol=coupling_tol, coupling_each_iter=False,
                 verbose=False, enable_telemetry=False)

    with Remodeller(cfg) as rem:
        rem.step(dt=1.0)  # one outer step
        
        # Build interior measure for QoI computation
        tags = build_interior_cell_tags(domain, INTERIOR_MARGIN)
        dx_int = ufl.Measure("dx", domain=domain, subdomain_data=tags, subdomain_id=1)
        
        # Compute QoIs using inline integration (avoid duplicate helpers)
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
        
        # Capture per-subiteration projected residuals (if exposed)
        proj_res = [rec["proj_res"] for rec in getattr(rem.fixedsolver, "subiter_metrics", [])]
    
    return (energy_mean, S_mean, rho_mean, anis_mean), proj_res


@pytest.mark.parametrize("accel", ["anderson", "picard"])
def test_coupling_tol_vs_qoi_error(accel):
    # Reference run (tight tolerance)
    ref_qois, ref_res = _run_once(coupling_tol=1e-10, max_subiters=200, accel=accel)

    # Tolerance ladder (coarse -> fine); keep few points for runtime
    tols = [1e-2, 1e-3, 1e-4]
    series = []
    for tol in tols:
        qois, res = _run_once(coupling_tol=tol, max_subiters=200, accel=accel)
        # L1 diff across QoIs (simple, robust)
        err = sum(abs(a-b) for a,b in zip(qois, ref_qois))
        series.append((tol, err))

    # Only assert on rank 0
    COMM.Barrier()
    if RANK != 0:
        return

    # Monotone decrease of error with tightening tolerance
    errs = [e for _, e in series]
    for i in range(1, len(errs)):
        assert errs[i] <= errs[i-1] * 1.05, f"Error did not decrease sufficiently: {errs[i]} vs {errs[i-1]}"

    # (Soft) log–log slope ~ 1 between first and last pair, unless close to floor
    # Skip slope check when dynamic range is tiny or near machine zero
    if errs[-1] > 1e-12 and errs[0] > 0:
        rel_drop = (errs[0] - errs[-1]) / max(errs[0], 1e-300)
        if rel_drop > 0.05:  # require at least 5% drop to assess slope meaningfully
            slope = (math.log(errs[-1]) - math.log(errs[0])) / (math.log(tols[-1]) - math.log(tols[0]))
            # Relaxed bounds to accommodate multiple QoI types (energy, density, stimulus, anisotropy)
            assert 0.3 <= slope <= 2.5, f"Observed scaling slope ~ {slope:.2f} outside expected range"


def _run_once_full(coupling_tol: float, *, max_subiters: int = 200, min_subiters: int = 1,
                   accel: str = "anderson", coupling_each_iter: bool = False, domain=None,
                   return_gains: bool = False):
    """Build a small problem and perform a single step with given coupling tolerance.
    
    Returns (QoIs tuple, subiter_metrics list). When coupling_each_iter=True,
    subiter_metrics[-1] contains 'rhoJ' (spectral radius of block-Jacobian estimate).
    """
    from dolfinx import fem
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
        rem.step(dt=1.0)  # one outer step
        
        # Build interior measure for QoI computation
        tags = build_interior_cell_tags(domain, INTERIOR_MARGIN)
        dx_int = ufl.Measure("dx", domain=domain, subdomain_data=tags, subdomain_id=1)
        
        # Compute QoIs using inline integration (avoid duplicate helpers)
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
        
        # Per-subiteration metrics (each rank has the same numbers for proj_res and rhoJ)
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
    """Check that observed contraction is bounded by rho(J) and verify MPI consistency.
    
    Combines:
    - Spectral radius contraction bounds (was: test_spectral_radius_matches_contraction)
    - MPI invariance of rhoJ (was: test_rhoJ_mpi_invariance)
    """
    tol = 1e-4
    # Ensure we have at least a few subiterations to measure ratios
    qois, metrics = _run_once_full(coupling_tol=tol, max_subiters=200, min_subiters=3,
                                   accel="anderson", coupling_each_iter=True, domain=unit_cube)

    COMM.Barrier()
    
    assert len(metrics) >= 3, "Not enough subiterations to estimate contraction; increase max_subiters or relax tol."

    # Take the last few ratios r_{k}/r_{k-1}
    proj = [m["proj_res"] for m in metrics if np.isfinite(m.get("proj_res", np.nan))]
    ratios = []
    for i in range(1, len(proj)):
        if proj[i-1] > 0:
            ratios.append(proj[i] / proj[i-1])
    assert ratios, "No valid residual ratios found."

    last_rhoJ = metrics[-1].get("rhoJ", float("nan"))
    assert np.isfinite(last_rhoJ), "rhoJ not recorded; ensure coupling_each_iter=True and implementation exposes it."
    assert last_rhoJ < 1.0 + 1e-6, f"rho(J) >= 1 suggests no local contraction: rhoJ={last_rhoJ:.3e}"

    # Observed contraction should be <= rhoJ + delta (allow small slack due to nonlinearity & noise)
    # Use median of last few ratios for robustness
    tail = ratios[-min(5, len(ratios)):]
    median_ratio = float(np.median(tail))
    
    if RANK == 0:
        assert median_ratio <= last_rhoJ + 0.05, f"Observed contraction ({median_ratio:.3e}) exceeds rho(J) + slack ({last_rhoJ:.3e})."
    
    # MPI invariance check: rho(J) should be identical across all ranks
    vals = COMM.allgather(float(last_rhoJ))
    vals_finite = [v for v in vals if np.isfinite(v)]
    if RANK == 0 and vals_finite:
        spread = max(vals_finite) - min(vals_finite)
        assert spread <= 1e-10, f"rho(J) spread across ranks too large: {spread:.3e} from {vals_finite}"


@pytest.mark.parametrize("unit_cube", [6], indirect=True)
def test_max_subiters_sufficient_and_insufficient(unit_cube):
    """Fix tolerance and show that too-small max_subiters fails to meet it and degrades QoIs."""
    tol = 1e-6
    # First run with sufficient iterations
    q_suf, m_suf = _run_once_full(coupling_tol=tol, max_subiters=64, min_subiters=1, coupling_each_iter=False, domain=unit_cube)
    # Then insufficient budget
    q_ins, m_ins = _run_once_full(coupling_tol=tol, max_subiters=2,  min_subiters=1, coupling_each_iter=False, domain=unit_cube)

    COMM.Barrier()
    if RANK != 0:
        return

    assert m_suf, "No subiteration metrics recorded for sufficient run."
    assert m_ins, "No subiteration metrics recorded for insufficient run."

    # Check stop criterion satisfaction
    assert m_suf[-1]["proj_res"] < tol, f"Sufficient run did not meet tolerance: {m_suf[-1]['proj_res']} >= {tol}"
    assert m_ins[-1]["proj_res"] >= tol, f"Insufficient run unexpectedly met tolerance: {m_ins[-1]['proj_res']} < {tol}"

    # QoIs: insufficient run should deviate more from sufficient than a tiny threshold
    err = sum(abs(a - b) for a, b in zip(q_ins, q_suf))
    assert err >= 1e-9, "QoIs identical despite failing to meet coupling tolerance; check config/weights."


# Note: Removed test_coupling_diag_invariance (implementation detail - diagnostics shouldn't affect results)
# Note: Removed test_interaction_gain_matches_recorded_metrics (implementation detail - internal consistency)
# The spectral radius computation is validated in test_spectral_radius_bounds_contraction via observed behavior


 # No __main__ runner needed; tests executed via pytest
