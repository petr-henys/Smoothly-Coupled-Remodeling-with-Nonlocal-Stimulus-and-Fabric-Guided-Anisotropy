#!/usr/bin/env python3
"""
Nondimensionalization parameter consistency and subsolver scaling tests.

Tests exact mapping between dimensional/nondimensional constants and physics invariance.
"""

import math
import pytest
import numpy as np

from dolfinx import fem
import basix
import ufl
from mpi4py import MPI

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.utils import build_dirichlet_bcs


def _float(c):
    """Helper to extract float value from fem.Constant or numeric."""
    try:
        return float(c)
    except TypeError:
        return float(c.value)


def test_constant_types_and_basic_ranges(unit_cube, facet_tags):
    """All nondimensional constants exist with correct types and basic ranges."""
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=False)

    # Types: all ND constants are fem.Constant
    for name in (
        "E0_nd", "dt_nd", "psi_ref_nd",
        "beta_par_nd", "beta_perp_nd",
        "cS_c", "tauS_c", "kappaS_c", "rS_gain_c",
        "cA_c", "tauA_c", "ell_c",
        "xi_aniso_c", "nu_c", "n_power_c",
    ):
        val = getattr(cfg, name)
        assert isinstance(val, fem.Constant), f"{name} is not fem.Constant"

    # Baseline positivity / sanity (not over-constraining)
    assert _float(cfg.E0_nd) == pytest.approx(1.0)
    assert _float(cfg.dt_nd) > 0
    assert _float(cfg.beta_par_nd) > 0
    assert _float(cfg.beta_perp_nd) > 0
    assert _float(cfg.cS_c) > 0
    assert _float(cfg.tauS_c) == pytest.approx(1.0)
    assert _float(cfg.kappaS_c) >= 0
    assert _float(cfg.rS_gain_c) >= 0
    assert _float(cfg.cA_c) > 0
    assert _float(cfg.tauA_c) > 0
    assert _float(cfg.ell_c) > 0
    # Dimensionless groups
    assert _float(cfg.nu_c) == pytest.approx(cfg.nu)
    assert _float(cfg.n_power_c) == pytest.approx(cfg.n_power)
    assert _float(cfg.xi_aniso_c) == pytest.approx(cfg.xi_aniso)


def test_dimensional_to_nondimensional_mapping_exact(unit_cube, facet_tags):
    """Verify exact formulas used for nondimensionalization."""
    # Choose distinct, nontrivial dimensional values
    cfg = Config(
        domain=unit_cube,
        facet_tags=facet_tags,
        L_c=2.5,
        u_c=3.0e-3,
        E0_dim=7.0e9,
        psi_ref_dim=123.4,
        # transport/diffusion
        beta_par_dim=4.2e-6,
        beta_perp_dim=7.3e-6,
        # stimulus S
        cS_dim=12.5,
        tauS_dim=2.5,  # => t_c = 0.4
        kappaS_dim=8.0e-5,
        rS_dim=6.0e-5,
        # orientation A
        cA_dim=3.3,
        tauA_dim=0.75,
        ell_dim=0.42,
        verbose=False,
    )

    # Base scales
    strain_scale = cfg.u_c / cfg.L_c
    sigma_c = cfg.E0_dim * strain_scale
    psi_c = cfg.E0_dim * (strain_scale ** 2)
    t_c = 1.0 / cfg.tauS_dim

    # Expected ND constants (current implementation: cS_c scaled by 1/t_c)
    expect = {
        "E0_nd": 1.0,
        "psi_ref_nd": cfg.psi_ref_dim / psi_c,
        "beta_par_nd": cfg.beta_par_dim * t_c / (cfg.L_c ** 2),
        "beta_perp_nd": cfg.beta_perp_dim * t_c / (cfg.L_c ** 2),
        "cS_c": cfg.cS_dim / t_c,  # Updated: now scaled by 1/t_c
        "tauS_c": 1.0,
        "kappaS_c": cfg.kappaS_dim * t_c / (cfg.L_c ** 2),
        "rS_gain_c": cfg.rS_dim * psi_c * t_c,
        "cA_c": cfg.cA_dim,
        "tauA_c": cfg.tauA_dim * t_c,
        "ell_c": cfg.ell_dim / cfg.L_c,
    }

    for name, val in expect.items():
        got = _float(getattr(cfg, name))
        assert math.isfinite(got), f"{name} not finite"
        assert got == pytest.approx(val, rel=1e-14, abs=1e-14), f"Mismatch in {name}"

    # Also check rho bounds nondimensionalization
    assert cfg.rho_min_nd == pytest.approx(cfg.rho_min_dim / cfg.rho_c)
    assert cfg.rho_max_nd == pytest.approx(cfg.rho_max_dim / cfg.rho_c)
    assert cfg.rho_min_nd < cfg.rho_max_nd


def test_roundtrip_reconstruct_dimensionals(unit_cube, facet_tags):
    """Invert the nondimensionalization to recover original dimensionals."""
    cfg = Config(
        domain=unit_cube, facet_tags=facet_tags,
        L_c=1.7, u_c=2.1e-3, E0_dim=5.6e9,
        psi_ref_dim=321.0,
        beta_par_dim=1.1e-5, beta_perp_dim=2.2e-5,
        cS_dim=9.0, tauS_dim=1.6, kappaS_dim=4.0e-5, rS_dim=7.5e-5,
        cA_dim=1.2, tauA_dim=2.5, ell_dim=0.37,
        verbose=False,
    )

    t_c = 1.0 / cfg.tauS_dim
    psi_c = cfg.psi_c

    # Reconstruct dimensionals from ND constants (current implementation)
    beta_par_dim_rt = _float(cfg.beta_par_nd) * (cfg.L_c ** 2) / t_c
    beta_perp_dim_rt = _float(cfg.beta_perp_nd) * (cfg.L_c ** 2) / t_c
    cS_dim_rt = _float(cfg.cS_c) * t_c  # Updated: cS_c now scaled by 1/t_c, so multiply back
    kappaS_dim_rt = _float(cfg.kappaS_c) * (cfg.L_c ** 2) / t_c
    rS_dim_rt = _float(cfg.rS_gain_c) / (psi_c * t_c)
    cA_dim_rt = _float(cfg.cA_c)
    tauA_dim_rt = _float(cfg.tauA_c) / t_c
    ell_dim_rt = _float(cfg.ell_c) * cfg.L_c
    psi_ref_dim_rt = _float(cfg.psi_ref_nd) * psi_c

    assert beta_par_dim_rt == pytest.approx(cfg.beta_par_dim, rel=1e-14, abs=1e-14)
    assert beta_perp_dim_rt == pytest.approx(cfg.beta_perp_dim, rel=1e-14, abs=1e-14)
    assert cS_dim_rt == pytest.approx(cfg.cS_dim, rel=1e-14, abs=1e-14)
    assert kappaS_dim_rt == pytest.approx(cfg.kappaS_dim, rel=1e-14, abs=1e-14)
    assert rS_dim_rt == pytest.approx(cfg.rS_dim, rel=1e-14, abs=1e-14)
    assert cA_dim_rt == pytest.approx(cfg.cA_dim, rel=1e-14, abs=1e-14)
    assert tauA_dim_rt == pytest.approx(cfg.tauA_dim, rel=1e-14, abs=1e-14)
    assert ell_dim_rt == pytest.approx(cfg.ell_dim, rel=1e-14, abs=1e-14)
    assert psi_ref_dim_rt == pytest.approx(cfg.psi_ref_dim, rel=1e-14, abs=1e-14)


def test_scaling_with_Lc(unit_cube, facet_tags):
    """Verify expected scaling laws when changing L_c only."""
    base = Config(domain=unit_cube, facet_tags=facet_tags, L_c=1.0, tauS_dim=1.0, verbose=False)
    scaled = Config(domain=unit_cube, facet_tags=facet_tags, L_c=2.0, tauS_dim=1.0, verbose=False)

    # Quantities ~ 1/L_c^2
    for name in ("beta_par_nd", "beta_perp_nd", "kappaS_c"):
        r = _float(getattr(scaled, name)) / _float(getattr(base, name))
        assert r == pytest.approx((base.L_c / scaled.L_c) ** 2)

    # Quantities ~ 1/L_c
    r_ell = _float(scaled.ell_c) / _float(base.ell_c)
    assert r_ell == pytest.approx(base.L_c / scaled.L_c)

    # psi_ref_nd ~ 1/psi_c ~ L_c^2 (since psi_c ~ E0*(u/L)^2)
    r_psiref = _float(scaled.psi_ref_nd) / _float(base.psi_ref_nd)
    assert r_psiref == pytest.approx((scaled.L_c / base.L_c) ** 2)


def test_scaling_with_tc(unit_cube, facet_tags):
    """Verify expected scaling laws when changing t_c (via tauS_dim)."""
    # t_c = 1 / tauS_dim
    base = Config(domain=unit_cube, facet_tags=facet_tags, tauS_dim=1.0, verbose=False)
    slower = Config(domain=unit_cube, facet_tags=facet_tags, tauS_dim=2.0, verbose=False)  # t_c halves

    t_ratio = (1.0 / slower.tauS_dim) / (1.0 / base.tauS_dim)  # t_c(slower)/t_c(base) = 0.5

    # cS_c now scales with 1/t_c (updated implementation)
    r_cS = _float(slower.cS_c) / _float(base.cS_c)
    assert r_cS == pytest.approx(1.0 / t_ratio)  # cS_c ~ 1/t_c, so ratio is inverted
    
    # cA_c is dimensional constant (no t_c scaling)
    r_cA = _float(slower.cA_c) / _float(base.cA_c)
    assert r_cA == pytest.approx(1.0)  # unchanged with t_c

    # Quantities ~ t_c
    for name in ("tauA_c", "rS_gain_c"):
        r = _float(getattr(slower, name)) / _float(getattr(base, name))
        assert r == pytest.approx(t_ratio)

    # Quantities ~ t_c / L_c^2 (L_c same here)
    for name in ("beta_par_nd", "beta_perp_nd", "kappaS_c"):
        r = _float(getattr(slower, name)) / _float(getattr(base, name))
        assert r == pytest.approx(t_ratio)


def test_property_relations_consistent(unit_cube, facet_tags):
    """Basic relationships among derived scales hold: psi_c/sigma_c = strain_scale."""
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, L_c=2.3, u_c=1.1e-3, E0_dim=9.1e9, verbose=False)
    assert cfg.psi_c / cfg.sigma_c == pytest.approx(cfg.strain_scale)
    assert cfg.sigma_c == pytest.approx(cfg.E0_dim * cfg.strain_scale)
    assert cfg.psi_c == pytest.approx(cfg.E0_dim * cfg.strain_scale ** 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def _constant_scalar(Q, value: float) -> fem.Function:
    f = fem.Function(Q)
    f.x.array[:] = float(value)
    f.x.scatter_forward()
    return f


def _constant_tensor(T, val_matrix: np.ndarray) -> fem.Function:
    A = fem.Function(T)
    gdim = T.mesh.geometry.dim
    assert val_matrix.shape == (gdim, gdim)
    flat = val_matrix.astype(np.float64).flatten()
    # Interpolate constant tensor
    A.interpolate(lambda x: np.tile(flat[:, None], (1, x.shape[1])))
    A.x.scatter_forward()
    return A


def _build_spaces(domain):
    gdim = domain.geometry.dim
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim, gdim))
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1)
    T = fem.functionspace(domain, P1_ten)
    return V, Q, T


@pytest.mark.parametrize("unit_cube", [4], indirect=True)
def test_mechanics_energy_invariant_under_Lu_scale(unit_cube, facet_tags):
    """Changing L_c and u_c by the same factor keeps physical energy invariant.

    Keep E0_dim, nu, n_power fixed; choose identical physical traction
    applied via dimensionless t = t_dim/sigma_c for each configuration.
    """
    domain = unit_cube
    V, Q, T = _build_spaces(domain)

    # Common fields: rho=1, A=I/d
    rho = _constant_scalar(Q, 1.0)
    gdim = domain.geometry.dim
    A_iso = _constant_tensor(T, np.eye(gdim) / gdim)
    uA = fem.Function(V)  # solution storage

    # Fixed Dirichlet on face id=1; Neumann on face id=2
    bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

    # Config A
    cfgA = Config(domain=domain, facet_tags=facet_tags,
                  L_c=1.0, u_c=1.0e-3, E0_dim=6.0e9, verbose=False)
    # Config B: scale L and u equally; psi_c and sigma_c unchanged
    cfgB = Config(domain=domain, facet_tags=facet_tags,
                  L_c=2.0, u_c=2.0e-3, E0_dim=6.0e9, verbose=False)

    # Physical traction (Pa); apply as dimensionless using each cfg.sigma_c
    t_dim = 2.5e6

    def solve_and_energy(cfg):
        tvec = np.zeros(gdim, dtype=np.float64)
        tvec[0] = -t_dim / cfg.sigma_c
        traction = (fem.Constant(cfg.domain, tvec), 2)
        mech = MechanicsSolver(uA, rho, A_iso, cfg, bcs, [traction])
        mech.setup()
        its, reason = mech.solve()
        assert reason >= 0, "Mechanics KSP did not converge"
        return mech.average_strain_energy()

    E_A = solve_and_energy(cfgA)
    E_B = solve_and_energy(cfgB)

    assert E_A == pytest.approx(E_B, rel=5e-5, abs=5e-5)


@pytest.mark.parametrize("unit_cube", [4], indirect=True)
def test_mechanics_energy_scales_with_psi_c_for_equal_nd_load(unit_cube, facet_tags):
    """With identical ND traction, physical energy scales with psi_c.

    Keep ND load fixed (same numeric traction vector) and compare energies
    for configs with different E0_dim (thus psi_c).
    """
    domain = unit_cube
    V, Q, T = _build_spaces(domain)
    rho = _constant_scalar(Q, 1.0)
    gdim = domain.geometry.dim
    A_iso = _constant_tensor(T, np.eye(gdim) / gdim)
    uA = fem.Function(V)
    bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

    # Two configs differing only by E0_dim (psi_c scales proportionally)
    cfg1 = Config(domain=domain, facet_tags=facet_tags,
                  E0_dim=5.0e9, verbose=False)
    cfg2 = Config(domain=domain, facet_tags=facet_tags,
                  E0_dim=12.5e9, verbose=False)

    # Same numeric ND traction for both
    t_nd = -0.2

    def energy(cfg):
        tvec = np.zeros(gdim, dtype=np.float64)
        tvec[0] = t_nd
        mech = MechanicsSolver(uA, rho, A_iso, cfg, bcs, [(fem.Constant(cfg.domain, tvec), 2)])
        mech.setup()
        mech.assemble_rhs()
        its, reason = mech.solve()
        assert reason >= 0
        return mech.average_strain_energy()

    E1 = energy(cfg1)
    E2 = energy(cfg2)
    assert (E2 / E1) == pytest.approx(cfg2.psi_c / cfg1.psi_c, rel=5e-5)


@pytest.mark.parametrize("unit_cube", [4], indirect=True)
def test_mechanics_energy_matches_direct_nd_assembly(unit_cube, facet_tags):
    """average_strain_energy equals ND assembly times psi_c (no double scaling)."""
    domain = unit_cube
    V, Q, T = _build_spaces(domain)
    rho = _constant_scalar(Q, 1.0)
    gdim = domain.geometry.dim
    A_iso = _constant_tensor(T, np.eye(gdim) / gdim)
    u = fem.Function(V)
    bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)

    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
    tvec = np.zeros(gdim, dtype=np.float64); tvec[0] = -0.15
    mech = MechanicsSolver(u, rho, A_iso, cfg, bcs, [(fem.Constant(domain, tvec), 2)])
    mech.setup()
    mech.assemble_rhs()
    its, reason = mech.solve()
    assert reason >= 0

    # Direct ND assembly -> physical units
    strain_energy_nd_expr = 0.5 * ufl.inner(mech.sigma(u, rho, A_iso), mech.eps(u))
    psi_local_nd = fem.assemble_scalar(fem.form(strain_energy_nd_expr * cfg.dx))
    psi_nd = mech.comm.allreduce(psi_local_nd, op=MPI.SUM)
    psi_dim_direct = float((psi_nd / max(mech.total_vol, 1e-300)) * cfg.psi_c)

    assert mech.average_strain_energy() == pytest.approx(psi_dim_direct, rel=5e-6, abs=5e-8)


@pytest.mark.parametrize("unit_cube", [4], indirect=True)
def test_stimulus_invariance_under_E0_scaling_compensated(unit_cube, facet_tags):
    """Stimulus solution invariant when E0 scaling is compensated.

    Scale E0_dim by k, set psi_ref_dim -> k * psi_ref_dim and rS_dim -> rS_dim / k,
    leaving all other dimensional parameters the same. The ND problem is identical.
    """
    k = 2.3
    domain = unit_cube
    V, Q, T = _build_spaces(domain)

    # Base parameters
    base_kwargs = dict(L_c=1.2, u_c=1.1e-3, E0_dim=7.0e9, tauS_dim=1.5,
                       cS_dim=10.0, kappaS_dim=8.0e-5, rS_dim=6.0e-5, psi_ref_dim=250.0)
    cfgA = Config(domain=domain, facet_tags=facet_tags, verbose=False, **base_kwargs)

    scaled_kwargs = base_kwargs.copy()
    scaled_kwargs.update(E0_dim=k * base_kwargs["E0_dim"],
                         rS_dim=base_kwargs["rS_dim"] / k,
                         psi_ref_dim=k * base_kwargs["psi_ref_dim"])
    cfgB = Config(domain=domain, facet_tags=facet_tags, verbose=False, **scaled_kwargs)

    # Same dt_nd (defaults to 1); psi_expr dimensionless constant
    psi_nd = fem.Constant(domain, float(0.37))
    S_old_A = _constant_scalar(Q, 0.1)
    S_old_B = _constant_scalar(Q, 0.1)

    S_A = fem.Function(Q)
    S_B = fem.Function(Q)

    solA = StimulusSolver(S_A, S_old_A, cfgA)
    solA.setup()
    solA.assemble_rhs(psi_nd)
    itsA, reasonA = solA.solve()
    assert reasonA >= 0

    solB = StimulusSolver(S_B, S_old_B, cfgB)
    solB.setup()
    solB.assemble_rhs(psi_nd)
    itsB, reasonB = solB.solve()
    assert reasonB >= 0

    assert np.allclose(S_A.x.array, S_B.x.array, rtol=5e-6, atol=5e-8)


@pytest.mark.parametrize("unit_cube", [4], indirect=True)
def test_density_invariance_under_Lc_tc_scaling(unit_cube, facet_tags):
    """Density solution invariant when t_c/L_c^2 and dt_nd are held constant.

    Scale L_c -> a L_c and tauS_dim -> tauS_dim / a^2 so that t_c -> a^2 t_c and
    t_c/L_c^2 stays unchanged. With dt_nd constant and S=0, the assembled system
    is identical.
    """
    a = 1.7
    domain = unit_cube
    V, Q, T = _build_spaces(domain)
    gdim = domain.geometry.dim
    A_iso = _constant_tensor(T, np.eye(gdim) / gdim)
    S_zero = _constant_scalar(Q, 0.0)
    rho_old_A = _constant_scalar(Q, 0.55)
    rho_old_B = _constant_scalar(Q, 0.55)
    rho_A = fem.Function(Q)
    rho_B = fem.Function(Q)

    cfgA = Config(domain=domain, facet_tags=facet_tags,
                  L_c=1.0, tauS_dim=1.0, verbose=False)
    cfgB = Config(domain=domain, facet_tags=facet_tags,
                  L_c=a * 1.0, tauS_dim=cfgA.tauS_dim / (a * a), verbose=False)

    # dt_nd defaults to 1 in both, ensuring equality; set S field into solver
    densA = DensitySolver(rho_A, rho_old_A, A_iso, S_zero, cfgA)
    densA.setup()
    densA.assemble_rhs()
    itsA, reasonA = densA.solve()
    assert reasonA >= 0

    densB = DensitySolver(rho_B, rho_old_B, A_iso, S_zero, cfgB)
    densB.setup()
    densB.assemble_rhs()
    itsB, reasonB = densB.solve()
    assert reasonB >= 0

    assert np.allclose(rho_A.x.array, rho_B.x.array, rtol=5e-6, atol=5e-8)


@pytest.mark.parametrize("unit_cube", [4], indirect=True)
def test_direction_invariance_under_joint_scaling(unit_cube, facet_tags):
    """Direction solver invariant under joint scaling preserving ND groups.

    Choose scaling L_c -> a L_c and tauS_dim -> tauS_dim / a^2 (so t_c -> a^2 t_c),
    with cA_dim -> a^2 cA_dim, tauA_dim -> tauA_dim / a^2, ell_dim -> a * ell_dim.
    Keep dt_nd equal (default 1). With u=0 and A_old=I/d, the LHS and RHS match.
    """
    a = 1.6
    domain = unit_cube
    V, Q, T = _build_spaces(domain)
    gdim = domain.geometry.dim
    A_old_A = _constant_tensor(T, np.eye(gdim) / gdim)
    A_old_B = _constant_tensor(T, np.eye(gdim) / gdim)
    A_sol_A = fem.Function(T)
    A_sol_B = fem.Function(T)

    # Mechanics solver only used to compute eps(u) inside update_rhs; use u=0
    rho = _constant_scalar(Q, 1.0)
    A_iso = _constant_tensor(T, np.eye(gdim) / gdim)
    bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    u_zero = fem.Function(V)

    cfgA = Config(domain=domain, facet_tags=facet_tags,
                  L_c=1.0, tauS_dim=1.0, cA_dim=1.2, tauA_dim=0.75, ell_dim=0.35,
                  verbose=False)

    cfgB = Config(domain=domain, facet_tags=facet_tags,
                  L_c=a * 1.0,
                  tauS_dim=cfgA.tauS_dim / (a * a),
                  cA_dim=a * a * cfgA.cA_dim,
                  tauA_dim=cfgA.tauA_dim / (a * a),
                  ell_dim=a * cfgA.ell_dim,
                  verbose=False)

    mechA = MechanicsSolver(u_zero, rho, A_iso, cfgA, bcs, [])
    mechA.setup()
    dirA = DirectionSolver(A_sol_A, A_old_A, cfgA)
    dirA.setup(); dirA.assemble_rhs(mechA.get_strain_tensor(u_zero))
    itsA, reasonA = dirA.solve()
    assert reasonA >= 0

    mechB = MechanicsSolver(u_zero, rho, A_iso, cfgB, bcs, [])
    mechB.setup()
    dirB = DirectionSolver(A_sol_B, A_old_B, cfgB)
    dirB.setup(); dirB.assemble_rhs(mechB.get_strain_tensor(u_zero))
    itsB, reasonB = dirB.solve()
    assert reasonB >= 0

    assert np.allclose(A_sol_A.x.array, A_sol_B.x.array, rtol=5e-6, atol=5e-8)
