#!/usr/bin/env python3
"""
Focused tests for nondimensional parameter consistency.

Covers:
- Exact mapping: dimensional -> nondimensional constants
- Round-trip: nondimensional constants -> recover dimensionals
- Scaling laws with L_c (length) and t_c (time)
- Basic type/positivity checks and property relations
"""

import math
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")

from dolfinx import fem

from simulation.config import Config


def _float(c):
    """Helper to extract float value from fem.Constant or numeric."""
    try:
        return float(c)
    except TypeError:
        return float(c.value)


@pytest.mark.unit
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


@pytest.mark.unit
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

    # Expected ND constants
    expect = {
        "E0_nd": 1.0,
        "psi_ref_nd": cfg.psi_ref_dim / psi_c,
        "beta_par_nd": cfg.beta_par_dim * t_c / (cfg.L_c ** 2),
        "beta_perp_nd": cfg.beta_perp_dim * t_c / (cfg.L_c ** 2),
        "cS_c": cfg.cS_dim / t_c,
        "tauS_c": 1.0,
        "kappaS_c": cfg.kappaS_dim * t_c / (cfg.L_c ** 2),
        "rS_gain_c": cfg.rS_dim * psi_c * t_c,
        "cA_c": cfg.cA_dim / t_c,
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


@pytest.mark.unit
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

    # Reconstruct dimensionals from ND constants
    beta_par_dim_rt = _float(cfg.beta_par_nd) * (cfg.L_c ** 2) / t_c
    beta_perp_dim_rt = _float(cfg.beta_perp_nd) * (cfg.L_c ** 2) / t_c
    cS_dim_rt = _float(cfg.cS_c) * t_c
    kappaS_dim_rt = _float(cfg.kappaS_c) * (cfg.L_c ** 2) / t_c
    rS_dim_rt = _float(cfg.rS_gain_c) / (psi_c * t_c)
    cA_dim_rt = _float(cfg.cA_c) * t_c
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_scaling_with_tc(unit_cube, facet_tags):
    """Verify expected scaling laws when changing t_c (via tauS_dim)."""
    # t_c = 1 / tauS_dim
    base = Config(domain=unit_cube, facet_tags=facet_tags, tauS_dim=1.0, verbose=False)
    slower = Config(domain=unit_cube, facet_tags=facet_tags, tauS_dim=2.0, verbose=False)  # t_c halves

    t_ratio = (1.0 / slower.tauS_dim) / (1.0 / base.tauS_dim)  # t_c(slower)/t_c(base) = 0.5

    # Quantities ~ 1/t_c
    for name in ("cS_c", "cA_c"):
        r = _float(getattr(slower, name)) / _float(getattr(base, name))
        assert r == pytest.approx(1.0 / t_ratio)

    # Quantities ~ t_c
    for name in ("tauA_c", "rS_gain_c"):
        r = _float(getattr(slower, name)) / _float(getattr(base, name))
        assert r == pytest.approx(t_ratio)

    # Quantities ~ t_c / L_c^2 (L_c same here)
    for name in ("beta_par_nd", "beta_perp_nd", "kappaS_c"):
        r = _float(getattr(slower, name)) / _float(getattr(base, name))
        assert r == pytest.approx(t_ratio)


@pytest.mark.unit
def test_property_relations_consistent(unit_cube, facet_tags):
    """Basic relationships among derived scales hold: psi_c/sigma_c = strain_scale."""
    cfg = Config(domain=unit_cube, facet_tags=facet_tags, L_c=2.3, u_c=1.1e-3, E0_dim=9.1e9, verbose=False)
    assert cfg.psi_c / cfg.sigma_c == pytest.approx(cfg.strain_scale)
    assert cfg.sigma_c == pytest.approx(cfg.E0_dim * cfg.strain_scale)
    assert cfg.psi_c == pytest.approx(cfg.E0_dim * cfg.strain_scale ** 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
