#!/usr/bin/env python3
"""
Tests for configuration validation and bounds checking.
"""

import pytest
from simulation.config import Config
from simulation.params import MaterialParams, SolverParams, DensityParams, NumericsParams


class TestConfigValidation:
    """Test Config parameter validation and bounds checking."""

    def test_config_requires_domain(self, facet_tags):
        """Config must have domain parameter."""
        with pytest.raises((ValueError, TypeError)):
            Config(
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            )  # Missing domain

    def test_config_requires_domain_not_none(self, facet_tags):
        """Config domain cannot be None."""
        with pytest.raises(ValueError, match="[Dd]omain"):
            Config(
                domain=None,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            )

    def test_config_accepts_valid_domain(self, unit_cube, facet_tags):
        """Config should accept valid mesh."""
        cfg = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        )
        assert cfg.domain is not None
        assert cfg.domain == unit_cube

    def test_config_rejects_negative_timestep(self, unit_cube, facet_tags):
        """set_dt should reject non-positive timestep."""
        cfg = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        )

        # Zero timestep
        with pytest.raises((ValueError, ZeroDivisionError)):
            cfg.set_dt(0.0)

        # Negative timestep
        with pytest.raises((ValueError, RuntimeError)):
            cfg.set_dt(-1.0)

    def test_config_poisson_ratio_in_valid_range(self, unit_cube, facet_tags):
        """Poisson ratio must be in physically valid range (-1, 0.5)."""
        # Test boundary values
        with pytest.raises((ValueError, RuntimeError)):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, nu0=0.6),
            )  # Too high

        with pytest.raises((ValueError, RuntimeError)):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, nu0=-1.5),
            )  # Too low

        # Valid values should work
        cfg1 = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, nu0=0.3),
        )
        assert cfg1.material.nu0 == 0.3

        cfg2 = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, nu0=0.0),
        )
        assert cfg2.material.nu0 == 0.0

    def test_config_positive_modulus(self, unit_cube, facet_tags):
        """Young's modulus must be positive."""
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, E0=-1000.0),
            )

        # Positive value should work
        cfg = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, E0=1000.0),
        )
        assert cfg.material.E0 == 1000.0

    def test_config_solver_type_validation(self, unit_cube, facet_tags):
        """KSP and PC types should be valid."""
        # Valid solvers
        cfg1 = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            solver=SolverParams(ksp_type="cg", pc_type="jacobi"),
        )
        assert cfg1.solver.ksp_type == "cg"

        cfg2 = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            solver=SolverParams(ksp_type="gmres", pc_type="ilu"),
        )
        assert cfg2.solver.ksp_type == "gmres"

    def test_config_accel_type_validation(self, unit_cube, facet_tags):
        """Acceleration type must be valid choice ('anderson' or 'picard')."""
        for accel in ["anderson", "picard"]:
            cfg = Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
                solver=SolverParams(accel_type=accel),
            )
            assert cfg.solver.accel_type == accel

        # 'none' is not accepted at config time
        with pytest.raises((ValueError, RuntimeError)):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
                solver=SolverParams(accel_type="none"),
            )

    def test_config_tolerance_values_positive(self, unit_cube, facet_tags):
        """Solver tolerances must be positive."""
        cfg = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
        )

        assert cfg.solver.ksp_rtol > 0
        assert cfg.solver.ksp_atol > 0
        assert cfg.solver.coupling_tol > 0
        assert cfg.numerics.smooth_eps > 0

    def test_config_iteration_limits_sensible(self, unit_cube, facet_tags):
        """Iteration limits should be positive integers."""
        cfg = Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            solver=SolverParams(max_subiters=100, min_subiters=1, ksp_max_it=500),
        )

        assert cfg.solver.max_subiters > 0
        assert cfg.solver.min_subiters > 0
        assert cfg.solver.min_subiters <= cfg.solver.max_subiters
        assert cfg.solver.ksp_max_it > 0

    def test_config_material_transition_parameters(self, unit_cube, facet_tags):
        """Transition parameters must be well-ordered and positive."""
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=-1.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
            )
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=1.2, rho_cort_min=1.2),
            )

    def test_numerics_params_validation(self, unit_cube, facet_tags):
        """NumericsParams validation."""
        # Invalid quadrature degree
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                numerics=NumericsParams(quadrature_degree=0),
            )

        # Invalid smooth_eps
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                numerics=NumericsParams(smooth_eps=0.0),
            )

    def test_geometry_params_validation(self, unit_cube, facet_tags):
        """GeometryParams validation."""
        from simulation.params import GeometryParams

        # Invalid fix_tag
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                geometry=GeometryParams(fix_tag=-1),
            )

        # Invalid load_tag
        with pytest.raises(ValueError):
            Config(
                domain=unit_cube,
                facet_tags=facet_tags,
                geometry=GeometryParams(load_tag=-1),
            )
