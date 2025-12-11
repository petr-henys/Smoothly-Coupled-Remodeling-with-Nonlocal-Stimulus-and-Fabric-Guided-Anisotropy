"""
Tests for simulation.calibrate_mechanostat module.

Tests the calibration utilities for psi_ref, k_rho, and Helmholtz length.
Uses unit cube with simple traction loading for reproducible results.
"""

import pytest
import numpy as np
from mpi4py import MPI

from dolfinx import mesh, fem
from dolfinx.fem import functionspace, Function
import basix.ufl

from simulation.config import Config
from simulation.utils import build_facetag, assign


# =============================================================================
# Simple Loader stub for testing (no femur geometry needed)
# =============================================================================

class SimpleLoaderStub:
    """
    Minimal loader stub for testing calibration without femur mesh.
    
    Applies constant traction on tag=2 (x=1 face) and zero on tag=1 (x=0 face).
    """
    
    def __init__(self, domain: mesh.Mesh, facet_tags: mesh.MeshTags):
        self.mesh = domain
        self.comm = domain.comm
        self.load_tag = 2  # x=1 face
        self.cut_tag = 1   # x=0 face (fixed, but we use cut for equilibrium)
        
        gdim = domain.geometry.dim
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
        V_traction = functionspace(domain, P1_vec)
        
        # Traction fields (will be set by loading cases)
        self.traction = Function(V_traction, name="traction")
        self.traction_cut = Function(V_traction, name="traction_cut")
        
        self._cached_cases = {}
        
    def precompute_loading_cases(self, loading_cases):
        """Store traction arrays for each loading case."""
        for case in loading_cases:
            # Apply traction in x-direction proportional to magnitude
            traction_val = np.zeros(3)
            if case.hip is not None:
                traction_val[0] = case.hip.magnitude * 1e-3  # Scale to reasonable stress
            
            # Store copy
            arr = np.zeros_like(self.traction.x.array)
            n_owned = self.traction.function_space.dofmap.index_map.size_local * 3
            for i in range(n_owned // 3):
                arr[3*i:3*i+3] = traction_val
            
            # Equilibrating reaction on cut (opposite direction)
            arr_cut = np.zeros_like(self.traction_cut.x.array)
            for i in range(n_owned // 3):
                arr_cut[3*i:3*i+3] = -traction_val  # Opposite for equilibrium
            
            self._cached_cases[case.name] = {
                "traction": arr.copy(),
                "traction_cut": arr_cut.copy(),
            }
    
    def set_loading_case(self, name: str):
        """Set traction fields from cached values."""
        cached = self._cached_cases[name]
        self.traction.x.array[:] = cached["traction"]
        self.traction_cut.x.array[:] = cached["traction_cut"]
        self.traction.x.scatter_forward()
        self.traction_cut.x.scatter_forward()


class SimpleLoadingCase:
    """Simple loading case for testing."""
    
    def __init__(self, name: str, weight: float = 1.0, magnitude: float = 1000.0):
        self.name = name
        self.weight = weight
        self.hip = type('Hip', (), {'magnitude': magnitude})()
        self.muscles = []


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def unit_cube_mesh():
    """Create unit cube mesh."""
    return mesh.create_unit_cube(
        MPI.COMM_WORLD, 4, 4, 4,
        ghost_mode=mesh.GhostMode.shared_facet
    )


@pytest.fixture
def simple_config(unit_cube_mesh):
    """Create config with unit cube."""
    from simulation.utils import build_facetag
    facet_tags = build_facetag(unit_cube_mesh)
    return Config(domain=unit_cube_mesh, facet_tags=facet_tags)


@pytest.fixture
def simple_loader(unit_cube_mesh):
    """Create simple loader stub."""
    facet_tags = build_facetag(unit_cube_mesh)
    return SimpleLoaderStub(unit_cube_mesh, facet_tags)


@pytest.fixture
def simple_loading_cases():
    """Create simple loading cases."""
    return [
        SimpleLoadingCase("case1", weight=1.0, magnitude=1000.0),
        SimpleLoadingCase("case2", weight=1.0, magnitude=1500.0),
    ]


# =============================================================================
# Tests for auto_helmholtz_length
# =============================================================================

class TestAutoHelmholtzLength:
    """Tests for Helmholtz length computation from mesh."""
    
    def test_returns_dict_with_required_keys(self, simple_config):
        """auto_helmholtz_length returns dict with h_min, helmholtz_L, factor."""
        from calibrate_mechanostat import auto_helmholtz_length
        
        result = auto_helmholtz_length(simple_config)
        
        assert "h_min" in result
        assert "helmholtz_L" in result
        assert "factor" in result
    
    def test_h_min_is_positive(self, simple_config):
        """h_min should be positive for non-empty mesh."""
        from calibrate_mechanostat import auto_helmholtz_length
        
        result = auto_helmholtz_length(simple_config)
        
        assert result["h_min"] > 0
    
    def test_helmholtz_L_equals_factor_times_h_min(self, simple_config):
        """L_h = factor * h_min."""
        from calibrate_mechanostat import auto_helmholtz_length
        
        factor = 3.0
        result = auto_helmholtz_length(simple_config, factor=factor)
        
        expected_L = factor * result["h_min"]
        assert np.isclose(result["helmholtz_L"], expected_L, rtol=1e-10)
    
    def test_uses_config_factor_if_not_provided(self, simple_config):
        """Uses cfg.helmholtz_factor if factor arg not provided."""
        from calibrate_mechanostat import auto_helmholtz_length
        
        simple_config.helmholtz_factor = 5.0
        result = auto_helmholtz_length(simple_config)
        
        assert result["factor"] == 5.0
    
    def test_h_min_consistent_across_ranks(self, simple_config):
        """h_min should be same on all MPI ranks."""
        from calibrate_mechanostat import auto_helmholtz_length
        
        result = auto_helmholtz_length(simple_config)
        h_min = result["h_min"]
        
        comm = simple_config.domain.comm
        all_h_min = comm.allgather(h_min)
        
        # All ranks should have same h_min
        assert all(np.isclose(h, h_min, rtol=1e-14) for h in all_h_min)


# =============================================================================
# Tests for compute_k_rho_from_relaxation
# =============================================================================

class TestComputeKRhoFromRelaxation:
    """Tests for k_rho computation from relaxation criterion."""
    
    def test_invalid_epsilon_raises(self, simple_config):
        """epsilon must be in (0, 1)."""
        from calibrate_mechanostat import compute_k_rho_from_relaxation
        
        # Create dummy S_linear field
        V = functionspace(simple_config.domain, ("DG", 0))
        S_linear = Function(V)
        S_linear.x.array[:] = 0.1
        S_linear.x.scatter_forward()
        
        with pytest.raises(ValueError, match="epsilon"):
            compute_k_rho_from_relaxation(simple_config, S_linear, epsilon=0.0)
        
        with pytest.raises(ValueError, match="epsilon"):
            compute_k_rho_from_relaxation(simple_config, S_linear, epsilon=1.0)
        
        with pytest.raises(ValueError, match="epsilon"):
            compute_k_rho_from_relaxation(simple_config, S_linear, epsilon=-0.1)
    
    def test_invalid_T_total_raises(self, simple_config):
        """T_total must be positive."""
        from calibrate_mechanostat import compute_k_rho_from_relaxation
        
        V = functionspace(simple_config.domain, ("DG", 0))
        S_linear = Function(V)
        S_linear.x.array[:] = 0.1
        S_linear.x.scatter_forward()
        
        with pytest.raises(ValueError, match="T_total"):
            compute_k_rho_from_relaxation(simple_config, S_linear, T_total=0.0)
        
        with pytest.raises(ValueError, match="T_total"):
            compute_k_rho_from_relaxation(simple_config, S_linear, T_total=-100.0)
    
    def test_returns_positive_k_rho(self, simple_config):
        """k_rho should be positive for positive |S_linear|."""
        from calibrate_mechanostat import compute_k_rho_from_relaxation
        
        V = functionspace(simple_config.domain, ("DG", 0))
        S_linear = Function(V)
        S_linear.x.array[:] = 0.5  # 50% deviation from reference
        S_linear.x.scatter_forward()
        
        k_rho = compute_k_rho_from_relaxation(
            simple_config, S_linear, T_total=1000.0, epsilon=0.1
        )
        
        assert k_rho > 0
    
    def test_k_rho_scales_with_T_total(self, simple_config):
        """k_rho should be inversely proportional to T_total."""
        from calibrate_mechanostat import compute_k_rho_from_relaxation
        
        V = functionspace(simple_config.domain, ("DG", 0))
        S_linear = Function(V)
        S_linear.x.array[:] = 0.3
        S_linear.x.scatter_forward()
        
        k_rho_1000 = compute_k_rho_from_relaxation(
            simple_config, S_linear, T_total=1000.0, epsilon=0.1
        )
        k_rho_2000 = compute_k_rho_from_relaxation(
            simple_config, S_linear, T_total=2000.0, epsilon=0.1
        )
        
        # k_rho ~ 1/T_total, so doubling T_total halves k_rho
        assert np.isclose(k_rho_1000 / k_rho_2000, 2.0, rtol=1e-10)
    
    def test_k_rho_formula(self, simple_config):
        """Verify k_rho = 2/(T·L1) * (-ln(epsilon)) formula."""
        from calibrate_mechanostat import compute_k_rho_from_relaxation
        
        V = functionspace(simple_config.domain, ("DG", 0))
        S_linear = Function(V)
        S_linear.x.array[:] = 0.25
        S_linear.x.scatter_forward()
        
        T_total = 500.0
        epsilon = 0.2
        
        k_rho = compute_k_rho_from_relaxation(
            simple_config, S_linear, T_total=T_total, epsilon=epsilon
        )
        
        # L1 = average |S_linear| = 0.25 (uniform field)
        L1 = 0.25
        expected_k_rho = (2.0 / (T_total * L1)) * (-np.log(epsilon))
        
        assert np.isclose(k_rho, expected_k_rho, rtol=1e-6)


# =============================================================================
# Tests for Config updates
# =============================================================================

class TestConfigHelmholtzFields:
    """Tests for helmholtz_L and helmholtz_factor in Config."""
    
    def test_config_has_helmholtz_L_field(self, simple_config):
        """Config should have helmholtz_L field."""
        assert hasattr(simple_config, "helmholtz_L")
    
    def test_config_has_helmholtz_factor_field(self, simple_config):
        """Config should have helmholtz_factor field."""
        assert hasattr(simple_config, "helmholtz_factor")
    
    def test_helmholtz_L_default_is_zero(self, simple_config):
        """helmholtz_L default should be 0 (auto mode)."""
        # Create fresh config to check default
        facet_tags = build_facetag(simple_config.domain)
        cfg = Config(domain=simple_config.domain, facet_tags=facet_tags)
        assert cfg.helmholtz_L == 0.0
    
    def test_helmholtz_factor_default(self, simple_config):
        """helmholtz_factor default should be 2.0."""
        facet_tags = build_facetag(simple_config.domain)
        cfg = Config(domain=simple_config.domain, facet_tags=facet_tags)
        assert cfg.helmholtz_factor == 2.0


# =============================================================================
# Integration test placeholder
# =============================================================================

@pytest.mark.integration
class TestCalibrateIntegration:
    """Integration tests requiring full mechanics solve.
    
    These tests are more expensive and verify the full calibration pipeline.
    Skipped if no femur geometry available.
    """
    
    @pytest.mark.skip(reason="Requires full loader infrastructure")
    def test_calibrate_mechanostat_full_pipeline(self):
        """Full calibration with real loading cases."""
        pass
    
    def test_auto_helmholtz_respects_config_update(self, simple_config):
        """Verify helmholtz_L can be set on config."""
        from calibrate_mechanostat import auto_helmholtz_length
        
        result = auto_helmholtz_length(simple_config)
        
        # Manually update config
        simple_config.helmholtz_L = result["helmholtz_L"]
        
        assert simple_config.helmholtz_L > 0
        assert np.isclose(
            simple_config.helmholtz_L, 
            result["factor"] * result["h_min"],
            rtol=1e-10
        )
