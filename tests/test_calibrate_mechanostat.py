"""
Tests for calibrate_mechanostat module.

Tests the calibration utilities for psi_ref, k_rho, and Helmholtz length.
Uses unit cube with simple loading for reproducible results.
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
# Simple Loader stub for testing
# =============================================================================

class SimpleLoaderStub:
    """Minimal loader stub for testing calibration."""
    
    def __init__(self, domain: mesh.Mesh, facet_tags: mesh.MeshTags):
        self.mesh = domain
        self.comm = domain.comm
        self.load_tag = 2  # x=1 face
        self.cut_tag = 1   # x=0 face
        
        gdim = domain.geometry.dim
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
        V_traction = functionspace(domain, P1_vec)
        
        self.traction = Function(V_traction, name="traction")
        self.traction_cut = Function(V_traction, name="traction_cut")
        self._cached_cases = {}
        
    def precompute_loading_cases(self, loading_cases):
        """Store traction arrays for each loading case."""
        for case in loading_cases:
            traction_val = np.zeros(3)
            if hasattr(case, 'magnitude'):
                traction_val[0] = case.magnitude * 1e-3
            
            n_owned = self.traction.function_space.dofmap.index_map.size_local * 3
            arr = np.zeros_like(self.traction.x.array)
            arr_cut = np.zeros_like(self.traction_cut.x.array)
            
            for i in range(n_owned // 3):
                arr[3*i:3*i+3] = traction_val
                arr_cut[3*i:3*i+3] = -traction_val
            
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
        self.magnitude = magnitude


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


@pytest.fixture
def constant_rho0(simple_config):
    """Create constant density field at rho0."""
    P1 = basix.ufl.element("Lagrange", simple_config.domain.basix_cell(), 1)
    Q = functionspace(simple_config.domain, P1)
    rho0 = Function(Q, name="rho0")
    assign(rho0, simple_config.rho0)
    return rho0


# =============================================================================
# Tests for compute_k_rho
# =============================================================================

class TestComputeKRho:
    """Tests for k_rho computation."""
    
    def test_invalid_T_relax_raises(self, simple_config, simple_loader, 
                                     simple_loading_cases, constant_rho0):
        """T_relax must be positive."""
        from calibrate_mechanostat import compute_k_rho
        
        with pytest.raises(ValueError, match="T_relax"):
            compute_k_rho(
                simple_config, simple_loader, simple_loading_cases,
                constant_rho0, psi_ref=0.1, T_relax=0.0
            )
    
    def test_invalid_target_fraction_raises(self, simple_config, simple_loader,
                                             simple_loading_cases, constant_rho0):
        """target_fraction must be in (0, 1)."""
        from calibrate_mechanostat import compute_k_rho
        
        with pytest.raises(ValueError, match="target_fraction"):
            compute_k_rho(
                simple_config, simple_loader, simple_loading_cases,
                constant_rho0, psi_ref=0.1, target_fraction=0.0
            )
        
        with pytest.raises(ValueError, match="target_fraction"):
            compute_k_rho(
                simple_config, simple_loader, simple_loading_cases,
                constant_rho0, psi_ref=0.1, target_fraction=1.0
            )
    
    def test_returns_positive_k_rho(self, simple_config, simple_loader,
                                     simple_loading_cases, constant_rho0):
        """k_rho should be positive."""
        from calibrate_mechanostat import compute_k_rho
        
        result = compute_k_rho(
            simple_config, simple_loader, simple_loading_cases,
            constant_rho0, psi_ref=0.1, T_relax=500.0
        )
        
        assert result["k_rho"] > 0
    
    def test_returns_required_keys(self, simple_config, simple_loader,
                                    simple_loading_cases, constant_rho0):
        """Result should contain all required keys."""
        from calibrate_mechanostat import compute_k_rho
        
        result = compute_k_rho(
            simple_config, simple_loader, simple_loading_cases,
            constant_rho0, psi_ref=0.1, T_relax=500.0
        )
        
        assert "k_rho" in result
        assert "tau_eff" in result
        assert "S_linear_rms" in result
        assert "saturation_factor" in result


# =============================================================================
# Tests for compute_psi_ref
# =============================================================================

class TestComputePsiRef:
    """Tests for psi_ref computation."""
    
    def test_returns_positive_psi_ref(self, simple_config, simple_loader,
                                       simple_loading_cases, constant_rho0):
        """psi_ref should be positive."""
        from calibrate_mechanostat import compute_psi_ref
        
        result = compute_psi_ref(
            simple_config, simple_loader, simple_loading_cases, constant_rho0
        )
        
        assert result["psi_ref"] > 0
    
    def test_returns_required_keys(self, simple_config, simple_loader,
                                    simple_loading_cases, constant_rho0):
        """Result should contain all required keys."""
        from calibrate_mechanostat import compute_psi_ref
        
        result = compute_psi_ref(
            simple_config, simple_loader, simple_loading_cases, constant_rho0
        )
        
        assert "psi_ref" in result
        assert "S_ref" in result
        assert "S_mean" in result
        assert "S_std" in result
        assert "psi_mean" in result


# =============================================================================
# Tests for calibrate_mechanostat
# =============================================================================

class TestCalibrateMechanostat:
    """Tests for full calibration."""
    
    def test_calibrate_returns_all_keys(self, simple_config, simple_loader,
                                         simple_loading_cases, constant_rho0):
        """Full calibration returns all required keys."""
        from calibrate_mechanostat import calibrate_mechanostat
        
        result = calibrate_mechanostat(
            simple_config, simple_loader, simple_loading_cases, constant_rho0,
            update_config=False
        )
        
        # psi_ref keys
        assert "psi_ref" in result
        assert "S_ref" in result
        
        # k_rho keys
        assert "k_rho" in result
        assert "tau_eff" in result
        
        # helmholtz_L should be passed through from config
        assert "helmholtz_L" in result
    
    def test_calibrate_updates_config(self, simple_config, simple_loader,
                                       simple_loading_cases, constant_rho0):
        """Calibration updates config when requested."""
        from calibrate_mechanostat import calibrate_mechanostat
        
        result = calibrate_mechanostat(
            simple_config, simple_loader, simple_loading_cases, constant_rho0,
            update_config=True
        )
        
        # psi_ref and k_rho should be updated
        assert simple_config.psi_ref == result["psi_ref"]
        assert simple_config.k_rho == result["k_rho"]
        # helmholtz_L is NOT calibrated, stays from config
        assert simple_config.helmholtz_L == result["helmholtz_L"]
    
    def test_calibrate_respects_update_config_false(self, simple_config, simple_loader,
                                                     simple_loading_cases, constant_rho0):
        """Calibration preserves config when update_config=False."""
        from calibrate_mechanostat import calibrate_mechanostat
        
        old_psi = simple_config.psi_ref
        old_k = simple_config.k_rho
        old_L = simple_config.helmholtz_L
        
        calibrate_mechanostat(
            simple_config, simple_loader, simple_loading_cases, constant_rho0,
            update_config=False
        )
        
        # Config should be unchanged
        assert simple_config.psi_ref == old_psi
        assert simple_config.k_rho == old_k
        assert simple_config.helmholtz_L == old_L


# =============================================================================
# Tests for utility functions
# =============================================================================

class TestUtilities:
    """Tests for internal utility functions."""
    
    def test_golden_section_finds_minimum(self):
        """Golden section should find minimum of unimodal function."""
        from calibrate_mechanostat import _golden_section_minimize
        
        # Minimum of (x - 3)^2 at x = 3
        f = lambda x: (x - 3.0)**2
        x_min = _golden_section_minimize(f, 0.0, 10.0, tol=1e-8)
        
        assert np.isclose(x_min, 3.0, rtol=1e-6)
    
    def test_golden_section_asymmetric_bounds(self):
        """Golden section works with asymmetric bounds."""
        from calibrate_mechanostat import _golden_section_minimize
        
        # Minimum of (x - 5)^2 at x = 5, with bounds [4, 20]
        f = lambda x: (x - 5.0)**2
        x_min = _golden_section_minimize(f, 4.0, 20.0, tol=1e-8)
        
        assert np.isclose(x_min, 5.0, rtol=1e-6)
