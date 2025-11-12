"""Tests for femur_remodeller_gait.py: coordinate scaling and physical force validation."""
import pytest
import numpy as np


import basix
from dolfinx import fem

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.config import Config

# Import the module under test
from simulation.femur_remodeller_gait import setup_femur_gait_loading


@pytest.fixture(scope="module")
def femur_setup():
    """Create femur mesh and function space (expensive, reuse across tests)."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, P1_vec)
    cfg = Config(domain=domain)
    return domain, V, cfg


@pytest.fixture(scope="module")
def gait_loader(femur_setup):
    """Create gait loader (reuse across tests)."""
    _, V, cfg = femur_setup
    return setup_femur_gait_loading(V, cfg, BW_kg=75.0, n_samples=9)


class TestCoordinateScaling:
    """Verify DOLFINx mesh and femurloader use consistent coordinate systems."""

    def test_dolfinx_mesh_in_millimeters(self, femur_setup):
        """Verify DOLFINx mesh coordinates are in millimeters (not meters)."""
        domain, _, _ = femur_setup
        geom = domain.geometry.x
        
        # Femur geometry should be O(100) mm, not O(0.1) m
        max_coord = np.max(np.abs(geom))
        assert 10.0 < max_coord < 500.0, \
            f"Mesh coords should be in mm (expected 10-500, got {max_coord})"

    def test_femurloader_mesh_in_millimeters(self, femur_setup):
        """Verify femurloader PyVista mesh is in millimeters."""
        import pyvista as pv
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        
        max_coord = np.max(np.abs(pv_mesh.points))
        assert 10.0 < max_coord < 500.0, \
            f"PyVista mesh should be in mm (expected 10-500, got {max_coord})"

    def test_coord_scale_is_unity(self, gait_loader):
        """Verify coord_scale=1.0 (no conversion needed)."""
        assert gait_loader.coord_scale == 1.0, \
            "coord_scale should be 1.0 since both DOLFINx and femurloader use mm"

    def test_geometry_bounds_match(self, femur_setup):
        """Verify DOLFINx and PyVista geometries have same bounds."""
        domain, _, _ = femur_setup
        import pyvista as pv
        
        dolfinx_geom = domain.geometry.x
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        pv_geom = pv_mesh.points
        
        # Should have same coordinate ranges (within tolerance)
        dolfinx_range = np.ptp(dolfinx_geom, axis=0)
        pv_range = np.ptp(pv_geom, axis=0)
        
        np.testing.assert_allclose(dolfinx_range, pv_range, rtol=0.01,
            err_msg="DOLFINx and PyVista geometry ranges should match")


class TestPhysicalForces:
    """Validate physical coherence of gait loading forces."""

    def test_hip_force_at_peak_stance(self, gait_loader, femur_setup):
        """Hip joint force at peak stance (~50%) should be ~3-4× body weight."""
        _, _, cfg = femur_setup
        BW_kg = 75.0
        BW_N = BW_kg * 9.81  # ~736 N
        
        # Update to peak stance phase (~50% gait cycle)
        gait_loader.update_loads(50.0)
        
        # Compute total force magnitude from traction field
        t_hip_vals = gait_loader.t_hip.x.array.reshape((-1, 3))
        force_magnitudes = np.linalg.norm(t_hip_vals, axis=1)
        max_traction_nd = np.max(force_magnitudes)
        
        # Traction is nondimensionalized: stress_nd = stress_Pa / sigma_c
        # sigma_c = E0_dim * strain_scale = 6.5e9 * 1e-3 = 6.5e6 Pa
        sigma_c = cfg.sigma_c
        max_traction_Pa = max_traction_nd * sigma_c
        
        # Typical hip contact area ~500-1000 mm², peak stress ~2-5 MPa
        # Max traction should be O(1-10 MPa)
        assert 0.5e6 < max_traction_Pa < 20e6, \
            f"Max hip traction should be 0.5-20 MPa, got {max_traction_Pa/1e6:.2f} MPa"

    def test_hip_force_total_magnitude(self, gait_loader):
        """Verify total hip force integrates to expected value."""
        # Hip force from OrthoLoad data should be ~3× body weight at peak
        gait_loader.update_loads(50.0)  # Peak stance
        
        # Get the force vector from gait interpolator
        F_css = gait_loader.hip_gait(50.0)  # Force in CSS frame
        F_magnitude = np.linalg.norm(F_css)
        
        BW_N = 75.0 * 9.81
        F_ratio = F_magnitude / BW_N
        
        # OrthoLoad peak hip forces typically 2-4× BW
        assert 2.0 < F_ratio < 5.0, \
            f"Hip force should be 2-4× BW at peak stance, got {F_ratio:.2f}× BW"

    def test_muscle_forces_reasonable(self, gait_loader):
        """Verify gluteus muscle forces are physically reasonable."""
        BW_N = 75.0 * 9.81
        
        # Check gluteus medius at peak (first 30% of gait)
        F_glmed = gait_loader.glmed_gait(25.0)
        glmed_magnitude = np.linalg.norm(F_glmed)
        glmed_ratio = glmed_magnitude / BW_N
        
        # Gluteus medius: typically 0.5-2× BW during stance
        assert 0.1 < glmed_ratio < 3.0, \
            f"Glmed force should be 0.1-3× BW, got {glmed_ratio:.2f}× BW"
        
        # Check gluteus maximus
        F_glmax = gait_loader.glmax_gait(25.0)
        glmax_magnitude = np.linalg.norm(F_glmax)
        glmax_ratio = glmax_magnitude / BW_N
        
        # Gluteus maximus: typically 0.2-1.5× BW
        assert 0.05 < glmax_ratio < 2.0, \
            f"Glmax force should be 0.05-2× BW, got {glmax_ratio:.2f}× BW"

    def test_force_progression_across_gait(self, gait_loader):
        """Verify forces vary smoothly across gait cycle."""
        phases = np.linspace(0, 100, 11)
        hip_forces = []
        
        for phase in phases:
            F_css = gait_loader.hip_gait(phase)
            hip_forces.append(np.linalg.norm(F_css))
        
        hip_forces = np.array(hip_forces)
        
        # Forces should vary (not constant)
        assert np.ptp(hip_forces) > 0.5 * np.max(hip_forces), \
            "Hip force should vary significantly across gait cycle"
        
        # Should have a peak (not monotonic)
        assert not np.all(np.diff(hip_forces) > 0), \
            "Hip force should not be monotonically increasing"
        assert not np.all(np.diff(hip_forces) < 0), \
            "Hip force should not be monotonically decreasing"

    def test_traction_field_nonzero(self, gait_loader, femur_setup):
        """Verify traction fields contain non-zero values after interpolation."""
        _, _, cfg = femur_setup
        gait_loader.update_loads(50.0)
        
        sigma_c = cfg.sigma_c  # Characteristic stress for nondimensionalization
        
        # Check all three traction fields
        for name, func in [("hip", gait_loader.t_hip),
                          ("glmed", gait_loader.t_glmed),
                          ("glmax", gait_loader.t_glmax)]:
            vals = func.x.array.reshape((-1, 3))
            nonzero_count = np.count_nonzero(vals)
            
            assert nonzero_count > 1000, \
                f"{name} traction should have >1000 nonzero values, got {nonzero_count}"
            
            max_magnitude_nd = np.max(np.linalg.norm(vals, axis=1))
            max_magnitude_Pa = max_magnitude_nd * sigma_c
            
            # Tractions should be O(0.1-10 MPa) in physical units
            assert max_magnitude_Pa > 1e4, \
                f"{name} traction magnitude should be >10 kPa, got {max_magnitude_Pa/1e3:.1f} kPa"


class TestGaitQuadrature:
    """Verify gait cycle quadrature integration."""

    def test_quadrature_weights_sum_to_one(self, gait_loader):
        """Trapezoid weights should sum to 1.0."""
        quadrature = gait_loader.get_quadrature()
        phases, weights = zip(*quadrature)
        
        assert np.isclose(sum(weights), 1.0), \
            f"Quadrature weights should sum to 1.0, got {sum(weights)}"

    def test_quadrature_covers_full_cycle(self, gait_loader):
        """Quadrature should span [0, 100]% gait cycle."""
        quadrature = gait_loader.get_quadrature()
        phases, _ = zip(*quadrature)
        
        assert min(phases) == 0.0, "Quadrature should start at 0%"
        assert max(phases) == 100.0, "Quadrature should end at 100%"

    def test_quadrature_sample_count(self, gait_loader):
        """Should have n_samples quadrature points."""
        quadrature = gait_loader.get_quadrature()
        assert len(quadrature) == gait_loader.n_samples, \
            f"Expected {gait_loader.n_samples} samples, got {len(quadrature)}"


class TestForceMaxima:
    """Compute and report maximum forces for each load type across gait cycle."""

    def test_report_max_forces_across_gait(self, gait_loader, femur_setup, capsys):
        """Compute max forces for all loads across gait cycle (informative test)."""
        _, _, cfg = femur_setup
        sigma_c = cfg.sigma_c
        BW_N = 75.0 * 9.81
        phases = np.linspace(0, 100, 21)
        
        hip_max = 0.0
        glmed_max = 0.0
        glmax_max = 0.0
        
        hip_max_phase = 0.0
        glmed_max_phase = 0.0
        glmax_max_phase = 0.0
        
        hip_traction_max_nd = 0.0
        glmed_traction_max_nd = 0.0
        glmax_traction_max_nd = 0.0
        
        for phase in phases:
            # Force vectors
            F_hip = np.linalg.norm(gait_loader.hip_gait(phase))
            F_glmed = np.linalg.norm(gait_loader.glmed_gait(phase))
            F_glmax = np.linalg.norm(gait_loader.glmax_gait(phase))
            
            if F_hip > hip_max:
                hip_max = F_hip
                hip_max_phase = phase
            if F_glmed > glmed_max:
                glmed_max = F_glmed
                glmed_max_phase = phase
            if F_glmax > glmax_max:
                glmax_max = F_glmax
                glmax_max_phase = phase
            
            # Traction fields
            gait_loader.update_loads(phase)
            t_hip = np.max(np.linalg.norm(gait_loader.t_hip.x.array.reshape((-1, 3)), axis=1))
            t_glmed = np.max(np.linalg.norm(gait_loader.t_glmed.x.array.reshape((-1, 3)), axis=1))
            t_glmax = np.max(np.linalg.norm(gait_loader.t_glmax.x.array.reshape((-1, 3)), axis=1))
            
            hip_traction_max_nd = max(hip_traction_max_nd, t_hip)
            glmed_traction_max_nd = max(glmed_traction_max_nd, t_glmed)
            glmax_traction_max_nd = max(glmax_traction_max_nd, t_glmax)
        
        print("\n" + "="*70)
        print("GAIT LOADING ANALYSIS (BW = 75 kg = 736 N)")
        print("="*70)
        print("\nFORCE MAGNITUDE MAXIMA:")
        print(f"  Hip joint:      {hip_max:6.1f} N ({hip_max/BW_N:.2f}× BW) at {hip_max_phase:.0f}% gait")
        print(f"  Gluteus medius: {glmed_max:6.1f} N ({glmed_max/BW_N:.2f}× BW) at {glmed_max_phase:.0f}% gait")
        print(f"  Gluteus maximus:{glmax_max:6.1f} N ({glmax_max/BW_N:.2f}× BW) at {glmax_max_phase:.0f}% gait")
        
        print(f"\nMAX TRACTION (CONTACT STRESS) ACROSS GAIT:")
        print(f"  Hip joint:      {hip_traction_max_nd * sigma_c / 1e6:5.2f} MPa (ND: {hip_traction_max_nd:.3f})")
        print(f"  Gluteus medius: {glmed_traction_max_nd * sigma_c / 1e6:5.2f} MPa (ND: {glmed_traction_max_nd:.3f})")
        print(f"  Gluteus maximus:{glmax_traction_max_nd * sigma_c / 1e6:5.2f} MPa (ND: {glmax_traction_max_nd:.3f})")
        
        print(f"\nREFERENCE VALUES:")
        print(f"  sigma_c = {sigma_c/1e6:.2f} MPa (characteristic stress)")
        print(f"  Hip contact area: ~500-1000 mm²")
        print(f"  Typical hip contact stress: 2-10 MPa")
        print("="*70)
        
        # This test always passes - it's informative only
        assert True
