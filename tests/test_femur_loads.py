"""
Test suite for femur_loads.py module.

Tests load application classes and utility functions including:
- Utility functions: build_load, vector_from_angles, gait_interpolator, orthoload2ISB
- GaussianSurfaceLoad base class functionality
- HIPJointLoad for hip joint force application
- MuscleLoad for muscle force application along splines
"""

from unittest.mock import patch

import numpy as np
import pytest


import pyvista as pv
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from scipy.interpolate import interp1d

from simulation.femur_loads import (
    build_load,
    vector_from_angles,
    gait_interpolator, 
    orthoload2ISB,
    GaussianSurfaceLoad,
    HIPJointLoad,
    MuscleLoad
)
from simulation.femur_css import FemurCSS


# Fixtures and test data generators
@pytest.fixture
def sample_femur_mesh():
    """Create a simple femur-like mesh for testing."""
    # Create a cylinder with a sphere on top (simplified femur)
    cylinder = pv.Cylinder(center=(0, -50, 0), direction=(0, 1, 0), 
                          radius=15, height=100)
    sphere = pv.Sphere(center=(0, 0, 0), radius=25)
    femur = cylinder + sphere
    return femur


@pytest.fixture
def sample_head_line():
    """Sample head line points for femoral head fitting."""
    return np.array([
        [-20.0, 0.0, 0.0],
        [20.0, 0.0, 0.0]
    ])


@pytest.fixture
def sample_le_me_line():
    """Sample lateral/medial epicondyle line."""
    return np.array([
        [0.0, -100.0, -30.0],  # Lateral epicondyle
        [0.0, -100.0, 30.0]    # Medial epicondyle
    ])


@pytest.fixture
def sample_femur_css(sample_femur_mesh, sample_head_line, sample_le_me_line):
    """Create a FemurCSS instance for testing."""
    return FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="left")


@pytest.fixture
def sample_gait_data():
    """Sample gait cycle data for testing."""
    gait_cycle = np.linspace(0, 100, 51)  # 0-100% gait cycle
    gait_values = np.sin(2 * np.pi * gait_cycle / 100)  # Simple sinusoidal pattern
    return np.column_stack([gait_cycle, gait_values])


@pytest.fixture
def sample_force_vector():
    """Sample force vector for testing."""
    return np.array([100.0, -500.0, 200.0])  # Realistic hip force components


@pytest.fixture
def sample_muscle_points():
    """Sample muscle attachment points for testing."""
    # Points along a simplified muscle attachment line
    t = np.linspace(0, 1, 5)
    points = np.zeros((5, 3))
    points[:, 0] = 10 * t  # anterior-posterior
    points[:, 1] = -20 - 30 * t  # superior-inferior 
    points[:, 2] = 15 * np.sin(np.pi * t)  # medial-lateral
    return points


# Test utility functions
class TestUtilityFunctions:
    """Test standalone utility functions."""
    
    def test_build_load_basic(self, sample_gait_data, sample_force_vector):
        """Test basic functionality of build_load."""
        result = build_load(sample_gait_data, sample_force_vector)
        
        # Check output shape and structure
        assert result.shape == (len(sample_gait_data), 4)
        assert_array_almost_equal(result[:, 0], sample_gait_data[:, 0])  # Time column
        
        # Check force scaling
        for i in range(3):
            expected_force = sample_force_vector[i] * sample_gait_data[:, 1]
            assert_array_almost_equal(result[:, i+1], expected_force)
    
    def test_build_load_zero_gait_values(self, sample_force_vector):
        """Test build_load with zero gait values."""
        gait_data = np.array([[0, 0], [50, 0], [100, 0]])
        result = build_load(gait_data, sample_force_vector)
        
        # All force components should be zero
        assert_array_almost_equal(result[:, 1:], 0.0)
    
    def test_vector_from_angles_zero_angles(self):
        """Test vector_from_angles with zero angles."""
        magnitude = 100.0
        vector = vector_from_angles(magnitude, 0.0, 0.0)
        
        expected = np.array([0.0, magnitude, 0.0])
        assert_array_almost_equal(vector, expected)
        assert_almost_equal(np.linalg.norm(vector), magnitude)
    
    def test_vector_from_angles_sagittal_only(self):
        """Test vector_from_angles with sagittal angle only."""
        magnitude = 100.0
        alpha_sag = 30.0
        vector = vector_from_angles(magnitude, alpha_sag, 0.0)
        
        # Should have x and y components, no z component
        assert vector[2] == 0.0
        assert vector[0] > 0  # Positive x for positive sagittal angle
        assert vector[1] > 0  # Positive y component
        assert_almost_equal(np.linalg.norm(vector), magnitude)
    
    def test_vector_from_angles_frontal_only(self):
        """Test vector_from_angles with frontal angle only."""
        magnitude = 100.0
        alpha_front = 45.0
        vector = vector_from_angles(magnitude, 0.0, alpha_front)
        
        # Should have y and z components, no x component
        assert vector[0] == 0.0
        assert vector[2] > 0  # Positive z for positive frontal angle
        assert vector[1] > 0  # Positive y component
        assert_almost_equal(np.linalg.norm(vector), magnitude)
    
    def test_vector_from_angles_both_angles(self):
        """Test vector_from_angles with both angles."""
        magnitude = 100.0
        alpha_sag = 15.0
        alpha_front = 30.0
        vector = vector_from_angles(magnitude, alpha_sag, alpha_front)
        
        # All components should be positive
        assert all(vector > 0)
        assert_almost_equal(np.linalg.norm(vector), magnitude)
    
    def test_vector_from_angles_negative_angles(self):
        """Test vector_from_angles with negative angles."""
        magnitude = 100.0
        alpha_sag = -20.0
        alpha_front = -10.0
        vector = vector_from_angles(magnitude, alpha_sag, alpha_front)
        
        assert vector[0] < 0  # Negative x for negative sagittal
        assert vector[1] > 0  # Y should still be positive
        assert vector[2] < 0  # Negative z for negative frontal
        assert_almost_equal(np.linalg.norm(vector), magnitude)
    

    def test_gait_interpolator_valid_input(self):
        """Test gait_interpolator with valid input."""
        gait_data = np.array([
            [0, 100, 200, 300],
            [25, 150, 250, 350],
            [50, 200, 300, 400],
            [75, 150, 250, 350],
            [100, 100, 200, 300]
        ])
        interpolator = gait_interpolator(gait_data)
        # Test interpolation at known points
        assert_array_almost_equal(interpolator(0), [100, 200, 300])
        assert_array_almost_equal(interpolator(50), [200, 300, 400])
        assert_array_almost_equal(interpolator(25), [150, 250, 350])

    def test_gait_interpolator_invalid_input_shape(self):
        """Test gait_interpolator with invalid input shape (wrong columns)."""
        invalid_data = np.array([[0, 100, 200], [50, 150, 250]])
        with pytest.raises(ValueError, match="Provide an \\(N,4\\) array"):
            gait_interpolator(invalid_data)

    def test_gait_interpolator_invalid_type(self):
        """Test gait_interpolator with non-numpy input (should raise TypeError)."""
        with pytest.raises(TypeError, match="Input must be a numpy ndarray"):
            gait_interpolator([[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_gait_interpolator_too_few_points(self):
        """Test gait_interpolator with less than 2 points (should raise ValueError)."""
        data = np.array([[0, 1, 2, 3]])
        with pytest.raises(ValueError, match="At least two data points are required"):
            gait_interpolator(data)

    def test_gait_interpolator_unsorted_data(self):
        """Test gait_interpolator with unsorted time data."""
        gait_data = np.array([
            [50, 200, 300, 400],
            [0, 100, 200, 300],
            [100, 100, 200, 300],
            [25, 150, 250, 350]
        ])
        interpolator = gait_interpolator(gait_data)
        assert_array_almost_equal(interpolator(0), [100, 200, 300])

    def test_gait_interpolator_exception_propagation(self, monkeypatch):
        """Test gait_interpolator error handling when interp1d fails."""
        gait_data = np.array([
            [0, 100, 200, 300],
            [50, 200, 300, 400]
        ])
        # Patch interp1d to raise an error
        import simulation.femur_loads as femur_loads_mod
        monkeypatch.setattr(femur_loads_mod, "interp1d", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("interp1d failed")))
        with pytest.raises(RuntimeError, match="interp1d failed"):
            femur_loads_mod.gait_interpolator(gait_data)
    
    def test_orthoload2ISB_conversion(self):
        """Test orthoload2ISB coordinate conversion."""
        # Sample data with time and 3 force components
        orthoload_data = np.array([
            [0, 100, 200, 300],
            [25, 150, 250, 350],
            [50, 200, 300, 400]
        ])
        
        isb_data = orthoload2ISB(orthoload_data)
        
        # Check shape is preserved
        assert isb_data.shape == orthoload_data.shape
        
        # Check column reordering: [0, 2, 3, 1]
        assert_array_almost_equal(isb_data[:, 0], orthoload_data[:, 0])  # Time unchanged
        assert_array_almost_equal(isb_data[:, 1], orthoload_data[:, 2])  # y+ -> x+
        assert_array_almost_equal(isb_data[:, 2], orthoload_data[:, 3])  # z+ -> y+
        assert_array_almost_equal(isb_data[:, 3], orthoload_data[:, 1])  # x+ -> z+


# Test GaussianSurfaceLoad base class
class TestGaussianSurfaceLoad:
    """Test GaussianSurfaceLoad base class."""
    
    def test_initialization(self, sample_femur_mesh, sample_femur_css):
        """Test GaussianSurfaceLoad initialization."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        # Check basic attributes
        assert load.css is sample_femur_css
        assert_array_almost_equal(load.head_center_world, sample_femur_css.fhc)
        assert load.head_radius == sample_femur_css.head_radius
        assert load._use_cell_data is True
        assert load._interp is None
        
        # Check mesh setup
        assert hasattr(load, 'mesh_world')
        assert hasattr(load, 'mesh_css')
        assert hasattr(load, 'centers_world')
        assert hasattr(load, 'centers_css')
        assert hasattr(load, 'areas')
        
        # Check mesh properties
        assert load.mesh_world.n_cells > 0
        assert load.mesh_css.n_cells == load.mesh_world.n_cells
        assert len(load.centers_world) == load.mesh_world.n_cells
        assert len(load.centers_css) == load.mesh_css.n_cells
        assert len(load.areas) == load.mesh_world.n_cells
    
    def test_initialization_point_data_mode(self, sample_femur_mesh, sample_femur_css):
        """Test initialization with use_cell_data=False."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css, use_cell_data=False)
        assert load._use_cell_data is False
    
    def test_resolve_force_vector_valid(self, sample_femur_mesh, sample_femur_css):
        """Test _resolve_force_vector with valid input."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([100, -500, 200])
        
        F_world, F_norm = load._resolve_force_vector(force_css)
        
        assert isinstance(F_world, np.ndarray)
        assert F_world.shape == (3,)
        assert F_norm > 0
        assert_almost_equal(np.linalg.norm(F_world), F_norm)
    
    def test_resolve_force_vector_none_input(self, sample_femur_mesh, sample_femur_css):
        """Test _resolve_force_vector with None input."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        with pytest.raises(ValueError, match="Force vector must be provided"):
            load._resolve_force_vector(None)
    
    def test_resolve_force_vector_too_small(self, sample_femur_mesh, sample_femur_css):
        """Test _resolve_force_vector with too small magnitude."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        tiny_force = np.array([1e-10, 1e-10, 1e-10])
        
        with pytest.raises(ValueError, match="World force vector magnitude too small"):
            load._resolve_force_vector(tiny_force)
    
    def test_interpolation_before_load_applied(self, sample_femur_mesh, sample_femur_css):
        """Test interpolation call before applying any load."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        test_points = np.array([[0, 0, 0], [1, 1, 1]])
        
        with pytest.raises(RuntimeError, match="Apply load first before interpolation"):
            load(test_points)
    
    def test_check_equilibrium_valid_traction(self, sample_femur_mesh, sample_femur_css):
        """Test equilibrium check with valid traction."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        # Create simple uniform traction
        n_cells = load.mesh_world.n_cells
        traction_vectors = np.ones((n_cells, 3)) * 0.1  # Small uniform traction
        total_force = np.sum(traction_vectors * load.areas[:, None], axis=0)
        expected_magnitude = np.linalg.norm(total_force)
        
        # Should not raise any exception
        load.check_equilibrium(traction_vectors, expected_magnitude)
    
    def test_check_equilibrium_invalid_shape(self, sample_femur_mesh, sample_femur_css):
        """Test equilibrium check with invalid traction shape."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        # Wrong shape (2D instead of 3D components)
        invalid_traction = np.ones((10, 2))
        
        with pytest.raises(ValueError, match="Traction vector must be Nx3 array"):
            load.check_equilibrium(invalid_traction, 100.0)
    
    def test_compute_traction_basic(self, sample_femur_mesh, sample_femur_css):
        """Test _compute_traction basic functionality."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        n_cells = load.mesh_world.n_cells
        weights = np.ones(n_cells)  # Uniform weights
        F_norm = 1000.0
        unit_force = np.array([0, 1, 0])  # Upward force
        
        traction_vectors = load._compute_traction(weights, F_norm, unit_force)
        
        assert traction_vectors.shape == (n_cells, 3)
        # All traction vectors should point upward
        assert np.all(traction_vectors[:, 1] > 0)  # Y component positive
        assert np.all(traction_vectors[:, 0] == 0)  # X component zero
        assert np.all(traction_vectors[:, 2] == 0)  # Z component zero
    
    def test_compute_traction_with_flip(self, sample_femur_mesh, sample_femur_css):
        """Test _compute_traction with force flipping."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        n_cells = load.mesh_world.n_cells
        weights = np.ones(n_cells)
        F_norm = 1000.0
        unit_force = np.array([0, 1, 0])
        
        traction_vectors = load._compute_traction(weights, F_norm, unit_force, flip=True)
        
        # All traction vectors should point downward (flipped)
        assert np.all(traction_vectors[:, 1] < 0)  # Y component negative
    
    def test_compute_traction_invalid_weights(self, sample_femur_mesh, sample_femur_css):
        """Test _compute_traction with invalid weights."""
        load = GaussianSurfaceLoad(sample_femur_mesh, sample_femur_css)
        
        n_cells = load.mesh_world.n_cells
        weights = np.zeros(n_cells)  # All zero weights
        F_norm = 1000.0
        unit_force = np.array([0, 1, 0])
        
        with pytest.raises(RuntimeError, match="Invalid normalization"):
            load._compute_traction(weights, F_norm, unit_force)


# Test HIPJointLoad class
class TestHIPJointLoad:
    """Test HIPJointLoad class functionality."""
    
    def test_initialization(self, sample_femur_mesh, sample_femur_css):
        """Test HIPJointLoad initialization."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        
        # Should inherit from GaussianSurfaceLoad
        assert isinstance(hip_load, GaussianSurfaceLoad)
        assert hip_load.css is sample_femur_css
    
    def test_apply_gaussian_load_basic(self, sample_femur_mesh, sample_femur_css):
        """Test basic Gaussian load application."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])  # Downward force in CSS
        
        mesh_with_traction = hip_load.apply_gaussian_load(force_css, sigma_deg=15.0)
        
        # Check output mesh
        assert isinstance(mesh_with_traction, pv.PolyData)
        assert 'traction' in mesh_with_traction.cell_data
        assert mesh_with_traction.cell_data['traction'].shape[1] == 3
        
        # Check interpolator was set up
        assert hip_load._interp is not None
        
        # Test interpolation
        test_points = hip_load.centers_world[:5]  # First 5 cell centers
        interpolated = hip_load(test_points)
        assert interpolated.shape == (5, 3)
    
    def test_apply_gaussian_load_with_flip(self, sample_femur_mesh, sample_femur_css):
        """Test Gaussian load application with force flipping."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])
        
        mesh_normal = hip_load.apply_gaussian_load(force_css, flip=False)
        
        # Create new instance for flipped test
        hip_load_flip = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        mesh_flipped = hip_load_flip.apply_gaussian_load(force_css, flip=True)
        
        # Traction should be opposite
        traction_normal = mesh_normal.cell_data['traction']
        traction_flipped = mesh_flipped.cell_data['traction']
        assert_array_almost_equal(traction_normal, -traction_flipped)
    
    def test_apply_gaussian_load_different_sigma(self, sample_femur_mesh, sample_femur_css):
        """Test Gaussian load with different sigma values."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])
        
        mesh_narrow = hip_load.apply_gaussian_load(force_css, sigma_deg=5.0)
        
        # Create new instance for wide sigma
        hip_load_wide = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        mesh_wide = hip_load_wide.apply_gaussian_load(force_css, sigma_deg=30.0)
        
        traction_narrow = mesh_narrow.cell_data['traction']
        traction_wide = mesh_wide.cell_data['traction']
        
        # Narrow distribution should have higher peak values
        max_narrow = np.max(np.linalg.norm(traction_narrow, axis=1))
        max_wide = np.max(np.linalg.norm(traction_wide, axis=1))
        assert max_narrow > max_wide
    
    @patch('pyvista.PolyData.ray_trace')
    def test_apply_gaussian_load_ray_miss(self, mock_ray_trace, sample_femur_mesh, sample_femur_css):
        """Test Gaussian load when ray trace misses surface."""
        # Mock ray_trace to return empty hits
        mock_ray_trace.return_value = (None, [])
        
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])
        
        with pytest.raises(RuntimeError, match="Ray cast missed surface"):
            hip_load.apply_gaussian_load(force_css)


# Test MuscleLoad class
class TestMuscleLoad:
    """Test MuscleLoad class functionality."""
    
    def test_initialization(self, sample_femur_mesh, sample_femur_css):
        """Test MuscleLoad initialization."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Should inherit from GaussianSurfaceLoad
        assert isinstance(muscle_load, GaussianSurfaceLoad)
        assert muscle_load.css is sample_femur_css
    
    def test_set_attachment_points_valid(self, sample_femur_mesh, sample_femur_css, sample_muscle_points):
        """Test setting valid attachment points."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load.set_attachment_points(sample_muscle_points)
        
        # Check internal attributes were set
        assert hasattr(muscle_load, '_tck')
        assert hasattr(muscle_load, '_curve')
        assert hasattr(muscle_load, '_tree')
        assert hasattr(muscle_load, '_seg_lengths')
        
        assert muscle_load._curve.shape[1] == 3  # 3D curve
        assert len(muscle_load._seg_lengths) > 0
    
    def test_set_attachment_points_invalid_shape(self, sample_femur_mesh, sample_femur_css):
        """Test setting attachment points with invalid shape."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Wrong shape (2D instead of 3D)
        invalid_points = np.array([[1, 2], [3, 4], [5, 6]])
        
        with pytest.raises(ValueError, match="Points must be Nx3 array"):
            muscle_load.set_attachment_points(invalid_points)
    
    def test_set_attachment_points_different_parameters(self, sample_femur_mesh, sample_femur_css, sample_muscle_points):
        """Test setting attachment points with different spline parameters."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Test with different degree and smoothing
        muscle_load.set_attachment_points(sample_muscle_points, degree=2, smooth=0.05)
        
        assert hasattr(muscle_load, '_tck')
        assert muscle_load._curve.shape[1] == 3
    
    def test_apply_gaussian_load_without_attachment_points(self, sample_femur_mesh, sample_femur_css):
        """Test applying load without setting attachment points first."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, 0, 500])  # Lateral force
        
        with pytest.raises(RuntimeError, match="Set attachment points first"):
            muscle_load.apply_gaussian_load(force_css)
    
    def test_apply_gaussian_load_valid(self, sample_femur_mesh, sample_femur_css, sample_muscle_points):
        """Test valid Gaussian load application for muscle."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load.set_attachment_points(sample_muscle_points)
        
        force_css = np.array([0, 0, 500])  # Lateral force
        
        mesh_with_traction = muscle_load.apply_gaussian_load(force_css, sigma=5.0)
        
        # Check output mesh
        assert isinstance(mesh_with_traction, pv.PolyData)
        assert 'traction' in mesh_with_traction.cell_data
        assert mesh_with_traction.cell_data['traction'].shape[1] == 3
        
        # Check interpolator was set up
        assert muscle_load._interp is not None
    
    def test_apply_gaussian_load_with_flip(self, sample_femur_mesh, sample_femur_css, sample_muscle_points):
        """Test muscle load application with force flipping."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load.set_attachment_points(sample_muscle_points)
        
        force_css = np.array([0, 0, 500])
        
        mesh_normal = muscle_load.apply_gaussian_load(force_css, flip=False)
        
        # Create new instance for flipped test
        muscle_load_flip = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load_flip.set_attachment_points(sample_muscle_points)
        mesh_flipped = muscle_load_flip.apply_gaussian_load(force_css, flip=True)
        
        # Both should have traction data
        assert 'traction' in mesh_normal.cell_data
        assert 'traction' in mesh_flipped.cell_data
        
        # Check that some cells have significant traction
        traction_normal = mesh_normal.cell_data['traction']
        traction_flipped = mesh_flipped.cell_data['traction']
        
        assert np.any(np.linalg.norm(traction_normal, axis=1) > 1e-6)
        assert np.any(np.linalg.norm(traction_flipped, axis=1) > 1e-6)
        
        # The magnitudes should be similar but directions different
        mag_normal = np.linalg.norm(traction_normal, axis=1)
        mag_flipped = np.linalg.norm(traction_flipped, axis=1)
        
        # Check that both have similar total magnitudes (force conservation)
        total_mag_normal = np.sum(mag_normal * muscle_load.areas)
        total_mag_flipped = np.sum(mag_flipped * muscle_load_flip.areas)
        
        assert_almost_equal(total_mag_normal, total_mag_flipped, decimal=1)
    
    def test_apply_gaussian_load_different_sigma(self, sample_femur_mesh, sample_femur_css, sample_muscle_points):
        """Test muscle load with different sigma values."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load.set_attachment_points(sample_muscle_points)
        
        force_css = np.array([0, 0, 500])
        
        mesh_narrow = muscle_load.apply_gaussian_load(force_css, sigma=2.0)
        
        # Create new instance for wide sigma
        muscle_load_wide = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load_wide.set_attachment_points(sample_muscle_points)
        mesh_wide = muscle_load_wide.apply_gaussian_load(force_css, sigma=10.0)
        
        traction_narrow = mesh_narrow.cell_data['traction']
        traction_wide = mesh_wide.cell_data['traction']
        
        # Narrow distribution should have higher peak values
        max_narrow = np.max(np.linalg.norm(traction_narrow, axis=1))
        max_wide = np.max(np.linalg.norm(traction_wide, axis=1))
        assert max_narrow > max_wide


# Integration tests
class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_hip_and_muscle_load_combination(self, sample_femur_mesh, sample_femur_css, sample_muscle_points):
        """Test applying both hip and muscle loads to the same mesh."""
        # Apply hip load
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        hip_force = np.array([0, -1000, 0])
        hip_mesh = hip_load.apply_gaussian_load(hip_force)
        
        # Apply muscle load
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load.set_attachment_points(sample_muscle_points)
        muscle_force = np.array([0, 0, 300])
        muscle_mesh = muscle_load.apply_gaussian_load(muscle_force)
        
        # Both should have traction data
        assert 'traction' in hip_mesh.cell_data
        assert 'traction' in muscle_mesh.cell_data
        
        # Can interpolate from both
        test_points = sample_femur_mesh.cell_centers().points[:10]
        hip_traction = hip_load(test_points)
        muscle_traction = muscle_load(test_points)
        
        assert hip_traction.shape == (10, 3)
        assert muscle_traction.shape == (10, 3)
    
    def test_gait_data_processing_workflow(self, sample_gait_data, sample_force_vector):
        """Test complete gait data processing workflow."""
        # Build load data
        load_data = build_load(sample_gait_data, sample_force_vector)
        
        # Create interpolator
        interpolator = gait_interpolator(load_data)
        
        # Test interpolation at different time points
        time_points = [10, 30, 60, 90]
        for t in time_points:
            forces = interpolator(t)
            assert forces.shape == (3,)
            assert not np.any(np.isnan(forces))
    
    def test_coordinate_system_consistency(self, sample_femur_mesh, sample_femur_css):
        """Test that loads respect coordinate system transformations."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        
        # Apply force in CSS coordinates
        force_css = np.array([100, -500, 200])
        mesh_with_traction = hip_load.apply_gaussian_load(force_css)
        
        # Transform force to world coordinates manually
        force_world_expected = sample_femur_css.css_to_world_vector(force_css)
        
        # Check that the total applied force matches expectation
        traction = mesh_with_traction.cell_data['traction']
        areas = hip_load.areas
        total_force = np.sum(traction * areas[:, None], axis=0)
        
        # Should be close to expected world force (within numerical tolerance)
        assert_array_almost_equal(total_force, force_world_expected, decimal=1)


# Edge cases and error handling
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_mesh(self, sample_head_line, sample_le_me_line):
        """Test with very small mesh."""
        # Create minimal tetrahedron with float64 points
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        cells = np.array([[4, 0, 1, 2, 3]])  # Single tetrahedron
        mesh = pv.UnstructuredGrid(cells, [pv.CellType.TETRA], points)
        
        css = FemurCSS(mesh, sample_head_line, sample_le_me_line)
        
        # Should still be able to create load objects
        hip_load = HIPJointLoad(mesh, css)
        muscle_load = MuscleLoad(mesh, css)
        
        assert isinstance(hip_load, HIPJointLoad)
        assert isinstance(muscle_load, MuscleLoad)
    
    def test_single_point_muscle_attachment(self, sample_femur_mesh, sample_femur_css):
        """Test muscle load with single attachment point."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Single point - this will fail in splprep due to insufficient points
        single_point = np.array([[0, 0, 0]])
        
        # Should raise an error for single point (can't fit spline)
        with pytest.raises(TypeError, match="1 <= k"):
            muscle_load.set_attachment_points(single_point)
    
    def test_two_point_muscle_attachment(self, sample_femur_mesh, sample_femur_css):
        """Test muscle load with minimum viable attachment points."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Two points - minimum for linear spline
        two_points = np.array([[0, 0, 0], [10, 10, 10]])
        
        # Should work with degree=1
        muscle_load.set_attachment_points(two_points, degree=1)
        
        force_css = np.array([0, 0, 100])
        mesh_with_traction = muscle_load.apply_gaussian_load(force_css)
        
        assert 'traction' in mesh_with_traction.cell_data
    
    def test_collinear_muscle_points(self, sample_femur_mesh, sample_femur_css):
        """Test muscle load with collinear attachment points."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Collinear points along y-axis
        collinear_points = np.array([
            [0, 0, 0],
            [0, 10, 0],
            [0, 20, 0],
            [0, 30, 0]
        ])
        
        muscle_load.set_attachment_points(collinear_points)
        
        force_css = np.array([100, 0, 0])
        mesh_with_traction = muscle_load.apply_gaussian_load(force_css)
        
        assert 'traction' in mesh_with_traction.cell_data


# Parametrized tests
class TestParametrized:
    """Parametrized tests for various input combinations."""
    
    @pytest.mark.parametrize("use_cell_data", [True, False])
    def test_gaussian_load_data_modes(self, sample_femur_mesh, sample_femur_css, use_cell_data):
        """Test GaussianSurfaceLoad with different data modes."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css, use_cell_data=use_cell_data)
        
        force_css = np.array([0, -1000, 0])
        mesh_with_traction = hip_load.apply_gaussian_load(force_css)
        
        # Both modes should create traction data
        assert 'traction' in mesh_with_traction.cell_data or 'traction' in mesh_with_traction.point_data
        
        # Test interpolation works
        test_points = hip_load.centers_world[:5]
        interpolated = hip_load(test_points)
        assert interpolated.shape == (5, 3)
    
    @pytest.mark.parametrize("sigma_deg", [5.0, 15.0, 30.0, 45.0])
    def test_hip_load_different_sigma_values(self, sample_femur_mesh, sample_femur_css, sigma_deg):
        """Test hip load with various sigma values."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])
        
        mesh_with_traction = hip_load.apply_gaussian_load(force_css, sigma_deg=sigma_deg)
        
        assert 'traction' in mesh_with_traction.cell_data
        traction = mesh_with_traction.cell_data['traction']
        
        # Check that traction magnitude varies with sigma
        max_traction = np.max(np.linalg.norm(traction, axis=1))
        assert max_traction > 0
    
    @pytest.mark.parametrize("sigma", [1.0, 3.0, 8.0, 15.0])
    def test_muscle_load_different_sigma_values(self, sample_femur_mesh, sample_femur_css, sample_muscle_points, sigma):
        """Test muscle load with various sigma values."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        muscle_load.set_attachment_points(sample_muscle_points)
        
        force_css = np.array([0, 0, 300])
        mesh_with_traction = muscle_load.apply_gaussian_load(force_css, sigma=sigma)
        
        assert 'traction' in mesh_with_traction.cell_data
        traction = mesh_with_traction.cell_data['traction']
        
        max_traction = np.max(np.linalg.norm(traction, axis=1))
        assert max_traction > 0
    
    @pytest.mark.parametrize("force_direction", [
        np.array([1, 0, 0]),    # Anterior
        np.array([0, -1, 0]),   # Inferior
        np.array([0, 0, 1]),    # Medial
        np.array([1, -1, 1]),   # Combined
    ])
    def test_different_force_directions(self, sample_femur_mesh, sample_femur_css, force_direction):
        """Test loads with different force directions."""
        magnitude = 1000.0
        force_css = magnitude * force_direction / np.linalg.norm(force_direction)
        
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        mesh_with_traction = hip_load.apply_gaussian_load(force_css)
        
        assert 'traction' in mesh_with_traction.cell_data
        
        # Check force equilibrium
        traction = mesh_with_traction.cell_data['traction']
        areas = hip_load.areas
        total_force = np.sum(traction * areas[:, None], axis=0)
        total_magnitude = np.linalg.norm(total_force)
        
        # Should preserve magnitude (within tolerance)
        assert_almost_equal(total_magnitude, magnitude, decimal=0)


# Performance tests (marked as slow)
class TestPerformance:
    """Performance tests for large-scale scenarios."""
    
    def test_large_mesh_hip_load(self):
        """Test hip load performance with large mesh."""
        # Create larger mesh
        large_mesh = pv.Sphere(theta_resolution=50, phi_resolution=50)
        head_line = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        le_me = np.array([[0.0, -2.0, -1.0], [0.0, -2.0, 1.0]])
        
        css = FemurCSS(large_mesh, head_line, le_me)
        hip_load = HIPJointLoad(large_mesh, css)
        
        force_css = np.array([0, -1000, 0])
        
        # Should complete without timeout
        mesh_with_traction = hip_load.apply_gaussian_load(force_css)
        
        assert mesh_with_traction.n_cells > 1000  # Ensure it's actually large
        assert 'traction' in mesh_with_traction.cell_data
    
    def test_many_muscle_points(self, sample_femur_mesh, sample_femur_css):
        """Test muscle load with many attachment points."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)
        
        # Create many points along a curve
        t = np.linspace(0, 1, 50)
        many_points = np.zeros((50, 3))
        many_points[:, 0] = 20 * np.sin(2 * np.pi * t)
        many_points[:, 1] = -50 - 30 * t
        many_points[:, 2] = 10 * np.cos(2 * np.pi * t)
        
        muscle_load.set_attachment_points(many_points)
        
        force_css = np.array([0, 0, 500])
        mesh_with_traction = muscle_load.apply_gaussian_load(force_css)
        
        assert 'traction' in mesh_with_traction.cell_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
