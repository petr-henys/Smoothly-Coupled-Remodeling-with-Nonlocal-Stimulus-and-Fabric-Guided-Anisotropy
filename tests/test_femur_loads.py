"""Tests for `simulation.femur_loads` (surface load application).

Tests load application classes and utility functions including:
- Utility functions: build_load, vector_from_angles, gait_interpolator, orthoload2ISB
- GaussianSurfaceLoad base class functionality  
- HIPJointLoad for hip joint force application
- MuscleLoad for muscle force application along splines

Split out from `test_gaitloader.py` for clarity.
"""

from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from simulation.femur_css import FemurCSS
from simulation.femur_loads import (
    GaussianSurfaceLoad,
    HIPJointLoad,
    MuscleLoad,
    build_load,
    gait_interpolator,
    orthoload2ISB,
    vector_from_angles,
)


# ===== Fixtures =====

@pytest.fixture
def sample_femur_mesh():
    """Create a simple femur-like mesh for testing."""
    cylinder = pv.Cylinder(center=(0, -50, 0), direction=(0, 1, 0), radius=15, height=100)
    sphere = pv.Sphere(center=(0, 0, 0), radius=25)
    femur = cylinder + sphere
    return femur


@pytest.fixture
def sample_head_line():
    """Sample head line points for femoral head fitting."""
    return np.array([
        [-20.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])


@pytest.fixture
def sample_le_me_line():
    """Sample lateral/medial epicondyle line."""
    return np.array([
        [0.0, -100.0, -30.0],
        [0.0, -100.0, 30.0],
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


# ===== Test utility functions =====

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
            assert_array_almost_equal(result[:, i + 1], expected_force)

    def test_build_load_zero_gait_values(self, sample_force_vector):
        """Test build_load with zero gait values."""
        gait_data = np.array([[0, 0], [50, 0], [100, 0]])
        result = build_load(gait_data, sample_force_vector)

        # All force components should be zero
        assert_array_almost_equal(result[:, 1:], 0.0)

    @pytest.mark.parametrize("alpha_sag,alpha_front", [(0.0, 0.0), (30.0, 45.0)])
    def test_vector_from_angles(self, alpha_sag, alpha_front):
        """Test vector_from_angles with angle combinations."""
        magnitude = 100.0
        vector = vector_from_angles(magnitude, alpha_sag, alpha_front)
        desc = f"alpha_sag={alpha_sag}, alpha_front={alpha_front}"

        # Verify magnitude is preserved
        assert_almost_equal(np.linalg.norm(vector), magnitude)

        # Verify expected component signs: positive y; x/z follow angle signs
        for i, expected_sign in enumerate([alpha_sag != 0, True, alpha_front != 0]):
            if expected_sign:
                assert vector[i] > 0, f"Component {i} should be positive for {desc}"
            else:
                assert abs(vector[i]) < 1e-10, f"Component {i} should be ~0 for {desc}"

    def test_gait_interpolator_valid_input(self):
        """Test gait_interpolator with valid input."""
        gait_data = np.array([
            [0, 100, 200, 300],
            [25, 150, 250, 350],
            [50, 200, 300, 400],
            [75, 150, 250, 350],
            [100, 100, 200, 300],
        ])
        interpolator = gait_interpolator(gait_data)
        # Test interpolation at known points
        assert_array_almost_equal(interpolator(0), [100, 200, 300])
        assert_array_almost_equal(interpolator(50), [200, 300, 400])
        assert_array_almost_equal(interpolator(25), [150, 250, 350])

    def test_orthoload2ISB_conversion(self):
        """Test orthoload2ISB coordinate conversion."""
        # Sample data with time and 3 force components
        orthoload_data = np.array([
            [0, 100, 200, 300],
            [25, 150, 250, 350],
            [50, 200, 300, 400],
        ])

        isb_data = orthoload2ISB(orthoload_data)

        # Check shape is preserved
        assert isb_data.shape == orthoload_data.shape

        # Check column reordering: [0, 2, 3, 1]
        assert_array_almost_equal(isb_data[:, 0], orthoload_data[:, 0])  # Time unchanged
        assert_array_almost_equal(isb_data[:, 1], orthoload_data[:, 2])  # y+ -> x+
        assert_array_almost_equal(isb_data[:, 2], orthoload_data[:, 3])  # z+ -> y+
        assert_array_almost_equal(isb_data[:, 3], orthoload_data[:, 1])  # x+ -> z+


# ===== Test GaussianSurfaceLoad base class =====

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
        assert hasattr(load, "mesh_world")
        assert hasattr(load, "mesh_css")
        assert hasattr(load, "centers_world")
        assert hasattr(load, "centers_css")
        assert hasattr(load, "areas")

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


# ===== Test HIPJointLoad class =====

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
        assert "traction" in mesh_with_traction.cell_data
        assert mesh_with_traction.cell_data["traction"].shape[1] == 3

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
        traction_normal = mesh_normal.cell_data["traction"]
        traction_flipped = mesh_flipped.cell_data["traction"]
        assert_array_almost_equal(traction_normal, -traction_flipped)

    def test_apply_gaussian_load_different_sigma(self, sample_femur_mesh, sample_femur_css):
        """Test Gaussian load with different sigma values."""
        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])

        mesh_narrow = hip_load.apply_gaussian_load(force_css, sigma_deg=5.0)

        # Create new instance for wide sigma
        hip_load_wide = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        mesh_wide = hip_load_wide.apply_gaussian_load(force_css, sigma_deg=30.0)

        traction_narrow = mesh_narrow.cell_data["traction"]
        traction_wide = mesh_wide.cell_data["traction"]

        # Narrow distribution should have higher peak values
        max_narrow = np.max(np.linalg.norm(traction_narrow, axis=1))
        max_wide = np.max(np.linalg.norm(traction_wide, axis=1))
        assert max_narrow > max_wide

    @patch("pyvista.PolyData.ray_trace")
    def test_apply_gaussian_load_ray_miss(self, mock_ray_trace, sample_femur_mesh, sample_femur_css):
        """Test Gaussian load when ray trace misses surface."""
        # Mock ray_trace to return empty hits
        mock_ray_trace.return_value = (None, [])

        hip_load = HIPJointLoad(sample_femur_mesh, sample_femur_css)
        force_css = np.array([0, -1000, 0])

        with pytest.raises(RuntimeError, match="Ray cast missed surface"):
            hip_load.apply_gaussian_load(force_css)


# ===== Test MuscleLoad class =====

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
        assert hasattr(muscle_load, "_tck")
        assert hasattr(muscle_load, "_curve")
        assert hasattr(muscle_load, "_tree")
        assert hasattr(muscle_load, "_seg_lengths")

        assert muscle_load._curve.shape[1] == 3  # 3D curve
        assert len(muscle_load._seg_lengths) > 0

    def test_set_attachment_points_invalid_shape(self, sample_femur_mesh, sample_femur_css):
        """Test setting attachment points with invalid shape."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)

        # Wrong shape (2D instead of 3D)
        invalid_points = np.array([[1, 2], [3, 4], [5, 6]])

        with pytest.raises(ValueError, match="Points must be Nx3 array"):
            muscle_load.set_attachment_points(invalid_points)

    def test_set_attachment_points_different_parameters(
        self, sample_femur_mesh, sample_femur_css, sample_muscle_points
    ):
        """Test setting attachment points with different spline parameters."""
        muscle_load = MuscleLoad(sample_femur_mesh, sample_femur_css)

        # Test with different degree and smoothing
        muscle_load.set_attachment_points(sample_muscle_points, degree=2, smooth=0.05)

        assert hasattr(muscle_load, "_tck")
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
        assert "traction" in mesh_with_traction.cell_data
        assert mesh_with_traction.cell_data["traction"].shape[1] == 3

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
        assert "traction" in mesh_normal.cell_data
        assert "traction" in mesh_flipped.cell_data

        # Check that some cells have significant traction
        traction_normal = mesh_normal.cell_data["traction"]
        traction_flipped = mesh_flipped.cell_data["traction"]

        assert np.any(np.linalg.norm(traction_normal, axis=1) > 1e-6)
        assert np.any(np.linalg.norm(traction_flipped, axis=1) > 1e-6)

        # The magnitudes should be similar but directions different
        mag_normal = np.linalg.norm(traction_normal, axis=1)
        mag_flipped = np.linalg.norm(traction_flipped, axis=1)

        # Check that both have similar total magnitudes (force conservation)
        total_mag_normal = np.sum(mag_normal * muscle_load.areas)
        total_mag_flipped = np.sum(mag_flipped * muscle_load_flip.areas)

        assert_almost_equal(total_mag_normal, total_mag_flipped, decimal=1)


# ===== Integration tests =====

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_hip_and_muscle_load_combination(
        self, sample_femur_mesh, sample_femur_css, sample_muscle_points
    ):
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
        assert "traction" in hip_mesh.cell_data
        assert "traction" in muscle_mesh.cell_data

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
        traction = mesh_with_traction.cell_data["traction"]
        areas = hip_load.areas
        total_force = np.sum(traction * areas[:, None], axis=0)

        # Should be close to expected world force (within numerical tolerance)
        assert_array_almost_equal(total_force, force_world_expected, decimal=1)


# ===== Edge cases and error handling =====

class TestLoadEdgeCases:
    """Test edge cases and error conditions."""

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

        assert "traction" in mesh_with_traction.cell_data
