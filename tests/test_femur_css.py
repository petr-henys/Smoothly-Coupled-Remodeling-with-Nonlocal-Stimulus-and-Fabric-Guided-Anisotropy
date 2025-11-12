"""
Test suite for femur_css.py module.

Tests coordinate system setup (CSS) for femur models including:
- JSON point loading
- Femoral head fitting
- Coordinate system construction
- Transformation matrices
- Vector transformations
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


import pyvista as pv
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from simulation.femur_css import FemurCSS, load_json_points, _fit_femoral_head, _unit


# Fixtures and test data generators
@pytest.fixture
def sample_json_points_file():
    """Create a temporary JSON file with sample control points."""
    data = {
        "markups": [{
            "controlPoints": [
                {"position": [10.0, 20.0, 30.0]},
                {"position": [15.0, 25.0, 35.0]},
                {"position": [20.0, 30.0, 40.0]}
            ]
        }]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    temp_path.unlink()


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
    # Two points roughly defining femoral head diameter
    return np.array([
        [-20.0, 0.0, 0.0],
        [20.0, 0.0, 0.0]
    ])


@pytest.fixture
def sample_le_me_line():
    """Sample lateral/medial epicondyle line."""
    # LE at negative Z (lateral), ME at positive Z (medial)
    return np.array([
        [0.0, -100.0, -30.0],  # Lateral epicondyle
        [0.0, -100.0, 30.0]    # Medial epicondyle
    ])


# Test helper functions
class TestHelperFunctions:
    """Test standalone helper functions."""
    
    def test_load_json_points_valid_file(self, sample_json_points_file):
        """Test loading points from valid JSON file."""
        points = load_json_points(sample_json_points_file)
        
        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 3)
        assert_array_almost_equal(points[0], [10.0, 20.0, 30.0])
        assert_array_almost_equal(points[1], [15.0, 25.0, 35.0])
        assert_array_almost_equal(points[2], [20.0, 30.0, 40.0])
    
    def test_load_json_points_missing_file(self):
        """Test loading points from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_json_points("non_existent_file.json")
    
    def test_unit_vector_normalization(self):
        """Test vector normalization."""
        # Test standard normalization
        v = np.array([3.0, 4.0, 0.0])
        v_unit = _unit(v)
        assert_almost_equal(np.linalg.norm(v_unit), 1.0)
        assert_array_almost_equal(v_unit, [0.6, 0.8, 0.0])
        
        # Test already normalized vector
        v_norm = np.array([1.0, 0.0, 0.0])
        assert_array_almost_equal(_unit(v_norm), v_norm)
    
    def test_unit_vector_zero_magnitude(self):
        """Test normalization of zero vector raises error."""
        v_zero = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Vector magnitude is zero"):
            _unit(v_zero)
    
    def test_unit_vector_tiny_magnitude(self):
        """Test normalization of near-zero vector raises error."""
        v_tiny = np.array([1e-10, 1e-10, 1e-10])
        with pytest.raises(ValueError, match="Vector magnitude is zero"):
            _unit(v_tiny)
    
    def test_fit_femoral_head(self, sample_femur_mesh, sample_head_line):
        """Test femoral head sphere fitting."""
        center, radius = _fit_femoral_head(sample_femur_mesh, sample_head_line)
        
        # Check return types
        assert isinstance(center, np.ndarray)
        assert center.shape == (3,)
        assert isinstance(radius, float)
        assert radius > 0
        
        # For our test sphere, center should be near origin
        assert_array_almost_equal(center, [0.0, 0.0, 0.0], decimal=1)
        # Radius should be reasonable for the fitted sphere
        assert 10 < radius < 20
    
    def test_fit_femoral_head_with_file_output(self, sample_femur_mesh, sample_head_line):
        """Test femoral head fitting with VTK file output."""
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            center, radius = _fit_femoral_head(
                sample_femur_mesh, sample_head_line, file=temp_path
            )
            
            # Check file was created
            assert temp_path.exists()
            
            # Load and verify sphere
            sphere = pv.read(str(temp_path))
            assert sphere.n_points > 0
            assert sphere.n_cells > 0
            
        finally:
            temp_path.unlink()


# Test FemurCSS class
class TestFemurCSS:
    """Test FemurCSS coordinate system class."""
    
    def test_init_valid_inputs(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test initialization with valid inputs."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="left")
        
        assert css.side == "left"
        assert isinstance(css.fhc, np.ndarray)
        assert css.fhc.shape == (3,)
        assert css.head_radius > 0
        assert hasattr(css, 'axes')
        assert set(css.axes.keys()) == {'x', 'y', 'z'}
    
    def test_init_invalid_side(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test initialization with invalid side parameter."""
        with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
            FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="middle")
    
    def test_coordinate_system_orthogonality(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test that coordinate system axes are orthonormal."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        x, y, z = css.axes['x'], css.axes['y'], css.axes['z']
        
        # Check unit vectors
        assert_almost_equal(np.linalg.norm(x), 1.0)
        assert_almost_equal(np.linalg.norm(y), 1.0)
        assert_almost_equal(np.linalg.norm(z), 1.0)
        
        # Check orthogonality
        assert_almost_equal(np.dot(x, y), 0.0, decimal=10)
        assert_almost_equal(np.dot(y, z), 0.0, decimal=10)
        assert_almost_equal(np.dot(x, z), 0.0, decimal=10)
        
        # Check right-handedness
        assert_array_almost_equal(np.cross(x, y), z)
        assert_array_almost_equal(np.cross(y, z), x)
        assert_array_almost_equal(np.cross(z, x), y)
    
    def test_side_dependent_axes(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test that coordinate axes differ between left and right sides."""
        css_left = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="left")
        css_right = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="right")
        
        # Y axes should be the same
        assert_array_almost_equal(css_left.axes['y'], css_right.axes['y'])
        
        # Z axes should be opposite
        assert_array_almost_equal(css_left.axes['z'], -css_right.axes['z'])
        
        # X axes should also differ (since X = Y × Z)
        assert not np.allclose(css_left.axes['x'], css_right.axes['x'])
    
    def test_transformation_matrix_properties(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test transformation matrix mathematical properties."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        # Get both transformation matrices
        M_to_css = css._transformation_matrix(world_to_css=True)
        M_to_world = css._transformation_matrix(world_to_css=False)
        
        # Check they are 4x4
        assert M_to_css.shape == (4, 4)
        assert M_to_world.shape == (4, 4)
        
        # Check they are inverses
        identity = M_to_css @ M_to_world
        assert_array_almost_equal(identity, np.eye(4))
        
        identity2 = M_to_world @ M_to_css
        assert_array_almost_equal(identity2, np.eye(4))
    
    def test_forward_inverse_transform_consistency(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test that forward and inverse transforms are consistent."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        # Transform mesh to CSS and back
        mesh_css = css.forward_transform(sample_femur_mesh)
        mesh_back = css.inverse_transform(mesh_css)
        
        # Points should match original (within numerical tolerance)
        assert_array_almost_equal(
            sample_femur_mesh.points, 
            mesh_back.points, 
            decimal=10
        )
    
    def test_vector_transformations(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test vector transformation methods."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        # Test with basis vectors
        for axis_name, axis_vec in css.axes.items():
            # World basis vector to CSS
            v_css = css.world_to_css_vector(axis_vec)
            
            # Should give standard basis in CSS
            expected = np.zeros(3)
            if axis_name == 'x':
                expected[0] = 1.0
            elif axis_name == 'y':
                expected[1] = 1.0
            else:  # z
                expected[2] = 1.0
            
            assert_array_almost_equal(v_css, expected)
            
            # And back
            v_world = css.css_to_world_vector(v_css)
            assert_array_almost_equal(v_world, axis_vec)
    
    def test_save_axes_vtk(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test saving coordinate axes to VTK file."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            css.save_axes_vtk(temp_path)
            
            # Check file exists
            assert temp_path.exists()
            
            # Load and verify
            axes_data = pv.read(str(temp_path))
            assert axes_data.n_points == 1
            assert 'x' in axes_data.array_names
            assert 'y' in axes_data.array_names
            assert 'z' in axes_data.array_names
            
            # Check axes match
            assert_array_almost_equal(axes_data['x'][0], css.axes['x'])
            assert_array_almost_equal(axes_data['y'][0], css.axes['y'])
            assert_array_almost_equal(axes_data['z'][0], css.axes['z'])
            
        finally:
            temp_path.unlink()
    
    def test_femoral_head_center_at_origin_in_css(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test that femoral head center is at origin in CSS coordinates."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        # Create a point at femoral head center
        fhc_point = pv.PolyData(css.fhc.reshape(1, 3))
        
        # Transform to CSS
        fhc_css = css.forward_transform(fhc_point)
        
        # Should be at origin
        assert_array_almost_equal(fhc_css.points[0], [0.0, 0.0, 0.0])


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow_with_head_sphere_output(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test complete workflow including sphere fitting output."""
        with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
            sphere_path = Path(f.name)
        
        try:
            # Initialize with sphere output
            css = FemurCSS(
                sample_femur_mesh, 
                sample_head_line, 
                sample_le_me_line,
                side="right",
                save_head_sphere=sphere_path
            )
            
            # Check sphere was saved
            assert sphere_path.exists()
            sphere = pv.read(str(sphere_path))
            assert sphere.n_points > 0
            
            # Verify coordinate system
            assert css.side == "right"
            assert hasattr(css, 'axes')
            
        finally:
            sphere_path.unlink()
    
    def test_mesh_transformation_preserves_properties(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        """Test that mesh transformations preserve geometric properties."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        # Original mesh properties
        orig_volume = sample_femur_mesh.volume
        orig_area = sample_femur_mesh.area
        orig_n_points = sample_femur_mesh.n_points
        orig_n_cells = sample_femur_mesh.n_cells
        
        # Transform to CSS
        mesh_css = css.forward_transform(sample_femur_mesh)
        
        # Check properties are preserved
        assert mesh_css.n_points == orig_n_points
        assert mesh_css.n_cells == orig_n_cells
        assert_almost_equal(mesh_css.volume, orig_volume, decimal=5)
        assert_almost_equal(mesh_css.area, orig_area, decimal=5)


# Parametrized tests
class TestParametrized:
    """Parametrized tests for various input combinations."""
    
    @pytest.mark.parametrize("side", ["left", "right"])
    def test_valid_sides(self, sample_femur_mesh, sample_head_line, sample_le_me_line, side):
        """Test initialization with both valid sides."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side=side)
        assert css.side == side.lower()
    
    @pytest.mark.parametrize("vector", [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([3.0, 4.0, 5.0]),
    ])
    def test_vector_roundtrip(self, sample_femur_mesh, sample_head_line, sample_le_me_line, vector):
        """Test vector transformation roundtrips."""
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)
        
        # World -> CSS -> World
        v_css = css.world_to_css_vector(vector)
        v_back = css.css_to_world_vector(v_css)
        assert_array_almost_equal(v_back, vector, decimal=10)
        
        # CSS -> World -> CSS
        v_world = css.css_to_world_vector(vector)
        v_back2 = css.world_to_css_vector(v_world)
        assert_array_almost_equal(v_back2, vector, decimal=10)


# Performance tests (optional, can be marked as slow)
class TestPerformance:
    """Performance tests for large meshes."""
    
    def test_large_mesh_transformation(self):
        """Test transformation performance with large mesh."""
        # Create a large mesh (100k points)
        large_mesh = pv.Sphere(theta_resolution=200, phi_resolution=200)
        head_line = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        le_me = np.array([[0.0, -2.0, -1.0], [0.0, -2.0, 1.0]])
        
        css = FemurCSS(large_mesh, head_line, le_me)
        
        # Time the transformation
        import time
        start = time.time()
        transformed = css.forward_transform(large_mesh)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
        assert transformed.n_points == large_mesh.n_points


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
