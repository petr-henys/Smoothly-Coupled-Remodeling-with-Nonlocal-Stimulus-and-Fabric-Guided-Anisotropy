"""Tests for `simulation.femur_css` (Femur anatomical CSS utilities).

Split out from `test_gaitloader.py` for clarity.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from simulation.femur_css import FemurCSS, load_json_points, _fit_femoral_head, _unit


@pytest.fixture
def sample_json_points_file():
    """Create a temporary JSON file with sample control points."""
    data = {
        "markups": [
            {
                "controlPoints": [
                    {"position": [10.0, 20.0, 30.0]},
                    {"position": [15.0, 25.0, 35.0]},
                    {"position": [20.0, 30.0, 40.0]},
                ]
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    yield temp_path
    temp_path.unlink()


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


class TestHelperFunctions:
    """Standalone helper-function tests for femur CSS utilities."""

    def test_load_json_points_valid_file(self, sample_json_points_file):
        points = load_json_points(sample_json_points_file)

        assert isinstance(points, np.ndarray)
        assert points.shape == (3, 3)
        assert_array_almost_equal(points[0], [10.0, 20.0, 30.0])
        assert_array_almost_equal(points[1], [15.0, 25.0, 35.0])
        assert_array_almost_equal(points[2], [20.0, 30.0, 40.0])

    def test_load_json_points_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_json_points("non_existent_file.json")

    def test_unit_vector_normalization(self):
        v = np.array([3.0, 4.0, 0.0])
        v_unit = _unit(v)
        assert_almost_equal(np.linalg.norm(v_unit), 1.0)
        assert_array_almost_equal(v_unit, [0.6, 0.8, 0.0])

        v_norm = np.array([1.0, 0.0, 0.0])
        assert_array_almost_equal(_unit(v_norm), v_norm)

    def test_unit_vector_zero_magnitude(self):
        v_zero = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="Vector magnitude is zero"):
            _unit(v_zero)

    def test_unit_vector_tiny_magnitude(self):
        v_tiny = np.array([1e-10, 1e-10, 1e-10])
        with pytest.raises(ValueError, match="Vector magnitude is zero"):
            _unit(v_tiny)

    def test_fit_femoral_head(self, sample_femur_mesh, sample_head_line):
        center, radius = _fit_femoral_head(sample_femur_mesh, sample_head_line)

        assert isinstance(center, np.ndarray)
        assert center.shape == (3,)
        assert isinstance(radius, float)
        assert radius > 0

        assert_array_almost_equal(center, [0.0, 0.0, 0.0], decimal=1)
        assert 10 < radius < 20

    def test_fit_femoral_head_with_file_output(self, sample_femur_mesh, sample_head_line):
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
            temp_path = Path(f.name)

        center, radius = _fit_femoral_head(sample_femur_mesh, sample_head_line, file=temp_path)

        assert temp_path.exists()
        sphere = pv.read(str(temp_path))
        assert sphere.n_points > 0
        assert sphere.n_cells > 0

        temp_path.unlink()


@pytest.mark.slow
class TestFemurCSS:
    """Tests for the `FemurCSS` coordinate system class."""

    def test_init_valid_inputs(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="left")

        assert css.side == "left"
        assert isinstance(css.fhc, np.ndarray)
        assert css.fhc.shape == (3,)
        assert css.head_radius > 0
        assert hasattr(css, "axes")
        assert set(css.axes.keys()) == {"x", "y", "z"}

    def test_init_invalid_side(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
            FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="middle")

    def test_coordinate_system_orthogonality(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        x, y, z = css.axes["x"], css.axes["y"], css.axes["z"]

        assert_almost_equal(np.linalg.norm(x), 1.0)
        assert_almost_equal(np.linalg.norm(y), 1.0)
        assert_almost_equal(np.linalg.norm(z), 1.0)

        assert_almost_equal(np.dot(x, y), 0.0, decimal=10)
        assert_almost_equal(np.dot(y, z), 0.0, decimal=10)
        assert_almost_equal(np.dot(x, z), 0.0, decimal=10)

        assert_array_almost_equal(np.cross(x, y), z)
        assert_array_almost_equal(np.cross(y, z), x)
        assert_array_almost_equal(np.cross(z, x), y)

    def test_side_dependent_axes(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css_left = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="left")
        css_right = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side="right")

        assert_array_almost_equal(css_left.axes["y"], css_right.axes["y"])
        assert_array_almost_equal(css_left.axes["z"], -css_right.axes["z"])
        assert not np.allclose(css_left.axes["x"], css_right.axes["x"])

    def test_transformation_matrix_properties(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        M_to_css = css._transformation_matrix(world_to_css=True)
        M_to_world = css._transformation_matrix(world_to_css=False)

        assert M_to_css.shape == (4, 4)
        assert M_to_world.shape == (4, 4)

        identity = M_to_css @ M_to_world
        assert_array_almost_equal(identity, np.eye(4))

        identity2 = M_to_world @ M_to_css
        assert_array_almost_equal(identity2, np.eye(4))

    def test_forward_inverse_transform_consistency(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        mesh_css = css.forward_transform(sample_femur_mesh)
        mesh_back = css.inverse_transform(mesh_css)

        assert_array_almost_equal(sample_femur_mesh.points, mesh_back.points, decimal=10)

    def test_vector_transformations(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        for axis_name, axis_vec in css.axes.items():
            v_css = css.world_to_css_vector(axis_vec)

            expected = np.zeros(3)
            if axis_name == "x":
                expected[0] = 1.0
            elif axis_name == "y":
                expected[1] = 1.0
            else:
                expected[2] = 1.0

            assert_array_almost_equal(v_css, expected)

            v_world = css.css_to_world_vector(v_css)
            assert_array_almost_equal(v_world, axis_vec)

    def test_save_axes_vtk(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
            temp_path = Path(f.name)

        css.save_axes_vtk(temp_path)

        assert temp_path.exists()

        axes_data = pv.read(str(temp_path))
        assert axes_data.n_points == 1
        assert "x" in axes_data.array_names
        assert "y" in axes_data.array_names
        assert "z" in axes_data.array_names

        assert_array_almost_equal(axes_data["x"][0], css.axes["x"])
        assert_array_almost_equal(axes_data["y"][0], css.axes["y"])
        assert_array_almost_equal(axes_data["z"][0], css.axes["z"])

        temp_path.unlink()

    def test_femoral_head_center_at_origin_in_css(self, sample_femur_mesh, sample_head_line, sample_le_me_line):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        fhc_point = pv.PolyData(css.fhc.reshape(1, 3))
        fhc_css = css.forward_transform(fhc_point)

        assert_array_almost_equal(fhc_css.points[0], [0.0, 0.0, 0.0])


class TestFemurCSSIntegration:
    """Integration tests for complete CSS workflows."""

    def test_complete_workflow_with_head_sphere_output(
        self,
        sample_femur_mesh,
        sample_head_line,
        sample_le_me_line,
    ):
        with tempfile.NamedTemporaryFile(suffix=".vtk", delete=False) as f:
            sphere_path = Path(f.name)

        css = FemurCSS(
            sample_femur_mesh,
            sample_head_line,
            sample_le_me_line,
            side="right",
            save_head_sphere=sphere_path,
        )

        assert sphere_path.exists()
        sphere = pv.read(str(sphere_path))
        assert sphere.n_points > 0

        assert css.side == "right"
        assert hasattr(css, "axes")

        sphere_path.unlink()

    def test_mesh_transformation_preserves_properties(
        self,
        sample_femur_mesh,
        sample_head_line,
        sample_le_me_line,
    ):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line)

        orig_volume = sample_femur_mesh.volume
        orig_area = sample_femur_mesh.area
        orig_n_points = sample_femur_mesh.n_points
        orig_n_cells = sample_femur_mesh.n_cells

        mesh_css = css.forward_transform(sample_femur_mesh)

        assert mesh_css.n_points == orig_n_points
        assert mesh_css.n_cells == orig_n_cells
        assert_almost_equal(mesh_css.volume, orig_volume, decimal=5)
        assert_almost_equal(mesh_css.area, orig_area, decimal=5)


class TestFemurCSSParametrized:
    """Parametrized tests for different `FemurCSS` inputs."""

    @pytest.mark.parametrize("side", ["left", "right"])
    def test_valid_sides(self, sample_femur_mesh, sample_head_line, sample_le_me_line, side):
        css = FemurCSS(sample_femur_mesh, sample_head_line, sample_le_me_line, side=side)
        assert css.side == side
