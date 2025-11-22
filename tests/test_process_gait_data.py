"""Unit and integration tests for `simulation.process_gait_data` utilities.

Split out from `test_gaitloader.py` for clarity.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from simulation.process_gait_data import (
    load_xy_datasets,
    segment_curves_grid,
    parse_hip_file,
    hip_to_xy_datasets,
    rescale_curve,
)


@pytest.fixture
def temp_hip_file(tmp_path):
    """Create minimal HIP file for testing."""
    content = """Peak Resultant Force F = 2500N
Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	500	1000	1500	2500	0.0
25	600	1100	1600	2400	0.25
50	700	1200	1700	2200	0.5
75	600	1100	1600	2400	0.75
100	500	1000	1500	2500	1.0
"""
    hip_file = tmp_path / "test_gait.HIP"
    hip_file.write_text(content)
    return hip_file


class TestLoadXYDatasets:
    """Tests for loading XY datasets from Excel files."""

    def test_load_single_dataset(self, tmp_path):
        df = pd.DataFrame({
            ("Dataset1", "X"): [0, 25, 50, 75, 100],
            ("Dataset1", "Y"): [100, 150, 200, 150, 100],
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine="openpyxl")

        datasets = load_xy_datasets(excel_file)

        assert "Dataset1" in datasets
        assert datasets["Dataset1"].shape == (5, 2)
        assert_array_almost_equal(datasets["Dataset1"][:, 0], [0, 25, 50, 75, 100])
        assert_array_almost_equal(datasets["Dataset1"][:, 1], [100, 150, 200, 150, 100])

    def test_load_multiple_datasets(self, tmp_path):
        df = pd.DataFrame({
            ("Dataset1", "X"): [0, 50, 100],
            ("Dataset1", "Y"): [10, 20, 30],
            ("Dataset2", "X"): [0, 50, 100],
            ("Dataset2", "Y"): [5, 15, 25],
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine="openpyxl")

        datasets = load_xy_datasets(excel_file)

        assert len(datasets) == 2
        assert "Dataset1" in datasets
        assert "Dataset2" in datasets

    def test_load_with_nan_values(self, tmp_path):
        df = pd.DataFrame({
            ("Dataset1", "X"): [0, 25, np.nan, 75, 100],
            ("Dataset1", "Y"): [100, 150, 200, np.nan, 100],
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine="openpyxl")

        datasets = load_xy_datasets(excel_file)

        assert datasets["Dataset1"].shape[0] < 5
        assert not np.any(np.isnan(datasets["Dataset1"]))


class TestSegmentCurvesGrid:
    """Tests for grid-based curve segmentation."""

    def test_single_curve_segmentation(self):
        points = np.array([
            [0.1, 0.0],
            [0.1, 0.5],
            [0.1, 1.0],
        ])

        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)

        assert len(curves) >= 1
        assert curves[0].shape[0] > 0

    def test_multiple_curves_segmentation(self):
        points = np.array([
            [0.1, 0.0],
            [0.1, 0.5],
            [0.1, 1.0],
            [0.9, 0.0],
            [0.9, 0.5],
            [0.9, 1.0],
        ])

        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)

        assert len(curves) >= 1

    def test_horizontal_segmentation(self):
        points = np.array([
            [0.0, 0.1],
            [0.5, 0.1],
            [1.0, 0.1],
            [0.0, 0.9],
            [0.5, 0.9],
            [1.0, 0.9],
        ])

        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=3)

        assert len(curves) >= 2

    def test_sorted_output(self):
        points = np.array([
            [0.5, 0.5],
            [0.1, 0.5],
            [0.9, 0.5],
            [0.3, 0.5],
        ])

        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)

        for curve in curves:
            if len(curve) > 1:
                x_coords = curve[:, 0]
                assert np.all(x_coords[:-1] <= x_coords[1:])

    def test_empty_cells_handled(self):
        points = np.array([
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ])

        curves = segment_curves_grid(points, n_vertical=3, m_horizontal=3)

        assert len(curves) <= 9
        assert all(len(curve) > 0 for curve in curves)

    def test_invalid_input_shape(self):
        points_1d = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="points must be \\(N,2\\)"):
            segment_curves_grid(points_1d, n_vertical=2, m_horizontal=2)

        points_3d = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        with pytest.raises(ValueError, match="points must be \\(N,2\\)"):
            segment_curves_grid(points_3d, n_vertical=2, m_horizontal=2)


class TestParseHIPFile:
    """Tests for HIP file parsing and physics validation."""

    def test_parse_valid_hip_file(self, temp_hip_file):
        result = parse_hip_file(temp_hip_file)

        assert "data" in result
        assert "metadata" in result
        assert "column_names" in result
        assert "physics_warnings" in result

        data = result["data"]
        assert data.shape[1] == 6
        assert data.shape[0] > 0

    def test_parse_hip_metadata(self, temp_hip_file):
        result = parse_hip_file(temp_hip_file)
        metadata = result["metadata"]
        assert "file_name" in metadata
        assert "peak_force" in metadata
        assert_almost_equal(metadata["peak_force"], 2500.0)

    def test_parse_hip_column_names(self, temp_hip_file):
        result = parse_hip_file(temp_hip_file)
        column_names = result["column_names"]
        assert "Cycle [%]" in column_names
        assert "Fx [N]" in column_names
        assert "Fy [N]" in column_names
        assert "Fz [N]" in column_names

    def test_physics_validation(self, tmp_path):
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	400	0.0
50	300	400	0	400	0.5
100	300	400	0	400	1.0
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)

        result = parse_hip_file(hip_file, validate_physics=True, fix_physics=False)
        assert len(result["physics_warnings"]) > 0
        assert any("violations" in warning.lower() for warning in result["physics_warnings"])

    def test_physics_fixing(self, tmp_path):
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	100	0.0
50	300	400	0	100	0.5
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)

        result = parse_hip_file(hip_file, fix_physics=True)

        data = result["data"]
        fx, fy, fz, f_calc = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        f_expected = np.sqrt(fx ** 2 + fy ** 2 + fz ** 2)

        assert_array_almost_equal(f_calc, f_expected, decimal=1)
        assert any("Recalculated" in warning for warning in result["physics_warnings"])

    def test_parse_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_hip_file("nonexistent.HIP")

    def test_parse_no_data_section(self, tmp_path):
        content = """Some header text
But no Cycle [%] line
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)

        with pytest.raises(ValueError, match="Could not find data section"):
            parse_hip_file(hip_file)

    def test_parse_no_valid_data_rows(self, tmp_path):
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
invalid	data	here
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)

        with pytest.raises(ValueError, match="No valid data rows found"):
            parse_hip_file(hip_file)

    def test_parse_incomplete_rows(self, tmp_path):
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	500	0.0
50	300	400		 	0.5
100	300	400	0	500	1.0
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)

        result = parse_hip_file(hip_file)
        assert result["data"].shape[0] == 2

    def test_physics_within_tolerance(self, tmp_path):
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	500.0	0.0
50	300	400	0	500.0	0.5
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)

        result = parse_hip_file(hip_file, validate_physics=True, fix_physics=False)
        violation_warnings = [
            w for w in result["physics_warnings"] if "violations" in w.lower()
        ]
        assert len(violation_warnings) == 0


class TestHipToXYDatasets:
    """Tests for conversion of HIP data to XY datasets."""

    def test_default_components(self, temp_hip_file):
        datasets = hip_to_xy_datasets(temp_hip_file)

        assert "Fx_vs_Cycle" in datasets
        assert "Fy_vs_Cycle" in datasets
        assert "Fz_vs_Cycle" in datasets
        assert "F_vs_Cycle" in datasets

    def test_specific_components(self, temp_hip_file):
        datasets = hip_to_xy_datasets(temp_hip_file, components=["Fx", "F"])

        assert "Fx_vs_Cycle" in datasets
        assert "F_vs_Cycle" in datasets
        assert "Fy_vs_Cycle" not in datasets
        assert "Fz_vs_Cycle" not in datasets

    def test_vs_time_axis(self, temp_hip_file):
        datasets = hip_to_xy_datasets(temp_hip_file, vs_time=True)

        assert "Fx_vs_Time" in datasets
        assert "Fy_vs_Time" in datasets

        fx_data = datasets["Fx_vs_Time"]
        assert fx_data[0, 0] == 0.0

    def test_unknown_component_warning(self, temp_hip_file, caplog):
        datasets = hip_to_xy_datasets(temp_hip_file, components=["Fx", "UnknownComponent"])

        assert "Fx_vs_Cycle" in datasets
        assert "UnknownComponent_vs_Cycle" not in datasets

    def test_dataset_structure(self, temp_hip_file):
        datasets = hip_to_xy_datasets(temp_hip_file, components=["Fx"])

        fx_data = datasets["Fx_vs_Cycle"]
        assert fx_data.ndim == 2
        assert fx_data.shape[1] == 2
        assert fx_data.shape[0] > 0


class TestRescaleCurve:
    """Tests for curve rescaling functionality."""

    def test_rescale_to_unit_square(self):
        points = np.array([[0, 0], [10, 5], [20, 10]])
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))

        assert_almost_equal(rescaled[0, 0], 0.0)
        assert_almost_equal(rescaled[-1, 0], 1.0)
        assert_almost_equal(rescaled[0, 1], 0.0)
        assert_almost_equal(rescaled[-1, 1], 1.0)

    def test_rescale_custom_ranges(self):
        points = np.array([[0, 0], [1, 1]])
        rescaled = rescale_curve(points, x_scale=(-5, 5), y_scale=(10, 20))

        assert_almost_equal(rescaled[0, 0], -5.0)
        assert_almost_equal(rescaled[-1, 0], 5.0)
        assert_almost_equal(rescaled[0, 1], 10.0)
        assert_almost_equal(rescaled[-1, 1], 20.0)

    def test_rescale_negative_to_positive(self):
        points = np.array([[-10, -5], [0, 0], [10, 5]])
        rescaled = rescale_curve(points, x_scale=(0, 100), y_scale=(0, 50))

        assert_almost_equal(rescaled[1, 0], 50.0)
        assert_almost_equal(rescaled[1, 1], 25.0)

    def test_rescale_preserves_relative_positions(self):
        points = np.array([[0, 0], [5, 10], [10, 20]])
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))

        assert_almost_equal(rescaled[1, 0], 0.5)
        assert_almost_equal(rescaled[1, 1], 0.5)


class TestGaitProcessingIntegration:
    """Simple integration tests combining multiple utilities."""

    def test_complete_hip_processing_workflow(self, temp_hip_file):
        result = parse_hip_file(temp_hip_file, fix_physics=True)
        assert result["data"].shape[0] > 0

        datasets = hip_to_xy_datasets(temp_hip_file, components=["F"])
        assert "F_vs_Cycle" in datasets

        f_data = datasets["F_vs_Cycle"]
        rescaled = rescale_curve(f_data, x_scale=(0, 1), y_scale=(0, 1))

        assert rescaled.shape == f_data.shape
        assert np.all((0.0 <= rescaled[:, 0]) & (rescaled[:, 0] <= 1.0))
        assert np.all((0.0 <= rescaled[:, 1]) & (rescaled[:, 1] <= 1.0))

    def test_excel_to_curve_segmentation(self, tmp_path):
        df = pd.DataFrame({
            ("Dataset", "X"): np.random.uniform(0, 1, 50),
            ("Dataset", "Y"): np.random.uniform(0, 1, 50),
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        excel_file = tmp_path / "test.xlsx"
        df.to_excel(excel_file, index=True, engine="openpyxl")

        datasets = load_xy_datasets(excel_file)
        points = datasets["Dataset"]

        curves = segment_curves_grid(points, n_vertical=3, m_horizontal=3)

        assert len(curves) > 0
        assert all(curve.shape[1] == 2 for curve in curves)


class TestGaitProcessingEdgeCases:
    """Edge cases and boundary conditions for gait processing utilities."""

    def test_segment_single_point(self):
        points = np.array([[0.5, 0.5]])
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)

        assert len(curves) >= 1
        assert curves[0].shape[0] == 1
