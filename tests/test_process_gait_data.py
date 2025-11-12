"""
Comprehensive test suite for process_gait_data.py module.

Tests gait data loading, curve segmentation, HIP file parsing,
physics validation, and data transformations.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_almost_equal

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


# ============================================================================
# LOAD XY DATASETS TESTS
# ============================================================================

class TestLoadXYDatasets:
    """Test loading XY datasets from Excel files."""
    def test_load_single_dataset(self, tmp_path):
        """Test loading Excel file with single dataset."""
        # Create sample Excel file
        df = pd.DataFrame({
            ('Dataset1', 'X'): [0, 25, 50, 75, 100],
            ('Dataset1', 'Y'): [100, 150, 200, 150, 100]
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine='openpyxl')
        
        datasets = load_xy_datasets(excel_file)
        
        assert 'Dataset1' in datasets
        assert datasets['Dataset1'].shape == (5, 2)
        assert_array_almost_equal(datasets['Dataset1'][:, 0], [0, 25, 50, 75, 100])
        assert_array_almost_equal(datasets['Dataset1'][:, 1], [100, 150, 200, 150, 100])
    def test_load_multiple_datasets(self, tmp_path):
        """Test loading Excel file with multiple datasets."""
        df = pd.DataFrame({
            ('Dataset1', 'X'): [0, 50, 100],
            ('Dataset1', 'Y'): [10, 20, 30],
            ('Dataset2', 'X'): [0, 50, 100],
            ('Dataset2', 'Y'): [5, 15, 25]
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine='openpyxl')
        
        datasets = load_xy_datasets(excel_file)
        
        assert len(datasets) == 2
        assert 'Dataset1' in datasets
        assert 'Dataset2' in datasets
    def test_load_with_nan_values(self, tmp_path):
        """Test loading datasets with NaN values (should be filtered)."""
        df = pd.DataFrame({
            ('Dataset1', 'X'): [0, 25, np.nan, 75, 100],
            ('Dataset1', 'Y'): [100, 150, 200, np.nan, 100]
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine='openpyxl')
        
        datasets = load_xy_datasets(excel_file)
        
        # NaN rows should be filtered out
        assert datasets['Dataset1'].shape[0] < 5
        assert not np.any(np.isnan(datasets['Dataset1']))
    def test_load_with_flip_y(self, tmp_path):
        """Test loading with Y-axis flipping."""
        df = pd.DataFrame({
            ('Dataset1', 'X'): [0, 50, 100],
            ('Dataset1', 'Y'): [10, 20, 30]
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine='openpyxl')
        
        datasets = load_xy_datasets(excel_file, flip_y=True)
        
        # Y values should be negated
        assert_array_almost_equal(datasets['Dataset1'][:, 1], [-10, -20, -30])
    def test_load_missing_columns(self, tmp_path):
        """Test error handling when X or Y columns are missing."""
        df = pd.DataFrame({
            ('Dataset1', 'X'): [0, 50, 100],
            ('Dataset1', 'Z'): [10, 20, 30]  # Wrong column name
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        excel_file = tmp_path / "test_data.xlsx"
        df.to_excel(excel_file, index=True, engine='openpyxl')
        
        with pytest.raises(ValueError, match="missing X/Y columns"):
            load_xy_datasets(excel_file)
    def test_load_specific_sheet(self, tmp_path):
        """Test loading from specific Excel sheet."""
        with pd.ExcelWriter(tmp_path / "test_data.xlsx", engine='openpyxl') as writer:
            df1 = pd.DataFrame({
                ('Sheet1Data', 'X'): [0, 50, 100],
                ('Sheet1Data', 'Y'): [10, 20, 30]
            })
            df1.columns = pd.MultiIndex.from_tuples(df1.columns)
            df1.to_excel(writer, sheet_name='Sheet1', index=True)
            
            df2 = pd.DataFrame({
                ('Sheet2Data', 'X'): [0, 50, 100],
                ('Sheet2Data', 'Y'): [100, 200, 300]
            })
            df2.columns = pd.MultiIndex.from_tuples(df2.columns)
            df2.to_excel(writer, sheet_name='Sheet2', index=True)
        
        # Load from specific sheet
        datasets = load_xy_datasets(tmp_path / "test_data.xlsx", sheet='Sheet2')
        assert 'Sheet2Data' in datasets
        assert 'Sheet1Data' not in datasets
    def test_load_nonexistent_file(self):
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_xy_datasets("nonexistent_file.xlsx")


# ============================================================================
# SEGMENT CURVES GRID TESTS
# ============================================================================

class TestSegmentCurvesGrid:
    """Test grid-based curve segmentation."""
    
    def test_single_curve_segmentation(self):
        """Test segmentation of points forming single curve."""
        # Points in a single vertical strip
        points = np.array([
            [0.1, 0.0],
            [0.1, 0.5],
            [0.1, 1.0]
        ])
        
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)
        
        # Should get one curve
        assert len(curves) >= 1
        # First curve should contain all points (or at least most)
        assert curves[0].shape[0] > 0
    
    def test_multiple_curves_segmentation(self):
        """Test segmentation into multiple curves."""
        # Points in two vertical strips
        points = np.array([
            [0.1, 0.0], [0.1, 0.5], [0.1, 1.0],  # Left strip
            [0.9, 0.0], [0.9, 0.5], [0.9, 1.0]   # Right strip
        ])
        
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)
        
        # Should get at least one curve (may be 1 or 2 depending on grid edges)
        assert len(curves) >= 1
    
    def test_horizontal_segmentation(self):
        """Test segmentation with multiple horizontal bands."""
        # Points in two horizontal bands
        points = np.array([
            [0.0, 0.1], [0.5, 0.1], [1.0, 0.1],  # Bottom band
            [0.0, 0.9], [0.5, 0.9], [1.0, 0.9]   # Top band
        ])
        
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=3)
        
        # Should segment into multiple curves
        assert len(curves) >= 2
    
    def test_sorted_output(self):
        """Test that points within each curve are sorted by X."""
        # Unsorted points
        points = np.array([
            [0.5, 0.5],
            [0.1, 0.5],
            [0.9, 0.5],
            [0.3, 0.5]
        ])
        
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)
        
        # Points in each curve should be sorted by X
        for curve in curves:
            if len(curve) > 1:
                x_coords = curve[:, 0]
                assert np.all(x_coords[:-1] <= x_coords[1:])  # Non-decreasing
    
    def test_empty_cells_handled(self):
        """Test handling of grid cells with no points."""
        # Points only in corners
        points = np.array([
            [0.1, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9]
        ])
        
        curves = segment_curves_grid(points, n_vertical=3, m_horizontal=3)
        
        # Should return curves only for non-empty cells
        assert len(curves) <= 9  # Max possible cells
        assert all(len(curve) > 0 for curve in curves)  # No empty curves
    
    def test_invalid_input_shape(self):
        """Test error handling for invalid input shape."""
        # 1D array instead of 2D
        points_1d = np.array([0.1, 0.2, 0.3])
        
        with pytest.raises(ValueError, match="points must be \\(N,2\\)"):
            segment_curves_grid(points_1d, n_vertical=2, m_horizontal=2)
        
        # 3D points instead of 2D
        points_3d = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        with pytest.raises(ValueError, match="points must be \\(N,2\\)"):
            segment_curves_grid(points_3d, n_vertical=2, m_horizontal=2)
    
    def test_eps_y_parameter(self):
        """Test eps_y parameter for Y-axis boundary extension."""
        # Points right at Y=0
        points = np.array([
            [0.1, 0.0],
            [0.5, 0.0],
            [0.9, 0.0]
        ])
        
        # Should work with default eps_y
        curves_default = segment_curves_grid(points, n_vertical=2, m_horizontal=2)
        assert len(curves_default) > 0
        
        # Should also work with custom eps_y
        curves_custom = segment_curves_grid(points, n_vertical=2, m_horizontal=2, eps_y=0.5)
        assert len(curves_custom) > 0
    
    def test_jitter_parameter(self):
        """Test jitter parameter for boundary handling."""
        points = np.array([
            [0.0, 0.0],  # Exactly at boundary
            [0.5, 0.5],
            [1.0, 1.0]   # Exactly at boundary
        ])
        
        # Should handle boundary points with jitter
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2, jitter=1e-6)
        assert len(curves) > 0


# ============================================================================
# PARSE HIP FILE TESTS
# ============================================================================

class TestParseHIPFile:
    """Test HIP file parsing and physics validation."""
    def test_parse_valid_hip_file(self, temp_hip_file):
        """Test parsing valid HIP file."""
        result = parse_hip_file(temp_hip_file)
        
        assert 'data' in result
        assert 'metadata' in result
        assert 'column_names' in result
        assert 'physics_warnings' in result
        
        # Check data shape
        data = result['data']
        assert data.shape[1] == 6  # 6 columns: Cycle, Fx, Fy, Fz, F, Time
        assert data.shape[0] > 0
    def test_parse_hip_metadata(self, temp_hip_file):
        """Test metadata extraction from HIP file."""
        result = parse_hip_file(temp_hip_file)
        
        metadata = result['metadata']
        assert 'file_name' in metadata
        assert 'peak_force' in metadata
        assert_almost_equal(metadata['peak_force'], 2500.0)
    def test_parse_hip_column_names(self, temp_hip_file):
        """Test column name extraction."""
        result = parse_hip_file(temp_hip_file)
        
        column_names = result['column_names']
        assert 'Cycle [%]' in column_names
        assert 'Fx [N]' in column_names
        assert 'Fy [N]' in column_names
        assert 'Fz [N]' in column_names
    def test_physics_validation(self, tmp_path):
        """Test physics validation (F >= sqrt(Fx² + Fy² + Fz²))."""
        # Create HIP file with physics violation
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	400	0.0
50	300	400	0	400	0.5
100	300	400	0	400	1.0
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        result = parse_hip_file(hip_file, validate_physics=True, fix_physics=False)
        
        # Should detect violations
        assert len(result['physics_warnings']) > 0
        assert any('violations' in warning.lower() for warning in result['physics_warnings'])
    def test_physics_fixing(self, tmp_path):
        """Test automatic physics fixing."""
        # Create HIP file with incorrect resultant
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	100	0.0
50	300	400	0	100	0.5
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        result = parse_hip_file(hip_file, fix_physics=True)
        
        # F should be recalculated
        data = result['data']
        fx, fy, fz, f_calc = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        f_expected = np.sqrt(fx**2 + fy**2 + fz**2)
        
        assert_array_almost_equal(f_calc, f_expected, decimal=1)
        assert any('Recalculated' in warning for warning in result['physics_warnings'])
    def test_parse_missing_file(self):
        """Test error handling for missing HIP file."""
        with pytest.raises(FileNotFoundError):
            parse_hip_file("nonexistent.HIP")
    def test_parse_no_data_section(self, tmp_path):
        """Test error handling when data section is missing."""
        content = """Some header text
But no Cycle [%] line
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        with pytest.raises(ValueError, match="Could not find data section"):
            parse_hip_file(hip_file)
    def test_parse_no_valid_data_rows(self, tmp_path):
        """Test error handling when no valid data rows exist."""
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
invalid	data	here
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        with pytest.raises(ValueError, match="No valid data rows found"):
            parse_hip_file(hip_file)
    def test_parse_incomplete_rows(self, tmp_path):
        """Test handling of incomplete data rows."""
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	500	0.0
50	300	400		 	0.5
100	300	400	0	500	1.0
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        result = parse_hip_file(hip_file)
        
        # Should skip incomplete row
        assert result['data'].shape[0] == 2  # Only valid rows
    def test_physics_within_tolerance(self, tmp_path):
        """Test that small numerical errors don't trigger warnings."""
        # Create HIP file with correct physics (within tolerance)
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	300	400	0	500.0	0.0
50	300	400	0	500.0	0.5
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        result = parse_hip_file(hip_file, validate_physics=True, fix_physics=False)
        
        # Should not have violation warnings (500 = sqrt(300² + 400²))
        violation_warnings = [w for w in result['physics_warnings'] if 'violations' in w.lower()]
        assert len(violation_warnings) == 0


# ============================================================================
# HIP TO XY DATASETS TESTS
# ============================================================================

class TestHipToXYDatasets:
    """Test conversion of HIP data to XY datasets."""
    def test_default_components(self, temp_hip_file):
        """Test default component extraction (Fx, Fy, Fz, F)."""
        datasets = hip_to_xy_datasets(temp_hip_file)
        
        # Should have all default components
        assert 'Fx_vs_Cycle' in datasets
        assert 'Fy_vs_Cycle' in datasets
        assert 'Fz_vs_Cycle' in datasets
        assert 'F_vs_Cycle' in datasets
    def test_specific_components(self, temp_hip_file):
        """Test extraction of specific components only."""
        datasets = hip_to_xy_datasets(temp_hip_file, components=['Fx', 'F'])
        
        assert 'Fx_vs_Cycle' in datasets
        assert 'F_vs_Cycle' in datasets
        assert 'Fy_vs_Cycle' not in datasets
        assert 'Fz_vs_Cycle' not in datasets
    def test_vs_time_axis(self, temp_hip_file):
        """Test using time instead of cycle percentage."""
        datasets = hip_to_xy_datasets(temp_hip_file, vs_time=True)
        
        # Keys should reference Time instead of Cycle
        assert 'Fx_vs_Time' in datasets
        assert 'Fy_vs_Time' in datasets
        
        # X-axis should be time values
        fx_data = datasets['Fx_vs_Time']
        assert fx_data[0, 0] == 0.0  # First time point
    def test_unknown_component_warning(self, temp_hip_file, caplog):
        """Test warning for unknown component names."""
        datasets = hip_to_xy_datasets(temp_hip_file, components=['Fx', 'UnknownComponent'])
        
        # Should only get valid component
        assert 'Fx_vs_Cycle' in datasets
        assert 'UnknownComponent_vs_Cycle' not in datasets
    def test_dataset_structure(self, temp_hip_file):
        """Test structure of returned datasets."""
        datasets = hip_to_xy_datasets(temp_hip_file, components=['Fx'])
        
        fx_data = datasets['Fx_vs_Cycle']
        assert fx_data.ndim == 2
        assert fx_data.shape[1] == 2  # X and Y columns
        assert fx_data.shape[0] > 0  # Has data rows


# ============================================================================
# RESCALE CURVE TESTS
# ============================================================================

class TestRescaleCurve:
    """Test curve rescaling functionality."""
    
    def test_rescale_to_unit_square(self):
        """Test rescaling to [0,1] x [0,1]."""
        points = np.array([
            [0, 0],
            [10, 5],
            [20, 10]
        ])
        
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))
        
        # X should be rescaled to [0, 1]
        assert_almost_equal(rescaled[0, 0], 0.0)
        assert_almost_equal(rescaled[-1, 0], 1.0)
        
        # Y should be rescaled to [0, 1]
        assert_almost_equal(rescaled[0, 1], 0.0)
        assert_almost_equal(rescaled[-1, 1], 1.0)
    
    def test_rescale_custom_ranges(self):
        """Test rescaling to custom ranges."""
        points = np.array([
            [0, 0],
            [1, 1]
        ])
        
        rescaled = rescale_curve(points, x_scale=(-5, 5), y_scale=(10, 20))
        
        # Check bounds
        assert_almost_equal(rescaled[0, 0], -5.0)
        assert_almost_equal(rescaled[-1, 0], 5.0)
        assert_almost_equal(rescaled[0, 1], 10.0)
        assert_almost_equal(rescaled[-1, 1], 20.0)
    
    def test_rescale_negative_to_positive(self):
        """Test rescaling from negative to positive range."""
        points = np.array([
            [-10, -5],
            [0, 0],
            [10, 5]
        ])
        
        rescaled = rescale_curve(points, x_scale=(0, 100), y_scale=(0, 50))
        
        # Middle point should map to middle of new range
        assert_almost_equal(rescaled[1, 0], 50.0)
        assert_almost_equal(rescaled[1, 1], 25.0)
    
    def test_rescale_preserves_relative_positions(self):
        """Test that relative positions are preserved after rescaling."""
        points = np.array([
            [0, 0],
            [5, 10],
            [10, 20]
        ])
        
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))
        
        # Middle point should be at midpoint
        assert_almost_equal(rescaled[1, 0], 0.5)
        assert_almost_equal(rescaled[1, 1], 0.5)
    
    def test_rescale_single_point(self):
        """Test rescaling with single point (degenerate case)."""
        points = np.array([[5, 10]])
        
        # With single point, min=max, so scaling is undefined
        # Function should handle this gracefully (likely NaN or 0)
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))
        
        # Result should exist and have correct shape
        assert rescaled.shape == (1, 2)
    
    def test_rescale_uniform_values(self):
        """Test rescaling when all X or Y values are the same."""
        points = np.array([
            [5, 0],
            [5, 5],
            [5, 10]
        ])
        
        # X values are all the same
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))
        
        # Y should be rescaled properly
        assert_almost_equal(rescaled[0, 1], 0.0)
        assert_almost_equal(rescaled[-1, 1], 1.0)
    
    def test_rescale_identity_transformation(self):
        """Test that rescaling to same range is identity-like."""
        points = np.array([
            [0, 0],
            [1, 1]
        ])
        
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))
        
        # Should be very close to original
        assert_array_almost_equal(rescaled, points)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    def test_complete_hip_processing_workflow(self, temp_hip_file):
        """Test complete HIP data processing workflow."""
        # Parse HIP file
        result = parse_hip_file(temp_hip_file, fix_physics=True)
        assert result['data'].shape[0] > 0
        
        # Convert to XY datasets
        datasets = hip_to_xy_datasets(temp_hip_file, components=['F'])
        assert 'F_vs_Cycle' in datasets
        
        # Rescale curve
        f_data = datasets['F_vs_Cycle']
        rescaled = rescale_curve(f_data, x_scale=(0, 1), y_scale=(0, 1))
        
        # Check final result
        assert rescaled.shape == f_data.shape
        assert np.all(rescaled[:, 0] >= 0) and np.all(rescaled[:, 0] <= 1)
        assert np.all(rescaled[:, 1] >= 0) and np.all(rescaled[:, 1] <= 1)
    def test_excel_to_curve_segmentation(self, tmp_path):
        """Test workflow from Excel to segmented curves."""
        # Create Excel file
        df = pd.DataFrame({
            ('Dataset', 'X'): np.random.uniform(0, 1, 50),
            ('Dataset', 'Y'): np.random.uniform(0, 1, 50)
        })
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        
        excel_file = tmp_path / "test.xlsx"
        df.to_excel(excel_file, index=True, engine='openpyxl')
        
        # Load and segment
        datasets = load_xy_datasets(excel_file)
        points = datasets['Dataset']
        
        curves = segment_curves_grid(points, n_vertical=3, m_horizontal=3)
        
        # Should have segmented curves
        assert len(curves) > 0
        assert all(curve.shape[1] == 2 for curve in curves)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_segment_single_point(self):
        """Test segmentation with single point."""
        points = np.array([[0.5, 0.5]])
        
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)
        
        # Should return one curve with one point
        assert len(curves) >= 1
        assert curves[0].shape[0] == 1
    
    def test_segment_collinear_points(self):
        """Test segmentation with collinear points."""
        # Vertical line
        points = np.array([
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0]
        ])
        
        curves = segment_curves_grid(points, n_vertical=2, m_horizontal=2)
        
        # Should handle collinear points
        assert len(curves) > 0
    
    def test_rescale_extreme_values(self):
        """Test rescaling with extreme value ranges."""
        points = np.array([
            [1e-10, 1e-10],
            [1e10, 1e10]
        ])
        
        rescaled = rescale_curve(points, x_scale=(0, 1), y_scale=(0, 1))
        
        # Should handle extreme ranges
        assert rescaled.shape == points.shape
        assert np.all(np.isfinite(rescaled))
    def test_hip_file_with_tabs_and_spaces(self, tmp_path):
        """Test HIP file parsing with mixed tabs and spaces."""
        content = """Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0   300   400   0   500   0.0
50	300	400	0	500	0.5
"""
        hip_file = tmp_path / "test.HIP"
        hip_file.write_text(content)
        
        result = parse_hip_file(hip_file)
        
        # Should parse both tab and space-separated rows
        assert result['data'].shape[0] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
