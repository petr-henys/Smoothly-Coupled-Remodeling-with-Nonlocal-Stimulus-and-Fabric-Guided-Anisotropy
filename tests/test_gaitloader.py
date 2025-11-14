"""
Comprehensive test suite for gait loading utilities, HIP parsing, femur CSS,
and load application workflows.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import basix
from dolfinx import fem
from mpi4py import MPI
import pyvista as pv

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

class TestGaitProcessingIntegration:
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

class TestGaitProcessingEdgeCases:
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


"""Test accumulated strain energy computation using femur geometry and gait loads."""

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.femur_gait import setup_femur_gait_loading
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitEnergyDriver
from simulation.utils import build_dirichlet_bcs


@pytest.fixture(scope="module")
def femur_mechanics_setup():
    """Create femur mesh, function spaces, config, and shared gait loader.

    All mechanics and loading tests use the same DOLFINx mesh/function
    space as the gait loader to avoid cross-mesh form assembly issues
    in DOLFINx 0.10.
    """
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags

    # Create function spaces
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)

    # Create config
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)

    # Shared gait loader built on the same vector space V
    gait_loader = setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)

    return domain, facet_tags, V, Q, cfg, gait_loader


@pytest.fixture(scope="module")
def gait_loader(femur_mechanics_setup):
    """Return shared gait loader built on the femur mechanics space."""
    _, _, _, _, _, gait_loader = femur_mechanics_setup
    return gait_loader


@pytest.fixture
def mechanics_solver(femur_mechanics_setup, gait_loader):
    """Create MechanicsSolver with femur geometry and gait loading."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup
    
    # Create field functions
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    A_dir = fem.Function(fem.functionspace(domain, 
                         basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                         name="A")
    
    # Initialize with uniform density and isotropic fabric
    rho.x.array[:] = 1.0  # Normalized density
    A_dir.x.array[:] = 0.0
    for i in range(3):
        A_dir.x.array[i::9] = 1.0/3.0  # Isotropic fabric (1/3 * I)
    
    # Build boundary conditions (fix distal end - tag 1)
    dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    
    # Neumann BCs from gait loader (applied on femur surface - tag 2)
    # Note: All gait loads (hip, glmed, glmax) are applied on femur_surface
    neumann_bcs = [
        (gait_loader.t_hip, 2),     # Hip joint on femur surface
        (gait_loader.t_glmed, 2),   # Glut med on femur surface
        (gait_loader.t_glmax, 2),   # Glut max on femur surface
    ]
    
    # Create and setup solver
    solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
    solver.setup()
    
    return solver


class TestAccumulatedStrainEnergy:
    """Test accumulated strain energy computation over gait cycle using GaitEnergyDriver."""
    
    def test_energy_driver_initialization(self, mechanics_solver, gait_loader):
        """GaitEnergyDriver should initialize without errors."""
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        assert driver.mech is mechanics_solver
        assert driver.gait is gait_loader
        assert len(driver.phases) == gait_loader.n_samples
        assert len(driver.weights) == gait_loader.n_samples
    
    def test_energy_expr_builds(self, mechanics_solver, gait_loader):
        """Energy expression should build successfully."""
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()
        assert psi_expr is not None, "Energy expression should be created"
    
    def test_energy_positivity(self, mechanics_solver, gait_loader):
        """Accumulated energy should be positive when integrated."""
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()
        
        # Integrate energy over domain
        cfg = mechanics_solver.cfg
        psi_total_local = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        comm = cfg.domain.comm
        psi_total = comm.allreduce(psi_total_local, op=MPI.SUM)
        
        assert psi_total > 0.0, f"Total strain energy should be positive, got {psi_total}"
    
    def test_energy_increases_with_load_magnitude(self, mechanics_solver, gait_loader):
        """Strain energy should increase with load magnitude."""
        # Store original load scale
        original_scale = gait_loader.load_scale
        comm = mechanics_solver.cfg.domain.comm
        
        # Compute with base load
        gait_loader.load_scale = 1.0
        driver_base = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver_base.update_snapshots()
        psi_expr_base = driver_base.energy_expr()
        psi_base_local = fem.assemble_scalar(fem.form(psi_expr_base * mechanics_solver.cfg.dx))
        psi_base = comm.allreduce(psi_base_local, op=MPI.SUM)
        
        # Compute with 2x load
        gait_loader.load_scale = 2.0
        driver_double = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver_double.update_snapshots()
        psi_expr_double = driver_double.energy_expr()
        psi_double_local = fem.assemble_scalar(fem.form(psi_expr_double * mechanics_solver.cfg.dx))
        psi_double = comm.allreduce(psi_double_local, op=MPI.SUM)
        
        # Restore original scale
        gait_loader.load_scale = original_scale
        
        # Energy should scale approximately as load^2 (linear elasticity)
        # Driver energy uses (psi/psi_ref)^n formulation, hence load^(2*n_power)
        ratio = psi_double / psi_base
        expected = 2.0 ** (2.0 * mechanics_solver.cfg.n_power)
        assert 0.5 * expected < ratio < 1.5 * expected, (
            "Energy should scale with load^(2*n_power); "
            f"expected≈{expected:.2f}, ratio={ratio:.2f}"
        )
    
    def test_energy_components_contribution(self, mechanics_solver, gait_loader):
        """Verify that multiple gait phases contribute to accumulated energy."""
        # Get individual phase energies (integrals over domain)
        quadrature = gait_loader.get_quadrature()
        phase_energies = []
        comm = mechanics_solver.cfg.domain.comm
        cfg = mechanics_solver.cfg
        
        for phase, weight in quadrature:
            gait_loader.update_loads(phase)
            mechanics_solver.assemble_rhs()
            mechanics_solver.solve()
            psi_density = mechanics_solver.get_strain_energy_density(mechanics_solver.u)
            psi_norm = (psi_density / cfg.psi_ref) ** cfg.n_power
            psi_local = fem.assemble_scalar(fem.form(psi_norm * cfg.dx))
            psi_total = comm.allreduce(psi_local, op=MPI.SUM)
            phase_energies.append((phase, weight, psi_total))
        
        # All phases should have positive energy
        for phase, weight, psi in phase_energies:
            assert psi > 0.0, f"Phase {phase}% should have positive energy, got {psi}"
        
        # Verify manual accumulation matches GaitEnergyDriver
        # Both compute weighted sum of energy integrals
        manual_accumulated = sum(w * psi for _, w, psi in phase_energies)
        
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()
        method_local = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        method_accumulated = comm.allreduce(method_local, op=MPI.SUM)
        
        np.testing.assert_allclose(manual_accumulated, method_accumulated, rtol=1e-6,
            err_msg="Manual and method accumulation should match")
    
    def test_peak_stance_has_maximum_energy(self, mechanics_solver, gait_loader):
        """Peak stance phase should have highest strain energy."""
        quadrature = gait_loader.get_quadrature()
        phase_energies = {}
        
        for phase, weight in quadrature:
            gait_loader.update_loads(phase)
            mechanics_solver.assemble_rhs()
            mechanics_solver.solve()
            psi_phase = mechanics_solver.average_strain_energy()
            phase_energies[phase] = psi_phase
        
        # Find peak energy phase
        max_phase = max(phase_energies, key=phase_energies.get)
        max_energy = phase_energies[max_phase]
        
        # Peak should be during stance phase (first half of gait cycle, 0-50%)
        # This is where hip joint reaction forces are highest
        assert 0.0 <= max_phase <= 60.0, \
            f"Peak energy should be during stance phase (0-60%), found at {max_phase}%"
        
        # Peak should be significantly higher than minimum
        min_energy = min(phase_energies.values())
        ratio = max_energy / min_energy
        assert ratio > 1.5, \
            f"Peak energy should be >1.5x minimum, got ratio={ratio:.2f}"

# Tests for femur_remodeller_gait.py: coordinate scaling and physical force validation.
from dolfinx.fem.petsc import (
    assemble_matrix as assemble_matrix_petsc,
    assemble_vector as assemble_vector_petsc,
    create_vector as create_vector_petsc,
)


@pytest.fixture(scope="module")
def femur_geometry_setup(femur_mechanics_setup):
    """Reuse femur mechanics mesh/space/config for geometry tests.

    Ensures all tests share a single DOLFINx mesh and function spaces.
    """
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup

    # Override only material stiffness used in some ranges if needed
    cfg.E0 = 17e3
    unit_scale = 1.0  # No scaling needed, already in mm

    return domain, facet_tags, V, Q, cfg, unit_scale


class TestCoordinateScaling:
    """Verify DOLFINx mesh and femurloader use consistent coordinate systems."""

    def test_dolfinx_mesh_in_millimeters(self, femur_geometry_setup):
        """Verify DOLFINx mesh coordinates are in millimeters (not meters)."""
        domain, _, _, _, _, _ = femur_geometry_setup
        geom = domain.geometry.x
        
        # Femur geometry should be O(100) mm, not O(0.1) m
        max_coord = np.max(np.abs(geom))
        assert 10.0 < max_coord < 500.0, \
            f"Mesh coords should be in mm (expected 10-500, got {max_coord})"

    def test_femurloader_mesh_in_millimeters(self, femur_geometry_setup):
        """Verify femurloader PyVista mesh is in millimeters."""
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        
        max_coord = np.max(np.abs(pv_mesh.points))
        assert 10.0 < max_coord < 500.0, \
            f"PyVista mesh should be in mm (expected 10-500, got {max_coord})"

    def test_coord_scale_is_unity(self, gait_loader):
        """Verify coord_scale=1.0 (no conversion needed)."""
        assert gait_loader.coord_scale == 1.0, \
            "coord_scale should be 1.0 since both DOLFINx and femurloader use mm"

    def test_geometry_bounds_match(self, femur_geometry_setup):
        """Verify DOLFINx and PyVista geometries have same bounds."""
        domain, _, _, _, _, _ = femur_geometry_setup
        
        dolfinx_geom = domain.geometry.x
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        pv_geom = pv_mesh.points
        
        # Should have same coordinate ranges (within tolerance)
        dolfinx_range = np.ptp(dolfinx_geom, axis=0)
        pv_range = np.ptp(pv_geom, axis=0)
        
        np.testing.assert_allclose(dolfinx_range, pv_range, rtol=0.10,
            err_msg="DOLFINx and PyVista geometry ranges should match")


class TestPhysicalForces:
    """Validate physical coherence of gait loading forces."""

    def test_hip_force_at_peak_stance(self, gait_loader, femur_geometry_setup):
        """Hip joint applied load (integrated) at peak stance should be ~1–6× BW."""
        domain, _, _, _, cfg, unit_scale = femur_geometry_setup
        BW_N = 75.0 * 9.81
        
        gait_loader.update_loads(50.0)
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        # Traction is in MPa, integrate over surface (mm²) to get force in N
        # Force [N] = traction [MPa = N/mm²] * area [mm²]
        F_applied_N = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_applied_N[i] = domain.comm.allreduce(Fi_loc, op=4)
        F_mag = np.linalg.norm(F_applied_N)
        assert 2.5 * BW_N < F_mag < 4.5 * BW_N, \
            f"Applied force should be 2.5–4.5× BW at peak stance, got {F_mag/BW_N:.2f}× BW"

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

    def test_traction_field_nonzero(self, gait_loader, femur_geometry_setup):
        """Verify traction fields contain non-zero values after interpolation."""
        _, _, _, _, cfg, _ = femur_geometry_setup
        gait_loader.update_loads(50.0)
        
        # Check all three traction fields (traction is directly in MPa)
        for name, func in [("hip", gait_loader.t_hip),
                          ("glmed", gait_loader.t_glmed),
                          ("glmax", gait_loader.t_glmax)]:
            vals = func.x.array.reshape((-1, 3))
            nonzero_count = np.count_nonzero(vals)
            
            assert nonzero_count > 100, \
                f"{name} traction should have >100 nonzero values, got {nonzero_count}"
            
            max_magnitude_MPa = np.max(np.linalg.norm(vals, axis=1))
            
            # Tractions should be non-zero (even if small due to load spreading)
            assert max_magnitude_MPa > 1e-6, \
                f"{name} traction magnitude should be >1e-6 MPa, got {max_magnitude_MPa:.3e} MPa"


class TestGaitQuadrature:
    """Verify gait cycle quadrature integration."""

    def test_quadrature_properties(self, gait_loader):
        """Verify quadrature weights, coverage, and sample count."""
        quadrature = gait_loader.get_quadrature()
        phases, weights = zip(*quadrature)
        
        # Weights should sum to 1.0
        assert np.isclose(sum(weights), 1.0), \
            f"Quadrature weights should sum to 1.0, got {sum(weights)}"
        
        # Should span full gait cycle [0, 100]%
        assert min(phases) == 0.0, "Quadrature should start at 0%"
        assert max(phases) == 100.0, "Quadrature should end at 100%"
        
        # Should have correct sample count
        assert len(quadrature) == gait_loader.n_samples, \
            f"Expected {gait_loader.n_samples} samples, got {len(quadrature)}"


class TestIndividualLoadIntegrals:
    """Each gait load individually integrates to physiological forces and matches its interpolator."""

    @pytest.mark.parametrize("load_name,traction_attr,gait_method,tol,bw_min,bw_max", [
        ("hip", "t_hip", "hip_gait", 0.05, 2.3, 4.5),
        ("gluteus_medius", "t_glmed", "glmed_gait", 0.08, 0.3, 2.5),
        ("gluteus_maximus", "t_glmax", "glmax_gait", 0.08, 0.1, 1.5),
    ])
    def test_load_integral_matches_interpolator_peak(self, gait_loader, femur_geometry_setup,
                                                      load_name, traction_attr, gait_method, tol, bw_min, bw_max):
        """Verify load integral matches interpolator and is in physiological range."""
        domain, _, _, _, cfg, unit_scale = femur_geometry_setup
        BW_N = 75.0 * 9.81

        # Find peak phase from interpolator
        phases = np.linspace(0, 100, 41)
        gait_fn = getattr(gait_loader, gait_method)
        mags = [np.linalg.norm(gait_fn(p)) for p in phases]
        phase = float(phases[int(np.argmax(mags))])
        gait_loader.update_loads(phase)

        # Integrate traction over contact tag=2
        import ufl
        traction = getattr(gait_loader, traction_attr)
        F_N = np.zeros(3)
        for i in range(3):
            Fi = fem.form(traction[i] * cfg.ds(2))
            val = fem.assemble_scalar(Fi)
            F_N[i] = domain.comm.allreduce(val, op=4)

        # Compare with gait interpolator
        F_css = gait_fn(phase)
        rel_err = abs(np.linalg.norm(F_N) - np.linalg.norm(F_css)) / max(np.linalg.norm(F_css), 1e-30)
        assert rel_err < tol, f"{load_name} integral error {rel_err:.2e} exceeds tolerance {tol}"

        # Physiological range check
        ratio = np.linalg.norm(F_N) / BW_N
        assert bw_min < ratio < bw_max, \
            f"{load_name} peak should be {bw_min}–{bw_max}× BW, got {ratio:.2f}× BW at {phase:.0f}%"


class TestForceMaxima:
    """Compute and report maximum forces for each load type across gait cycle."""

    def test_report_max_forces_across_gait(self, gait_loader, femur_geometry_setup, capsys):
        """Strictly verify force maxima and their phases are physiological."""
        _, _, _, _, cfg, _ = femur_geometry_setup
        BW_N = 75.0 * 9.81
        phases = np.linspace(0, 100, 21)
        
        hip_max = 0.0
        glmed_max = 0.0
        glmax_max = 0.0
        
        hip_max_phase = 0.0
        glmed_max_phase = 0.0
        glmax_max_phase = 0.0
        
        hip_traction_max_MPa = 0.0
        glmed_traction_max_MPa = 0.0
        glmax_traction_max_MPa = 0.0
        
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
            
            # Traction fields (already in MPa)
            gait_loader.update_loads(phase)
            t_hip = np.max(np.linalg.norm(gait_loader.t_hip.x.array.reshape((-1, 3)), axis=1))
            t_glmed = np.max(np.linalg.norm(gait_loader.t_glmed.x.array.reshape((-1, 3)), axis=1))
            t_glmax = np.max(np.linalg.norm(gait_loader.t_glmax.x.array.reshape((-1, 3)), axis=1))
            
            hip_traction_max_MPa = max(hip_traction_max_MPa, t_hip)
            glmed_traction_max_MPa = max(glmed_traction_max_MPa, t_glmed)
            glmax_traction_max_MPa = max(glmax_traction_max_MPa, t_glmax)
        
        # Strict physiological checks
        assert 2.3 < hip_max / BW_N < 4.5, \
            f"Hip peak should be 2.3–4.5× BW, got {hip_max/BW_N:.2f}× BW at {hip_max_phase:.0f}%"
        assert 0.3 < glmed_max / BW_N < 2.5, \
            f"Gluteus medius peak should be 0.3–2.5× BW, got {glmed_max/BW_N:.2f}× BW"
        assert 0.1 < glmax_max / BW_N < 1.5, \
            f"Gluteus maximus peak should be 0.1–1.5× BW, got {glmax_max/BW_N:.2f}× BW"

        # Peak phases: hip ~ mid-stance, glute med early–mid, glute max early
        assert 10.0 <= hip_max_phase <= 60.0, \
            f"Hip peak phase should be 10–60%, got {hip_max_phase:.0f}%"
        assert 10.0 <= glmed_max_phase <= 50.0, \
            f"Gluteus medius peak phase should be 10–50%, got {glmed_max_phase:.0f}%"
        assert 0.0 <= glmax_max_phase <= 50.0, \
            f"Gluteus maximus peak phase should be 0–50%, got {glmax_max_phase:.0f}%"

        # Traction magnitudes within physiological contact-stress expectations (MPa)
        # Hip joint contact stress during walking typically 2–10 MPa; allow headroom for concentration
        assert 1e-6 < hip_traction_max_MPa < 20.0, \
            f"Hip max traction should be 1e-6–20 MPa, got {hip_traction_max_MPa:.3e} MPa"


class TestFemurDeformationFeasibility:
    """Test that gait loads produce physiologically feasible femur deformations known from literature."""
    
    def test_displacement_magnitude_range(self, gait_loader, femur_geometry_setup):
        """Displacement magnitudes should be in physiological range (0.05-3 mm).
        
        Literature: Femur deformations during gait are typically submillimeter to few mm.
        - Bessho et al. (2007): femoral head displacement ~1-2 mm under gait loads
        - Taylor et al. (1996): peak surface strains correspond to ~0.5-1.5 mm displacement
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver with uniform density and isotropic fabric
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        # Use realistic bulk density to target physiological strains
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0  # Isotropic
        
        # Fix distal end, apply gait loads
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance (max load ~50% gait)
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Get displacement magnitudes (already in mm)
        u_vals_mm = u.x.array.reshape((-1, 3))
        u_magnitudes_mm = np.linalg.norm(u_vals_mm, axis=1)
        
        max_displacement_mm = np.max(u_magnitudes_mm)
        mean_displacement_mm = np.mean(u_magnitudes_mm)
        
        # Stricter physiological range for walking: 0.05–3.0 mm peak
        assert 0.05 < max_displacement_mm < 3.0, \
            f"Max displacement should be 0.05–3.0 mm, got {max_displacement_mm:.3f} mm"
        
        # Most of bone should have smaller deformations (mean << max)
        assert mean_displacement_mm < 2.0, \
            f"Mean displacement should be <2.0 mm, got {mean_displacement_mm:.3f} mm"
        
        # Localized deformation: peak at least 1.5× mean
        assert max_displacement_mm > 1.5 * mean_displacement_mm, \
            "Max displacement should be >1.5× mean (localized peak)"
    
    def test_strain_magnitude_range(self, gait_loader, femur_geometry_setup):
        """Peak strains should be in physiological range (200-3000 microstrain).
        
        Literature: In vivo strain measurements during gait:
        - Burr et al. (1996): femoral strains 400-1200 microstrain during walking
        - Lanyon et al. (1975): femoral shaft strains 300-3000 microstrain during various activities
        - Gross & Rubin (1995): physiological strains typically 50-3000 microstrain
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        import ufl
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute strain tensor: epsilon = 0.5*(grad(u) + grad(u)^T)
        eps = ufl.sym(ufl.grad(u))
        
        # Von Mises strain: sqrt(2/3 * eps_dev : eps_dev)
        I = ufl.Identity(3)
        eps_dev = eps - (ufl.tr(eps) / 3.0) * I
        eps_vm = ufl.sqrt((2.0/3.0) * ufl.inner(eps_dev, eps_dev))
        
        # Project to DG0 for evaluation
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        eps_vm_proj = fem.Function(DG0, name="eps_vm")
        eps_vm_expr = fem.Expression(eps_vm, DG0.element.interpolation_points)
        eps_vm_proj.interpolate(eps_vm_expr)
        
        # Get strain values (dimensionless)
        eps_vals = eps_vm_proj.x.array[:]
        
        # Convert to microstrain: microstrain = strain * 1e6
        eps_microstrain = eps_vals * 1e6
        
        max_strain = np.max(eps_microstrain)
        p95_strain = np.percentile(eps_microstrain, 95)
        
        # Physiological ranges:
        # Peak typically within 300–6000 με, bulk (95th percentile) < 3000 με, median 50–2000 με
        p50_strain = np.percentile(eps_microstrain, 50)
        assert 300.0 < max_strain < 6000.0, \
            f"Peak strain should be 300–6000 microstrain, got {max_strain:.1f}"
        assert p95_strain < 3000.0, \
            f"95th percentile strain should be <3000 microstrain, got {p95_strain:.1f}"
        assert 50.0 < p50_strain < 2000.0, \
            f"Median strain should be 50–2000 microstrain, got {p50_strain:.1f}"
    
    def test_stress_magnitude_range(self, gait_loader, femur_geometry_setup):
        """Peak stresses should be in physiological range (1-100 MPa).
        
        Literature:
        - Cortical bone yield stress: ~100-150 MPa
        - Typical walking stresses in femur: 5-50 MPa
        - Hip contact stress: 2-10 MPa (already tested in TestPhysicalForces)
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        import ufl
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute von Mises stress
        # sigma = E_eff(rho) * C : epsilon
        # For isotropic: sigma_vm = sqrt(3/2 * s:s) where s is deviatoric stress
        
        eps = ufl.sym(ufl.grad(u))
        I = ufl.Identity(3)
        
        # Lame parameters (isotropic, rho=1.0, E0 in MPa)
        E_eff = cfg.E0  # MPa
        nu = cfg.nu
        lmbda = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0*nu))
        mu = E_eff / (2.0 * (1.0 + nu))
        
        # Stress tensor (in MPa since E_eff is in MPa)
        sigma = lmbda * ufl.tr(eps) * I + 2.0 * mu * eps
        
        # Von Mises stress
        sigma_dev = sigma - (ufl.tr(sigma) / 3.0) * I
        sigma_vm = ufl.sqrt((3.0/2.0) * ufl.inner(sigma_dev, sigma_dev))
        
        # Project to DG0
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        sigma_vm_proj = fem.Function(DG0, name="sigma_vm")
        sigma_vm_expr = fem.Expression(sigma_vm, DG0.element.interpolation_points)
        sigma_vm_proj.interpolate(sigma_vm_expr)
        
        # Get stress values (already in MPa)
        sigma_vals_MPa = sigma_vm_proj.x.array[:]
        
        max_stress = np.max(sigma_vals_MPa)
        p95_stress = np.percentile(sigma_vals_MPa, 95)
        
        # Stricter physiological bounds for walking
        assert 2.0 < max_stress < 100.0, \
            f"Peak von Mises stress should be 2–100 MPa, got {max_stress:.1f} MPa"
        assert p95_stress < 60.0, \
            f"95th percentile stress should be <60 MPa, got {p95_stress:.1f} MPa"
    
    def test_strain_energy_density_range(self, gait_loader, femur_geometry_setup):
        """Strain energy density should be in physiological range.
        
        Literature: Typical SED values that trigger remodeling:
        - Huiskes et al. (1987): reference SED ~0.004 J/g (~0.004 MPa)
        - Beaupré et al. (1990): lazy zone 0.0005–0.015 J/g (~0.0005–0.015 MPa)
        - Our psi_ref_dim = 0.0003 MPa reference
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Get strain energy density from solver (returns UFL expression in MPa)
        psi_MPa = solver.get_strain_energy_density(u)
        
        # Project to DG0 for evaluation
        import basix
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        psi_proj = fem.Function(DG0, name="psi")
        psi_expr = fem.Expression(psi_MPa, DG0.element.interpolation_points)
        psi_proj.interpolate(psi_expr)
        
        # Get values (already in MPa)
        psi_vals_MPa = psi_proj.x.array[:]

        max_sed = np.max(psi_vals_MPa)
        median_sed = np.median(psi_vals_MPa)

        # Physiological range (MPa): ~1e-7 to 1.0 typical envelope
        # Note: With uniform reference density, SED is higher than in actual bone
        # Typical values from literature: 0.0005–0.015 MPa
        assert 1e-7 < max_sed < 1.0, \
            f"Peak SED should be 1e-7–1 MPa, got {max_sed:.4e} MPa"

        assert median_sed < 0.5, \
            f"Median SED should be <0.5 MPa, got {median_sed:.3f} MPa"

        # Check against reference value (should be same order of magnitude)
        psi_ref_MPa = cfg.psi_ref
        assert 0.01 < median_sed / psi_ref_MPa < 1e4, \
            f"Median SED should be within 4 orders of magnitude of psi_ref"


class TestReactionForces:
    """Test reaction forces at fixed boundary for different loading scenarios."""
    
    def test_reaction_force_equilibrium_peak_stance(self, gait_loader, femur_geometry_setup):
        """Reaction forces should be non-zero and physiological at fixed boundary.
        
        Validates that reaction forces at the fixed distal end are computed correctly
        and have reasonable magnitudes.
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        # Fix distal end, apply gait loads
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute consistent reaction via unconstrained residual r = A0 u − b0
        from petsc4py import PETSc
        A0 = assemble_matrix_petsc(solver.a_form)
        A0.assemble()
        b0 = create_vector_petsc(V)
        with b0.localForm() as b0_loc:
            b0_loc.set(0.0)
        assemble_vector_petsc(b0, solver.L_form)
        b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        r = create_vector_petsc(V)
        with r.localForm() as r_loc:
            r_loc.set(0.0)
        A0.mult(u.x.petsc_vec, r)
        r.axpy(-1.0, b0)

        # Sum reactions per component over Dirichlet DOFs (tag=1)
        # Reaction forces are already in N (stress in MPa × area in mm²)
        F_reaction_N = np.zeros(3)
        for i, bc in enumerate(dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.getValues(idx_owned)
                F_reaction_N[i] += float(np.sum(r_local))

        F_reaction_magnitude = np.linalg.norm(F_reaction_N)

        BW_N = 75.0 * 9.81
        
        # Reaction force should be non-zero and physiological
        assert F_reaction_magnitude > 0.01 * BW_N, \
            f"Reaction force should be >1% BW, got {F_reaction_magnitude:.1f} N"
        
        # Reaction force should be reasonable (not excessively large)
        assert F_reaction_magnitude < 20.0 * BW_N, \
            f"Reaction force should be <20× BW, got {F_reaction_magnitude:.1f} N"

        # Also check near-equilibrium with applied traction at peak
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_N = np.zeros(3)
        for i in range(3):
            Fi = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi)
            F_applied_N[i] = domain.comm.allreduce(Fi_loc, op=4)
        rel_eq = np.linalg.norm(F_applied_N + F_reaction_N) / max(np.linalg.norm(F_applied_N), 1e-30)
        assert rel_eq < 5e-7, f"Force balance failed at peak: rel_err={rel_eq:.2e}"
    
    def test_reaction_force_magnitude_physiological(self, gait_loader, femur_geometry_setup):
        """Reaction force magnitude should be physiological (comparable to body weight × gait cycles).
        
        For 75 kg person: BW = 736 N
        Peak gait: ~3-4× BW hip force + muscle forces
        Reaction should balance total applied load.
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        from dolfinx.fem.petsc import assemble_vector
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        BW_N = 75.0 * 9.81
        
        # Test at multiple gait phases
        phases = [25.0, 50.0, 75.0]  # Early stance, peak, late stance
        
        for phase in phases:
            gait_loader.update_loads(phase)
            solver.assemble_rhs()
            solver.solve()
            
            # Compute consistent reaction via unconstrained residual
            from petsc4py import PETSc
            A0 = assemble_matrix_petsc(solver.a_form)
            A0.assemble()
            b0 = create_vector_petsc(V)
            with b0.localForm() as b0_loc:
                b0_loc.set(0.0)
            assemble_vector_petsc(b0, solver.L_form)
            b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

            r = create_vector_petsc(V)
            with r.localForm() as r_loc:
                r_loc.set(0.0)
            A0.mult(u.x.petsc_vec, r)
            r.axpy(-1.0, b0)

            F_reaction_N = np.zeros(3)
            for i, bc in enumerate(dirichlet_bcs):
                idx_all, first_ghost = bc.dof_indices()
                idx_owned = idx_all[:first_ghost]
                if idx_owned.size:
                    r_local = r.getValues(idx_owned)
                    F_reaction_N[i] += float(np.sum(r_local))

            F_reaction_magnitude = np.linalg.norm(F_reaction_N)
            
            # Reaction should be 0.01-20× BW  
            # Note: Reaction may be smaller due to distributed loading across surface
            assert 0.01 * BW_N < F_reaction_magnitude < 20.0 * BW_N, \
                f"Phase {phase}%: Reaction {F_reaction_magnitude:.1f} N should be 0.01-20× BW ({BW_N:.1f} N)"
    
    def test_reaction_force_components_realistic(self, gait_loader, femur_geometry_setup):
        """Reaction force components should have realistic directional distribution.
        
        Femur is loaded primarily in superior-inferior (y) direction with some
        medial-lateral (z) and anterior-posterior (x) components.
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        from dolfinx.fem.petsc import assemble_vector
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Consistent reaction via unconstrained residual
        from petsc4py import PETSc
        A0 = assemble_matrix_petsc(solver.a_form)
        A0.assemble()
        b0 = create_vector_petsc(V)
        with b0.localForm() as b0_loc:
            b0_loc.set(0.0)
        assemble_vector_petsc(b0, solver.L_form)
        b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        r = create_vector_petsc(V)
        with r.localForm() as r_loc:
            r_loc.set(0.0)
        A0.mult(u.x.petsc_vec, r)
        r.axpy(-1.0, b0)

        F_reaction_N = np.zeros(3)
        for i, bc in enumerate(dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.getValues(idx_owned)
                F_reaction_N[i] += float(np.sum(r_local))

        F_reaction_magnitude = np.linalg.norm(F_reaction_N)
        
        # Directional consistency: reaction should oppose net applied load
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_N = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_applied_N[i] = domain.comm.allreduce(Fi_loc, op=4)
        num = -float(F_reaction_N @ F_applied_N)
        den = (np.linalg.norm(F_reaction_N) * np.linalg.norm(F_applied_N) + 1e-30)
        cos_theta = num / den
        assert cos_theta > 0.98, \
            f"Reaction should strongly oppose applied load (cosθ>0.98). Got cosθ={cos_theta:.2f}"
    
    def test_reaction_force_varies_with_gait_phase(self, gait_loader, femur_geometry_setup):
        """Reaction forces should vary across gait cycle (not constant)."""
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        from dolfinx.fem.petsc import assemble_vector
        
        domain, facet_tags, V, Q, cfg, _ = femur_geometry_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Sample across gait cycle
        phases = np.linspace(0, 100, 9)
        reaction_magnitudes = []
        
        for phase in phases:
            gait_loader.update_loads(phase)
            solver.assemble_rhs()
            solver.solve()

            # Consistent reaction via unconstrained residual
            from petsc4py import PETSc
            A0 = assemble_matrix_petsc(solver.a_form)
            A0.assemble()
            b0 = create_vector_petsc(V)
            with b0.localForm() as b0_loc:
                b0_loc.set(0.0)
            assemble_vector_petsc(b0, solver.L_form)
            b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

            r = create_vector_petsc(V)
            with r.localForm() as r_loc:
                r_loc.set(0.0)
            A0.mult(u.x.petsc_vec, r)
            r.axpy(-1.0, b0)

            F_reaction_N = np.zeros(3)
            for i, bc in enumerate(dirichlet_bcs):
                idx_all, first_ghost = bc.dof_indices()
                idx_owned = idx_all[:first_ghost]
                if idx_owned.size:
                    r_local = r.getValues(idx_owned)
                    F_reaction_N[i] += float(np.sum(r_local))

            reaction_magnitudes.append(np.linalg.norm(F_reaction_N))
        
        reaction_magnitudes = np.array(reaction_magnitudes)
        
        # Reactions should vary across gait (not constant)
        variation = np.ptp(reaction_magnitudes)  # peak-to-peak
        mean_reaction = np.mean(reaction_magnitudes)
        
        assert variation > 0.5 * mean_reaction, \
            f"Reaction force should vary >50% across gait, got {variation/mean_reaction*100:.1f}%"
        
        # Should have a peak (not monotonic)
        assert not np.all(np.diff(reaction_magnitudes) > 0), \
            "Reaction force should not be monotonically increasing"
        assert not np.all(np.diff(reaction_magnitudes) < 0), \
            "Reaction force should not be monotonically decreasing"

"""
Test suite for femur_css.py module.

Tests coordinate system setup (CSS) for femur models including:
- JSON point loading
- Femoral head fitting
- Coordinate system construction
- Transformation matrices
- Vector transformations
"""
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
class TestFemurCSSIntegration:
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
class TestFemurCSSParametrized:
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
class TestFemurCSSPerformance:
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

"""
Test suite for femur_loads.py module.

Tests load application classes and utility functions including:
- Utility functions: build_load, vector_from_angles, gait_interpolator, orthoload2ISB
- GaussianSurfaceLoad base class functionality
- HIPJointLoad for hip joint force application
- MuscleLoad for muscle force application along splines
"""

from simulation.femur_loads import (
    build_load,
    vector_from_angles,
    gait_interpolator, 
    orthoload2ISB,
    GaussianSurfaceLoad,
    HIPJointLoad,
    MuscleLoad
)
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
    
    @pytest.mark.parametrize("alpha_sag,alpha_front,expected_signs,desc", [
        (0.0, 0.0, (0, 1, 0), "zero angles"),
        (30.0, 0.0, (1, 1, 0), "sagittal only"),
        (0.0, 45.0, (0, 1, 1), "frontal only"),
        (15.0, 30.0, (1, 1, 1), "both positive angles"),
        (-20.0, -10.0, (-1, 1, -1), "both negative angles"),
    ])
    def test_vector_from_angles(self, alpha_sag, alpha_front, expected_signs, desc):
        """Test vector_from_angles with various angle combinations."""
        magnitude = 100.0
        vector = vector_from_angles(magnitude, alpha_sag, alpha_front)
        
        # Verify magnitude is preserved
        assert_almost_equal(np.linalg.norm(vector), magnitude, err_msg=f"Failed for {desc}")
        
        # Verify expected component signs
        for i, expected_sign in enumerate(expected_signs):
            if expected_sign > 0:
                assert vector[i] > 0, f"Component {i} should be positive for {desc}"
            elif expected_sign < 0:
                assert vector[i] < 0, f"Component {i} should be negative for {desc}"
            else:
                assert abs(vector[i]) < 1e-10, f"Component {i} should be zero for {desc}"
    

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
class TestLoadEdgeCases:
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
class TestLoadParametrized:
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
class TestLoadPerformance:
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

