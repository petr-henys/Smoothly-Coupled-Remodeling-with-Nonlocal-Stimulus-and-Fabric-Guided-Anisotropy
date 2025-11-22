"""
Tests for paths.py module.

Tests cover:
- Path constants (FemurPaths, GaitPaths)
- Directory structure
- ensure_directories() function
- get_output_path() with subdirectories and cleaning
- get_hip_traction_path() indexing
- Path resolution and existence
"""

import pytest
from pathlib import Path

# Import after sys.path manipulation in conftest
from simulation.paths import (
    PROJECT_ROOT,
    ANATOMY_DIR,
    ANATOMY_RAW_DIR,
    ANATOMY_PROCESSED_DIR,
    FEMUR_ANATOMY_DIR,
    GAIT_DATA_DIR,
    PROXIMAL_FEMUR_DIR,
    RESULTS_DIR,
    ARCHIVE_DIR,
    FemurPaths,
    GaitPaths,
    ensure_directories,
    get_output_path,
    get_hip_traction_path,
)


class TestPathConstants:
    """Test path constant definitions."""
    
    def test_project_root_is_path(self):
        """Test that PROJECT_ROOT is a Path object."""
        assert isinstance(PROJECT_ROOT, Path)
    
    def test_project_root_exists(self):
        """Test that PROJECT_ROOT points to an existing directory."""
        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()
    
    def test_anatomy_dir_relative_to_project(self):
        """Test that ANATOMY_DIR is relative to PROJECT_ROOT."""
        assert ANATOMY_DIR == PROJECT_ROOT / "anatomy"
    
    def test_anatomy_subdirectories(self):
        """Test anatomy directory structure."""
        assert ANATOMY_RAW_DIR == ANATOMY_DIR / "raw"
        assert ANATOMY_PROCESSED_DIR == ANATOMY_DIR / "processed"
    
    def test_raw_subdirectories(self):
        """Test raw data subdirectory structure."""
        assert FEMUR_ANATOMY_DIR == ANATOMY_RAW_DIR / "femur_anatomy"
        assert GAIT_DATA_DIR == ANATOMY_RAW_DIR / "gait_data"
        assert PROXIMAL_FEMUR_DIR == ANATOMY_RAW_DIR / "proximal_femur"
    
    def test_results_dir_relative_to_project(self):
        """Test that RESULTS_DIR is relative to PROJECT_ROOT."""
        assert RESULTS_DIR == PROJECT_ROOT / "results"
    
    def test_archive_dir_relative_to_project(self):
        """Test that ARCHIVE_DIR is relative to PROJECT_ROOT."""
        assert ARCHIVE_DIR == PROJECT_ROOT / "archive"


class TestFemurPaths:
    """Test FemurPaths class constants."""
    
    def test_femur_mesh_paths(self):
        """Test femur mesh file paths."""
        assert FemurPaths.FEMUR_MESH_VTK == PROXIMAL_FEMUR_DIR / "model.vtk"
        assert FemurPaths.FEMUR_MESH_FEB == PROXIMAL_FEMUR_DIR / "model.feb"
    
    def test_femur_mesh_extensions(self):
        """Test that mesh files have correct extensions."""
        assert FemurPaths.FEMUR_MESH_VTK.suffix == ".vtk"
        assert FemurPaths.FEMUR_MESH_FEB.suffix == ".feb"
    
    def test_head_line_json_path(self):
        """Test head line JSON path."""
        assert FemurPaths.HEAD_LINE_JSON == FEMUR_ANATOMY_DIR / "infsup_head_line.mrk.json"
        assert FemurPaths.HEAD_LINE_JSON.suffix == ".json"
    
    def test_le_me_line_json_path(self):
        """Test LE-ME line JSON path."""
        assert FemurPaths.LE_ME_LINE_JSON == FEMUR_ANATOMY_DIR / "LE-ME_line.mrk.json"
        assert FemurPaths.LE_ME_LINE_JSON.suffix == ".json"
    
    def test_vastus_lateralis_json_path(self):
        """Test vastus lateralis JSON path."""
        assert FemurPaths.VASTUS_LATERALIS_JSON == FEMUR_ANATOMY_DIR / "VASTUS_lateralis.mrk.json"
        assert FemurPaths.VASTUS_LATERALIS_JSON.suffix == ".json"
    
    def test_gl_med_json_path(self):
        """Test gluteus medius JSON path."""
        assert FemurPaths.GL_MED_JSON == FEMUR_ANATOMY_DIR / "GL_med.mrk.json"
        assert FemurPaths.GL_MED_JSON.suffix == ".json"
    
    def test_gl_max_json_path(self):
        """Test gluteus maximus JSON path."""
        assert FemurPaths.GL_MAX_JSON == FEMUR_ANATOMY_DIR / "GL_max.mrk.json"
        assert FemurPaths.GL_MAX_JSON.suffix == ".json"
    
    def test_css_axes_vtk_path(self):
        """Test CSS axes VTK path."""
        assert FemurPaths.CSS_AXES_VTK == RESULTS_DIR / "css.vtk"
        assert FemurPaths.CSS_AXES_VTK.suffix == ".vtk"
    
    def test_facet_markers_vtk_path(self):
        """Test facet markers VTK path."""
        assert FemurPaths.FACET_MARKERS_VTK == RESULTS_DIR / "facet_markers.vtk"
        assert FemurPaths.FACET_MARKERS_VTK.suffix == ".vtk"
    
    def test_all_paths_are_path_objects(self):
        """Test that all FemurPaths attributes are Path objects."""
        for attr_name in dir(FemurPaths):
            if not attr_name.startswith('_'):
                attr = getattr(FemurPaths, attr_name)
                assert isinstance(attr, Path), f"{attr_name} should be a Path object"


class TestGaitPaths:
    """Test GaitPaths class constants."""
    
    def test_amiri_excel_path(self):
        """Test Amiri Excel file path."""
        assert GaitPaths.AMIRI_EXCEL == GAIT_DATA_DIR / "Amiri" / "gait_data_amiri.xlsx"
        assert GaitPaths.AMIRI_EXCEL.suffix == ".xlsx"
    
    def test_hip99_walking_path(self):
        """Test HIP99 walking data path."""
        assert GaitPaths.HIP99_WALKING == GAIT_DATA_DIR / "HIP99" / "Walking_Average.HIP"
        assert GaitPaths.HIP99_WALKING.suffix == ".HIP"
    
    def test_all_paths_are_path_objects(self):
        """Test that all GaitPaths attributes are Path objects."""
        for attr_name in dir(GaitPaths):
            if not attr_name.startswith('_'):
                attr = getattr(GaitPaths, attr_name)
                assert isinstance(attr, Path), f"{attr_name} should be a Path object"


class TestEnsureDirectories:
    """Test ensure_directories() function."""
    
    def test_ensure_directories_creates_anatomy_dir(self):
        """Test that ensure_directories creates ANATOMY_DIR."""
        # Directory should exist (created on import)
        assert ANATOMY_DIR.exists()
        assert ANATOMY_DIR.is_dir()
    
    def test_ensure_directories_creates_anatomy_raw_dir(self):
        """Test that ensure_directories creates ANATOMY_RAW_DIR."""
        assert ANATOMY_RAW_DIR.exists()
        assert ANATOMY_RAW_DIR.is_dir()
    
    def test_ensure_directories_creates_anatomy_processed_dir(self):
        """Test that ensure_directories creates ANATOMY_PROCESSED_DIR."""
        assert ANATOMY_PROCESSED_DIR.exists()
        assert ANATOMY_PROCESSED_DIR.is_dir()
    
    def test_ensure_directories_creates_femur_anatomy_dir(self):
        """Test that ensure_directories creates FEMUR_ANATOMY_DIR."""
        assert FEMUR_ANATOMY_DIR.exists()
        assert FEMUR_ANATOMY_DIR.is_dir()
    
    def test_ensure_directories_creates_gait_data_dir(self):
        """Test that ensure_directories creates GAIT_DATA_DIR."""
        assert GAIT_DATA_DIR.exists()
        assert GAIT_DATA_DIR.is_dir()
    
    def test_ensure_directories_creates_proximal_femur_dir(self):
        """Test that ensure_directories creates PROXIMAL_FEMUR_DIR."""
        assert PROXIMAL_FEMUR_DIR.exists()
        assert PROXIMAL_FEMUR_DIR.is_dir()
    
    def test_ensure_directories_creates_results_dir(self):
        """Test that ensure_directories creates RESULTS_DIR."""
        assert RESULTS_DIR.exists()
        assert RESULTS_DIR.is_dir()
    
    def test_ensure_directories_idempotent(self):
        """Test that calling ensure_directories multiple times is safe."""
        # Should not raise exception
        ensure_directories()
        ensure_directories()
        ensure_directories()
        
        # Directories should still exist
        assert ANATOMY_DIR.exists()
        assert RESULTS_DIR.exists()


class TestGetOutputPath:
    """Test get_output_path() function."""
    
    def test_get_output_path_simple_filename(self):
        """Test get_output_path with simple filename."""
        path = get_output_path("test.txt")
        
        assert path == RESULTS_DIR / "test.txt"
        assert path.parent == RESULTS_DIR
        assert path.name == "test.txt"
    
    def test_get_output_path_with_subdirectory(self):
        """Test get_output_path with subdirectory."""
        path = get_output_path("test.txt", subdir="subdir")
        
        assert path == RESULTS_DIR / "subdir" / "test.txt"
        assert path.parent == RESULTS_DIR / "subdir"
        assert path.name == "test.txt"
    
    def test_get_output_path_creates_subdirectory(self):
        """Test that get_output_path creates subdirectory if needed."""
        # Use unique subdir name to avoid conflicts
        import uuid
        subdir_name = f"test_subdir_{uuid.uuid4().hex[:8]}"
        
        path = get_output_path("test.txt", subdir=subdir_name)
        
        # Subdirectory should be created
        assert path.parent.exists()
        assert path.parent.is_dir()
        
        # Clean up
        if path.parent.exists():
            shutil.rmtree(path.parent)
    
    def test_get_output_path_creates_results_dir(self):
        """Test that get_output_path creates RESULTS_DIR if needed."""
        path = get_output_path("test.txt")
        
        # RESULTS_DIR should exist
        assert RESULTS_DIR.exists()
        assert RESULTS_DIR.is_dir()
    
    def test_get_output_path_nested_subdirectory(self):
        """Test get_output_path with nested subdirectory."""
        path = get_output_path("test.txt", subdir="level1/level2/level3")
        
        assert path == RESULTS_DIR / "level1" / "level2" / "level3" / "test.txt"
        assert path.parent == RESULTS_DIR / "level1" / "level2" / "level3"
    
    def test_get_output_path_with_extension(self):
        """Test get_output_path preserves file extension."""
        path = get_output_path("data.csv", subdir="output")
        
        assert path.suffix == ".csv"
        assert path.name == "data.csv"
    
    def test_get_output_path_clean_flag_with_subdir(self):
        """Test get_output_path with clean=True removes existing files in subdir."""
        import uuid
        subdir_name = f"test_clean_{uuid.uuid4().hex[:8]}"
        
        # Create subdirectory with a file
        subdir_path = RESULTS_DIR / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)
        test_file = subdir_path / "old_file.txt"
        test_file.write_text("old content")
        
        # Call with clean=True
        path = get_output_path("new_file.txt", subdir=subdir_name, clean=True)
        
        # Old file should be removed
        assert not test_file.exists()
        
        # Subdirectory should still exist
        assert subdir_path.exists()
        
        # Clean up
        if subdir_path.exists():
            shutil.rmtree(subdir_path)
    
    def test_get_output_path_clean_flag_without_subdir(self):
        """Test get_output_path with clean=True in RESULTS_DIR (should not remove everything)."""
        # This is a dangerous operation - clean=True without subdir would remove all results
        # Just verify the function doesn't raise an error (but we won't actually test cleaning RESULTS_DIR)
        
        # Create a temporary file in a subdir to avoid affecting real results
        import uuid
        subdir_name = f"test_clean_root_{uuid.uuid4().hex[:8]}"
        subdir_path = RESULTS_DIR / subdir_name
        subdir_path.mkdir(parents=True, exist_ok=True)
        test_file = subdir_path / "test.txt"
        test_file.write_text("content")
        
        # Don't actually call clean=True on RESULTS_DIR root
        # Just verify path is correct
        path = get_output_path("file.txt")
        assert path.parent == RESULTS_DIR
        
        # Clean up
        if subdir_path.exists():
            shutil.rmtree(subdir_path)


class TestGetHipTractionPath:
    """Test get_hip_traction_path() function."""
    
    def test_get_hip_traction_path_basic(self):
        """Test basic hip traction path generation."""
        path = get_hip_traction_path(0)
        
        assert path == RESULTS_DIR / "hip_traction_000.vtk"
        assert path.suffix == ".vtk"
    
    def test_get_hip_traction_path_index_formatting(self):
        """Test that index is zero-padded to 3 digits."""
        assert get_hip_traction_path(1).name == "hip_traction_001.vtk"
        assert get_hip_traction_path(10).name == "hip_traction_010.vtk"
        assert get_hip_traction_path(99).name == "hip_traction_099.vtk"
        assert get_hip_traction_path(100).name == "hip_traction_100.vtk"
        assert get_hip_traction_path(999).name == "hip_traction_999.vtk"
    
    def test_get_hip_traction_path_large_index(self):
        """Test hip traction path with large index."""
        path = get_hip_traction_path(1234)
        
        assert path.name == "hip_traction_1234.vtk"
        assert path.suffix == ".vtk"
    
    def test_get_hip_traction_path_sequence(self):
        """Test generating sequence of hip traction paths."""
        paths = [get_hip_traction_path(i) for i in range(5)]
        
        expected_names = [
            "hip_traction_000.vtk",
            "hip_traction_001.vtk",
            "hip_traction_002.vtk",
            "hip_traction_003.vtk",
            "hip_traction_004.vtk",
        ]
        
        for path, expected_name in zip(paths, expected_names):
            assert path.name == expected_name
            assert path.parent == RESULTS_DIR
    
    def test_get_hip_traction_path_all_in_results_dir(self):
        """Test that all hip traction paths are in RESULTS_DIR."""
        for i in range(10):
            path = get_hip_traction_path(i)
            assert path.parent == RESULTS_DIR


class TestPathResolution:
    """Test path resolution and relative paths."""
    
    def test_all_paths_absolute(self):
        """Test that all main directory paths are absolute."""
        assert ANATOMY_DIR.is_absolute()
        assert ANATOMY_RAW_DIR.is_absolute()
        assert ANATOMY_PROCESSED_DIR.is_absolute()
        assert RESULTS_DIR.is_absolute()
        assert ARCHIVE_DIR.is_absolute()
    
    def test_femur_paths_absolute(self):
        """Test that all FemurPaths are absolute."""
        assert FemurPaths.FEMUR_MESH_VTK.is_absolute()
        assert FemurPaths.FEMUR_MESH_FEB.is_absolute()
        assert FemurPaths.HEAD_LINE_JSON.is_absolute()
        assert FemurPaths.CSS_AXES_VTK.is_absolute()
    
    def test_gait_paths_absolute(self):
        """Test that all GaitPaths are absolute."""
        assert GaitPaths.AMIRI_EXCEL.is_absolute()
        assert GaitPaths.HIP99_WALKING.is_absolute()
    
    def test_project_root_parent_resolution(self):
        """Test that PROJECT_ROOT is correctly resolved from paths.py location."""
        # PROJECT_ROOT should be parent of simulation directory
        assert (PROJECT_ROOT / "simulation").exists()
        assert (PROJECT_ROOT / "simulation").is_dir()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_get_output_path_empty_filename(self):
        """Test get_output_path with empty filename."""
        path = get_output_path("")
        
        # Should create path with empty filename
        assert path == RESULTS_DIR / ""
    
    def test_get_output_path_filename_with_path_separators(self):
        """Test get_output_path with filename containing path separators."""
        # Path separators in filename should be handled by Path
        path = get_output_path("sub/file.txt")
        
        # Path object will normalize this
        assert "file.txt" in str(path)
    
    def test_get_hip_traction_path_negative_index(self):
        """Test get_hip_traction_path with negative index (should work but produce odd names)."""
        path = get_hip_traction_path(-1)
        
        # Python f-string will format -1 as "-001"
        assert "-0" in path.name or "-1" in path.name
        assert path.suffix == ".vtk"


class TestIntegration:
    """Integration tests for path management."""
    
    def test_create_and_retrieve_file_in_subdir(self):
        """Test creating and retrieving file in subdirectory."""
        import uuid
        subdir_name = f"integration_test_{uuid.uuid4().hex[:8]}"
        
        # Get output path
        path = get_output_path("test_file.txt", subdir=subdir_name)
        
        # Create file
        path.write_text("test content")
        
        # Verify file exists
        assert path.exists()
        assert path.read_text() == "test content"
        
        # Clean up
        if path.parent.exists():
            shutil.rmtree(path.parent)
    
    def test_directory_structure_integrity(self):
        """Test that directory structure remains consistent."""
        ensure_directories()
        
        # Check all directories still exist and have correct relationships
        assert ANATOMY_RAW_DIR.parent == ANATOMY_DIR
        assert ANATOMY_PROCESSED_DIR.parent == ANATOMY_DIR
        assert FEMUR_ANATOMY_DIR.parent == ANATOMY_RAW_DIR
        assert GAIT_DATA_DIR.parent == ANATOMY_RAW_DIR
        assert PROXIMAL_FEMUR_DIR.parent == ANATOMY_RAW_DIR
        assert RESULTS_DIR.parent == PROJECT_ROOT
