"""Test suite for febio_parser.py - simplified parser for tet4 meshes with surface tags."""

import numpy as np
import pytest
from pathlib import Path
from simulation.febio_parser import FEBio2Dolfinx


@pytest.fixture
def temp_febio_file(tmp_path):
    """Create minimal FEBio file for testing."""
    febio_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="2.5">
  <Mesh>
    <Nodes>
      <node id="1">0.0,0.0,0.0</node>
      <node id="2">1.0,0.0,0.0</node>
      <node id="3">0.0,1.0,0.0</node>
      <node id="4">0.0,0.0,1.0</node>
    </Nodes>
    <Elements type="tet4" mat="1" name="Part1">
      <elem id="1">1,2,3,4</elem>
    </Elements>
    <Surface name="Surface1">
      <tri3 id="1">1,2,3</tri3>
    </Surface>
  </Mesh>
</febio_spec>
"""
    febio_file = tmp_path / "test_mesh.feb"
    febio_file.write_text(febio_content)
    return febio_file


@pytest.fixture
def temp_directory(tmp_path):
    """Provide temporary directory for test outputs."""
    return tmp_path


class TestBasicParsing:
    """Test basic FEBio file parsing functionality."""
    def test_minimal_febio_file(self, temp_febio_file):
        """Test parsing minimal valid FEBio file."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert parser.nodes is not None
        assert parser.elements is not None
        assert parser.surfaces is not None
        assert parser.mesh_dolfinx is not None
        
        assert isinstance(parser.nodes, np.ndarray)
        assert parser.nodes.shape[1] == 3
        
        assert isinstance(parser.elements, np.ndarray)
        assert parser.elements.shape[1] == 4
        
        assert "Surface1" in parser.surfaces
    def test_missing_file(self):
        """Test error handling for missing FEBio file."""
        with pytest.raises(FileNotFoundError):
            FEBio2Dolfinx("nonexistent_file.feb")


class TestElementTypes:
    """Test element type handling."""
    def test_tet4_elements(self, temp_febio_file):
        """Test tet4 element extraction."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        assert parser.elements.shape[1] == 4


class TestSurfaceTags:
    """Test surface boundary tag creation."""
    def test_single_surface(self, temp_febio_file):
        """Test creation of single surface tag."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert hasattr(parser, 'meshtags')
        assert hasattr(parser, 'surface_tags')
        assert "Surface1" in parser.surface_tags
        assert parser.surface_tags["Surface1"] >= 1


class TestMeshExport:
    """Test mesh export functionality."""
    def test_save_mesh_vtk(self, temp_febio_file, temp_directory):
        """Test saving mesh with surface tags to VTK."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        vtk_file = temp_directory / "surface_tags.vtk"
        parser.save_mesh_vtk(str(vtk_file))
        
        assert vtk_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
