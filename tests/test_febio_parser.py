"""
Comprehensive test suite for febio_parser.py module.

Tests FEBio XML parsing, DOLFINx mesh creation, boundary tags, material
mapping, discrete sets, and error handling.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


import pyvista as pv
from mpi4py import MPI
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from simulation.febio_parser import FEBio2Dolfinx


# ============================================================================
# BASIC PARSING TESTS
# ============================================================================

class TestBasicParsing:
    """Test basic FEBio file parsing functionality."""
    
    @pytest.mark.febio
    def test_minimal_febio_file(self, temp_febio_file):
        """Test parsing minimal valid FEBio file."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        # Check basic attributes exist
        assert str(parser.feb_file) == str(temp_febio_file)
        assert parser.nodes is not None
        assert parser.elements is not None
        assert parser.surfaces is not None
        assert parser.mesh_dolfinx is not None
        
        # Check node extraction (nodes is now numpy array, not dict)
        assert isinstance(parser.nodes, np.ndarray)
        assert parser.nodes.shape == (4, 3)  # 4 nodes, 3D coords
        
        # Check element extraction (elements is now numpy array, not dict)
        assert isinstance(parser.elements, np.ndarray)
        assert parser.elements.shape == (1, 4)  # 1 element, 4 nodes
        
        # Check surface extraction (surfaces is still dict)
        assert "Surface1" in parser.surfaces
        assert parser.surfaces["Surface1"].shape == (1, 3)  # 1 triangle, 3 nodes
    
    @pytest.mark.febio
    def test_missing_file(self):
        """Test error handling for missing FEBio file."""
        with pytest.raises(FileNotFoundError):
            FEBio2Dolfinx("nonexistent_file.feb")
    
    @pytest.mark.febio
    def test_invalid_xml(self):
        """Test error handling for invalid XML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write("<invalid><unclosed>")
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(Exception):  # XML parsing error
                FEBio2Dolfinx(str(temp_path))
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_missing_nodes(self):
        """Test error handling for FEBio file without nodes."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
    </Mesh>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="No nodes found in FEBio mesh"):
                FEBio2Dolfinx(str(temp_path))
        finally:
            temp_path.unlink()


# ============================================================================
# NODE RE-INDEXING TESTS
# ============================================================================

class TestNodeReindexing:
    """Test node ID re-indexing functionality."""
    
    @pytest.mark.febio
    def test_non_contiguous_node_ids(self):
        """Test handling of non-contiguous node IDs."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="10">0.0,0.0,0.0</node>
            <node id="25">1.0,0.0,0.0</node>
            <node id="30">0.0,1.0,0.0</node>
            <node id="45">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">10,25,30,45</elem>
        </Elements>
        <Surface name="Surface1">
            <tri3 id="1">10,25,30</tri3>
        </Surface>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
            <v>0.3</v>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Nodes should be re-indexed to contiguous array
            assert parser.nodes.shape == (45, 3)  # max_id=45, so array has 45 rows
            
            # Elements should use re-indexed IDs (0-based in parser.elements)
            assert np.all(parser.elements >= 0)
            assert np.all(parser.elements < 45)
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_zero_based_node_ids(self):
        """Test handling of 0-based node IDs (non-standard)."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="0">0.0,0.0,0.0</node>
            <node id="1">1.0,0.0,0.0</node>
            <node id="2">0.0,1.0,0.0</node>
            <node id="3">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">0,1,2,3</elem>
        </Elements>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Should handle re-indexing correctly (nodes is numpy array)
            assert parser.nodes.shape == (4, 3)
            assert parser.elements.shape == (1, 4)
            
        finally:
            temp_path.unlink()


# ============================================================================
# ELEMENT TYPE TESTS
# ============================================================================

class TestElementTypes:
    """Test handling of different element types."""
    
    @pytest.mark.febio
    def test_tet4_elements(self, temp_febio_file):
        """Test tet4 element extraction."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        conn, etype = parser.elements["Part1"]
        assert etype == "tet4"
        assert conn.shape[1] == 4  # 4 nodes per tet4
    
    @pytest.mark.febio
    def test_tet10_elements(self):
        """Test tet10 element handling (linearization to tet4)."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
            <node id="5">0.5,0.0,0.0</node>
            <node id="6">0.5,0.5,0.0</node>
            <node id="7">0.0,0.5,0.0</node>
            <node id="8">0.0,0.0,0.5</node>
            <node id="9">0.5,0.0,0.5</node>
            <node id="10">0.0,0.5,0.5</node>
        </Nodes>
        <Elements type="tet10" name="Part1">
            <elem id="1">1,2,3,4,5,6,7,8,9,10</elem>
        </Elements>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # tet10 should be linearized to tet4 (first 4 nodes)
            conn, etype = parser.elements["Part1"]
            assert etype == "tet10"
            assert conn.shape[1] == 10  # Original tet10 connectivity stored
            
            # DOLFINx mesh should be created successfully
            assert parser.mesh_dolfinx is not None
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_unsupported_element_type(self):
        """Test error handling for unsupported element types."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="hex8" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
    </Mesh>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported element type"):
                FEBio2Dolfinx(str(temp_path))
        finally:
            temp_path.unlink()


# ============================================================================
# SURFACE TAG TESTS
# ============================================================================

class TestSurfaceTags:
    """Test surface boundary tag creation."""
    
    @pytest.mark.febio
    def test_single_surface(self, temp_febio_file):
        """Test creation of single surface tag."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert hasattr(parser, 'facet_tag')
        assert hasattr(parser, 'ds')
        assert "Surface1" in parser.surface_tags
        
        tag_id = parser.get_surface_tag("Surface1")
        assert tag_id >= 1  # 1-based tags
    
    @pytest.mark.febio
    def test_multiple_surfaces(self):
        """Test handling of multiple named surfaces."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
            <node id="5">1.0,1.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
            <elem id="2">2,3,4,5</elem>
        </Elements>
        <Surface name="BottomSurface">
            <tri3 id="1">1,2,3</tri3>
        </Surface>
        <Surface name="TopSurface">
            <tri3 id="2">3,4,5</tri3>
        </Surface>
        <Surface name="SideSurface">
            <tri3 id="3">1,2,4</tri3>
        </Surface>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # All surfaces should be tagged
            assert len(parser.surface_tags) == 3
            assert "BottomSurface" in parser.surface_tags
            assert "TopSurface" in parser.surface_tags
            assert "SideSurface" in parser.surface_tags
            
            # Each should have unique tag ID
            tags = [parser.get_surface_tag(name) for name in parser.surface_tags]
            assert len(set(tags)) == 3  # All unique
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_get_surface_tag_invalid_name(self, temp_febio_file):
        """Test error handling for invalid surface name."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        with pytest.raises(KeyError, match="Surface 'NonExistent' not found"):
            parser.get_surface_tag("NonExistent")
    
    @pytest.mark.febio
    def test_ds_named_measure(self, temp_febio_file):
        """Test ds_named measure creation."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        ds_surf1 = parser.ds_named("Surface1")
        assert ds_surf1 is not None
        # Should be UFL measure
        assert hasattr(ds_surf1, 'subdomain_id')


# ============================================================================
# VOLUME TAG TESTS
# ============================================================================

class TestVolumeTags:
    """Test volume domain tag creation."""
    
    @pytest.mark.febio
    def test_single_volume_domain(self, temp_febio_file):
        """Test single volume domain tagging."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert hasattr(parser, 'cell_tag')
        assert hasattr(parser, 'dx')
        assert "Part1" in parser.domain_tags
        
        tag_id = parser.get_domain_tag("Part1")
        assert tag_id >= 0  # 0-based tags for domains
    
    @pytest.mark.febio
    def test_multiple_volume_domains(self, complex_febio_xml):
        """Test multiple volume domains with robust tagging."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Should have both domains
            assert "Bone" in parser.domain_tags
            assert "Cartilage" in parser.domain_tags
            
            # Each should have unique tag ID
            bone_tag = parser.get_domain_tag("Bone")
            cartilage_tag = parser.get_domain_tag("Cartilage")
            assert bone_tag != cartilage_tag
            
            # Check region labels
            assert "Bone" in parser.region_labels
            assert "Cartilage" in parser.region_labels
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_dx_named_measure(self, temp_febio_file):
        """Test dx_named measure creation."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        dx_part1 = parser.dx_named("Part1")
        assert dx_part1 is not None
        assert hasattr(dx_part1, 'subdomain_id')
    
    @pytest.mark.febio
    def test_get_domain_tag_invalid_name(self, temp_febio_file):
        """Test error handling for invalid domain name."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        with pytest.raises(KeyError, match="Domain 'NonExistent' not found"):
            parser.get_domain_tag("NonExistent")


# ============================================================================
# MATERIAL MAPPING TESTS
# ============================================================================

class TestMaterialMapping:
    """Test material property mapping to cells."""
    
    @pytest.mark.febio
    def test_material_extraction(self, temp_febio_file):
        """Test material property extraction from XML."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert "Material1" in parser.materials
        mat_props = parser.materials["Material1"]
        assert "E" in mat_props
        assert "v" in mat_props
        assert_almost_equal(mat_props["E"], 1000.0)
        assert_almost_equal(mat_props["v"], 0.3)
    
    @pytest.mark.febio
    def test_material_per_cell_assignment(self, temp_febio_file):
        """Test material assignment to mesh cells."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert hasattr(parser, 'E_cell')
        assert hasattr(parser, 'nu_cell')
        assert hasattr(parser, 'material_labels')
        
        # Should have material data for all cells
        num_cells = parser.mesh_dolfinx.topology.index_map(3).size_local
        assert len(parser.E_cell) == num_cells
        assert len(parser.nu_cell) == num_cells
        assert len(parser.material_labels) == num_cells
    
    @pytest.mark.febio
    def test_multiple_materials(self, complex_febio_xml):
        """Test handling of multiple materials."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Should have both materials
            assert "BoneMaterial" in parser.materials
            assert "CartilageMaterial" in parser.materials
            
            # Check different properties
            assert parser.materials["BoneMaterial"]["E"] == 10000.0
            assert parser.materials["CartilageMaterial"]["E"] == 500.0
            
            # Cells should have different materials assigned
            unique_materials = set(parser.material_labels)
            assert len(unique_materials) >= 2  # At least two different materials
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_missing_material_properties(self):
        """Test handling of missing material properties."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Should handle missing materials gracefully with defaults
            assert hasattr(parser, 'E_cell')
            # Default values should be set
            assert np.all(parser.E_cell >= 0)
            
        finally:
            temp_path.unlink()


# ============================================================================
# DISCRETE SET TESTS
# ============================================================================

class TestDiscreteSets:
    """Test discrete set (ligament) handling."""
    
    @pytest.mark.febio
    def test_discrete_set_extraction(self, complex_febio_xml):
        """Test extraction of discrete sets from XML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            assert "Ligament1" in parser.discrete_sets
            assert "Ligament1" in parser.discrete_set_names
            
            # Check discrete set data structure
            ligament_data = parser.discrete_sets["Ligament1"]
            assert ligament_data.shape[1] == 2  # Pairs of node indices
            assert ligament_data.shape[0] == 2  # Two elements
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_discrete_set_names_property(self, complex_febio_xml):
        """Test discrete_set_names property."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            names = parser.discrete_set_names
            assert isinstance(names, list)
            assert "Ligament1" in names
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_save_discrete_sets_to_vtk(self, complex_febio_xml, temp_directory):
        """Test saving discrete sets to VTK file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            vtk_file = temp_directory / "ligaments.vtm"
            multiblock = parser.save_discrete_sets_to_vtk(str(vtk_file))
            
            assert vtk_file.exists()
            assert multiblock is not None
            assert len(multiblock) >= 1  # At least one block
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_no_discrete_sets(self, temp_febio_file):
        """Test handling when no discrete sets exist."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        assert len(parser.discrete_sets) == 0
        assert len(parser.discrete_set_names) == 0
        
        # save_discrete_sets_to_vtk should return None
        result = parser.save_discrete_sets_to_vtk("dummy.vtm")
        assert result is None


# ============================================================================
# MESH EXPORT TESTS
# ============================================================================

class TestMeshExport:
    """Test mesh export functionality."""
    
    @pytest.mark.febio
    def test_save_mesh_with_tags(self, temp_febio_file, temp_directory):
        """Test saving mesh with region and surface tags."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        parser.save_mesh_with_tags(temp_directory)
        
        # Check file was created
        mesh_file = temp_directory / "mesh_complete.vtk"
        assert mesh_file.exists()
        
        # Load and verify
        loaded_mesh = pv.read(str(mesh_file))
        assert loaded_mesh.n_cells > 0
        
        # Check cell data exists
        assert "RegionID" in loaded_mesh.cell_data
        assert "RegionName" in loaded_mesh.cell_data
        assert "MaterialName" in loaded_mesh.cell_data
        assert "DisplayName" in loaded_mesh.cell_data
    
    @pytest.mark.febio
    def test_save_mesh_with_surface_tags(self, complex_febio_xml, temp_directory):
        """Test mesh export with surface information."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            parser.save_mesh_with_tags(temp_directory)
            
            mesh_file = temp_directory / "mesh_complete.vtk"
            loaded_mesh = pv.read(str(mesh_file))
            
            # Should have surface ID data
            assert "SurfaceID" in loaded_mesh.cell_data or "DisplayName" in loaded_mesh.cell_data
            
        finally:
            temp_path.unlink()


# ============================================================================
# REGION INDEX MAP TESTS
# ============================================================================

class TestRegionIndexMap:
    """Test region index mapping functionality."""
    
    @pytest.mark.febio
    def test_region_index_map_property(self, temp_febio_file):
        """Test region_index_map property."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        region_map = parser.region_index_map
        assert isinstance(region_map, dict)
        assert "Part1" in region_map
        
        # Should contain cell indices
        indices = region_map["Part1"]
        assert isinstance(indices, np.ndarray)
        assert len(indices) > 0
        assert np.all(indices >= 0)
    
    @pytest.mark.febio
    def test_multiple_region_indices(self, complex_febio_xml):
        """Test region indices for multiple domains."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            region_map = parser.region_index_map
            assert "Bone" in region_map
            assert "Cartilage" in region_map
            
            # Indices should not overlap
            bone_indices = set(region_map["Bone"])
            cartilage_indices = set(region_map["Cartilage"])
            assert len(bone_indices & cartilage_indices) == 0  # No overlap
            
        finally:
            temp_path.unlink()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.mark.febio
    def test_complete_parsing_workflow(self, complex_febio_xml, temp_directory):
        """Test complete parsing and export workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(complex_febio_xml)
            temp_path = Path(f.name)
        
        try:
            # Parse
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Verify mesh
            assert parser.mesh_dolfinx is not None
            num_cells = parser.mesh_dolfinx.topology.index_map(3).size_local
            assert num_cells > 0
            
            # Verify tags
            assert len(parser.surface_tags) > 0
            assert len(parser.domain_tags) > 0
            
            # Verify materials
            assert len(parser.materials) > 0
            
            # Export
            parser.save_mesh_with_tags(temp_directory)
            if parser.discrete_sets:
                parser.save_discrete_sets_to_vtk(str(temp_directory / "ligaments.vtm"))
            
            # Verify exports
            assert (temp_directory / "mesh_complete.vtk").exists()
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_repr_method(self, temp_febio_file):
        """Test __repr__ method."""
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        repr_str = repr(parser)
        assert "FEBio2Dolfinx" in repr_str
        assert "nodes=" in repr_str
        assert "elements=" in repr_str
        assert "surfaces=" in repr_str


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.febio
    def test_single_element_mesh(self):
        """Test handling of minimal single-element mesh."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            assert parser.mesh_dolfinx is not None
            num_cells = parser.mesh_dolfinx.topology.index_map(3).size_local
            assert num_cells == 1
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_no_surfaces(self):
        """Test handling when no surfaces are defined."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Should handle no surfaces gracefully
            assert len(parser.surfaces) == 0
            assert len(parser.surface_tags) == 0
            
        finally:
            temp_path.unlink()
    
    @pytest.mark.febio
    def test_empty_surface_definition(self):
        """Test handling of empty surface definition."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
        <Surface name="EmptySurface">
        </Surface>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
        </material>
    </Material>
</febio_spec>
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
            f.write(xml_content)
            temp_path = Path(f.name)
        
        try:
            parser = FEBio2Dolfinx(str(temp_path))
            
            # Empty surface should not be added
            assert "EmptySurface" not in parser.surfaces
            
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
