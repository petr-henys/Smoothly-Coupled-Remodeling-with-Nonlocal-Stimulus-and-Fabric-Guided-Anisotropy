"""Test suite for febio_parser.py - simplified parser for tet4 meshes with surface tags."""

import numpy as np
import pytest
from pathlib import Path
from simulation.febio_parser import FEBio2Dolfinx


@pytest.fixture
def temp_febio_file(tmp_path):
    """Create minimal FEBio file for testing (with enough elements for MPI partitioning)."""
    febio_content = """<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec version="2.5">
  <Mesh>
    <Nodes>
      <node id="1">0.0,0.0,0.0</node>
      <node id="2">1.0,0.0,0.0</node>
      <node id="3">0.0,1.0,0.0</node>
      <node id="4">0.0,0.0,1.0</node>
      <node id="5">1.0,1.0,0.0</node>
      <node id="6">1.0,0.0,1.0</node>
      <node id="7">0.0,1.0,1.0</node>
      <node id="8">1.0,1.0,1.0</node>
    </Nodes>
    <Elements type="tet4" mat="1" name="Part1">
      <elem id="1">1,2,3,4</elem>
      <elem id="2">2,3,4,5</elem>
      <elem id="3">2,4,5,6</elem>
      <elem id="4">3,4,5,7</elem>
      <elem id="5">4,5,6,8</elem>
      <elem id="6">4,5,7,8</elem>
    </Elements>
    <Surface name="Surface1">
      <tri3 id="1">1,2,3</tri3>
      <tri3 id="2">2,3,5</tri3>
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
    
    def test_meshtags_correctness(self, temp_febio_file):
        """Test that meshtags are correctly assigned in serial and parallel."""
        from mpi4py import MPI
        
        parser = FEBio2Dolfinx(str(temp_febio_file))
        comm = MPI.COMM_WORLD
        
        # Check that meshtags have correct dimension (facets)
        assert parser.meshtags.dim == 2
        
        # Check that all tagged facets have valid tag values
        for tag_val in parser.meshtags.values:
            assert tag_val in parser.surface_tags.values()
        
        # Gather total number of tagged facets across all ranks
        local_count = len(parser.meshtags.indices)
        total_count = comm.allreduce(local_count, op=MPI.SUM)
        
        # Expected: 2 triangles in Surface1
        if comm.Get_rank() == 0:
            assert total_count == 2, f"Expected 2 tagged facets, got {total_count}"
    
    def test_meshtags_consistency(self, temp_febio_file):
        """Test that surface matching is consistent between serial and parallel runs."""
        from mpi4py import MPI
        
        parser = FEBio2Dolfinx(str(temp_febio_file))
        comm = MPI.COMM_WORLD
        
        # Gather all tagged facet midpoints
        mesh = parser.mesh_dolfinx
        fdim = 2
        
        if len(parser.meshtags.indices) > 0:
            from dolfinx import mesh as dmesh
            local_midpoints = dmesh.compute_midpoints(mesh, fdim, parser.meshtags.indices)
            local_tags = parser.meshtags.values
        else:
            local_midpoints = np.empty((0, 3), dtype=np.float64)
            local_tags = np.empty(0, dtype=np.int32)
        
        # Gather to rank 0
        all_midpoints = comm.gather(local_midpoints, root=0)
        all_tags = comm.gather(local_tags, root=0)
        
        if comm.Get_rank() == 0:
            # Concatenate all data
            if all_midpoints:
                all_midpoints = np.vstack([m for m in all_midpoints if len(m) > 0])
                all_tags = np.hstack([t for t in all_tags if len(t) > 0])
                
                # Check that we have the expected number of facets
                assert len(all_midpoints) == 2, f"Expected 2 facets, got {len(all_midpoints)}"
                
                # All should have tag 1 (Surface1)
                assert np.all(all_tags == 1), f"All tags should be 1, got {all_tags}"


class TestMeshExport:
    """Test mesh export functionality."""
    def test_save_mesh_vtk(self, temp_febio_file, temp_directory):
        """Test saving mesh with surface tags to VTK."""
        from mpi4py import MPI
        
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        vtk_file = temp_directory / "surface_tags.vtk"
        parser.save_mesh_vtk(str(vtk_file))
        
        # Should write single file in both serial and parallel
        comm = MPI.COMM_WORLD
        comm.Barrier()  # Ensure all ranks finish writing
        
        if comm.Get_rank() == 0:
            assert vtk_file.exists(), f"VTK file {vtk_file} not found"
            
            # Verify no rank-specific files were created
            vtk_files = list(temp_directory.glob("surface_tags*.vtk"))
            assert len(vtk_files) == 1, f"Expected 1 VTK file, found {len(vtk_files)}: {vtk_files}"
            assert vtk_files[0].name == "surface_tags.vtk", f"Expected surface_tags.vtk, got {vtk_files[0].name}"
    
    def test_vtk_contains_surface_data(self, temp_febio_file, temp_directory):
        """Test that VTK file contains correct surface data."""
        from mpi4py import MPI
        import pyvista as pv
        
        parser = FEBio2Dolfinx(str(temp_febio_file))
        
        vtk_file = temp_directory / "surface_with_data.vtk"
        parser.save_mesh_vtk(str(vtk_file))
        
        comm = MPI.COMM_WORLD
        comm.Barrier()
        
        if comm.Get_rank() == 0:
            # Read and verify VTK file
            mesh = pv.read(str(vtk_file))
            
            # Should have 2 triangular cells (from test fixture)
            assert mesh.n_cells == 2, f"Expected 2 cells, got {mesh.n_cells}"
            
            # Should have SurfaceID and SurfaceName data
            assert "SurfaceID" in mesh.cell_data, "Missing SurfaceID in cell data"
            assert "SurfaceName" in mesh.cell_data, "Missing SurfaceName in cell data"
            
            # All cells should be tagged with Surface1 (tag=1)
            assert np.all(mesh.cell_data["SurfaceID"] == 1), f"Expected all tags=1, got {mesh.cell_data['SurfaceID']}"
            assert np.all(mesh.cell_data["SurfaceName"] == "Surface1"), f"Expected all 'Surface1', got {mesh.cell_data['SurfaceName']}"


class TestMeshScaling:
    """Test mesh coordinate scaling."""
    def test_scale_factor_applied(self, temp_febio_file):
        """Test that scale factor correctly scales coordinates."""
        # Parse with default scale
        parser_default = FEBio2Dolfinx(str(temp_febio_file), scale=1.0)
        nodes_default = parser_default.nodes.copy()
        
        # Parse with 1000x scale (m to mm)
        parser_scaled = FEBio2Dolfinx(str(temp_febio_file), scale=1000.0)
        nodes_scaled = parser_scaled.nodes.copy()
        
        # Check that scaled nodes are exactly 1000x the default
        np.testing.assert_allclose(nodes_scaled, nodes_default * 1000.0, rtol=1e-10)
    
    def test_scale_preserves_origin(self, temp_febio_file):
        """Test that scaling preserves the origin point (0,0,0)."""
        parser = FEBio2Dolfinx(str(temp_febio_file), scale=1000.0)
        
        # First node is at origin in test file
        np.testing.assert_allclose(parser.nodes[0], [0.0, 0.0, 0.0], atol=1e-10)
    
    def test_scale_affects_mesh_dimensions(self, temp_febio_file):
        """Test that mesh bounding box is scaled correctly."""
        parser_default = FEBio2Dolfinx(str(temp_febio_file), scale=1.0)
        parser_scaled = FEBio2Dolfinx(str(temp_febio_file), scale=1000.0)
        
        bbox_default = parser_default.nodes.max(axis=0) - parser_default.nodes.min(axis=0)
        bbox_scaled = parser_scaled.nodes.max(axis=0) - parser_scaled.nodes.min(axis=0)
        
        np.testing.assert_allclose(bbox_scaled, bbox_default * 1000.0, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

