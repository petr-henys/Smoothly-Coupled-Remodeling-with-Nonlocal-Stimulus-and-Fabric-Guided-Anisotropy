"""Parse FEBio .feb files and create DOLFINx meshes with boundary tags."""
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv
from basix.ufl import element as basix_element
from dolfinx import mesh, plot
from mpi4py import MPI
from scipy.spatial import KDTree


class FEBio2Dolfinx:
    """Parse FEBio .feb file and create DOLFINx mesh with surface tags."""

    def __init__(self, feb_file: str):
        """Parse FEBio file, create DOLFINx mesh, and match surface tags."""
        self.logger = logging.getLogger(__name__)
        self.feb_file = Path(feb_file)
        self.logger.info("Parsing FEBio file: %s", self.feb_file)
        
        tree = ET.parse(self.feb_file)
        self.mesh_xml = tree.getroot().find("Mesh")
        
        self._extract_nodes_and_elements()
        self._extract_surfaces()
        
        self.mesh_dolfinx = self._create_dolfinx_mesh()
        self.meshtags = self._match_surface_tags()
        
        self.logger.info("FEBio import complete: %d surfaces", len(self.surface_tags))

    def _extract_nodes_and_elements(self) -> None:
        """Extract nodes and tet4 elements from FEBio XML."""
        node_elems = self.mesh_xml.findall(".//Nodes/node")
        if not node_elems:
            raise ValueError("No nodes found in FEBio mesh")
        
        # Build node array (1-indexed in FEBio)
        max_id = max(int(n.get("id")) for n in node_elems)
        self.nodes = np.empty((max_id, 3), dtype=np.float64)
        for n in node_elems:
            idx = int(n.get("id")) - 1
            coords = [float(x) for x in n.text.split(",")]
            self.nodes[idx] = coords
        
        # Extract tet4 elements
        tet_elements = []
        for grp in self.mesh_xml.findall("Elements"):
            etype = grp.get("type")
            if etype != "tet4":
                raise ValueError(f"Only tet4 elements supported, found: {etype}")
            for elem in grp.findall("elem"):
                node_ids = [int(x) - 1 for x in elem.text.split(",")]  # Convert to 0-based
                tet_elements.append(node_ids)
        
        self.elements = np.array(tet_elements, dtype=np.int64)
        self.logger.debug("Extracted %d nodes, %d tet4 elements", len(self.nodes), len(self.elements))
    
    def _extract_surfaces(self) -> None:
        """Extract surface triangle definitions from FEBio XML."""
        self.surfaces = {}
        for surf in self.mesh_xml.findall("Surface"):
            name = surf.get("name")
            if not name:
                continue
            
            tri_elems = surf.findall("tri3")
            if not tri_elems:
                continue
            
            triangles = []
            for tri in tri_elems:
                node_ids = [int(x) - 1 for x in tri.text.split(",")]  # 0-based
                triangles.append(node_ids[:3])  # Only first 3 nodes for tri3
            
            self.surfaces[name] = np.array(triangles, dtype=np.int64)
        
        self.logger.debug("Extracted %d surfaces: %s", len(self.surfaces), list(self.surfaces.keys()))

    def _create_dolfinx_mesh(self) -> mesh.Mesh:
        """Create DOLFINx mesh from tet4 elements and nodes."""
        element = basix_element("Lagrange", "tetrahedron", 1, shape=(3,))
        domain = mesh.create_mesh(MPI.COMM_WORLD, self.elements, element, self.nodes)
        self.logger.debug("Created DOLFINx mesh: %d cells", domain.topology.index_map(3).size_global)
        return domain

    def _match_surface_tags(self) -> mesh.MeshTags:
        """Match FEBio surface triangles to DOLFINx boundary facets using KDTree."""
        fdim = 2  # Facet dimension for 3D mesh
        tdim = 3  # Cell dimension
        self.mesh_dolfinx.topology.create_entities(fdim)
        self.mesh_dolfinx.topology.create_connectivity(fdim, tdim)
        
        # Get boundary facets and their midpoints
        boundary_facets = mesh.exterior_facet_indices(self.mesh_dolfinx.topology)
        facet_midpoints = mesh.compute_midpoints(self.mesh_dolfinx, fdim, boundary_facets)
        tree = KDTree(facet_midpoints)
        
        # Match each surface
        all_facet_indices = []
        all_facet_markers = []
        self.surface_tags = {}
        
        for marker, (surf_name, surf_triangles) in enumerate(self.surfaces.items(), start=1):
            self.surface_tags[surf_name] = marker
            
            # Compute midpoints of FEBio triangles
            tri_midpoints = self.nodes[surf_triangles].mean(axis=1)
            
            # Find nearest DOLFINx facets
            distances, indices = tree.query(tri_midpoints)
            matched_facets = boundary_facets[indices]
            
            # Tolerance check (1% of mesh bounding box diagonal)
            bbox_diag = np.linalg.norm(self.nodes.max(axis=0) - self.nodes.min(axis=0))
            tolerance = 0.01 * bbox_diag
            valid_mask = distances < tolerance
            
            matched_facets = matched_facets[valid_mask]
            n_rejected = (~valid_mask).sum()
            
            all_facet_indices.append(matched_facets)
            all_facet_markers.append(np.full(len(matched_facets), marker, dtype=np.int32))
            
            self.logger.info("Surface '%s' (tag=%d): matched %d/%d triangles (%d rejected)",
                             surf_name, marker, len(matched_facets), len(surf_triangles), n_rejected)
        
        # Combine all tags
        if not all_facet_indices:
            return mesh.meshtags(self.mesh_dolfinx, fdim, 
                                 np.array([], dtype=np.int32), 
                                 np.array([], dtype=np.int32))
        
        facet_indices = np.hstack(all_facet_indices).astype(np.int32)
        facet_values = np.hstack(all_facet_markers).astype(np.int32)
        sort_order = np.argsort(facet_indices)
        
        return mesh.meshtags(self.mesh_dolfinx, fdim, 
                             facet_indices[sort_order], 
                             facet_values[sort_order])

    def save_mesh_vtk(self, output_path: str | Path) -> None:
        """Save boundary facets with surface tags as VTK surface mesh."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fdim = 2
        facet_to_vertex = self.mesh_dolfinx.topology.connectivity(fdim, 0)
        vertices = self.mesh_dolfinx.geometry.x
        
        # Extract tagged facets
        triangles = []
        tag_ids = []
        tag_names = []
        inv_tags = {v: k for k, v in self.surface_tags.items()}
        
        for facet_idx, tag_value in zip(self.meshtags.indices, self.meshtags.values):
            tri_vertices = facet_to_vertex.links(facet_idx)
            triangles.append(tri_vertices)
            tag_ids.append(tag_value)
            tag_names.append(inv_tags[tag_value])
        
        # Build PyVista mesh
        triangles = np.array(triangles, dtype=np.int64)
        faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
        
        surface_mesh = pv.PolyData(vertices, faces)
        surface_mesh.cell_data["SurfaceID"] = np.array(tag_ids, dtype=np.int32)
        surface_mesh.cell_data["SurfaceName"] = np.array(tag_names, dtype=str)
        surface_mesh.save(str(output_path))
        
        self.logger.info("Saved %d surface facets to %s", len(triangles), output_path)

    def __repr__(self) -> str:
        return f"FEBio2Dolfinx({self.feb_file.name}, surfaces={list(self.surface_tags.keys())})"
