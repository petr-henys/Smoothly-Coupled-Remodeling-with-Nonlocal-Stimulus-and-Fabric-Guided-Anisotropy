"""Parse FEBio .feb files and create DOLFINx meshes with boundary tags."""
from __future__ import annotations

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
    """Parse FEBio XML and create DOLFINx mesh with surface meshtags.

    Supports tet4 elements only. Returns mesh and boundary facet tags.
    """

    def __init__(self, feb_file: str):
        """Parse FEBio file and create DOLFINx mesh with surface tags."""
        self.logger = logging.getLogger(__name__)
        self.feb_file = feb_file
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            self.logger.info("Parsing FEBio file: %s", feb_file)
            tree = ET.parse(feb_file)
            root = tree.getroot()
            self.mesh_xml = root.find("Mesh")

            self._reindex_nodes()
            self._extract_geometry()
        else:
            self.nodes = None
            self.elements = None
            self.surfaces = None

        self.mesh_dolfinx = self._create_dolfinx_mesh()
        self.meshtags = self._create_surface_tags(self.mesh_dolfinx)

        if rank == 0:
            self.logger.info("FEBio parsing complete: %d surfaces", len(self.surfaces))

    def _reindex_nodes(self) -> None:
        """Re-index node IDs to be contiguous starting from 1."""
        node_elems = self.mesh_xml.findall(".//Nodes/node")
        if not node_elems:
            # Defer to _extract_geometry's explicit error message for consistency
            raise ValueError("No <node> elements found in FEBio Mesh/Nodes section")

        old_ids = sorted({int(n.get("id")) for n in node_elems})
        id_map = {old: new for new, old in enumerate(old_ids, 1)}

        for node in self.mesh_xml.findall(".//Nodes/*"):
            node.set("id", str(id_map[int(node.get("id"))]))

        for tag in ("Elements/elem", "Surface/tri3", "Surface/tri6", "DiscreteSet/*"):
            for e in self.mesh_xml.findall(tag):
                e.text = ",".join(str(id_map[int(i)]) for i in e.text.strip().split(","))

    def _extract_geometry(self) -> None:
        """Extract nodes, elements, and surfaces from XML."""
        # Nodes
        node_elems = self.mesh_xml.findall(".//Nodes/node")
        if not node_elems:
            raise ValueError("No <node> elements found in FEBio Mesh/Nodes section")
        
        max_id = max(int(n.get("id")) for n in node_elems)
        self.nodes = np.empty((max_id, 3), dtype=float)
        for n in node_elems:
            idx = int(n.get("id")) - 1
            self.nodes[idx] = list(map(float, n.text.split(",")))
        self.logger.debug("Extracted %d nodes", self.nodes.shape[0])

        # Elements
        self.elements = []
        for grp in self.mesh_xml.findall("Elements"):
            etype = grp.get("type", "")
            conn = (
                np.array(
                    [list(map(int, e.text.split(","))) for e in grp.findall("elem")],
                    dtype=int,
                )
                - 1
            )
            self.elements.append((conn, etype))
        self.logger.debug("Extracted %d element groups", len(self.elements))

        # Surfaces
        self.surfaces = {}
        for surf in self.mesh_xml.findall("Surface"):
            name = surf.get("name", "")
            elems = surf.findall("tri3") or surf.findall("tri6")
            if elems:
                conn = (
                    np.array(
                        [list(map(int, e.text.split(","))) for e in elems], dtype=int
                    )
                    - 1
                )
                self.surfaces[name] = conn
        self.logger.debug("Extracted %d surfaces", len(self.surfaces))

    def _create_dolfinx_mesh(self) -> mesh.Mesh:
        """Create DOLFINx mesh from FEBio tet4 elements."""
        self.logger.debug("Creating DOLFINx mesh from FEBio elements")
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            # Collect all elements (tet4 only)
            conns = []
            for conn, etype in self.elements:
                if etype != "tet4":
                    raise ValueError(f"Only tet4 elements supported, found: {etype}")
                conns.append(conn.astype(np.int64))

            conn_all = np.vstack(conns)
            self.logger.debug("Total tet4 elements: %d", conn_all.shape[0])
            
            nodes = self.nodes
            num_nodes = nodes.shape[0]
        else:
            conn_all = np.zeros((0, 4), dtype=np.int64)
            nodes = np.zeros((0, 3), dtype=float)
            num_nodes = 0

        # Broadcast number of nodes to define element shape correctly on all ranks
        num_nodes = comm.bcast(num_nodes, root=0)

        # Create DOLFINx mesh
        # Note: create_mesh distributes the mesh from the provided data.
        # We provide full data on rank 0 and empty on others.
        cell_el = basix_element("Lagrange", "tetrahedron", 1, shape=(3,))
        dom = mesh.create_mesh(MPI.COMM_WORLD, conn_all, cell_el, nodes)
        
        self.logger.debug("Created DOLFINx mesh")
        return dom

    def _create_surface_tags(self, domain: mesh.Mesh) -> mesh.MeshTags:
        """Map FEBio surfaces to DOLFINx boundary facet meshtags."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Prepare surface geometry on Rank 0 for broadcast
        if rank == 0:
            surface_data = []
            for sname, fconn in self.surfaces.items():
                # Extract coordinates for this surface's triangles
                # fconn is (N_tri, 3) indices into self.nodes
                coords = self.nodes[fconn] # (N_tri, 3, 3)
                surface_data.append((sname, coords))
        else:
            surface_data = None
            
        # Broadcast surface definitions (much smaller than full mesh)
        surface_data = comm.bcast(surface_data, root=0)
        
        # Reconstruct self.surfaces on other ranks just for the keys/names if needed,
        # but we mainly need the geometry to match.
        # We'll populate self.surfaces keys for consistency in __repr__
        if rank != 0:
            self.surfaces = {sname: None for sname, _ in surface_data}

        self.logger.debug("Creating surface tags for %d surfaces", len(surface_data))
        fdim = domain.topology.dim - 1
        domain.topology.create_entities(fdim)
        domain.topology.create_connectivity(fdim, domain.topology.dim)

        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        mids_dom = mesh.compute_midpoints(domain, fdim, boundary_facets)
        
        if len(mids_dom) > 0:
            tree = KDTree(mids_dom)
        else:
            tree = None

        facet_indices = []
        facet_markers = []
        self.surface_tags = {}
        
        # Tolerance for matching surface midpoints (e.g. 0.001 mm)
        tol = 1e-3

        for marker, (sname, coords) in enumerate(surface_data, start=1):
            self.surface_tags[sname] = marker
            
            if tree is None:
                continue

            # coords is (N_tri, 3, 3), mean over axis 1 gives (N_tri, 3) midpoints
            mids_src = coords.mean(axis=1)
            dist, i = tree.query(mids_src, k=1)
            
            # Filter by distance to ensure we only tag facets that actually match
            mask = dist < tol
            valid_indices = i[mask]
            
            if len(valid_indices) > 0:
                facets = boundary_facets[valid_indices]
                facet_indices.append(facets)
                facet_markers.append(np.full_like(facets, marker, dtype=np.int32))

        if not facet_indices:
            return mesh.meshtags(
                domain, fdim, np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            )

        fi = np.hstack(facet_indices).astype(np.int32)
        fv = np.hstack(facet_markers).astype(np.int32)
        order = np.argsort(fi)

        tags = mesh.meshtags(domain, fdim, fi[order], fv[order])
        self.logger.debug("Created surface tags: %s", list(self.surface_tags.keys()))
        return tags

    def save_mesh_vtk(self, output_path: str | Path) -> None:
        """Save mesh with surface tags to VTK for inspection.
        
        Args:
            output_path: Path to output VTK file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PyVista grid from DOLFINx mesh
        grid = pv.UnstructuredGrid(*plot.vtk_mesh(self.mesh_dolfinx))
        
        # Add surface tag information to cells that touch boundaries
        fdim = self.mesh_dolfinx.topology.dim - 1
        tdim = self.mesh_dolfinx.topology.dim
        num_cells = self.mesh_dolfinx.topology.index_map(tdim).size_local
        
        # Initialize cell data arrays
        cell_surface_ids = np.full(num_cells, -1, dtype=np.int32)
        cell_surface_names = np.full(num_cells, "", dtype=object)
        
        # Map facets to cells
        self.mesh_dolfinx.topology.create_connectivity(fdim, tdim)
        f_to_c = self.mesh_dolfinx.topology.connectivity(fdim, tdim)
        
        # Inverse mapping: tag -> surface name
        inv_surface_tags = {v: k for k, v in self.surface_tags.items()}
        
        # Map each tagged facet to its adjacent cell(s)
        for facet_idx, surface_tag in zip(self.meshtags.indices, self.meshtags.values):
            cells = f_to_c.links(facet_idx)
            surface_name = inv_surface_tags.get(surface_tag, f"surface_{surface_tag}")
            for cell in cells:
                if cell < num_cells:  # Only local cells
                    if cell_surface_ids[cell] == -1:
                        cell_surface_ids[cell] = surface_tag
                        cell_surface_names[cell] = surface_name
                    else:
                        # Concatenate multiple surface names
                        cell_surface_names[cell] += f";{surface_name}"
        
        # Add to grid
        grid.cell_data["SurfaceID"] = cell_surface_ids
        grid.cell_data["SurfaceName"] = cell_surface_names.astype(str)
        
        # Save
        grid.save(str(output_path))
        self.logger.info("Saved mesh with surface tags to %s", output_path)

    def __repr__(self) -> str:
        return (
            f"<FEBio2Dolfinx surfaces={list(self.surfaces.keys())}>"
        )
