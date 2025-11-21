"""Parse FEBio .feb files and build DOLFINx meshes with boundary tags."""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv
from dolfinx import mesh
from mpi4py import MPI
from scipy.spatial import KDTree

from simulation.logger import get_logger


class FEBio2Dolfinx:
    """Parse FEBio XML and build DOLFINx mesh with surface boundary tags via KDTree matching.

    The mesh coordinates are assumed to already be provided in millimetres. No
    additional scaling is applied so that downstream loads and material
    parameters remain physically consistent.
    """

    def __init__(self, feb_file: str):
        """Parse FEBio file and build DOLFINx mesh with matched surface tags.
        
        Args:
            feb_file: Path to FEBio .feb file
        """
        self.logger = get_logger(MPI.COMM_WORLD, verbose=True, name="FEBio2Dolfinx")
        self.feb_file = Path(feb_file)
        
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Check file existence on rank 0 and broadcast status to avoid deadlock
        file_exists = None
        if rank == 0:
            file_exists = self.feb_file.exists()
        file_exists = comm.bcast(file_exists, root=0)
        
        if not file_exists:
            raise FileNotFoundError(f"FEBio file not found: {self.feb_file}")

        if rank == 0:
            self.logger.info(f"Parsing FEBio file: {self.feb_file} (units: mm)")
            tree = ET.parse(self.feb_file)
            self.mesh_xml = tree.getroot().find("Mesh")
            self._extract_nodes_and_elements()
            self._extract_surfaces()
            self._log_unit_hint()
            # Clean up XML tree
            del self.mesh_xml
            del tree
        else:
            self.nodes = None
            self.elements = None
            self.surfaces = None

        self._broadcast_mesh_data()
        
        self.mesh_dolfinx = self._create_dolfinx_mesh()
        self.meshtags = self._match_surface_tags()
        
        if rank == 0:
            self.logger.info(f"FEBio import complete: {len(self.surface_tags)} surfaces")

    def _broadcast_mesh_data(self):
        """Broadcast nodes and surfaces from rank 0 to all ranks. Elements remain on rank 0."""
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Broadcast nodes (needed for surface matching)
        n_nodes = self.nodes.shape[0] if rank == 0 else None
        n_nodes = comm.bcast(n_nodes, root=0)

        if rank != 0:
            self.nodes = np.empty((n_nodes, 3), dtype=np.float64)
            self.elements = np.empty((0, 4), dtype=np.int64)

        comm.Bcast(self.nodes, root=0)
        self.surfaces = comm.bcast(self.surfaces, root=0)

    def _log_unit_hint(self) -> None:
        """Lightweight heuristic to flag likely unit mistakes (e.g., m vs mm)."""
        bbox = self.nodes.max(axis=0) - self.nodes.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        if diag < 1e-2 or diag > 5e3:
            self.logger.warning(
                "Mesh bounding box diagonal is {:.3g} mm; check source units (expected millimetres).",
                diag,
            )
        else:
            self.logger.info("Mesh extent: [{:.3f}, {:.3f}, {:.3f}] mm", *bbox)

    def _create_dolfinx_mesh(self) -> mesh.Mesh:
        """Build DOLFINx mesh from tet4 connectivity and node coordinates."""
        from basix.ufl import element as basix_element
        
        element = basix_element("Lagrange", "tetrahedron", 1, shape=(3,))
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # Only pass cells and nodes on rank 0 to let dolfinx distribute the mesh
        if rank == 0:
            cells = self.elements
            x = self.nodes
        else:
            cells = self.elements  # Empty on rank > 0
            x = np.empty((0, 3), dtype=np.float64)
            
        partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet)
        domain = mesh.create_mesh(comm, cells, element, x, partitioner=partitioner)
        return domain

    def _extract_nodes_and_elements(self) -> None:
        """Extract nodes and tet4 elements from XML (1-indexed → 0-indexed)."""
        node_elems = self.mesh_xml.findall(".//Nodes/node")
        if not node_elems:
            raise ValueError("No nodes found in FEBio mesh")
        
        # Build node array
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
                node_ids = [int(x) - 1 for x in elem.text.split(",")]
                tet_elements.append(node_ids)
        
        self.elements = np.array(tet_elements, dtype=np.int64)
    
    def _extract_surfaces(self) -> None:
        """Extract surface triangle facets from XML."""
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
                node_ids = [int(x) - 1 for x in tri.text.split(",")]
                triangles.append(node_ids[:3])
            
            self.surfaces[name] = np.array(triangles, dtype=np.int64)

    def _match_surface_tags(self) -> mesh.MeshTags:
        """Match FEBio surface triangles to DOLFINx boundary facets in an MPI-safe way.

        This implementation:
        - computes local exterior boundary facets and their midpoints on each rank,
        - gathers all facet midpoints and their owning ranks on rank 0,
        - performs a global KDTree search on rank 0 to match FEBio surface midpoints
          to the nearest mesh facet midpoints,
        - scatters the matched facet indices and markers back to individual ranks,
        - and finally builds a local `mesh.meshtags` object on each rank.

        The resulting `self.meshtags` is consistent across different MPI layouts.
        """
        comm = self.mesh_dolfinx.comm
        rank = comm.Get_rank()
        size = comm.Get_size()

        fdim = 2  # Facet dimension for 3D mesh
        tdim = 3  # Cell (volume) dimension

        # Ensure facet entities/connectivities exist
        self.mesh_dolfinx.topology.create_entities(fdim)
        self.mesh_dolfinx.topology.create_connectivity(fdim, tdim)

        # --- 1) Local exterior facets and their midpoints ---
        boundary_facets_local = mesh.exterior_facet_indices(self.mesh_dolfinx.topology)
        facet_midpoints_local = mesh.compute_midpoints(self.mesh_dolfinx, fdim, boundary_facets_local)
        n_local_facets = boundary_facets_local.size

        # Gather number of facets per rank so we can reconstruct owners on rank 0
        counts_facets = comm.gather(int(n_local_facets), root=0)

        # Gather local facet indices to rank 0
        if rank == 0:
            counts_facets = counts_facets or []
            total_facets = sum(counts_facets)
            displs_facets = np.concatenate(([0], np.cumsum(counts_facets[:-1], dtype=np.int32))) if counts_facets else np.array([0], dtype=np.int32)
            all_facets = np.empty(total_facets, dtype=np.int32) if total_facets > 0 else np.empty(0, dtype=np.int32)
        else:
            displs_facets = None
            all_facets = None

        comm.Gatherv(
            boundary_facets_local.astype(np.int32),
            [all_facets, counts_facets, displs_facets, MPI.INT],
            root=0,
        )

        # Gather facet midpoints (flattened) to rank 0
        local_midpoints_flat = facet_midpoints_local.ravel()
        if rank == 0:
            counts_coords = [int(c * facet_midpoints_local.shape[1]) for c in counts_facets] if counts_facets else []
            total_coords = sum(counts_coords)
            displs_coords = np.concatenate(([0], np.cumsum(counts_coords[:-1], dtype=np.int32))) if counts_coords else np.array([0], dtype=np.int32)
            all_midpoints_flat = np.empty(total_coords, dtype=np.float64) if total_coords > 0 else np.empty(0, dtype=np.float64)
        else:
            counts_coords = None
            displs_coords = None
            all_midpoints_flat = None

        comm.Gatherv(
            local_midpoints_flat,
            [all_midpoints_flat, counts_coords, displs_coords, MPI.DOUBLE],
            root=0,
        )

        # --- 2) Global KDTree on rank 0 to match FEBio triangles to mesh facets ---
        if rank == 0:
            all_midpoints = all_midpoints_flat.reshape(-1, facet_midpoints_local.shape[1]) if all_midpoints_flat.size > 0 else np.empty((0, 3), dtype=np.float64)
            owners = np.concatenate([np.full(c, r, dtype=np.int32) for r, c in enumerate(counts_facets)]) if counts_facets else np.empty(0, dtype=np.int32)
            tree = KDTree(all_midpoints) if all_midpoints.shape[0] > 0 else None

            # Prepare per-rank containers for matched facets/markers
            per_rank_indices = [[] for _ in range(size)]
            per_rank_values = [[] for _ in range(size)]

            # Surface name -> integer marker mapping (same on all ranks)
            self.surface_tags = {}

            # Tolerance based on global bounding box
            bbox_diag = float(np.linalg.norm(self.nodes.max(axis=0) - self.nodes.min(axis=0)))
            tolerance = 0.01 * bbox_diag

            for marker, (surf_name, surf_triangles) in enumerate(self.surfaces.items(), start=1):
                self.surface_tags[surf_name] = marker

                if surf_triangles.size == 0 or tree is None or all_midpoints.shape[0] == 0:
                    self.logger.warning(f"Surface '{surf_name}' has no triangles or no boundary facets available.")
                    continue

                # Midpoints of FEBio surface triangles in physical coordinates
                tri_midpoints = self.nodes[surf_triangles].mean(axis=1)

                # For each triangle midpoint, find closest mesh facet midpoint
                distances, indices = tree.query(tri_midpoints)

                # Filter by tolerance
                valid_mask = distances < tolerance
                matched_global_idx = indices[valid_mask]
                n_matched = int(matched_global_idx.size)
                n_total = int(tri_midpoints.shape[0])
                n_rejected = int((~valid_mask).sum())

                if n_matched == 0:
                    self.logger.warning(f"Surface '{surf_name}' (tag={marker}) – no triangles matched within tolerance {tolerance:.3e}.")
                    continue

                matched_facets = all_facets[matched_global_idx]
                matched_owners = owners[matched_global_idx]

                # Distribute matched facets to owning ranks
                for r in range(size):
                    rank_mask = matched_owners == r
                    if not np.any(rank_mask):
                        continue
                    local_facets_r = matched_facets[rank_mask].astype(np.int32)
                    markers_r = np.full(local_facets_r.size, marker, dtype=np.int32)
                    per_rank_indices[r].append(local_facets_r)
                    per_rank_values[r].append(markers_r)

                self.logger.info(f"Surface '{surf_name}' (tag={marker}) matched {n_matched}/{n_total} triangles ({n_rejected} rejected)")

            # Concatenate per-rank lists into arrays so we can scatter them
            indices_per_rank = []
            values_per_rank = []
            for r in range(size):
                if per_rank_indices[r]:
                    indices_per_rank.append(np.concatenate(per_rank_indices[r]))
                    values_per_rank.append(np.concatenate(per_rank_values[r]))
                else:
                    indices_per_rank.append(np.empty(0, dtype=np.int32))
                    values_per_rank.append(np.empty(0, dtype=np.int32))
        else:
            indices_per_rank = None
            values_per_rank = None
            self.surface_tags = {}

        # Broadcast surface_tags dictionary to all ranks
        self.surface_tags = comm.bcast(self.surface_tags, root=0)

        # --- 3) Scatter facet indices and markers to each rank ---
        local_indices = np.asarray(comm.scatter(indices_per_rank, root=0), dtype=np.int32)
        local_values = np.asarray(comm.scatter(values_per_rank, root=0), dtype=np.int32)

        # Sort indices
        if local_indices.size > 0:
            order = np.argsort(local_indices)
            local_indices = local_indices[order]
            local_values = local_values[order]

        return mesh.meshtags(self.mesh_dolfinx, fdim, local_indices, local_values)

    def save_mesh_vtk(self, output_path: str | Path) -> None:
        """Save surfaces from original FEBio data as VTK file (rank 0 only)."""
        output_path = Path(output_path)
        comm = self.mesh_dolfinx.comm
        rank = comm.Get_rank()
        
        # Gather all tag IDs from all ranks
        local_tag_ids = set(self.meshtags.values)
        all_local_tags = comm.gather(local_tag_ids, root=0)
        
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Union of all tag IDs across ranks
            matched_tag_ids = set().union(*all_local_tags)
            inv_tags = {v: k for k, v in self.surface_tags.items()}
            
            # Collect triangles from matched surfaces
            all_triangles = []
            all_tags = []
            
            for surf_name, surf_triangles in self.surfaces.items():
                tag_id = self.surface_tags[surf_name]
                if tag_id not in matched_tag_ids:
                    continue
                for tri in surf_triangles:
                    all_triangles.append(tri)
                    all_tags.append(tag_id)
            
            if len(all_triangles) > 0:
                all_triangles = np.array(all_triangles, dtype=np.int64)
                all_tags = np.array(all_tags, dtype=np.int32)
                
                # Build PyVista mesh from FEBio nodes and connectivity
                n_triangles = all_triangles.shape[0]
                faces = np.hstack([np.full((n_triangles, 1), 3), all_triangles]).ravel()
                
                surface_mesh = pv.PolyData(self.nodes, faces)
                surface_mesh.cell_data["SurfaceID"] = all_tags
                surface_mesh.cell_data["SurfaceName"] = [inv_tags[int(tid)] for tid in all_tags]
                surface_mesh.save(str(output_path))
                
                unique_verts = np.unique(all_triangles.ravel())
                self.logger.info(f"Saved {n_triangles} surface facets ({len(unique_verts)} unique vertices) to {output_path}")
            else:
                surface_mesh = pv.PolyData(self.nodes)
                surface_mesh.save(str(output_path))
                self.logger.warning(f"Saved empty surface mesh (no tagged facets) to {output_path}")
        
        comm.Barrier()

    def __repr__(self) -> str:
        return f"FEBio2Dolfinx({self.feb_file.name}, surfaces={list(self.surface_tags.keys())})"
