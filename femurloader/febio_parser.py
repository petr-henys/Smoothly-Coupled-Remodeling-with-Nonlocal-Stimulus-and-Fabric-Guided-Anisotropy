"""Parse FEBio .feb files and create DOLFINx meshes with boundary tags."""
from __future__ import annotations

import logging
from pathlib import Path

import xml.etree.ElementTree as ET

import numpy as np
import pyvista as pv
import ufl
from basix.ufl import element as basix_element
from dolfinx import mesh
from mpi4py import MPI
from scipy.spatial import KDTree


class FEBio2Dolfinx:
    """Parse FEBio XML and create DOLFINx mesh with surface and volume tags.

    Supports tet4/tet10 elements. Re-indexes nodes to be contiguous starting from 1.
    Volume tags are computed robustly via centroid nearest-neighbor to avoid
    mislabeling after DOLFINx cell renumbering.
    """

    def __init__(self, feb_file: str):
        """Parse FEBio file and create DOLFINx mesh."""
        self.logger = logging.getLogger(__name__)
        self.feb_file = feb_file
        self.logger.info("Parsing FEBio file: %s", feb_file)
        tree = ET.parse(feb_file)
        root = tree.getroot()
        self.root = root
        self.mesh_xml = root.find("Mesh")

        self.nodes: dict[str, np.ndarray] = {}
        self.elements: dict[str, tuple[np.ndarray, str]] = {}
        self.surfaces: dict[str, np.ndarray] = {}
        self.element_sets: dict[str, np.ndarray] = {}
        self.discrete_sets: dict[str, np.ndarray] = {}
        self._discrete_set_names: list[str] = []
        self.materials: dict[str, dict[str, float]] = {}
        self.domain_materials: dict[str, str] = {}

        self._reindex_nodes()
        self._extract_geometry()
        self.mesh_dolfinx = self._create_dolfinx_mesh()
        self.meshtags = self._create_surface_tags(self.mesh_dolfinx)

        self.logger.info(
            "FEBio parsing complete: %d surfaces, %d discrete sets (ligaments)",
            len(self.surfaces),
            len(self.discrete_sets),
        )

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
        """Extract nodes, elements, surfaces, and discrete sets from XML."""
        # Nodes (ordered by node id to ensure consistency with connectivity)
        node_elems = self.mesh_xml.findall(".//Nodes/node")
        if not node_elems:
            raise ValueError("No <node> elements found in FEBio Mesh/Nodes section")
        
        max_id = max(int(n.get("id")) for n in node_elems)

        coords = np.empty((max_id, 3), dtype=float)
        for n in node_elems:
            idx = int(n.get("id")) - 1
            coords[idx] = list(map(float, n.text.split(",")))

        # Store a single ordered node array
        self.nodes = {"AllNodes": coords}
        self.logger.debug("Extracted %d nodes (id-ordered)", coords.shape[0])

        # Elements
        for grp in self.mesh_xml.findall("Elements"):
            name = grp.get("name", "")
            etype = grp.get("type", "")
            conn = (
                np.array(
                    [list(map(int, e.text.split(","))) for e in grp.findall("elem")],
                    dtype=int,
                )
                - 1
            )
            self.elements[name] = (conn, etype)
        self.logger.debug("Extracted %d element groups", len(self.elements))

        # Surfaces
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
        self.logger.debug("Extracted %d surfaces: %s", len(self.surfaces), list(self.surfaces.keys()))

        # Element sets
        for eset in self.mesh_xml.findall("ElementSet"):
            name = eset.get("name", "")
            ids = np.fromstring(eset.text, sep=",", dtype=np.int32)
            self.element_sets[name] = ids

        # Discrete sets
        self._discrete_set_names.clear()
        for dset in self.mesh_xml.findall("DiscreteSet"):
            name = dset.get("name", "") or "unnamed"
            pairs = [
                list(map(int, child.text.split(",")))
                for child in dset
                if child.text
            ]
            pairs = [p for p in pairs if len(p) == 2]
            if pairs:
                self.discrete_sets[name] = np.asarray(pairs, dtype=int)
                self._discrete_set_names.append(name)

        self.logger.info(
            "Extracted %d discrete sets: %s", len(self.discrete_sets), ", ".join(self._discrete_set_names)
        )

        # Materials
        mat_root = self.root.find("Material")
        if mat_root is not None:
            for m in mat_root.findall("material"):
                mname = m.get("name", "")
                E_val = float(m.findtext("E", default="0"))
                v_val = float(m.findtext("v", default="0.3"))
                self.materials[mname] = {"E": E_val, "v": v_val}
        self.logger.debug("Extracted %d materials", len(self.materials))

        # MeshDomains
        md_root = self.root.find("MeshDomains")
        if md_root is not None:
            for sd in md_root.findall("SolidDomain"):
                self.domain_materials[sd.get("name", "")] = sd.get("mat", "")

    def _create_dolfinx_mesh(self) -> mesh.Mesh:
        """Create DOLFINx mesh from element groups with robust cell/material tags.

        DOLFINx může přečíslovat buňky; proto mapujeme skupiny přes NN centroidy.
        Pro geometrii linearizujeme tet10 → tet4 (první 4 rohové uzly).
        """
        self.logger.debug("Creating DOLFINx mesh from FEBio elements")

        # ------------- body points -------------
        pts = np.vstack(list(self.nodes.values())).astype(np.float64)
        self.logger.debug("Total nodes: %d", pts.shape[0])

        # ------------- sběr elementů (tet4/tet10) -------------
        conns_lin: list[np.ndarray] = []           # konektivita pro vytvoření sítě (tet4)
        start = 0
        for name, (conn, etype) in self.elements.items():
            if etype == "tet4":
                conn_lin = conn.astype(np.int64)
            elif etype == "tet10":
                # FEBio tet10: prvních 4 uzlů jsou vrcholy – pro geometrii použijeme tet4
                conn_lin = conn[:, :4].astype(np.int64)
                self.logger.warning(
                    "Group '%s' is tet10 → using first 4 corner nodes as linear tet4 geometry.", name
                )
            else:
                raise ValueError(f"Unsupported element type: {etype}")
            conns_lin.append(conn_lin)

        conn_all = np.vstack(conns_lin)
        self.logger.debug("Total elements (linearized): %d", conn_all.shape[0])

        # ------------- vytvoření DOLFINx mesh (bez tagů) -------------
        cell_el = basix_element("Lagrange", "tetrahedron", 1, shape=(pts.shape[1],))
        dom = mesh.create_mesh(MPI.COMM_WORLD, conn_all, cell_el, pts)

        # ------------- robustní mapování skupin → buňky -------------
        # DOLFINx může buňky přečíslovat, takže TAGY NELZE přiřadit podle prostého pořadí!
        # Uděláme nearest-neighbor (centroid zdrojového prvku → nejbližší centroid buňky v DOLFINx).
        tdim = dom.topology.dim
        num_cells = dom.topology.index_map(tdim).size_local
        all_cells = np.arange(num_cells, dtype=np.int32)
        mid_dom = mesh.compute_midpoints(dom, tdim, all_cells)  # (num_cells, 3)
        tree = KDTree(mid_dom)

        # příprava výstupních polí per buňka
        per_cell_tag = np.full(num_cells, -1, dtype=np.int32)
        per_cell_dist = np.full(num_cells, np.inf)
        self.domain_tags: dict[str, int] = {}      # name -> tag id
        region_labels: list[str] = []

        # Budoucí materiálové parametry per buňka
        E_cells = np.zeros(num_cells, dtype=float)
        nu_cells = np.zeros(num_cells, dtype=float)
        material_labels = np.empty(num_cells, dtype=object)

        next_tag = 0
        for name, (conn_src, _etype) in self.elements.items():
            if name not in self.domain_tags:
                self.domain_tags[name] = next_tag
                region_labels.append(name)
                next_tag += 1
            tag = self.domain_tags[name]

            # centroidy zdrojových elementů (vycházíme z původní konektivity)
            mids_src = pts[conn_src].mean(axis=1)  # (n_el_in_group, 3)

            # NN přiřazení na dolfinx buňky
            dist, loc = tree.query(mids_src, k=1)  # loc: index buňky v dom
            # Konflikty řešíme "nejbližší vítězí"
            for d, c in zip(dist, loc):
                if d < per_cell_dist[c]:
                    per_cell_dist[c] = d
                    per_cell_tag[c] = tag

        # sanity-check: máme otagované všechny buňky?
        n_unassigned = int(np.sum(per_cell_tag < 0))
        if n_unassigned > 0:
            self.logger.warning(
                "There are %d unassigned cells after tagging; these will be set to tag -1.", n_unassigned
            )

        # ------------- sestavení MeshTags pro buňky (dx) -------------
        tagged_idx = np.where(per_cell_tag >= 0)[0].astype(np.int32)
        tagged_vals = per_cell_tag[tagged_idx].astype(np.int32)
        order = np.argsort(tagged_idx)
        self.cell_tag = mesh.meshtags(dom, tdim, tagged_idx[order], tagged_vals[order])
        self.dx = ufl.Measure("dx", domain=dom, subdomain_data=self.cell_tag)

        # ------------- materiály per buňka -------------
        inv_tags = {v: k for k, v in self.domain_tags.items()}  # tag -> group name
        for c in range(num_cells):
            tag = per_cell_tag[c]
            gname = inv_tags.get(tag, "")
            mname = self.domain_materials.get(gname, "")
            props = self.materials.get(mname, {"E": 0.0, "v": 0.3})
            E_cells[c] = float(props["E"])
            nu_cells[c] = float(props["v"])
            material_labels[c] = mname or gname

        # ------------- uložení metadat do objektu -------------
        self.E_cell = E_cells
        self.nu_cell = nu_cells
        self.material_labels = material_labels.astype(str)

        self.region_labels = np.asarray(region_labels, dtype=str)
        self.region_label_map = {label: idx for idx, label in enumerate(self.region_labels)}

        # Mapa region → indexy buněk (z per_cell_tag)
        self._region_index_map = {
            inv_tags[idx]: np.where(per_cell_tag == idx)[0].astype(np.int32)
            for idx in range(len(self.region_labels))
        }

        # Per-buňkový vektor region_id pro vizualizaci (pro .vtk export)
        self.region_ids = per_cell_tag.copy()

        self.logger.debug(
            "Created DOLFINx mesh with %d regions (robust tagged): %s",
            len(self.region_labels),
            list(self.region_labels),
        )
        return dom

    def _create_surface_tags(self, domain: mesh.Mesh) -> mesh.MeshTags:
        """Map FEBio surfaces to DOLFINx boundary facets."""
        self.logger.debug("Creating surface tags for %d surfaces", len(self.surfaces))
        fdim = domain.topology.dim - 1
        domain.topology.create_entities(fdim)
        domain.topology.create_connectivity(fdim, domain.topology.dim)

        boundary_facets = mesh.exterior_facet_indices(domain.topology)
        mids_dom = mesh.compute_midpoints(domain, fdim, boundary_facets)
        tree = KDTree(mids_dom)

        all_nodes = np.vstack(list(self.nodes.values()))
        facet_indices = []
        facet_markers = []
        self.surface_tags = {}  # Map surface names to numeric tags

        for marker, (sname, fconn) in enumerate(self.surfaces.items(), start=1):
            self.surface_tags[sname] = marker
            mids_src = all_nodes[fconn].mean(axis=1)
            dist, i = tree.query(mids_src, k=1)
            facets = boundary_facets[i]
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker, dtype=np.int32))

        if not facet_indices:
            return mesh.meshtags(
                domain, fdim, np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            )

        fi = np.hstack(facet_indices).astype(np.int32)
        fv = np.hstack(facet_markers).astype(np.int32)
        order = np.argsort(fi)

        self.facet_tag = mesh.meshtags(domain, fdim, fi[order], fv[order])
        self.ds = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tag)

        self.logger.debug(
            "Created surface tags: %s", {name: tag for name, tag in self.surface_tags.items()}
        )

        return self.facet_tag

    def get_surface_tag(self, name: str) -> int:
        """Get numeric tag for a named surface.

        Args:
            name: Surface name from FEBio file

        Returns:
            Numeric tag (1-based index)

        Raises:
            KeyError: If surface name not found
        """
        if name not in self.surface_tags:
            available = ", ".join(self.surface_tags.keys())
            raise KeyError(f"Surface '{name}' not found. Available surfaces: {available}")
        return self.surface_tags[name]

    def ds_named(self, name: str) -> ufl.Measure:
        """Get ds measure for a named surface.

        Args:
            name: Surface name from FEBio file

        Returns:
            UFL measure for the named surface
        """
        tag = self.get_surface_tag(name)
        return self.ds(tag)

    def get_domain_tag(self, name: str) -> int:
        """Get numeric tag for a named volume domain.

        Args:
            name: Domain name from FEBio file (e.g., "PelvisBone", "SIJCartilageLeft")

        Returns:
            Numeric tag (0-based index)

        Raises:
            KeyError: If domain name not found
        """
        if name not in self.domain_tags:
            available = ", ".join(self.domain_tags.keys())
            raise KeyError(f"Domain '{name}' not found. Available domains: {available}")
        return self.domain_tags[name]

    def dx_named(self, name: str) -> ufl.Measure:
        """Get dx measure for a named volume domain.

        Args:
            name: Domain name from FEBio file (e.g., "PelvisBone", "SIJCartilageLeft")

        Returns:
            UFL measure for the named domain
        """
        tag = self.get_domain_tag(name)
        return self.dx(tag)

    def save_mesh_with_tags(self, output_dir: Path) -> None:
        """Save mesh with material regions and boundary surface tags in ONE VTK file.
        
        Saves volume mesh with cell data including both region information and
        surface tag information (for cells on boundaries). Cells touching multiple
        surfaces will have all surface names concatenated.
        """
        from dolfinx import plot

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Volume mesh with cell data
        grid = pv.UnstructuredGrid(*plot.vtk_mesh(self.mesh_dolfinx))
        grid.cell_data["RegionID"] = self.region_ids
        region_names = np.array([
            self.region_labels[rid] if rid >= 0 else "unassigned" 
            for rid in self.region_ids
        ])
        grid.cell_data["RegionName"] = region_names
        grid.cell_data["MaterialName"] = self.material_labels
        
        # Add surface information to cells that touch boundaries
        fdim = self.mesh_dolfinx.topology.dim - 1
        has_facets = (
            hasattr(self.facet_tag, 'indices') and 
            self.facet_tag.indices.size > 0
        ) or (
            hasattr(self.facet_tag, 'values') and 
            self.facet_tag.values.size > 0
        )
        
        if has_facets:
            # Map surface tags from facets to cells
            tdim = self.mesh_dolfinx.topology.dim
            num_cells = self.mesh_dolfinx.topology.index_map(tdim).size_local
            
            # Use lists to collect all surfaces per cell
            cell_surface_ids = [[] for _ in range(num_cells)]
            cell_surface_names = [[] for _ in range(num_cells)]
            
            # Create facet-to-cell connectivity
            self.mesh_dolfinx.topology.create_connectivity(fdim, tdim)
            f_to_c = self.mesh_dolfinx.topology.connectivity(fdim, tdim)
            
            # Get facet indices and values
            if hasattr(self.facet_tag, 'indices'):
                facet_indices = self.facet_tag.indices
            else:
                facet_indices = np.arange(len(self.facet_tag.values), dtype=np.int32)
            
            # Create inverse mapping: tag -> surface name
            inv_surface_tags = {v: k for k, v in self.surface_tags.items()}
            
            # Map each tagged facet to its adjacent cell(s)
            for facet_idx, surface_tag in zip(facet_indices, self.facet_tag.values):
                cells = f_to_c.links(facet_idx)
                surface_name = inv_surface_tags.get(surface_tag, f"surface_{surface_tag}")
                for cell in cells:
                    if cell < num_cells:  # Only local cells
                        if surface_tag not in cell_surface_ids[cell]:
                            cell_surface_ids[cell].append(surface_tag)
                            cell_surface_names[cell].append(surface_name)
            
            # Convert to arrays - use first surface for ID, concatenate names
            cell_surface_id_array = np.array([
                ids[0] if ids else -1 for ids in cell_surface_ids
            ], dtype=np.int32)
            
            cell_surface_name_array = np.array([
                ";".join(names) if names else "" for names in cell_surface_names
            ], dtype=object)
            
            # Add surface ID to grid
            grid.cell_data["SurfaceID"] = cell_surface_id_array
            
            # Create DisplayName that prioritizes surface names for visualization
            display_names = np.array([
                cell_surface_name_array[i] if cell_surface_name_array[i] else region_names[i]
                for i in range(num_cells)
            ], dtype=object)
            grid.cell_data["DisplayName"] = display_names.astype(str)
        else:
            # No surfaces, DisplayName = RegionName
            grid.cell_data["DisplayName"] = region_names
        
        # Save single combined mesh
        mesh_path = output_dir / "mesh_complete.vtk"
        grid.save(str(mesh_path))
        self.logger.debug("Saved mesh with regions and surfaces to %s", mesh_path)

    def save_discrete_sets_to_vtk(
        self, filename: str = "ligaments.vtm", *, scalar_name: str | None = None
    ) -> pv.MultiBlock | None:
        """Save discrete sets as PyVista polylines to VTM file."""
        if not self.discrete_sets:
            return None

        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        global_points = np.vstack(list(self.nodes.values()))
        mb = pv.MultiBlock()

        for blk_idx, set_name in enumerate(self.discrete_set_names, 1):
            pairs = self.discrete_sets[set_name]
            pairs0 = pairs - 1
            unique_nodes = np.unique(pairs0)
            g2l = {g: l for l, g in enumerate(unique_nodes)}
            remapped = np.vectorize(g2l.__getitem__)(pairs0)
            n_pairs = remapped.shape[0]
            lines = np.column_stack((np.full(n_pairs, 2), remapped)).ravel().astype(np.int64)
            pts_local = global_points[unique_nodes]
            block = pv.PolyData(pts_local, lines=lines)
            block.field_data["name"] = np.array([set_name], dtype="U")
            if scalar_name:
                block[scalar_name] = np.full(n_pairs, blk_idx, dtype=np.int32)
            mb[set_name] = block

        mb.save(str(output_path))
        return mb

    @property
    def discrete_set_names(self) -> list[str]:
        """Ordered list of discrete set names."""
        return list(self._discrete_set_names)

    @property
    def region_index_map(self) -> dict[str, np.ndarray]:
        """Region label to cell indices mapping."""
        return {label: idxs.copy() for label, idxs in self._region_index_map.items()}

    def __repr__(self) -> str:
        return (
            f"<FEBio2Dolfinx '{Path(self.feb_file).name}' "
            f"nodes={list(self.nodes.keys())} "
            f"elements={list(self.elements.keys())} "
            f"surfaces={list(self.surfaces.keys())} "
            f"discrete_sets={list(self.discrete_sets.keys())}>"
        )

if __name__ == "__main__":
    mdl = FEBio2Dolfinx("anatomy_data/full_model.feb")
