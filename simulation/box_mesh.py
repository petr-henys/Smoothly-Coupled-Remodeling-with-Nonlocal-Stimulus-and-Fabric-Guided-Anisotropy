"""Box mesh generation with boundary facet tagging.

Creates a box mesh with proper facet tags for:
- Bottom (z=0): Fixed support (Dirichlet BC)
- Top (z=H): Pressure loading (Neumann BC)
- Sides: Free (no BC)

Units: mm (consistent with project convention)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from dolfinx import mesh
from mpi4py import MPI


@dataclass
class BoxGeometry:
    """Box geometry specification.
    
    Attributes:
        Lx: Length in x-direction [mm]
        Ly: Length in y-direction [mm]  
        Lz: Height in z-direction [mm]
        nx: Number of elements in x
        ny: Number of elements in y
        nz: Number of elements in z
    """
    Lx: float = 10.0
    Ly: float = 10.0
    Lz: float = 30.0
    nx: int = 5
    ny: int = 5
    nz: int = 15


class BoxMeshBuilder:
    """Build a box mesh with tagged boundaries for bone remodeling simulation.
    
    Facet tag convention:
        1: Bottom (z=0) - Fixed support
        2: Top (z=Lz) - Pressure loading  
        3: Side x=0
        4: Side x=Lx
        5: Side y=0
        6: Side y=Ly
    """
    
    # Tag constants for clarity
    TAG_BOTTOM = 1   # z = 0 (Dirichlet: fixed)
    TAG_TOP = 2      # z = Lz (Neumann: pressure)
    TAG_X_MIN = 3    # x = 0 (free)
    TAG_X_MAX = 4    # x = Lx (free)
    TAG_Y_MIN = 5    # y = 0 (free)
    TAG_Y_MAX = 6    # y = Ly (free)
    
    def __init__(self, geometry: BoxGeometry | None = None, comm: MPI.Comm | None = None):
        """Initialize box mesh builder.
        
        Args:
            geometry: Box geometry specification (defaults to BoxGeometry())
            comm: MPI communicator (defaults to COMM_WORLD)
        """
        self.geometry = geometry or BoxGeometry()
        self.comm = comm or MPI.COMM_WORLD
        
        self._mesh: mesh.Mesh | None = None
        self._facet_tags: mesh.MeshTags | None = None
    
    def build(self) -> Tuple[mesh.Mesh, mesh.MeshTags]:
        """Build the box mesh with facet tags.
        
        Returns:
            Tuple of (mesh, facet_tags)
        """
        g = self.geometry
        
        # Create box mesh (hexahedra for better accuracy)
        self._mesh = mesh.create_box(
            self.comm,
            [np.array([0.0, 0.0, 0.0]), np.array([g.Lx, g.Ly, g.Lz])],
            [g.nx, g.ny, g.nz],
            cell_type=mesh.CellType.tetrahedron,
            ghost_mode=mesh.GhostMode.shared_facet,
        )
        
        # Create facet tags
        self._facet_tags = self._build_facet_tags()
        
        return self._mesh, self._facet_tags
    
    def _build_facet_tags(self) -> mesh.MeshTags:
        """Create facet tags for all boundaries."""
        g = self.geometry
        m = self._mesh
        fdim = m.topology.dim - 1
        
        # Define boundary locator functions with tolerance
        tol = 1e-10 * max(g.Lx, g.Ly, g.Lz)
        
        boundaries = [
            (self.TAG_BOTTOM, lambda x: np.isclose(x[2], 0.0, atol=tol)),
            (self.TAG_TOP, lambda x: np.isclose(x[2], g.Lz, atol=tol)),
            (self.TAG_X_MIN, lambda x: np.isclose(x[0], 0.0, atol=tol)),
            (self.TAG_X_MAX, lambda x: np.isclose(x[0], g.Lx, atol=tol)),
            (self.TAG_Y_MIN, lambda x: np.isclose(x[1], 0.0, atol=tol)),
            (self.TAG_Y_MAX, lambda x: np.isclose(x[1], g.Ly, atol=tol)),
        ]
        
        # Collect all facet indices and markers
        facet_indices = []
        facet_markers = []
        
        for marker, locator in boundaries:
            facets = mesh.locate_entities_boundary(m, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        
        # Combine and sort
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        
        # Ensure connectivity exists
        m.topology.create_connectivity(fdim, m.topology.dim)
        
        return mesh.meshtags(
            m, fdim, 
            facet_indices[sorted_facets], 
            facet_markers[sorted_facets]
        )
    
    @property
    def mesh(self) -> mesh.Mesh:
        """Return the mesh (builds if not already built)."""
        if self._mesh is None:
            self.build()
        return self._mesh
    
    @property
    def facet_tags(self) -> mesh.MeshTags:
        """Return the facet tags (builds if not already built)."""
        if self._facet_tags is None:
            self.build()
        return self._facet_tags


def create_box_mesh(
    Lx: float = 10.0,
    Ly: float = 10.0, 
    Lz: float = 30.0,
    nx: int = 5,
    ny: int = 5,
    nz: int = 15,
    comm: MPI.Comm | None = None,
) -> Tuple[mesh.Mesh, mesh.MeshTags]:
    """Convenience function to create a box mesh with facet tags.
    
    Args:
        Lx: Length in x-direction [mm]
        Ly: Length in y-direction [mm]
        Lz: Height in z-direction [mm]
        nx: Number of elements in x
        ny: Number of elements in y
        nz: Number of elements in z
        comm: MPI communicator
        
    Returns:
        Tuple of (mesh, facet_tags)
        
    Facet tags:
        1: Bottom (z=0) - Fixed
        2: Top (z=Lz) - Loaded
    """
    geometry = BoxGeometry(Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz)
    builder = BoxMeshBuilder(geometry, comm)
    return builder.build()
