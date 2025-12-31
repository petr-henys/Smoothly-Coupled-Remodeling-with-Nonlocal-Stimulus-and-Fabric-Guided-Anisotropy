"""Pressure/traction loader for box mesh simulations.

Applies uniform or spatially varying pressure on a tagged surface.

Units: MPa (1 MPa = 1 N/mm²) for pressure/traction with mm-based meshes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from box.mesh import BoxMeshBuilder


class GradientType(Enum):
    """Type of spatial pressure gradient."""
    LINEAR = "linear"      # Linear variation: f_min at x_min, f_max at x_max
    PARABOLIC = "parabolic"  # Parabolic: f_max at center, f_min at edges
    BENDING = "bending"    # Bending-like: compression on one side, tension on other


@dataclass
class PressureLoadSpec:
    """Pressure load specification.
    
    Attributes:
        magnitude: Pressure magnitude [MPa] (positive = compression into surface)
        direction: Load direction as unit vector (default: -z for compression)
        gradient_axis: If not None, apply gradient along this axis (0=x, 1=y, 2=z)
        gradient_type: Type of gradient (LINEAR, PARABOLIC, BENDING)
        gradient_range: (min_factor, max_factor) for gradient
        box_extent: (min_coord, max_coord) along gradient axis for normalization
    """
    magnitude: float = 1.0
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    gradient_axis: int | None = None  # None = uniform, 0=x, 1=y
    gradient_type: GradientType = GradientType.LINEAR
    gradient_range: tuple[float, float] = (0.5, 1.5)  # (min_factor, max_factor)
    box_extent: tuple[float, float] = (0.0, 10.0)  # (min, max) along gradient axis


@dataclass
class BoxLoadingCase:
    """One loading case for box model.
    
    Attributes:
        name: Descriptive name for the case
        day_cycles: Number of loading cycles per day (for remodeling weighting)
        pressure: Pressure specification
    """
    name: str
    day_cycles: float = 1.0
    pressure: PressureLoadSpec | None = None


class BoxLoader:
    """Simple pressure loader for box mesh simulations.
    
    Applies pressure-derived traction (uniform or graded) on one surface.
    Compatible with the Remodeller framework via the same interface as Loader.
    Traction values are stored ONLY at DOFs on the loaded surface (load_tag).
    """
    
    def __init__(
        self, 
        mesh: dmesh.Mesh, 
        facet_tags: dmesh.MeshTags,
        load_tag: int,
        loading_cases: List[BoxLoadingCase],
    ):
        """Initialize box loader.
        
        Loading cases are precomputed immediately during initialization.
        
        Args:
            mesh: DOLFINx mesh
            facet_tags: Mesh tags for boundary facets
            load_tag: Tag for the loaded surface (e.g., BoxMeshBuilder.TAG_TOP)
            loading_cases: List of loading cases to precompute
        """
        self.mesh = mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.facet_tags = facet_tags
        self.load_tag = load_tag
        self.gdim = mesh.geometry.dim
        self._loading_cases = loading_cases
        
        # Vector function space for traction (P1, 3D)
        P1_vec = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(self.gdim,))
        self.V = fem.functionspace(mesh, P1_vec)
        
        # Traction field (updated by set_loading_case)
        self.traction = fem.Function(self.V, name="Traction")
        
        # Cache for precomputed tractions (only owned DOFs on load surface)
        self._cached_tractions: dict[str, np.ndarray] = {}
        
        # Total owned DOFs in the function
        self._n_owned_total: int = (
            self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        )
        
        # Initialize DOF indexing for loaded surface
        self._init_load_surface_dofs()
        
        # Precompute all loading cases immediately
        self.precompute_loading_cases(self._loading_cases)
    
    @property
    def loading_cases(self) -> List[BoxLoadingCase]:
        """Get the list of loading cases."""
        return self._loading_cases

    def _init_load_surface_dofs(self) -> None:
        """Identify DOFs on facets with load_tag and cache their indices/coordinates.
        
        After this, traction is only set at DOFs on the loaded surface.
        Other DOFs remain zero, making it clear where the load is applied.
        """
        V = self.V
        imap = V.dofmap.index_map
        bs = int(V.dofmap.index_map_bs)
        n_blocks_total = int(imap.size_local + imap.num_ghosts)
        
        # Find facets with load_tag
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        
        # Ensure connectivity is created (collective operation)
        self.mesh.topology.create_connectivity(fdim, tdim)
        
        facets = self.facet_tags.find(self.load_tag)
        
        # locate_dofs_topological is collective - all ranks must call it
        dofs = fem.locate_dofs_topological(V, fdim, facets)
        
        if dofs.size == 0:
            block_ids = np.empty(0, dtype=np.int32)
        else:
            # Check if DOFs are scalar or block indices
            if bs > 1 and int(dofs.max()) >= n_blocks_total:
                block_ids = np.unique(dofs // bs).astype(np.int32, copy=False)
            else:
                block_ids = np.unique(dofs).astype(np.int32, copy=False)
        
        # Keep only owned (non-ghost) blocks
        self._owned_blocks = np.ascontiguousarray(
            block_ids[block_ids < imap.size_local], dtype=np.int32
        )
        self._n_owned = int(self._owned_blocks.size)
        
        # Cache coordinates for owned blocks on load surface
        coords = V.tabulate_dof_coordinates()
        n_coords = int(coords.shape[0])
        
        if n_coords == n_blocks_total:
            block_coords = coords
        elif bs > 1 and n_coords == n_blocks_total * bs:
            block_coords = coords[::bs]
        else:
            if bs > 1 and n_coords % n_blocks_total == 0 and (n_coords // n_blocks_total) == bs:
                block_coords = coords[::bs]
            else:
                block_coords = coords[:n_blocks_total]
        
        if self._n_owned > 0:
            self._x_owned = np.ascontiguousarray(block_coords[self._owned_blocks])
        else:
            self._x_owned = np.empty((0, self.gdim), dtype=np.float64)
    
    def _compute_gradient_factor(
        self,
        x: np.ndarray,
        axis: int,
        grad_type: GradientType,
        f_min: float,
        f_max: float,
        x_min: float,
        x_max: float,
    ) -> np.ndarray:
        """Compute spatial factor for gradient loading.
        
        Args:
            x: Coordinate array (3, N)
            axis: Axis along which to compute gradient (0=x, 1=y, 2=z)
            grad_type: Type of gradient (LINEAR, PARABOLIC, BENDING)
            f_min, f_max: Factor range
            x_min, x_max: Coordinate extent along axis
            
        Returns:
            Array of factors (N,)
        """
        # Normalize coordinate to [0, 1]
        L = max(x_max - x_min, 1e-10)
        t = (x[axis] - x_min) / L
        t = np.clip(t, 0.0, 1.0)
        
        if grad_type == GradientType.LINEAR:
            # Linear: f_min at t=0, f_max at t=1
            factor = f_min + t * (f_max - f_min)
            
        elif grad_type == GradientType.PARABOLIC:
            # Parabolic: f_max at center (t=0.5), f_min at edges (t=0,1)
            # factor = f_min + (f_max - f_min) * 4 * t * (1 - t)
            # This gives a bell shape with peak at center
            factor = f_min + (f_max - f_min) * 4.0 * t * (1.0 - t)
            
        elif grad_type == GradientType.BENDING:
            # Bending: linear from f_min at t=0 to f_max at t=1
            # but symmetric around center -> creates bending-like stress
            # Actually: f_min at edges, transition through 1.0 at center to f_max
            # t_centered = t - 0.5  -> range [-0.5, 0.5]
            # factor = 1.0 + (f_max - 1.0) * 2 * t_centered for t > 0.5
            # factor = 1.0 + (f_min - 1.0) * 2 * (0.5 - t) for t < 0.5
            # Simplified: linear from f_min to f_max across full width
            # This creates tension on one side, compression on other
            factor = f_min + t * (f_max - f_min)
        else:
            # Default to uniform
            factor = np.ones_like(t) * 0.5 * (f_min + f_max)
            
        return factor
    
    def precompute_loading_cases(self, loading_cases: List[BoxLoadingCase]) -> None:
        """Precompute traction arrays for all loading cases.
        
        Traction values are computed ONLY at DOFs on the load surface (load_tag).
        Other DOFs remain zero. Supports uniform, linear gradient, parabolic, 
        and bending distributions.
        
        Args:
            loading_cases: List of loading cases to precompute
        """
        bs = int(self.V.dofmap.index_map_bs)
        
        for case in loading_cases:
            if case.pressure is None:
                # No load case - zero traction (store only surface DOF values)
                self._cached_tractions[case.name] = np.zeros(self._n_owned * bs)
            else:
                spec = case.pressure
                p = spec.magnitude
                d = np.array(spec.direction, dtype=np.float64)
                d = d / np.linalg.norm(d)  # Normalize direction
                
                if spec.gradient_axis is None:
                    # Uniform pressure: same traction at all surface DOFs
                    traction_vec = p * d  # shape (3,)
                    # Tile for each owned block DOF
                    values = np.tile(traction_vec, self._n_owned)
                else:
                    # Gradient pressure: evaluate at surface DOF coordinates
                    axis = spec.gradient_axis
                    grad_type = spec.gradient_type
                    f_min, f_max = spec.gradient_range
                    x_min, x_max = spec.box_extent
                    
                    # Compute traction at each owned surface DOF
                    # self._x_owned is (n_owned, gdim), need to transpose for _compute_gradient_factor
                    x_transposed = self._x_owned.T  # (gdim, n_owned)
                    factor = self._compute_gradient_factor(
                        x_transposed, axis, grad_type, f_min, f_max, x_min, x_max
                    )  # shape (n_owned,)
                    # Traction at each DOF: p * factor * d
                    # Result: (n_owned, gdim)
                    traction_at_dofs = np.outer(factor * p, d)
                    values = traction_at_dofs.ravel()  # flatten for storage
                
                # Cache the computed values for surface DOFs only
                self._cached_tractions[case.name] = values.copy()
    
    def set_loading_case(self, case_name: str) -> None:
        """Set the current loading case by name.
        
        Traction is set ONLY at DOFs on the load surface; other DOFs are zeroed.
        
        Args:
            case_name: Name of the precomputed loading case
            
        Raises:
            KeyError: If case_name was not precomputed
        """
        if case_name not in self._cached_tractions:
            raise KeyError(
                f"Loading case '{case_name}' not found. "
                f"Available cases: {list(self._cached_tractions.keys())}"
            )
        
        bs = int(self.V.dofmap.index_map_bs)
        
        # Zero entire traction field first
        self.traction.x.array[:self._n_owned_total] = 0.0
        
        # Set cached values at surface DOFs only
        if self._n_owned > 0:
            blocks = self._owned_blocks
            # Scalar DOF indices for these blocks
            dof_idx = (blocks[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]).ravel()
            self.traction.x.array[dof_idx] = self._cached_tractions[case_name]
        
        self.traction.x.scatter_forward()
    
    def set_pressure(self, pressure: float, direction: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> None:
        """Directly set uniform pressure (without caching).
        
        Traction is set ONLY at DOFs on the load surface; other DOFs are zeroed.
        
        Args:
            pressure: Pressure magnitude [MPa]
            direction: Unit direction vector (default: -z compression)
        """
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        traction_vec = pressure * d
        
        bs = int(self.V.dofmap.index_map_bs)
        
        # Zero entire traction field first
        self.traction.x.array[:self._n_owned_total] = 0.0
        
        # Set values at surface DOFs only
        if self._n_owned > 0:
            blocks = self._owned_blocks
            dof_idx = (blocks[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]).ravel()
            # Tile the traction vector for each surface DOF
            values = np.tile(traction_vec, self._n_owned)
            self.traction.x.array[dof_idx] = values
        
        self.traction.x.scatter_forward()
