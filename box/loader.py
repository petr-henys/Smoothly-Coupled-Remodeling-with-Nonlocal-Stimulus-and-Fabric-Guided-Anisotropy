"""Pressure/traction loader for box mesh simulations.

Applies uniform or spatially varying pressure on one or more tagged surfaces.
Supports multi-wall loading for hydrostatic pressure or triaxial stress states.

Units: MPa (1 MPa = 1 N/mm²) for pressure/traction with mm-based meshes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple

import numpy as np
import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh

class GradientType(Enum):
    """Type of spatial pressure gradient."""
    LINEAR = "linear"      # Linear variation: f_min at x_min, f_max at x_max
    PARABOLIC = "parabolic"  # Parabolic: f_max at center, f_min at edges
    BENDING = "bending"    # Bending-like: compression on one side, tension on other


@dataclass
class PressureLoadSpec:
    """Pressure load specification.
    
    Each PressureLoadSpec knows which surface(s) it applies to:
    
    For single-wall loading:
        - Set `load_tag` to the facet tag (e.g., BoxMeshBuilder.TAG_TOP)
        - Use `direction` to specify the traction direction
        
    For multi-wall loading (e.g., hydrostatic pressure):
        - Use `wall_tags` to specify which walls receive load
        - Use `wall_directions` to specify inward normal for each wall
        - `load_tag` is ignored when `wall_tags` is non-empty
    
    Attributes:
        magnitude: Pressure magnitude [MPa] (positive = compression into surface)
        load_tag: Facet tag for single-wall loading. Required unless wall_tags is set.
        direction: Load direction as unit vector (default: -z for compression).
                   Used in single-wall mode (when wall_tags is empty).
        wall_tags: Tuple of facet tags for multi-wall loading. Overrides load_tag.
        wall_directions: Direction (inward normal) for each wall in wall_tags.
                         Must have same length as wall_tags.
        gradient_axis: If not None, apply gradient along this axis (0=x, 1=y, 2=z)
        gradient_type: Type of gradient (LINEAR, PARABOLIC, BENDING)
        gradient_range: (min_factor, max_factor) for gradient
        box_extent: (min_coord, max_coord) along gradient axis for normalization
    """
    magnitude: float = 1.0
    load_tag: int | None = None  # Facet tag for single-wall loading
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    wall_tags: tuple[int, ...] = ()  # Non-empty = multi-wall mode
    wall_directions: tuple[tuple[float, float, float], ...] = ()  # One per wall_tag
    gradient_axis: int | None = None  # None = uniform, 0=x, 1=y
    gradient_type: GradientType = GradientType.LINEAR
    gradient_range: tuple[float, float] = (0.5, 1.5)  # (min_factor, max_factor)
    box_extent: tuple[float, float] = (0.0, 10.0)  # (min, max) along gradient axis
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if len(self.wall_tags) != len(self.wall_directions):
            raise ValueError(
                f"wall_tags ({len(self.wall_tags)}) and wall_directions "
                f"({len(self.wall_directions)}) must have the same length"
            )
        # In single-wall mode, load_tag must be set
        if not self.wall_tags and self.load_tag is None:
            raise ValueError(
                "Either load_tag must be set (single-wall) or wall_tags must be "
                "non-empty (multi-wall)"
            )
    
    def get_all_tags(self) -> tuple[int, ...]:
        """Return all facet tags this spec applies to."""
        if self.wall_tags:
            return self.wall_tags
        elif self.load_tag is not None:
            return (self.load_tag,)
        return ()


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
    """Pressure loader for box mesh simulations.
    
    Applies pressure-derived traction (uniform or graded) on one or more surfaces.
    Compatible with the Remodeller framework via the same interface as Loader.
    Traction values are stored ONLY at DOFs on the loaded surfaces.
    
    The loader automatically determines which surfaces need loading from the
    PressureLoadSpec in each loading case. Each spec defines its own load_tag
    (single-wall) or wall_tags (multi-wall).
    """
    
    def __init__(
        self, 
        mesh: dmesh.Mesh, 
        facet_tags: dmesh.MeshTags,
        loading_cases: List[BoxLoadingCase],
    ):
        """Initialize box loader.
        
        Loading cases are precomputed immediately during initialization.
        Tags are automatically extracted from each loading case's PressureLoadSpec.
        
        Args:
            mesh: DOLFINx mesh
            facet_tags: Mesh tags for boundary facets
            loading_cases: List of loading cases to precompute. Each case's
                          PressureLoadSpec defines which surface(s) it applies to.
        """
        self.mesh = mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.facet_tags = facet_tags
        self.gdim = mesh.geometry.dim
        self._loading_cases = loading_cases
        
        # Extract all unique tags from loading cases
        self.load_tags = self._extract_tags_from_cases(loading_cases)
        
        # Backward compatibility: expose first tag as load_tag
        self.load_tag = self.load_tags[0] if self.load_tags else 0
        
        # Vector function space for traction (P1, 3D)
        P1_vec = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(self.gdim,))
        self.V = fem.functionspace(mesh, P1_vec)
        
        # Traction field (updated by set_loading_case)
        self.traction = fem.Function(self.V, name="Traction")
        
        # Cache for precomputed tractions (only owned DOFs on load surfaces)
        self._cached_tractions: dict[str, np.ndarray] = {}
        
        # Total owned DOFs in the function
        self._n_owned_total: int = (
            self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        )
        
        # Initialize DOF indexing for loaded surfaces (per-tag)
        self._init_load_surface_dofs()
        
        # Precompute all loading cases immediately
        self.precompute_loading_cases(self._loading_cases)
    
    @staticmethod
    def _extract_tags_from_cases(loading_cases: List[BoxLoadingCase]) -> Tuple[int, ...]:
        """Extract all unique facet tags from loading cases."""
        tags: set[int] = set()
        for case in loading_cases:
            if case.pressure is not None:
                tags.update(case.pressure.get_all_tags())
        return tuple(sorted(tags))
    
    @property
    def loading_cases(self) -> List[BoxLoadingCase]:
        """Get the list of loading cases."""
        return self._loading_cases

    def _init_load_surface_dofs(self) -> None:
        """Identify DOFs on facets with load_tags and cache their indices/coordinates.
        
        Stores per-tag DOF blocks and coordinates for multi-wall loading.
        After this, traction is only set at DOFs on the loaded surfaces.
        Other DOFs remain zero, making it clear where the load is applied.
        """
        V = self.V
        imap = V.dofmap.index_map
        bs = int(V.dofmap.index_map_bs)
        n_blocks_total = int(imap.size_local + imap.num_ghosts)
        
        tdim = self.mesh.topology.dim
        fdim = tdim - 1
        
        # Ensure connectivity is created (collective operation)
        self.mesh.topology.create_connectivity(fdim, tdim)
        
        # Cache coordinates once
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
        
        # Per-tag storage
        self._owned_blocks_per_tag: Dict[int, np.ndarray] = {}
        self._x_owned_per_tag: Dict[int, np.ndarray] = {}
        self._n_owned_per_tag: Dict[int, int] = {}
        
        # Collect all owned blocks across all tags (for unified access)
        all_owned_blocks_list: List[np.ndarray] = []
        
        for tag in self.load_tags:
            facets = self.facet_tags.find(tag)
            
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
            owned_blocks = np.ascontiguousarray(
                block_ids[block_ids < imap.size_local], dtype=np.int32
            )
            n_owned = int(owned_blocks.size)
            
            self._owned_blocks_per_tag[tag] = owned_blocks
            self._n_owned_per_tag[tag] = n_owned
            
            if n_owned > 0:
                self._x_owned_per_tag[tag] = np.ascontiguousarray(block_coords[owned_blocks])
                all_owned_blocks_list.append(owned_blocks)
            else:
                self._x_owned_per_tag[tag] = np.empty((0, self.gdim), dtype=np.float64)
        
        # Backward compatibility: unified owned blocks (union of all tags, unique)
        if all_owned_blocks_list:
            all_blocks = np.unique(np.concatenate(all_owned_blocks_list))
            self._owned_blocks = np.ascontiguousarray(all_blocks, dtype=np.int32)
        else:
            self._owned_blocks = np.empty(0, dtype=np.int32)
        self._n_owned = int(self._owned_blocks.size)
        
        # Unified coordinates (for backward compatibility)
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
        
        Traction values are computed ONLY at DOFs on the load surfaces.
        Other DOFs remain zero. Supports:
        - Single-wall loading (backward compatible)
        - Multi-wall loading via wall_tags in PressureLoadSpec
        - Uniform, linear gradient, parabolic, and bending distributions.
        
        Args:
            loading_cases: List of loading cases to precompute
        """
        bs = int(self.V.dofmap.index_map_bs)
        
        for case in loading_cases:
            if case.pressure is None:
                # No load case - zero traction (store only surface DOF values)
                self._cached_tractions[case.name] = np.zeros(self._n_owned_total)
            else:
                spec = case.pressure
                p = spec.magnitude
                
                # Initialize full array with zeros
                values = np.zeros(self._n_owned_total, dtype=np.float64)
                
                if spec.wall_tags:
                    # Multi-wall mode: apply pressure to each specified wall
                    for wall_tag, wall_dir in zip(spec.wall_tags, spec.wall_directions):
                        if wall_tag not in self._owned_blocks_per_tag:
                            continue  # Skip tags not in our load_tags
                        
                        d = np.array(wall_dir, dtype=np.float64)
                        d = d / np.linalg.norm(d)  # Normalize direction
                        
                        owned_blocks = self._owned_blocks_per_tag[wall_tag]
                        n_owned = self._n_owned_per_tag[wall_tag]
                        x_owned = self._x_owned_per_tag[wall_tag]
                        
                        if n_owned == 0:
                            continue
                        
                        if spec.gradient_axis is None:
                            # Uniform pressure on this wall
                            traction_vec = p * d
                            wall_values = np.tile(traction_vec, n_owned)
                        else:
                            # Gradient pressure on this wall
                            axis = spec.gradient_axis
                            grad_type = spec.gradient_type
                            f_min, f_max = spec.gradient_range
                            x_min, x_max = spec.box_extent
                            
                            x_transposed = x_owned.T
                            factor = self._compute_gradient_factor(
                                x_transposed, axis, grad_type, f_min, f_max, x_min, x_max
                            )
                            traction_at_dofs = np.outer(factor * p, d)
                            wall_values = traction_at_dofs.ravel()
                        
                        # Write to appropriate DOF indices
                        dof_idx = (owned_blocks[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]).ravel()
                        values[dof_idx] = wall_values
                else:
                    # Single-wall mode: use load_tag from the spec
                    d = np.array(spec.direction, dtype=np.float64)
                    d = d / np.linalg.norm(d)
                    
                    tag = spec.load_tag
                    if tag is None or tag not in self._owned_blocks_per_tag:
                        continue
                    
                    owned_blocks = self._owned_blocks_per_tag[tag]
                    n_owned = self._n_owned_per_tag[tag]
                    x_owned = self._x_owned_per_tag[tag]
                    
                    if n_owned > 0:
                        if spec.gradient_axis is None:
                            # Uniform pressure
                            traction_vec = p * d
                            wall_values = np.tile(traction_vec, n_owned)
                        else:
                            # Gradient pressure
                            axis = spec.gradient_axis
                            grad_type = spec.gradient_type
                            f_min, f_max = spec.gradient_range
                            x_min, x_max = spec.box_extent
                            
                            x_transposed = x_owned.T
                            factor = self._compute_gradient_factor(
                                x_transposed, axis, grad_type, f_min, f_max, x_min, x_max
                            )
                            traction_at_dofs = np.outer(factor * p, d)
                            wall_values = traction_at_dofs.ravel()
                        
                        dof_idx = (owned_blocks[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]).ravel()
                        values[dof_idx] = wall_values
                
                # Cache the computed values
                self._cached_tractions[case.name] = values.copy()
    
    def set_loading_case(self, case_name: str) -> None:
        """Set the current loading case by name.
        
        Traction is set ONLY at DOFs on the load surfaces; other DOFs are zeroed.
        
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
        
        # Cached values are now full-length arrays with zeros at non-surface DOFs
        self.traction.x.array[:self._n_owned_total] = self._cached_tractions[case_name][:self._n_owned_total]
        
        self.traction.x.scatter_forward()
    
    def set_pressure(self, pressure: float, direction: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> None:
        """Directly set uniform pressure on all load surfaces (without caching).
        
        Traction is set ONLY at DOFs on the load surfaces; other DOFs are zeroed.
        Applies the same direction to all surfaces (single-wall mode behavior).
        
        For multi-wall loading with different directions per wall, use
        precompute_loading_cases with PressureLoadSpec.wall_tags instead.
        
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
        
        # Set values at surface DOFs for all load_tags
        for tag in self.load_tags:
            owned_blocks = self._owned_blocks_per_tag[tag]
            n_owned = self._n_owned_per_tag[tag]
            
            if n_owned > 0:
                dof_idx = (owned_blocks[:, None] * bs + np.arange(bs, dtype=np.int32)[None, :]).ravel()
                values = np.tile(traction_vec, n_owned)
                self.traction.x.array[dof_idx] = values
        
        self.traction.x.scatter_forward()
