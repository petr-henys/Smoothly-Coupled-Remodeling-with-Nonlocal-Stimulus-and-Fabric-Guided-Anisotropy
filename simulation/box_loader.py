"""Pressure loader for box mesh simulations.

Provides a simple loader that applies uniform pressure to the top surface
of a box mesh. Much simpler than the femur-specific Loader class.

Units: MPa for pressure (consistent with project convention)
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

from simulation.box_mesh import BoxMeshBuilder


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
    
    Applies uniform pressure (or traction) to the top surface of a box.
    Compatible with the Remodeller framework via the same interface as Loader.
    """
    
    def __init__(
        self, 
        mesh: dmesh.Mesh, 
        facet_tags: dmesh.MeshTags,
        load_tag: int = BoxMeshBuilder.TAG_TOP,
    ):
        """Initialize box loader.
        
        Args:
            mesh: DOLFINx mesh
            facet_tags: Mesh tags for boundary facets
            load_tag: Tag for the loaded surface (default: top surface)
        """
        self.mesh = mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.facet_tags = facet_tags
        self.load_tag = load_tag
        self.gdim = mesh.geometry.dim
        
        # Vector function space for traction (P1, 3D)
        P1_vec = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(self.gdim,))
        self.V = fem.functionspace(mesh, P1_vec)
        
        # Traction field (updated by set_loading_case)
        self.traction = fem.Function(self.V, name="Traction")
        
        # Cache for precomputed tractions
        self._cached_tractions: dict[str, np.ndarray] = {}
        self._n_owned: int = 0
        
        # Compute owned DOFs
        self._n_owned = (
            self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
        )
    
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
        
        Supports uniform, linear gradient, parabolic, and bending distributions.
        
        Args:
            loading_cases: List of loading cases to precompute
        """
        for case in loading_cases:
            if case.pressure is None:
                # No load case - zero traction
                self._cached_tractions[case.name] = np.zeros(self._n_owned)
            else:
                spec = case.pressure
                p = spec.magnitude
                d = np.array(spec.direction, dtype=np.float64)
                d = d / np.linalg.norm(d)  # Normalize direction
                
                if spec.gradient_axis is None:
                    # Uniform pressure
                    traction_vec = p * d
                    
                    def _uniform_traction(x):
                        return np.tile(traction_vec.reshape(3, 1), (1, x.shape[1]))
                    
                    temp_fn = fem.Function(self.V)
                    temp_fn.interpolate(_uniform_traction)
                else:
                    # Gradient pressure along specified axis
                    axis = spec.gradient_axis
                    grad_type = spec.gradient_type
                    f_min, f_max = spec.gradient_range
                    x_min, x_max = spec.box_extent
                    
                    # Capture variables for closure
                    _axis = axis
                    _grad_type = grad_type
                    _f_min, _f_max = f_min, f_max
                    _x_min, _x_max = x_min, x_max
                    _p = p
                    _d = d
                    _self = self
                    
                    def _gradient_traction(x):
                        factor = _self._compute_gradient_factor(
                            x, _axis, _grad_type, _f_min, _f_max, _x_min, _x_max
                        )
                        # Traction = factor * p * direction
                        traction = np.outer(_d, factor * _p)
                        return traction
                    
                    temp_fn = fem.Function(self.V)
                    temp_fn.interpolate(_gradient_traction)
                
                # Cache the owned DOFs
                self._cached_tractions[case.name] = temp_fn.x.array[:self._n_owned].copy()
    
    def set_loading_case(self, case_name: str) -> None:
        """Set the current loading case by name.
        
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
        
        # Copy cached traction to the function
        self.traction.x.array[:self._n_owned] = self._cached_tractions[case_name]
        self.traction.x.scatter_forward()
    
    def set_pressure(self, pressure: float, direction: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> None:
        """Directly set uniform pressure (without caching).
        
        Convenience method for simple use cases.
        
        Args:
            pressure: Pressure magnitude [MPa]
            direction: Unit direction vector (default: -z compression)
        """
        d = np.array(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        traction_vec = pressure * d
        
        def _uniform_traction(x):
            return np.tile(traction_vec.reshape(3, 1), (1, x.shape[1]))
        
        self.traction.interpolate(_uniform_traction)
        self.traction.x.scatter_forward()
