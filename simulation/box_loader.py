"""Pressure loader for box mesh simulations.

Provides a simple loader that applies uniform pressure to the top surface
of a box mesh. Much simpler than the femur-specific Loader class.

Units: MPa for pressure (consistent with project convention)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from simulation.box_mesh import BoxMeshBuilder


@dataclass
class PressureLoadSpec:
    """Pressure load specification.
    
    Attributes:
        magnitude: Pressure magnitude [MPa] (positive = compression into surface)
        direction: Load direction as unit vector (default: -z for compression)
        gradient_axis: If not None, apply linear gradient along this axis (0=x, 1=y, 2=z)
        gradient_range: (min_factor, max_factor) for gradient (e.g., (0.5, 1.5) means 50%-150%)
        box_extent: (min_coord, max_coord) along gradient axis for normalization
    """
    magnitude: float = 1.0
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
    gradient_axis: int | None = None  # None = uniform, 0=x, 1=y
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
    
    def precompute_loading_cases(self, loading_cases: List[BoxLoadingCase]) -> None:
        """Precompute traction arrays for all loading cases.
        
        Supports both uniform and gradient pressure distributions.
        
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
                    f_min, f_max = spec.gradient_range
                    x_min, x_max = spec.box_extent
                    
                    def _gradient_traction(x):
                        # Normalize coordinate to [0, 1]
                        t = (x[axis] - x_min) / max(x_max - x_min, 1e-10)
                        t = np.clip(t, 0.0, 1.0)
                        # Linear interpolation of factor
                        factor = f_min + t * (f_max - f_min)
                        # Traction = factor * p * direction
                        traction = np.outer(d, factor * p)
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
