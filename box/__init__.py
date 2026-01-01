"""Box model package for bone remodeling simulations.

Provides mesh generation, loading, and solver factory for box-shaped specimens.
"""

from box.mesh import BoxGeometry, BoxMeshBuilder, create_box_mesh
from box.loader import (
    BoxLoader,
    BoxLoadingCase,
    GradientType,
    PressureLoadSpec,
)
from box.scenarios import (
    get_single_pressure_case,
    get_gradient_pressure_case,
    get_parabolic_pressure_case,
    get_bending_like_case,
    get_physiological_compression_cases,
    get_overload_scenarios,
    get_disuse_scenarios,
    get_cyclic_loading_cases,
    get_hydrostatic_pressure_case,
    get_triaxial_pressure_case,
)
from box.factory import BoxSolverFactory, BoxDriver

__all__ = [
    # Mesh
    "BoxGeometry",
    "BoxMeshBuilder",
    "create_box_mesh",
    # Loader
    "BoxLoader",
    "BoxLoadingCase",
    "GradientType",
    "PressureLoadSpec",
    # Scenarios
    "get_single_pressure_case",
    "get_gradient_pressure_case",
    "get_parabolic_pressure_case",
    "get_bending_like_case",
    "get_physiological_compression_cases",
    "get_overload_scenarios",
    "get_disuse_scenarios",
    "get_cyclic_loading_cases",
    "get_hydrostatic_pressure_case",
    "get_triaxial_pressure_case",
    # Factory
    "BoxSolverFactory",
    "BoxDriver",
]
