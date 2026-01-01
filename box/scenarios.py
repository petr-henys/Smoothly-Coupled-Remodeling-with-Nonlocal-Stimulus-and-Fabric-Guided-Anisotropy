"""Predefined loading scenarios for box mesh simulations.

Provides simple pressure loading cases for bone remodeling studies.
These can be used as-is or as templates for custom scenarios.

Physical context:
- Simulates uniaxial compression similar to trabecular bone specimens
- Pressure values are in MPa (typical physiological: 0.1-5 MPa)
- Negative z-direction = compression from top
"""

from __future__ import annotations

from typing import List

from box.loader import BoxLoadingCase, GradientType, PressureLoadSpec


def get_single_pressure_case(
    pressure: float = 1.0,
    name: str = "static_compression",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a single uniform compression case.
    
    Args:
        pressure: Pressure magnitude [MPa] (positive = compression)
        name: Case name
        day_cycles: Loading cycles per day
        
    Returns:
        BoxLoadingCase for uniform compression
    """
    return BoxLoadingCase(
        name=name,
        day_cycles=day_cycles,
        pressure=PressureLoadSpec(
            magnitude=pressure,
            direction=(0.0, 0.0, -1.0),  # Compression in -z
        ),
    )


def get_gradient_pressure_case(
    pressure: float = 1.0,
    gradient_axis: int = 0,
    gradient_range: tuple[float, float] = (0.5, 1.5),
    gradient_type: GradientType = GradientType.LINEAR,
    box_extent: tuple[float, float] = (0.0, 10.0),
    name: str = "gradient_compression",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a pressure case with spatial gradient across the surface.
    
    This creates non-uniform loading that drives density adaptation.
    Areas with higher load will densify, areas with lower load will resorb.
    
    Args:
        pressure: Base pressure magnitude [MPa]
        gradient_axis: Axis for gradient (0=x, 1=y)
        gradient_range: (min_factor, max_factor) for pressure variation
        gradient_type: Type of gradient (LINEAR, PARABOLIC, BENDING)
        box_extent: (min_coord, max_coord) along gradient axis
        name: Case name
        day_cycles: Loading cycles per day
        
    Returns:
        BoxLoadingCase with gradient pressure
    """
    return BoxLoadingCase(
        name=name,
        day_cycles=day_cycles,
        pressure=PressureLoadSpec(
            magnitude=pressure,
            direction=(0.0, 0.0, -1.0),
            gradient_axis=gradient_axis,
            gradient_type=gradient_type,
            gradient_range=gradient_range,
            box_extent=box_extent,
        ),
    )


def get_parabolic_pressure_case(
    pressure: float = 1.0,
    gradient_axis: int = 0,
    center_factor: float = 2.0,
    edge_factor: float = 0.5,
    box_extent: tuple[float, float] = (0.0, 10.0),
    name: str = "parabolic_compression",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a parabolic pressure distribution - peak at center, low at edges.
    
    Creates a bell-shaped pressure distribution that produces:
    - High SED at center -> bone densification
    - Low SED at edges -> bone resorption
    
    This mimics localized loading (e.g., contact patch).
    
    Args:
        pressure: Base pressure magnitude [MPa]
        gradient_axis: Axis for parabola (0=x, 1=y)
        center_factor: Factor at center (peak)
        edge_factor: Factor at edges (min)
        box_extent: (min_coord, max_coord) along axis
        name: Case name
        day_cycles: Loading cycles per day
        
    Returns:
        BoxLoadingCase with parabolic pressure
    """
    return BoxLoadingCase(
        name=name,
        day_cycles=day_cycles,
        pressure=PressureLoadSpec(
            magnitude=pressure,
            direction=(0.0, 0.0, -1.0),
            gradient_axis=gradient_axis,
            gradient_type=GradientType.PARABOLIC,
            gradient_range=(edge_factor, center_factor),  # min at edges, max at center
            box_extent=box_extent,
        ),
    )


def get_bending_like_case(
    pressure: float = 1.0,
    gradient_axis: int = 0,
    tension_factor: float = 0.2,
    compression_factor: float = 1.8,
    box_extent: tuple[float, float] = (0.0, 10.0),
    name: str = "bending_load",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a bending-like load - compression on one side, reduced on other.
    
    Simulates eccentric loading that creates bending-like stress distribution:
    - One side gets high compression -> high SED -> densification
    - Other side gets low load -> low SED -> resorption
    
    This is useful for studying adaptive remodeling under asymmetric loading.
    
    Args:
        pressure: Base pressure magnitude [MPa]
        gradient_axis: Axis across which load varies (0=x, 1=y)
        tension_factor: Factor on low-load side (< 1.0)
        compression_factor: Factor on high-load side (> 1.0)
        box_extent: (min_coord, max_coord) along axis
        name: Case name
        day_cycles: Loading cycles per day
        
    Returns:
        BoxLoadingCase with bending-like pressure
    """
    return BoxLoadingCase(
        name=name,
        day_cycles=day_cycles,
        pressure=PressureLoadSpec(
            magnitude=pressure,
            direction=(0.0, 0.0, -1.0),
            gradient_axis=gradient_axis,
            gradient_type=GradientType.LINEAR,  # Linear gives bending-like distribution
            gradient_range=(tension_factor, compression_factor),
            box_extent=box_extent,
        ),
    )


def get_physiological_compression_cases() -> List[BoxLoadingCase]:
    """Standard physiological-like compression scenarios.
    
    Simulates daily loading typical of trabecular bone:
    - Low load: resting/low activity periods
    - Medium load: normal walking
    - High load: stair climbing, jumping
    
    Returns:
        List of BoxLoadingCase for physiological loading
    """
    # Low load: typical resting / light activity
    case_low = BoxLoadingCase(
        name="low_load",
        day_cycles=0.3,  # 30% of time at low load
        pressure=PressureLoadSpec(
            magnitude=0.5,  # 0.5 MPa
            direction=(0.0, 0.0, -1.0),
        ),
    )
    
    # Medium load: normal walking
    case_medium = BoxLoadingCase(
        name="medium_load", 
        day_cycles=0.5,  # 50% of time at medium load
        pressure=PressureLoadSpec(
            magnitude=2.0,  # 2 MPa
            direction=(0.0, 0.0, -1.0),
        ),
    )
    
    # High load: stair climbing, brief impacts
    case_high = BoxLoadingCase(
        name="high_load",
        day_cycles=0.2,  # 20% of time at high load
        pressure=PressureLoadSpec(
            magnitude=5.0,  # 5 MPa
            direction=(0.0, 0.0, -1.0),
        ),
    )
    
    return [case_low, case_medium, case_high]


def get_overload_scenarios() -> List[BoxLoadingCase]:
    """Overload scenarios for studying hypertrophy.
    
    Simulates increased mechanical loading to study bone formation.
    
    Returns:
        List of BoxLoadingCase for overload conditions
    """
    case_overload = BoxLoadingCase(
        name="overload",
        day_cycles=1.0,
        pressure=PressureLoadSpec(
            magnitude=8.0,  # 8 MPa - above normal physiological
            direction=(0.0, 0.0, -1.0),
        ),
    )
    
    return [case_overload]


def get_disuse_scenarios() -> List[BoxLoadingCase]:
    """Disuse scenarios for studying bone loss.
    
    Simulates reduced mechanical loading (bed rest, microgravity).
    
    Returns:
        List of BoxLoadingCase for disuse conditions  
    """
    case_disuse = BoxLoadingCase(
        name="disuse",
        day_cycles=1.0,
        pressure=PressureLoadSpec(
            magnitude=0.1,  # 0.1 MPa - minimal loading
            direction=(0.0, 0.0, -1.0),
        ),
    )
    
    return [case_disuse]


def get_cyclic_loading_cases(
    pressure_min: float = 0.5,
    pressure_max: float = 3.0,
    n_levels: int = 3,
) -> List[BoxLoadingCase]:
    """Generate multiple loading levels for cycle-weighted stimulus.
    
    Creates evenly spaced loading cases between min and max pressure,
    with equal day_cycles weighting.
    
    Args:
        pressure_min: Minimum pressure [MPa]
        pressure_max: Maximum pressure [MPa]
        n_levels: Number of loading levels
        
    Returns:
        List of BoxLoadingCase with varying pressure levels
    """
    import numpy as np
    
    pressures = np.linspace(pressure_min, pressure_max, n_levels)
    cases = []
    
    for i, p in enumerate(pressures):
        cases.append(BoxLoadingCase(
            name=f"level_{i+1}",
            day_cycles=1.0 / n_levels,  # Equal weighting
            pressure=PressureLoadSpec(
                magnitude=p,
                direction=(0.0, 0.0, -1.0),
            ),
        ))
    
    return cases


def get_hydrostatic_pressure_case(
    pressure: float = 1.0,
    name: str = "hydrostatic",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a hydrostatic pressure case (uniform compression from all sides).
    
    This creates triaxial loading where pressure is applied on:
    - Top (z=Lz): inward normal (0, 0, -1)
    - X_MAX (x=Lx): inward normal (-1, 0, 0)
    - Y_MAX (y=Ly): inward normal (0, -1, 0)
    
    Note: Bottom (z=0) is typically fixed (Dirichlet), and X_MIN/Y_MIN
    are often constrained by symmetry. Adjust wall_tags as needed.
    
    To use this, the BoxLoader must be initialized with all three load_tags:
        load_tags = [BoxMeshBuilder.TAG_TOP, BoxMeshBuilder.TAG_X_MAX, 
                     BoxMeshBuilder.TAG_Y_MAX]
    
    Args:
        pressure: Pressure magnitude [MPa] (positive = compression)
        name: Case name
        day_cycles: Loading cycles per day
        
    Returns:
        BoxLoadingCase for hydrostatic pressure (triaxial compression)
    """
    from box.mesh import BoxMeshBuilder
    
    return BoxLoadingCase(
        name=name,
        day_cycles=day_cycles,
        pressure=PressureLoadSpec(
            magnitude=pressure,
            wall_tags=(
                BoxMeshBuilder.TAG_TOP,
                BoxMeshBuilder.TAG_X_MAX,
                BoxMeshBuilder.TAG_Y_MAX,
            ),
            wall_directions=(
                (0.0, 0.0, -1.0),   # Top: press down
                (-1.0, 0.0, 0.0),   # X_MAX: press inward (-x)
                (0.0, -1.0, 0.0),   # Y_MAX: press inward (-y)
            ),
        ),
    )


def get_triaxial_pressure_case(
    pressure_z: float = 1.0,
    pressure_x: float = 0.5,
    pressure_y: float = 0.5,
    name: str = "triaxial",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a triaxial pressure case with independent pressures on each axis.
    
    Applies different pressure magnitudes on each face:
    - Top (z=Lz): pressure_z with inward normal (0, 0, -1)
    - X_MAX (x=Lx): pressure_x with inward normal (-1, 0, 0)  
    - Y_MAX (y=Ly): pressure_y with inward normal (0, -1, 0)
    
    For hydrostatic stress state, use equal pressures on all axes.
    
    Note: This uses a single magnitude scaled differently per wall. For truly
    independent magnitudes, create separate BoxLoadingCase instances.
    
    Args:
        pressure_z: Axial pressure [MPa] (z-direction, main loading axis)
        pressure_x: Confining pressure [MPa] (x-direction)
        pressure_y: Confining pressure [MPa] (y-direction)
        name: Case name
        day_cycles: Loading cycles per day
        
    Returns:
        BoxLoadingCase for triaxial loading
    """
    from box.mesh import BoxMeshBuilder
    
    # We use magnitude=1 and encode actual pressures in direction vectors
    # Direction = pressure_i * unit_inward_normal
    return BoxLoadingCase(
        name=name,
        day_cycles=day_cycles,
        pressure=PressureLoadSpec(
            magnitude=1.0,  # Actual pressure encoded in directions
            wall_tags=(
                BoxMeshBuilder.TAG_TOP,
                BoxMeshBuilder.TAG_X_MAX,
                BoxMeshBuilder.TAG_Y_MAX,
            ),
            wall_directions=(
                (0.0, 0.0, -pressure_z),
                (-pressure_x, 0.0, 0.0),
                (0.0, -pressure_y, 0.0),
            ),
        ),
    )
