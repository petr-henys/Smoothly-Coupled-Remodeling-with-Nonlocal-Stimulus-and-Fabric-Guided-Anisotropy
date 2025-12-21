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

from simulation.box_loader import BoxLoadingCase, PressureLoadSpec


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
    box_extent: tuple[float, float] = (0.0, 10.0),
    name: str = "gradient_compression",
    day_cycles: float = 1.0,
) -> BoxLoadingCase:
    """Create a pressure case with linear gradient across the surface.
    
    This creates non-uniform loading that drives density adaptation.
    Areas with higher load will densify, areas with lower load will resorb.
    
    Args:
        pressure: Base pressure magnitude [MPa]
        gradient_axis: Axis for gradient (0=x, 1=y)
        gradient_range: (min_factor, max_factor) for pressure variation
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
            gradient_range=gradient_range,
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
