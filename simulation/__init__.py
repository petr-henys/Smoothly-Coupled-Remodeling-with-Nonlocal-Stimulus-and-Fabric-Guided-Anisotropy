"""Bone remodeling simulation package.

Core components: Config, Remodeller, CouplingBlock protocol, GaitDriver,
solvers (Mechanics, Fabric, Stimulus, Density), ProgressReporter.
"""

from simulation.config import Config
from simulation.protocols import CouplingBlock
from simulation.params import (
    MaterialParams,
    DensityParams,
    StimulusParams,
    FabricParams,
    SolverParams,
    TimeParams,
    NumericsParams,
    OutputParams,
)
from simulation.progress import ProgressReporter
from femur.scenarios import get_standard_gait_cases, load_scenarios_from_yaml

__all__ = [
    # Config and parameters
    "Config",
    "MaterialParams",
    "DensityParams",
    "StimulusParams",
    "FabricParams",
    "SolverParams",
    "TimeParams",
    "NumericsParams",
    "OutputParams",
    # Protocols
    "CouplingBlock",
    # Progress reporting
    "ProgressReporter",
    # Scenarios
    "get_standard_gait_cases",
    "load_scenarios_from_yaml",
]
