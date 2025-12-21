"""Simulation package for bone remodeling.

This package provides the core simulation components:
- Config: Simulation parameters organized into logical groups
- Remodeller: Top-level orchestrator for coupled simulations
- Solvers: MechanicsSolver, DensitySolver, StimulusSolver, FabricSolver
- Drivers: GaitDriver for multi-load mechanical analysis
- Protocols: CouplingBlock interface for solver blocks
- ProgressReporter: Rich progress display
- Scenarios: Predefined gait loading configurations
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
from simulation.scenarios import get_standard_gait_cases, load_scenarios_from_yaml

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
