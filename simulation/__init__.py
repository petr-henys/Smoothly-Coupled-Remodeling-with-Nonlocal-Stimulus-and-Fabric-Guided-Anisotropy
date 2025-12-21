"""Simulation package for bone remodeling.

This package provides the core simulation components:
- Config: Simulation parameters organized into logical groups
- Remodeller: Top-level orchestrator for coupled simulations
- Solvers: MechanicsSolver, DensitySolver, StimulusSolver, FabricSolver
- Drivers: GaitDriver for multi-load mechanical analysis
- Protocols: CouplingBlock interface for solver blocks
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

__all__ = [
    "Config",
    "CouplingBlock",
    "MaterialParams",
    "DensityParams",
    "StimulusParams",
    "FabricParams",
    "SolverParams",
    "TimeParams",
    "NumericsParams",
    "OutputParams",
]
