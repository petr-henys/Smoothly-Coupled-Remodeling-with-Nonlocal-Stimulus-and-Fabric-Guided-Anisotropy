"""PDE solvers for bone remodeling simulation.

This module provides the core solver classes:
- MechanicsSolver: Linear elasticity with anisotropic stiffness
- FabricSolver: Log-fabric tensor evolution
- StimulusSolver: Mechanostat stimulus field
- DensitySolver: Bone density evolution
"""

from simulation.solvers.base import BaseLinearSolver
from simulation.solvers.mechanics import MechanicsSolver
from simulation.solvers.fabric import FabricSolver
from simulation.solvers.stimulus import StimulusSolver
from simulation.solvers.density import DensitySolver
