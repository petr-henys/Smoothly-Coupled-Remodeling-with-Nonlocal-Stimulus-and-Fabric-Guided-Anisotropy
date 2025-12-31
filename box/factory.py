"""Factory for box model solvers.

Provides a SolverFactory implementation for box mesh simulations with:
- Bottom support (Dirichlet BC on z=0)
- Pressure loading on top (Neumann BC on z=Lz)

Mirrors the DefaultSolverFactory but uses BoxLoader instead of femur Loader.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

from dolfinx import fem

from box.loader import BoxLoader, BoxLoadingCase
from box.mesh import BoxMeshBuilder
from simulation.drivers import GaitDriver
from simulation.solvers import (
    MechanicsSolver,
    FabricSolver, 
    StimulusSolver,
    DensitySolver,
)
from simulation.utils import build_dirichlet_bcs

if TYPE_CHECKING:
    from simulation.config import Config


class BoxDriver(GaitDriver):
    """Multi-load mechanics driver for box model.
    
    Identical to GaitDriver but accepts BoxLoader instead of Loader.
    The underlying mechanics and SED computation are the same.
    """
    
    def __init__(
        self,
        mech: MechanicsSolver,
        config: "Config",
        loader: BoxLoader,
    ):
        """Initialize box driver.
        
        Args:
            mech: Mechanics solver instance
            config: Simulation configuration
            loader: Box pressure loader (with loading_cases already precomputed)
        """
        super().__init__(mech, config, loader)


class BoxSolverFactory:
    """Solver factory for box mesh simulations.
    
    Creates solvers configured for a box geometry with:
    - Bottom surface fixed (z=0)
    - Top surface loaded with pressure (z=Lz)
    """
    
    def __init__(self, cfg: "Config"):
        """Initialize factory with configuration.
        
        Args:
            cfg: Simulation configuration with domain and facet_tags set
        """
        self.cfg = cfg
    
    def create_mechanics_solver(
        self,
        u: fem.Function,
        rho: fem.Function,
        L: fem.Function,
        loader: BoxLoader,
    ) -> MechanicsSolver:
        """Create mechanics solver with box boundary conditions.
        
        Args:
            u: Displacement field
            rho: Density field
            L: Log-fabric tensor field
            loader: Box pressure loader
            
        Returns:
            Configured MechanicsSolver
        """
        # Dirichlet BC: fixed bottom surface (z=0)
        bc_mech = build_dirichlet_bcs(
            u.function_space,
            self.cfg.facet_tags,
            id_tag=BoxMeshBuilder.TAG_BOTTOM,  # z=0 fixed
            value=0.0,
        )
        
        # Neumann BC: pressure on top surface
        neumann_bcs = [(loader.traction, loader.load_tag)]
        
        return MechanicsSolver(
            u, rho, self.cfg, bc_mech, neumann_bcs, L=L
        )
    
    def create_driver(
        self,
        mech_solver: MechanicsSolver,
        loader: BoxLoader,
    ) -> BoxDriver:
        """Create box mechanics driver.
        
        Args:
            mech_solver: Mechanics solver instance
            loader: Box pressure loader (with loading_cases precomputed)
            
        Returns:
            BoxDriver for multi-load mechanics
        """
        return BoxDriver(mech_solver, self.cfg, loader=loader)
    
    def create_fabric_solver(
        self,
        L: fem.Function,
        L_old: fem.Function,
        Qbar: fem.Function,
    ) -> FabricSolver:
        """Create fabric evolution solver.
        
        Args:
            L: Current log-fabric tensor
            L_old: Previous log-fabric tensor
            Qbar: Averaged stress-squared tensor from mechanics
            
        Returns:
            FabricSolver instance
        """
        return FabricSolver(L, L_old, Qbar, self.cfg)
    
    def create_stimulus_solver(
        self,
        S: fem.Function,
        S_old: fem.Function,
        psi: fem.Function,
        rho: fem.Function,
    ) -> StimulusSolver:
        """Create stimulus solver.
        
        Args:
            S: Current stimulus field
            S_old: Previous stimulus field
            psi: Strain energy density from mechanics
            rho: Current density field
            
        Returns:
            StimulusSolver instance
        """
        return StimulusSolver(S, S_old, psi, rho, self.cfg)
    
    def create_density_solver(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        S: fem.Function,
    ) -> DensitySolver:
        """Create density evolution solver.
        
        Args:
            rho: Current density field
            rho_old: Previous density field  
            S: Stimulus field
            
        Returns:
            DensitySolver instance
        """
        return DensitySolver(rho, rho_old, S, self.cfg)
