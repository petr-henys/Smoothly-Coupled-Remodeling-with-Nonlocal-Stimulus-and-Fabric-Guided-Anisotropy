"""Solver factory for dependency injection."""

from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

from dolfinx import fem

from simulation.drivers import GaitDriver
from simulation.solvers import MechanicsSolver, FabricSolver, StimulusSolver, DensitySolver
from simulation.utils import build_dirichlet_bcs

if TYPE_CHECKING:
    from simulation.config import Config
    from femur.loader import Loader


class SolverFactory(Protocol):
    """Protocol for solver creation (Dependency Injection)."""

    def create_mechanics_solver(
        self,
        u: fem.Function,
        rho: fem.Function,
        L: fem.Function,
        loader: Loader,
    ) -> MechanicsSolver:
        """Create the mechanics solver."""
        ...

    def create_driver(
        self,
        mech_solver: MechanicsSolver,
        loader: Loader,
    ) -> GaitDriver:
        """Create the mechanics driver."""
        ...

    def create_fabric_solver(
        self,
        L: fem.Function,
        L_old: fem.Function,
        Qbar: fem.Function,
    ) -> FabricSolver:
        """Create the fabric evolution solver."""
        ...

    def create_stimulus_solver(
        self,
        S: fem.Function,
        S_old: fem.Function,
        psi: fem.Function,
        rho: fem.Function,
    ) -> StimulusSolver:
        """Create the stimulus solver."""
        ...

    def create_density_solver(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        S: fem.Function,
    ) -> DensitySolver:
        """Create the density evolution solver."""
        ...


class DefaultSolverFactory:
    """Default solver factory implementation."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def create_mechanics_solver(
        self,
        u: fem.Function,
        rho: fem.Function,
        L: fem.Function,
        loader: Loader,
    ) -> MechanicsSolver:
        # Boundary conditions
        bc_mech = build_dirichlet_bcs(
            u.function_space, 
            self.cfg.facet_tags, 
            id_tag=self.cfg.geometry.fix_tag, 
            value=0.0
        )
        neumann_bcs = [(loader.traction, loader.load_tag)]

        return MechanicsSolver(
            u, rho, self.cfg, bc_mech, neumann_bcs, L=L
        )

    def create_driver(
        self,
        mech_solver: MechanicsSolver,
        loader: Loader,
    ) -> GaitDriver:
        return GaitDriver(mech_solver, self.cfg, loader=loader)

    def create_fabric_solver(
        self,
        L: fem.Function,
        L_old: fem.Function,
        Qbar: fem.Function,
    ) -> FabricSolver:
        return FabricSolver(L, L_old, Qbar, self.cfg)

    def create_stimulus_solver(
        self,
        S: fem.Function,
        S_old: fem.Function,
        psi: fem.Function,
        rho: fem.Function,
    ) -> StimulusSolver:
        return StimulusSolver(S, S_old, psi, rho, self.cfg)

    def create_density_solver(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        S: fem.Function,
    ) -> DensitySolver:
        return DensitySolver(rho, rho_old, S, self.cfg)
