"""Energy and structure tensor drivers for remodeling stimulus and direction solvers.

This module provides driver objects that translate mechanics results (displacements u) into:

- strain–energy density ψ(u)  [MPa]
- structure tensor M(u) = ε(u)ᵀ ε(u)

Used as inputs for:
- StimulusSolver  (source term from ψ)
- DirectionSolver (evolution of fabric tensor from M)
"""

from __future__ import annotations

from typing import Protocol

from simulation.config import Config
from dolfinx import fem
import ufl


class StrainDriver(Protocol):
    """Protocol for drivers that provide mechanical fields to remodeling PDEs."""

    def energy_expr(self) -> ufl.core.expr.Expr: ...
    def structure_expr(self) -> ufl.core.expr.Expr: ...
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> None: ...  # no-op for instantaneous drivers


class InstantEnergyDriver:
    """Instantaneous energy ψ(u) and structure tensor M = εᵀε from *current* mechanics state.

    Single-regime, explicit formulation: energy density is evaluated directly
    from the current displacement field `mech.u` without any gait averaging.
    """

    def __init__(self, mech):
        self.mech = mech

    def invalidate(self) -> None:
        return None

    def update_snapshots(self) -> None:
        return None

    def energy_expr(self) -> ufl.core.expr.Expr:
        """Strain-energy density ψ(u) [MPa]."""
        return self.mech.get_strain_energy_density()

    def structure_expr(self) -> ufl.core.expr.Expr:
        """Structure tensor M = εᵀ ε from current mechanics state."""
        e = self.mech.get_strain_tensor()
        return ufl.dot(ufl.transpose(e), e)


class GaitEnergyDriver:
    """Gait-averaged driver with eager setup and explicit updates only for `u`."""

    def __init__(self, mech, gait_loader, config: Config):
        self.mech = mech
        self.gait = gait_loader
        self.cfg = config
        self.psi_ref = float(config.psi_ref)
        self.exponent = float(config.n_power)

        quad = list(self.gait.get_quadrature())
        if not quad:
            raise ValueError("Gait quadrature must provide at least one sample.")

        self.phases = [float(phase) for phase, _ in quad]
        self.weights = [float(weight) for _, weight in quad]
        V = self.mech.u.function_space
        self.u_snap = [fem.Function(V, name=f"u_snap_{i}") for i in range(len(self.phases))]

        self._tractions = (self.gait.t_hip, self.gait.t_glmed, self.gait.t_glmax)
        self.loads = self._precompute_loads()
        self.psi_expr, self.M_expr = self._build_expressions()

    def invalidate(self) -> None:
        return None

    def update_snapshots(self) -> None:
        """Solve mechanics for each gait phase and refresh displacement snapshots."""
        for idx in range(len(self.phases)):
            self._apply_load(idx)
            self.mech.assemble_rhs()
            self.mech.solve()
            self.u_snap[idx].x.array[:] = self.mech.u.x.array
            self.u_snap[idx].x.scatter_forward()
        self.mech.u.x.scatter_forward()

    def energy_expr(self) -> ufl.core.expr.Expr:
        return self.psi_expr

    def structure_expr(self) -> ufl.core.expr.Expr:
        return self.M_expr

    def _precompute_loads(self) -> list[tuple]:
        loads: list[tuple] = []
        for phase in self.phases:
            self.gait.update_loads(phase)
            loads.append(tuple(traction.x.array.copy() for traction in self._tractions))
        return loads

    def _apply_load(self, idx: int) -> None:
        for traction, data in zip(self._tractions, self.loads[idx]):
            traction.x.array[:] = data
            traction.x.scatter_forward()

    def _build_expressions(self) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr]:
        psi_terms = []
        structure_terms = []
        for u_i, weight in zip(self.u_snap, self.weights):
            psi_i = self.mech.get_strain_energy_density(u_i)
            e_i = self.mech.get_strain_tensor(u_i)
            structure_i = ufl.dot(ufl.transpose(e_i), e_i)
            psi_terms.append(weight * (psi_i / self.psi_ref) ** self.exponent)
            structure_terms.append(weight * structure_i)
        return sum(psi_terms), sum(structure_terms)
