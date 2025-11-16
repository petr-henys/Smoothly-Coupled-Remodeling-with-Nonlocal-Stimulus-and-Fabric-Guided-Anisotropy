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

import numpy as np
from mpi4py import MPI
from simulation.config import Config
from dolfinx import fem
import ufl


class StrainDriver(Protocol):
    """Protocol for drivers that provide mechanical fields to remodeling PDEs."""

    def energy_expr(self) -> ufl.core.expr.Expr: ...
    def structure_expr(self) -> ufl.core.expr.Expr: ...
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> dict | None: ...  # no-op for instantaneous drivers


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
        self.comm = self.mech.u.function_space.mesh.comm

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
        self._last_stats: dict | None = None

    def invalidate(self) -> None:
        return None

    def update_snapshots(self) -> dict:
        """Solve mechanics for each gait phase and refresh displacement snapshots."""
        times: list[float] = []
        iters: list[float] = []
        for idx in range(len(self.phases)):
            start = MPI.Wtime()
            self._apply_load(idx)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            elapsed = self._elapsed_max(start)
            times.append(float(elapsed))
            iters.append(float(its))
            self.u_snap[idx].x.array[:] = self.mech.u.x.array
            self.u_snap[idx].x.scatter_forward()
        self.mech.u.x.scatter_forward()

        stats = self._build_stats(iters, times)
        self._last_stats = stats
        return stats

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
        """Build gait-aggregated expressions for ψ [MPa] and structure tensor M.

        ψ is an L^p-type average of strain energy density over the gait cycle:
            ψ_gait = psi_ref * ( Σ w (ψ/psi_ref)^n / Σ w )^(1/n)
        so that ψ_gait has units [MPa] and is consistent with StimulusSolver.
        """
        if self.exponent <= 0.0:
            raise ValueError(f"n_power must be positive, got {self.exponent}")

        psi_p_terms = []
        structure_terms = []
        total_weight = 0.0

        for u_i, weight in zip(self.u_snap, self.weights):
            psi_i = self.mech.get_strain_energy_density(u_i)  # [MPa]
            e_i = self.mech.get_strain_tensor(u_i)
            structure_i = ufl.dot(ufl.transpose(e_i), e_i)

            # Dimensionless p-th power of normalised energy
            psi_p_terms.append(weight * (psi_i / self.psi_ref) ** self.exponent)
            structure_terms.append(weight * structure_i)
            total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("Gait quadrature weights must sum to a positive value.")

        # Average the p-th power and map back to MPa via L^p norm
        psi_p_avg = sum(psi_p_terms) / total_weight        # dimensionless
        psi_expr = self.psi_ref * psi_p_avg ** (1.0 / self.exponent)  # [MPa]

        # Use weighted average for structure tensor as well (dimensionless)
        M_expr = sum(structure_terms) / total_weight

        return psi_expr, M_expr


    def _build_stats(self, iters: list[float], times: list[float]) -> dict:
        total_time = float(sum(times))
        median_time = float(np.median(times)) if times else 0.0
        median_iters = float(np.median(iters)) if iters else 0.0
        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": total_time,
            "median_time": median_time,
            "median_iters": median_iters,
        }

    def _elapsed_max(self, t0: float) -> float:
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)
