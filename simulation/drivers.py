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
    """Gait-averaged Carter–Beaupré daily stress stimulus (SED-based) + structure tensor.

    This driver implements the *classical* Carter–Beaupré daily stress stimulus
    using strain–energy density ψ(u) instead of an effective stress:

        J_day(x) = Σ_i n_i ψ_i(x)^m / psi_ref^(m-1)     [MPa]        (Carter–Beaupré)

    where
        – i indexes loading types / phases during the gait cycle,
        – ψ_i(x) is the strain–energy density at phase i  [MPa],
        – n_i is the number of cycles of loading type i per day,
        – m = cfg.n_power is the stress/energy exponent,
        – psi_ref = cfg.psi_ref is the reference energy density [MPa].

    In this implementation we approximate:
        n_i ≈ N_cyc_per_day · w_i

    where w_i are quadrature weights over the gait cycle (Σ w_i = 1) and
    N_cyc_per_day = cfg.gait_cycles_per_day.

    Then

        J_day(x) = psi_ref * N_cyc_per_day * ⟨(ψ(x)/psi_ref)^m⟩_cycle   [MPa]

    which is exactly what Carter & Beaupré do (up to using ψ instead of σ).

    Interface:
        - energy_expr()     → J_day(x)    (Carter–Beaupré daily stress stimulus) [MPa]
        - daily_stress_expr → alias for J_day(x) (for clarity / post-processing)
        - structure_expr()  → gait-averaged structure tensor M(x) = ⟨εᵀε⟩_cycle
    """

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

        # UFL expressions (built from snapshots)
        self.psi_expr: ufl.core.expr.Expr
        self.M_expr: ufl.core.expr.Expr
        self._J_day_expr: ufl.core.expr.Expr  # explicit Carter–Beaupré daily stress stimulus

        self.psi_expr, self.M_expr, self._J_day_expr = self._build_expressions()
        self._last_stats: dict | None = None

    def invalidate(self) -> None:
        """Rebuild expressions if psi_ref or exponent change in Config."""
        dirty = False
        if abs(self.psi_ref - float(self.cfg.psi_ref)) > 1e-9:
            self.psi_ref = float(self.cfg.psi_ref)
            dirty = True

        if abs(self.exponent - float(self.cfg.n_power)) > 1e-9:
            self.exponent = float(self.cfg.n_power)
            dirty = True

        if dirty:
            self.psi_expr, self.M_expr, self._J_day_expr = self._build_expressions()

    def update_snapshots(self) -> dict:
        """Solve mechanics at each gait phase and refresh displacement snapshots."""
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

        # Domain-average of the Carter–Beaupré daily stress (psi_expr == J_day)
        psi_int_local = fem.assemble_scalar(fem.form(self.psi_expr * self.cfg.dx))
        psi_int = self.comm.allreduce(psi_int_local, op=MPI.SUM)

        vol_local = fem.assemble_scalar(fem.form(1.0 * self.cfg.dx))
        vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        psi_avg = psi_int / vol

        stats = self._build_stats(iters, times, psi_avg)
        self._last_stats = stats
        return stats

    # ----- Interface expected by StimulusSolver / DirectionSolver -----

    def energy_expr(self) -> ufl.core.expr.Expr:
        """Return Carter–Beaupré daily stress stimulus J_day(x) [MPa]."""
        return self.psi_expr

    def daily_stress_expr(self) -> ufl.core.expr.Expr:
        """Explicit accessor for J_day(x) (alias for energy_expr)."""
        return self._J_day_expr

    def structure_expr(self) -> ufl.core.expr.Expr:
        """Return gait-averaged structure tensor M(x) = ⟨εᵀε⟩_cycle."""
        return self.M_expr

    # ----- Internal helpers -----

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

    def _build_expressions(self) -> tuple[ufl.core.expr.Expr, ufl.core.expr.Expr, ufl.core.expr.Expr]:
        """Build Carter–Beaupré daily stress stimulus J_day(x) and M(x).

        Using the original Carter–Beaupré idea:

            J_day(x) = Σ_i n_i ψ_i(x)^m / psi_ref^(m-1)      [MPa]
                     ≈ psi_ref * N_cyc * ⟨(ψ(x)/psi_ref)^m⟩_cycle

        where we approximate:
            n_i ≈ N_cyc * w_i,    Σ w_i = 1  (gait quadrature),
            N_cyc = cfg.gait_cycles_per_day.
        """
        if self.exponent <= 0.0:
            raise ValueError(f"n_power must be positive, got {self.exponent}")

        N_cyc = float(self.cfg.gait_cycles_per_day)
        if N_cyc <= 0.0:
            raise ValueError(f"gait_cycles_per_day must be positive, got {N_cyc}")

        psi_p_terms = []
        structure_terms = []
        total_weight = 0.0

        for u_i, weight in zip(self.u_snap, self.weights):
            # Instantaneous SED at phase i [MPa]
            psi_i = self.mech.get_strain_energy_density(u_i)

            # Structure tensor at phase i
            e_i = self.mech.get_strain_tensor(u_i)
            structure_i = ufl.dot(ufl.transpose(e_i), e_i)

            # Dimensionless term (ψ_i/psi_ref)^m weighted over gait cycle
            psi_p_terms.append(weight * (psi_i / self.psi_ref) ** self.exponent)
            structure_terms.append(weight * structure_i)
            total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("Gait quadrature weights must sum to a positive value.")

        # Cycle-averaged dimensionless term ⟨(ψ/psi_ref)^m⟩_cycle
        J_cycle = sum(psi_p_terms) / total_weight

        # Carter–Beaupré daily stress stimulus J_day(x) [MPa]
        #   J_day = psi_ref * Σ_i n_i (ψ_i/psi_ref)^m
        #         ≈ psi_ref * N_cyc * ⟨(ψ/psi_ref)^m⟩_cycle
        J_day = self.psi_ref * N_cyc * J_cycle

        # Use weighted average for structure tensor as well (dimensionless)
        M_expr = sum(structure_terms) / total_weight

        # NOTE:
        #   - psi_expr is *defined* here as the Carter–Beaupré daily stress stimulus.
        #   - You can plug psi_expr directly into StimulusSolver (treat psi_ref in Config
        #     as the daily-stress setpoint), or, if you want a separate S-driver, use
        #     daily_stress_expr() explicitly and adjust rS_gain/psi_ref units accordingly.
        psi_expr = J_day

        return psi_expr, M_expr, J_day

    def _build_stats(self, iters: list[float], times: list[float], psi_avg: float) -> dict:
        total_time = float(sum(times))
        median_time = float(np.median(times)) if times else 0.0
        median_iters = float(np.median(iters)) if iters else 0.0
        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": total_time,
            "median_time": median_time,
            "median_iters": median_iters,
            "psi_avg": psi_avg,   # now interpreted as domain-average J_day
        }

    def _elapsed_max(self, t0: float) -> float:
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)



    def _build_stats(self, iters: list[float], times: list[float], psi_avg: float) -> dict:
        total_time = float(sum(times))
        median_time = float(np.median(times)) if times else 0.0
        median_iters = float(np.median(iters)) if iters else 0.0
        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": total_time,
            "median_time": median_time,
            "median_iters": median_iters,
            "psi_avg": psi_avg,
        }

    def _elapsed_max(self, t0: float) -> float:
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)
