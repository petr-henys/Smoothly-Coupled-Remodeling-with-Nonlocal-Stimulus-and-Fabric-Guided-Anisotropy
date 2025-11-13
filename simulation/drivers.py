"""Energy and structure tensor drivers for remodeling stimulus and direction solvers.

This module provides driver objects that translate mechanics results (displacements u) into:

- strain–energy density ψ(u)  [MPa]
- structure tensor M(u) = ε(u)ᵀ ε(u)

Used as inputs for:
- StimulusSolver  (source term from ψ)
- DirectionSolver (evolution of fabric tensor from M)
"""

from __future__ import annotations

from typing import Protocol, List, Tuple
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
    """Gait-averaged drivers with a single prebuilt UFL graph.

    - Builds persistent snapshots u_snap[i] (Functions on the displacement space) and *one*
      UFL expression for ψ_eff and M_eff that reference these snapshots.
    - On each GS sweep (i.e. whenever (ρ, A) change), call update_snapshots(), which
      loops phases: update_loads → assemble_rhs → solve → copy into u_snap[i].
    - energy_expr() and structure_expr() simply return the prebuilt UFL expressions.
    - IMPORTANT: No cycles-per-day scaling here (keep units clean: ψ in MPa). Put CPD into rS_gain.
    """

    def __init__(self, mech, gait_loader, *, m_exponent: int = 2, psi_ref: float | None = None):
        self.mech = mech
        self.gait = gait_loader
        self.m = int(m_exponent)
        self.psi_ref = float(psi_ref)

        # Persistent snapshots and metadata
        self._u_snap: List[fem.Function] | None = None
        self._weights: List[float] | None = None
        self._phases: List[float] | None = None

        # Prebuilt UFL
        self._psi_expr: ufl.core.expr.Expr | None = None
        self._M_expr: ufl.core.expr.Expr | None = None

        # Track displacement space identity (in case mesh/FS changes)
        self._V_id = id(self.mech.u.function_space)

    def invalidate(self) -> None:
        """Force rebuild of snapshots and UFL exprs (only if space changed)."""
        V_id_now = id(self.mech.u.function_space)
        if V_id_now != self._V_id:
            self._u_snap = None
            self._psi_expr = None
            self._M_expr = None
            self._V_id = V_id_now

    # Public API expected by FixedPointSolver
    def update_snapshots(self) -> None:
        """Recompute u_snap[i] for current (ρ, A) by looping gait phases."""
        self._ensure_prebuilt()

        # Save current u (avoid side-effects)
        u_saved = fem.Function(self.mech.u.function_space)
        u_saved.x.array[:] = self.mech.u.x.array

        for i, phase in enumerate(self._phases):
            self.gait.update_loads(phase)
            self.mech.assemble_rhs()
            self.mech.solve()
            self._u_snap[i].x.array[:] = self.mech.u.x.array
            self._u_snap[i].x.scatter_forward()

        # Restore u
        self.mech.u.x.array[:] = u_saved.x.array
        self.mech.u.x.scatter_forward()

    def energy_expr(self) -> ufl.core.expr.Expr:
        self._ensure_prebuilt()
        assert self._psi_expr is not None
        return self._psi_expr

    def structure_expr(self) -> ufl.core.expr.Expr:
        self._ensure_prebuilt()
        assert self._M_expr is not None
        return self._M_expr

    # Internal helpers
    def _ensure_prebuilt(self) -> None:
        if self._u_snap is not None and self._psi_expr is not None and self._M_expr is not None:
            return
        self._build_snapshots_and_exprs()

    def _build_snapshots_and_exprs(self) -> None:
        # Create persistent snapshots + quadrature metadata
        self._u_snap = []
        self._weights = []
        self._phases = []
        for phase, w in self.gait.get_quadrature():
            self._u_snap.append(fem.Function(self.mech.u.function_space))
            self._weights.append(float(w))
            self._phases.append(float(phase))

        # Prebuild UFL expressions (one pass used for both ψ and M)
        psi_terms = []
        M_terms = []
        for u_i, w in zip(self._u_snap, self._weights):
            # scalar energy density (MPa)
            psi_i = self.mech.get_strain_energy_density(u_i)

            # strain tensor for structure proxy
            e_i = self.mech.get_strain_tensor(u_i)
            M_i = ufl.dot(ufl.transpose(e_i), e_i)

            # optionally normalized and exponentiated
            if self.psi_ref is not None and self.psi_ref > 0.0:
                psi_term = (psi_i / self.psi_ref) ** self.m
            else:
                psi_term = psi_i ** self.m

            psi_terms.append(w * psi_term)
            M_terms.append(w * M_i)

        self._psi_expr = sum(psi_terms)
        self._M_expr = sum(M_terms)
