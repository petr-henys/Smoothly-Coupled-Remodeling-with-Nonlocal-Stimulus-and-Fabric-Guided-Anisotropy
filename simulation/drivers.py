"""Energy and structure tensor drivers for remodeling stimulus and direction solvers.

This module provides small driver objects that translate mechanics results
(displacements u) into:

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


class InstantEnergyDriver:
    """Instantaneous energy ψ(u) and structure tensor M = εᵀε from current mechanics state.

    Single-regime, explicit formulation: energy density is evaluated directly
    from the *current* displacement field `mech.u` without any gait averaging.
    """

    def __init__(self, mech):
        self.mech = mech

    def invalidate(self) -> None:
        """Nothing to invalidate: always uses current mechanics state."""
        return None

    def energy_expr(self) -> ufl.core.expr.Expr:
        """Strain-energy density ψ(u) [MPa]."""
        return self.mech.get_strain_energy_density()

    def structure_expr(self) -> ufl.core.expr.Expr:
        """Structure tensor M = εᵀ ε from current mechanics state."""
        e = self.mech.get_strain_tensor()
        return ufl.dot(ufl.transpose(e), e)


class GaitEnergyDriver:
    """Gait-averaged energy and structure via quadrature over frozen displacement snapshots.

    Explicit daily accumulation model:

    - Computes *average* strain–energy density over the gait cycle by phase
      quadrature.
    - Does NOT include cycles-per-day scaling here; literature practice is to
      aggregate cycles in the stimulus source term (rS_gain [1/(MPa·day)]).

    Implementation notes
    --------------------
    - The expensive part is solving the mechanics problem for each gait phase.
      We therefore:
        * build the UFL expressions ψ̄(ρ, A, {u_i}) and M̄(ρ, A, {u_i}) **once**,
        * keep a list of snapshot Functions u_i for each quadrature point,
        * on each `invalidate()` just recompute the snapshots (mechanics solves)
          but **do not** rebuild the UFL graphs.
    - This preserves accuracy (full quadrature over phases, full re-solve for
      new ρ, A) while avoiding repeated UFL construction and duplicate solves
      when both ψ and M are needed.
    """

    def __init__(self, mech, gait_loader, cycles_per_day: float = 1.0):
        self.mech = mech
        self.gait = gait_loader
        # Currently not used in ψ̄; retained for completeness / possible post-processing
        self.cpd = float(cycles_per_day)

        # Lazy-built UFL expressions and snapshots
        self._psi_expr: ufl.core.expr.Expr | None = None
        self._M_expr: ufl.core.expr.Expr | None = None

        self._snapshots: List[fem.Function] | None = None
        self._quadrature: List[Tuple[float, float]] | None = None

        # Mark that snapshots are not consistent with current (ρ, A) yet
        self._dirty: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def invalidate(self) -> None:
        """Mark driver as dirty when (ρ, A) change in the outer fixed-point loop.

        We keep the UFL graphs (ψ̄, M̄) but force an update of the displacement
        snapshots at the next call to energy_expr()/structure_expr().
        """
        self._dirty = True

    def energy_expr(self) -> ufl.core.expr.Expr:
        """Gait-averaged strain-energy density ψ̄(u; ρ, A) [MPa]."""
        self._ensure_built_and_fresh()
        assert self._psi_expr is not None
        return self._psi_expr

    def structure_expr(self) -> ufl.core.expr.Expr:
        """Gait-averaged structure tensor M̄(u; ρ, A) = ⟨εᵀ ε⟩ over gait cycle."""
        self._ensure_built_and_fresh()
        assert self._M_expr is not None
        return self._M_expr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_snapshots(self) -> None:
        """Allocate snapshot functions and cache quadrature, if not yet done."""
        if self._snapshots is not None:
            return

        quad = self.gait.get_quadrature()  # List[(phase, weight)]
        self._quadrature = list(quad)

        V = self.mech.u.function_space
        self._snapshots = [
            fem.Function(V, name=f"u_gait_snap_{i}") for i in range(len(self._quadrature))
        ]

    def _update_snapshots(self) -> None:
        """Recompute displacement snapshots for all gait quadrature points.

        This is the heavy part: for each phase p_j we:
            - update Neumann loads via gait_loader,
            - assemble RHS,
            - solve K(ρ, A) u_j = f_j,
            - copy solution into u_snapshot[j].

        The current mechanics state is restored at the end.
        """
        self._ensure_snapshots()
        assert self._quadrature is not None
        assert self._snapshots is not None

        # Save current displacement state
        V = self.mech.u.function_space
        u_saved = fem.Function(V)
        u_saved.x.array[:] = self.mech.u.x.array

        # Loop over gait phases
        for (phase, _w), u_snap in zip(self._quadrature, self._snapshots):
            self.gait.update_loads(phase)
            self.mech.assemble_rhs()
            self.mech.solve()
            # Copy solution into snapshot and scatter
            u_snap.x.array[:] = self.mech.u.x.array
            u_snap.x.scatter_forward()

        # Restore saved state
        self.mech.u.x.array[:] = u_saved.x.array
        self.mech.u.x.scatter_forward()

        self._dirty = False

    def _build_expressions(self) -> None:
        """Build UFL expressions for ψ̄ and M̄ using snapshot functions.

        This is done only once. Later, `invalidate()` + `_update_snapshots()`
        will refresh the snapshot values without rebuilding these graphs.
        """
        self._ensure_snapshots()
        assert self._quadrature is not None
        assert self._snapshots is not None

        # Strain–energy density and structure tensor averaged over phases
        psi_eff = ufl.as_ufl(0.0)

        d = self.mech.gdim
        M_eff = ufl.zero((d, d))

        for (phase, w), u_snap in zip(self._quadrature, self._snapshots):
            # NOTE: These calls construct UFL expressions; they do not touch
            # the current numeric values of u_snap / ρ / A.
            psi_i = self.mech.get_strain_energy_density(u_snap)
            e_i = self.mech.get_strain_tensor(u_snap)
            psi_eff = psi_eff + w * psi_i
            M_eff = M_eff + w * ufl.dot(ufl.transpose(e_i), e_i)

        # No cycles-per-day scaling here; that belongs in rS_gain if needed.
        self._psi_expr = psi_eff
        self._M_expr = M_eff

    def _ensure_built_and_fresh(self) -> None:
        """Make sure (ψ̄, M̄) graphs exist and snapshots match the current (ρ, A)."""
        if self._psi_expr is None or self._M_expr is None:
            self._build_expressions()
        if self._dirty:
            self._update_snapshots()



class GaitEnergyDriver:
    """Gait-averaged mechanical drivers with *prebuilt* UFL expressions.

    This driver builds UFL expressions once:
        psi_eff = sum_i w_i * psi(u_snap[i], rho, A)
        M_eff   = sum_i w_i * (epsilon(u_snap[i])^T * epsilon(u_snap[i]))
    where u_snap[i] are persistent Functions on the mechanics displacement space.

    On each GS sweep (i.e., whenever (rho, A) change), call update_snapshots()
    to recompute u_snap[i] by solving the mechanics with the current material.
    No UFL graph rebuilds happen in GS; only RHS assemble + linear solves.
    """

    def __init__(self, mechsolver, gait_loader, cycles_per_day: float = 1.0, m_exponent: int = 2, psi_ref: float | None = None):
        self.mech = mechsolver
        self.gait = gait_loader
        self.cycles_per_day = float(cycles_per_day)
        self.m = int(m_exponent)
        self.psi_ref = float(psi_ref) if psi_ref is not None else None

        from dolfinx import fem
        self._u_snap: list[fem.Function] | None = None
        self._weights: list[float] | None = None
        self._phases: list[float] | None = None

        # Prebuilt UFL expressions
        self._psi_expr = None
        self._M_expr = None

        # cache of function space id to detect need for rebuild
        self._V_id = id(self.mech.u.function_space)

    def invalidate(self):
        """Mark expressions invalid if displacement space changed (mesh/FS rebuild)."""
        V_id_now = id(self.mech.u.function_space)
        if V_id_now != self._V_id:
            self._u_snap = None
            self._psi_expr = None
            self._M_expr = None
            self._V_id = V_id_now

    def _ensure_prebuilt(self):
        """Build u_snap[i] and UFL expressions once."""
        if self._u_snap is not None and self._psi_expr is not None and self._M_expr is not None:
            return

        from dolfinx import fem
        import ufl

        # Prepare persistent snapshots and quadrature metadata
        snaps = []
        weights = []
        phases = []
        for phase, w in self.gait.get_quadrature():
            snaps.append(fem.Function(self.mech.u.function_space))
            weights.append(float(w))
            phases.append(float(phase))
        self._u_snap = snaps
        self._weights = weights
        self._phases = phases

        # Build UFL expressions once
        psi_terms = []
        M_terms = []
        for u_i, w in zip(self._u_snap, self._weights):
            psi_i = self.mech.get_strain_energy_density(u_i)  # UFL scalar
            # Structure tensor ~ E^T E (or any symmetric PSD measure you use)
            E = self.mech.green_strain(u_i) if hasattr(self.mech, "green_strain") else ufl.sym(ufl.grad(u_i))
            M_i = ufl.dot(E.T, E)  # tensor
            psi_terms.append(w * ((psi_i / self.psi_ref)**self.m if self.psi_ref else psi_i**self.m))
            M_terms.append(w * M_i)

        # Cycles-per-day scaling on energy side (kept in driver by design)
        psi_eff = sum(psi_terms)
        if self.cycles_per_day != 1.0:
            psi_eff = self.cycles_per_day * psi_eff

        M_eff = sum(M_terms)

        self._psi_expr = psi_eff
        self._M_expr = M_eff

    def update_snapshots(self):
        """Recompute u_snap[i] by cycling gait phases with current (rho, A)."""
        self._ensure_prebuilt()

        # optional: save/restore current u to avoid side effects
        from dolfinx import fem
        u_saved = fem.Function(self.mech.u.function_space)
        u_saved.x.array[:] = self.mech.u.x.array

        for phase in self._phases:
            self.gait.update_loads(phase)
            self.mech.assemble_rhs()
            self.mech.solve()

            # copy to the corresponding snapshot
            i = self._phases.index(phase)  # small list; ok
            self._u_snap[i].x.array[:] = self.mech.u.x.array

        # restore u
        self.mech.u.x.array[:] = u_saved.x.array

    def energy_expr(self):
        """Return prebuilt UFL expression for gait-averaged driver (depends on u_snap)."""
        self._ensure_prebuilt()
        return self._psi_expr

    def structure_expr(self):
        """Return prebuilt UFL expression for structure tensor (depends on u_snap)."""
        self._ensure_prebuilt()
        return self._M_expr
