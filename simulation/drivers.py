"""Energy and structure tensor drivers for remodeling stimulus and direction solvers."""

from __future__ import annotations
from typing import Protocol
from dolfinx import fem
import ufl

class StrainDriver(Protocol):
    def energy_expr(self) -> ufl.core.expr.Expr: ...
    def structure_expr(self) -> ufl.core.tensor.Tensor: ...
    def invalidate(self) -> None: ...

class InstantEnergyDriver:
    """Instantaneous energy ψ(u) and structure tensor M=εᵀε from current mechanics state."""
    def __init__(self, mech):
        self.mech = mech
    def energy_expr(self):
        """Strain energy density ψ(u)."""
        return self.mech.get_strain_energy_density()
    def structure_expr(self):
        """Structure tensor M = εᵀε."""
        e = self.mech.get_strain_tensor()
        return ufl.dot(ufl.transpose(e), e)
    def invalidate(self): 
        pass

class GaitEnergyDriver:
    """Gait-averaged energy and structure via quadrature over frozen displacement snapshots."""
    def __init__(self, mech, gait_loader, cycles_per_day: float = 1.0):
        self.mech = mech
        self.gait = gait_loader
        self.cpd  = float(cycles_per_day)
        self._psi_expr = None
        self._M_expr = None

    def invalidate(self):
        self._psi_expr = None
        self._M_expr = None

    def energy_expr(self):
        """Gait-averaged energy density ⟨ψ⟩."""
        if self._psi_expr is None:
            self._psi_expr = self._build_energy_expr()
        return self._psi_expr

    def structure_expr(self):
        """Gait-averaged structure tensor ⟨εᵀε⟩."""
        if self._M_expr is None:
            self._M_expr = self._build_structure_expr()
        return self._M_expr

    def _build_energy_expr(self):
        """Build gait-averaged energy UFL expression via phase quadrature."""
        psi_eff = ufl.as_ufl(0.0)
        for phase, w in self.gait.get_quadrature():
            self.gait.update_loads(phase)
            self.mech.assemble_rhs()
            self.mech.solve()
            u_snap = fem.Function(self.mech.u.function_space)
            u_snap.x.array[:] = self.mech.u.x.array
            psi_eff += w * self.mech.get_strain_energy_density(u_snap)
        return self.cpd * psi_eff

    def _build_structure_expr(self):
        """Build gait-averaged structure tensor UFL expression via phase quadrature."""
        d = self.mech.gdim
        M_eff = ufl.zero((d, d))
        for phase, w in self.gait.get_quadrature():
            self.gait.update_loads(phase)
            self.mech.assemble_rhs()
            self.mech.solve()
            u_snap = fem.Function(self.mech.u.function_space)
            u_snap.x.array[:] = self.mech.u.x.array
            e = self.mech.get_strain_tensor(u_snap)
            M_eff = M_eff + w * ufl.dot(ufl.transpose(e), e)
        return M_eff
