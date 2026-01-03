"""Stimulus solver with diffusion, decay, and saturating mechanostat drive."""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
import ufl

from simulation.solvers.base import BaseLinearSolver
from simulation.stats import SweepStats
from simulation.utils import smooth_max, smooth_abs, smoothstep01

if TYPE_CHECKING:
    from simulation.config import Config


class StimulusSolver(BaseLinearSolver):
    """Stimulus S: diffusion + decay with saturating mechanostat drive.

    Solves (implicit Euler):
        τ/dt (S - S_old) + S - τ·D ΔS = drive(m)

    where drive = S_max · tanh((m - m_ref)/m_ref / κ) with optional lazy-zone.
    """

    _label = "stim"

    def __init__(
        self,
        S: fem.Function,
        S_old: fem.Function,
        psi_field,
        rho: fem.Function,
        config: "Config",
    ):
        super().__init__(config, S, [], [])
        self.S = self.state
        self.S_old = S_old
        self.psi = psi_field
        self.rho = rho

        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

        self.tau_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_tau))
        self.D_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_D))
        self.S_max_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_S_max))
        self.kappa_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_kappa))

        self._use_diffusion = float(self.cfg.stimulus.stimulus_D) > 0.0

    def _compile_forms(self):
        dt = self.dt_c
        tau = self.tau_c

        S_trial = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)

        alpha = tau / dt
        a_ufl = (alpha + 1.0) * S_trial * v * self.dx
        if self._use_diffusion:
            a_ufl += (tau * self.D_c) * ufl.dot(ufl.grad(S_trial), ufl.grad(v)) * self.dx
        self.a_form = fem.form(a_ufl)

        # Production term (explicit in time)
        eps = float(self.cfg.numerics.smooth_eps)
        rho_safe = smooth_max(self.rho, self.cfg.density.rho_min, eps)
        m = self.psi / rho_safe

        # Blended reference stimulus
        denom = float(self.cfg.material.rho_cort_min - self.cfg.material.rho_trab_max)
        t = (rho_safe - self.cfg.material.rho_trab_max) / denom
        w = smoothstep01(t, eps)
        
        psi_ref_trab = float(self.cfg.stimulus.psi_ref_trab)
        psi_ref_cort = float(self.cfg.stimulus.psi_ref_cort)
        m_ref = psi_ref_trab * (1.0 - w) + psi_ref_cort * w

        delta = (m - m_ref) / m_ref
        delta_abs = smooth_abs(delta, eps)

        delta0 = float(self.cfg.stimulus.stimulus_delta0)
        if delta0 > 0.0:
            delta0_safe = max(delta0, 1e-12)
            gate = 1.0 - ufl.exp(-((delta_abs / delta0_safe) ** 2))
            delta_eff = delta * gate
        else:
            delta_eff = delta

        drive = self.S_max_c * ufl.tanh(delta_eff / self.kappa_c)

        L_ufl = (alpha * self.S_old + drive) * v * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        self.dt_c.value = float(self.cfg.dt)
        super().assemble_lhs()

    def assemble_rhs(self):
        self.S_old.x.scatter_forward()
        self.psi.x.scatter_forward()
        self.rho.x.scatter_forward()

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    # -------------------------------------------------------------------------
    # CouplingBlock protocol
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return (self.S,)

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return (self.S_old,)

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        return (self.S,)

    def solve(self) -> SweepStats:
        if float(self.dt_c.value) != float(self.cfg.dt):
            self.assemble_lhs()
        self.assemble_rhs()
        return self._solve()

    def sweep(self) -> SweepStats:
        return self.solve()
