"""Density evolution solver with diffusion and bounded formation/resorption kinetics."""

from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
import ufl

from simulation.solvers.base import BaseLinearSolver
from simulation.stats import SweepStats
from simulation.utils import smooth_max, smoothstep01, smooth_clamp

if TYPE_CHECKING:
    from simulation.config import Config


class DensitySolver(BaseLinearSolver):
    """Density ρ: diffusion + bounded formation/resorption kinetics (implicit Euler).

    Solves:
        ρ/dt - D_ρ Δρ + reaction·ρ = ρ_old/dt + k_form·S⁺ + k_res·S⁻

    The reaction term ensures bounds are attracting:
    - Formation vanishes as ρ → ρ_max
    - Resorption vanishes as ρ → ρ_min

    Surface availability A(ρ) modulates kinetics based on vascular porosity.
    """

    _label = "dens"

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        stimulus: fem.Function,
        config: "Config",
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.S = stimulus
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

    def _compile_forms(self):
        dt = self.dt_c

        eps = float(self.cfg.numerics.smooth_eps)

        S_pos = smooth_max(self.S, 0.0, eps)
        S_neg = smooth_max(-self.S, 0.0, eps)

        # Surface availability A(rho_old)
        if bool(self.cfg.density.surface_use):
            rho_t = float(self.cfg.density.rho_tissue)
            f_raw = 1.0 - (self.rho_old / rho_t)
            f = smooth_clamp(f_raw, 0.0, 1.0, eps)

            rho_trab_max = float(self.cfg.material.rho_trab_max)
            rho_cort_min = float(self.cfg.material.rho_cort_min)

            # Trabecular (Martin-type) specific surface
            S_trab = (
                32.3 * f
                - 93.9 * f**2
                + 134.0 * f**3
                - 101.0 * f**4
                + 28.8 * f**5
            )

            # Cortical proxy
            f_tr = max(1.0 - rho_trab_max / rho_t, 1e-6)
            S_trab_tr = (
                32.3 * f_tr
                - 93.9 * f_tr**2
                + 134.0 * f_tr**3
                - 101.0 * f_tr**4
                + 28.8 * f_tr**5
            )
            surface_cort_scale = S_trab_tr / (f_tr**0.5)
            S_cort = surface_cort_scale * ufl.sqrt(f + eps)

            denom = max(rho_cort_min - rho_trab_max, 1e-12)
            t = (self.rho_old - rho_trab_max) / denom
            w_cort = smoothstep01(t)

            S_v = (1.0 - w_cort) * S_trab + w_cort * S_cort
            S_v = smooth_max(S_v, 0.0, eps)

            A_min = float(self.cfg.density.surface_A_min)
            S0 = float(self.cfg.density.surface_S0)
            A_surf = A_min + (1.0 - A_min) * (S_v / (S_v + S0))
        else:
            A_surf = 1.0

        rho_min = float(self.cfg.density.rho_min)
        rho_max = float(self.cfg.density.rho_max)

        k_form = float(self.cfg.density.k_rho_form) * A_surf
        k_res = float(self.cfg.density.k_rho_resorb) * A_surf

        reaction = (k_form * S_pos / rho_max) + (k_res * S_neg / rho_min)

        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.density.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction * self.trial * self.test * self.dx
        )
        self.a_form = fem.form(a_ufl)

        L_ufl = ((self.rho_old / dt) + (k_form * S_pos) + (k_res * S_neg)) * self.test * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        self.dt_c.value = float(self.cfg.dt)
        self.S.x.scatter_forward()
        self.rho_old.x.scatter_forward()
        super().assemble_lhs()

    def assemble_rhs(self):
        self.S.x.scatter_forward()
        self.rho_old.x.scatter_forward()

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
        return (self.rho,)

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return (self.rho_old,)

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        return (self.rho,)

    def solve(self) -> SweepStats:
        self.assemble_lhs()
        self.assemble_rhs()
        return self._solve()

    def sweep(self) -> SweepStats:
        return self.solve()
