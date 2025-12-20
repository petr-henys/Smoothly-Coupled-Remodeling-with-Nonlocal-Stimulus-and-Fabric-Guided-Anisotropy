"""Linear subsolvers for mechanics, stimulus, and density."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)
import ufl

from simulation.utils import build_nullspace, smooth_max, smoothstep01
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config


class _BaseLinearSolver:
    """Base class for PETSc KSP linear solvers."""

    def __init__(
        self,
        cfg: "Config",
        state_function: fem.Function,
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
    ):
        self.state = state_function
        self.function_space = state_function.function_space
        self.mesh = self.function_space.mesh
        self.comm = self.mesh.comm
        self.rank = self.comm.rank
        self.gdim = self.mesh.geometry.dim

        self.cfg = cfg
        self.dx = self.cfg.dx
        self.ds = self.cfg.ds
        self.logger = get_logger(self.comm, name=self.__class__.__name__, log_file=self.cfg.log_file)
        self.smooth_eps: float = self.cfg.smooth_eps

        self.trial = ufl.TrialFunction(self.function_space)
        self.test = ufl.TestFunction(self.function_space)

        self.last_reason: int = 0

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: PETSc.KSP = None
        self.A: PETSc.Mat = None
        self.b: PETSc.Vec = None

        self.a_form: fem.Form = None
        self.L_form: fem.Form = None

    def destroy(self):
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None
        if self.A is not None:
            self.A.destroy()
            self.A = None
        if self.b is not None:
            self.b.destroy()
            self.b = None

    def setup(self):
        self._compile_forms()
        self.A = create_matrix(self.a_form)
        self.b = create_vector(self.function_space)
        self.assemble_lhs()
        self._setup_ksp()

    def assemble_lhs(self) -> None:
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()
        if self.ksp is not None:
            self.ksp.setOperators(self.A)

    def assemble_rhs(self) -> None:
        raise NotImplementedError

    def _solve(self) -> int:
        self.ksp.solve(self.b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()

        reason = int(self.ksp.getConvergedReason())
        self.last_reason = reason
        return reason

    def _maybe_warn(self, reason: int, label: str):
        if reason < 0:
            self.logger.warning(f"{label} solver failed to converge (reason: {reason})")

    def create_ksp(self, prefix: str, ksp_options: dict[str, object]) -> PETSc.KSP:
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOptionsPrefix(prefix + "_")

        opts = PETSc.Options()
        for k, v in ksp_options.items():
            if v is not None:
                opts[f"{prefix}_{k}"] = v

        opts[f"{prefix}_ksp_rtol"] = self.cfg.ksp_rtol
        opts[f"{prefix}_ksp_atol"] = self.cfg.ksp_atol
        opts[f"{prefix}_ksp_max_it"] = self.cfg.ksp_max_it

        self.ksp.setInitialGuessNonzero(True)
        if self.A is not None:
            self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()
        self.ksp.setUp()
        return self.ksp


class MechanicsSolver(_BaseLinearSolver):
    """Linear elasticity with density-dependent stiffness."""

    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        config: "Config",
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho
        self._nullspace = None

    def _compile_forms(self):
        a_ufl = ufl.inner(self.sigma(self.trial, self.rho), self.eps(self.test)) * self.dx
        self.a_form = fem.form(a_ufl)

        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_ufl = ufl.inner(zero_vec, self.test) * self.ds
        n = ufl.FacetNormal(self.mesh)

        for t, tag in self.neumann_bcs:
            if len(t.ufl_shape) == 0:
                L_ufl = L_ufl + ufl.inner(-t * n, self.test) * self.ds(tag)
            else:
                L_ufl = L_ufl + ufl.inner(t, self.test) * self.ds(tag)

        self.L_form = fem.form(L_ufl)

    def assemble_lhs(self) -> None:
        self.rho.x.scatter_forward()
        super().assemble_lhs()

    def _setup_ksp(self):
        self._nullspace = build_nullspace(self.function_space)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(self._nullspace)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)

    @staticmethod
    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho):
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)
        rho_rel = rho_eff / self.cfg.rho_ref

        denom = float(self.cfg.rho_cort_min - self.cfg.rho_trab_max)
        t = (rho_eff - self.cfg.rho_trab_max) / denom
        w = smoothstep01(t)
        k = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w

        E = self.cfg.E0 * (rho_rel ** k)
        nu = self.cfg.nu0
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return 2.0 * mu * self.eps(u) + lmbda * ufl.tr(self.eps(u)) * ufl.Identity(self.gdim)

    def assemble_rhs(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.dirichlet_bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.dirichlet_bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return int(reason)

    # Fixed interface expected by the coupling solver
    @property
    def state_fields(self):
        return ()

    def sweep(self):
        reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return {"label": "mech", "reason": int(reason)}


class StimulusSolver(_BaseLinearSolver):
    """Update stimulus field `S` via diffusion/decay with a saturating drive.

    Model (dimensionless `S`):
      dS/dt = D_S ΔS - (1/tau_S) S + (1/tau_S) S_max tanh(delta/kappa),
      with the local quasi-static limit S = S_max tanh(delta/kappa) for tau_S=0.
    where delta is computed from the specific energy `m = psi/rho_safe` relative to `m_ref`.
    """

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

        self.logger = get_logger(self.comm, name="Stimulus", log_file=self.cfg.log_file)

        # Time step is adaptive, so keep dt as a Constant (updates reassemble the LHS).
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

        # Stimulus parameters (Constants for UFL forms)
        self.tau_c = fem.Constant(self.mesh, float(self.cfg.stimulus_tau))
        self.D_c = fem.Constant(self.mesh, float(self.cfg.stimulus_D))
        self.S_max_c = fem.Constant(self.mesh, float(self.cfg.stimulus_S_max))
        self.kappa_c = fem.Constant(self.mesh, float(self.cfg.stimulus_kappa))
        self.delta0_c = fem.Constant(self.mesh, float(self.cfg.stimulus_delta0))

        # Compile-time flag: avoid forming grad-grad if D_S is identically zero (helps DG spaces).
        self._use_diffusion = float(self.cfg.stimulus_D) > 0.0

    def _compile_forms(self):
        dt = self.dt_c
        tau = self.tau_c

        S_trial = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)

        # (tau/dt) * S^{n+1} + S^{n+1}  on the LHS, plus diffusion if enabled.
        alpha = tau / dt
        a_ufl = (alpha + 1.0) * S_trial * v * self.dx
        if self._use_diffusion:
            a_ufl += (tau * self.D_c) * ufl.dot(ufl.grad(S_trial), ufl.grad(v)) * self.dx
        self.a_form = fem.form(a_ufl)

        # Production term (explicit in time): lazy-zone + saturation
        #   delta = (m - m_ref) / m_ref
        #   |delta| <= delta0 -> drive = 0
        #   otherwise drive = S_max * tanh((|delta|-delta0)/kappa) * sign(delta)
        eps = float(self.cfg.smooth_eps)
        rho_safe = smooth_max(self.rho, self.cfg.rho_min, eps)
        m = self.psi / rho_safe
        m_ref = float(self.cfg.psi_ref) / float(self.cfg.rho_ref)
        delta = (m - m_ref) / m_ref
        delta_abs = ufl.sqrt(delta * delta + eps * eps)
        delta_excess = smooth_max(delta_abs - self.delta0_c, 0.0, eps)
        sgn = delta / delta_abs
        drive = self.S_max_c * ufl.tanh(delta_excess / self.kappa_c) * sgn

        L_ufl = (alpha * self.S_old + drive) * v * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        # dt may change between steps; update Constant and reassemble LHS
        self.dt_c.value = float(self.cfg.dt)
        super().assemble_lhs()

    def assemble_rhs(self):
        self.S_old.x.scatter_forward()
        if hasattr(self.psi, "x"):
            self.psi.x.scatter_forward()
        self.rho.x.scatter_forward()

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    @property
    def state_fields(self):
        return (self.S,)

    def solve(self) -> int:
        # Reassemble LHS when dt changes (adaptive stepping).
        if float(self.dt_c.value) != float(self.cfg.dt):
            self.assemble_lhs()
        self.assemble_rhs()
        reason = self._solve()
        self._maybe_warn(reason, "Stimulus")
        return int(reason)

    def sweep(self):
        reason = self.solve()
        return {"label": "stim", "reason": int(reason)}


class DensitySolver(_BaseLinearSolver):
    """Density evolution (implicit Euler diffusion + reaction driven by stimulus S).

    d(rho)/dt = D * Laplacian(rho) + k_rho * S
    """

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

        # Smooth helpers (avoid hard clamps / if-branches in UFL)
        eps = float(self.cfg.smooth_eps)

        def smooth_min(a, b, eps_):
            # smooth_min(a,b) = a + b - smooth_max(a,b)
            return a + b - smooth_max(a, b, eps_)

        # Positive / negative stimulus parts (non-negative)
        S_pos = smooth_max(self.S, 0.0, eps)
        S_neg = smooth_max(-self.S, 0.0, eps)

        # Surface availability A(rho_old) from apparent density via a vascular-porosity proxy.
        if bool(self.cfg.surface_use):
            rho_t = float(self.cfg.rho_tissue)
            f_raw = 1.0 - (self.rho_old / rho_t)
            f = smooth_min(smooth_max(f_raw, 0.0, eps), 1.0, eps)

            # Martin polynomial for specific surface S_V(f) [1/mm]
            S_v = (
                32.3 * f
                - 93.9 * f**2
                + 134.0 * f**3
                - 101.0 * f**4
                + 28.8 * f**5
            )
            S_v = smooth_max(S_v, 0.0, eps)

            A_min = float(self.cfg.surface_A_min)
            S0 = float(self.cfg.surface_S0)
            x = S_v / S0
            x = smooth_min(x, 1.0, eps)  # linear in S_v, capped
            A_surf = A_min + (1.0 - A_min) * x
        else:
            A_surf = 1.0

        # Soft-bounded kinetics: formation vanishes as rho -> rho_max; resorption vanishes as rho -> rho_min.
        rho_min = float(self.cfg.rho_min)
        rho_max = float(self.cfg.rho_max)

        k_form = float(self.cfg.k_rho_form) * A_surf
        k_res = float(self.cfg.k_rho_resorb) * A_surf

        # Linear reaction coefficient on rho^{n+1}
        reaction = (k_form * S_pos / rho_max) + (k_res * S_neg / rho_min)

        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction * self.trial * self.test * self.dx
        )
        self.a_form = fem.form(a_ufl)

        # RHS constants (note: the reaction term ensures the bounds are attracting).
        L_ufl = ((self.rho_old / dt) + (k_form * S_pos) + (k_res * S_neg)) * self.test * self.dx
        self.L_form = fem.form(L_ufl)
    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        self.dt_c.value = float(self.cfg.dt)
        super().assemble_lhs()

    def assemble_rhs(self):
        self.S.x.scatter_forward()
        self.rho_old.x.scatter_forward()

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    @property
    def state_fields(self):
        return (self.rho,)

    def solve(self) -> int:
        # LHS depends on dt, stimulus, and rho_old (via surface availability) -> reassemble each call
        self.assemble_lhs()
        self.assemble_rhs()
        reason = self._solve()
        self._maybe_warn(reason, "Density")
        return int(reason)
    def sweep(self):
        reason = self.solve()
        return {"label": "dens", "reason": int(reason)}