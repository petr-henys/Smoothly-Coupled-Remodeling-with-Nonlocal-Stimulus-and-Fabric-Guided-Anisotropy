"""PDE subsolvers: mechanics (u), stimulus (S), density (ρ), direction (A)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)
import ufl

from simulation.utils import build_nullspace
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config


# --- Smooth regularization helpers (C^∞ approximations) ---

def smooth_abs(x, eps: float):
    """C^∞ approximation of |x|."""
    return ufl.sqrt(x * x + eps * eps)


def smooth_plus(x, eps: float):
    """C^∞ approximation of max(x, 0)."""
    sabs = smooth_abs(x, eps)
    return 0.5 * (x + sabs)


def smooth_max(x, xmin, eps: float):
    """C^∞ approximation of max(x, xmin)."""
    dx = x - xmin
    return xmin + 0.5 * (dx + ufl.sqrt(dx * dx + eps * eps))


def smooth_heaviside(x, eps: float):
    """C^∞ approximation of step function H(x)."""
    return 0.5 * (1.0 + x / ufl.sqrt(x * x + eps * eps))


# --- PSD + unit-trace projection helpers ---

def unittrace_psd_from_any(T, dim: int, eps: float):
    """Project arbitrary tensor T to unit-trace PSD via TᵀT + εI."""
    I = ufl.Identity(dim)
    M = ufl.dot(ufl.transpose(T), T) + eps * I
    return M / ufl.tr(M)


def unittrace_psd(B, dim: int, eps: float):
    """Project PSD tensor B to unit-trace via B + εI."""
    I = ufl.Identity(dim)
    M = B + eps * I
    return M / ufl.tr(M)


def get_distal_damping_mask(mesh, z_min: float, height: float = 15.0, transition: float = 5.0):
    """
    Returns a UFL expression for a spatial mask that dampens values near the distal boundary.

    The mask is:
      - 0.0 for z < z_min + height
      - Transitions from 0.0 to 1.0 for z in [z_min + height, z_min + height + transition]
      - 1.0 for z > z_min + height + transition
    """
    x = ufl.SpatialCoordinate(mesh)
    z = x[2]
    z_start = z_min + height
    
    if transition <= 1e-14:
        # Step function if transition is effectively zero
        return ufl.conditional(ufl.ge(z, z_start), 1.0, 0.0)
    
    t = (z - z_start) / transition
    return ufl.max_value(0.0, ufl.min_value(1.0, t))


# --- Base linear solver class ---

class _BaseLinearSolver:
    """Base KSP solver with setup, assembly, solve, and stats tracking."""

    def __init__(
        self,
        cfg: Config,
        state_function: fem.Function,
        dirichlet_bcs: list[fem.DirichletBC],
        neumann_bcs: list[tuple[fem.Function, int]],
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
        self.logger = get_logger(self.comm, verbose=self.cfg.verbose, name=self.__class__.__name__)
        self.smooth_eps: float = self.cfg.smooth_eps

        self.trial = ufl.TrialFunction(self.function_space)
        self.test = ufl.TestFunction(self.function_space)

        self.total_iters = 0
        self.ksp_steps = 0
        self.last_reason: int | None = None
        self.last_iters: int | None = None

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: PETSc.KSP | None = None
        self.A: PETSc.Mat | None = None
        self.b: PETSc.Vec | None = create_vector(self.function_space)
        self.a_form: ufl.Form | None = None

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

    def _reset_stats(self):
        self.last_reason = None
        self.last_iters = None

    def build_lhs_form(self):
        """Return UFL bilinear form for the current LHS."""
        raise NotImplementedError(f"{self.__class__.__name__}.build_lhs_form() not implemented")

    def assemble_lhs(self):
        """(Re)build and assemble LHS matrix from subclass-provided form."""
        self.a_form = fem.form(self.build_lhs_form())
        if self.A is None:
            self.A = create_matrix(self.a_form)
        else:
            self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()

        if self.ksp is not None:
            self.ksp.setOperators(self.A)
            self.ksp.setUp()
        self._reset_stats()

    def _solve(self) -> tuple[int, int]:
        self.ksp.solve(self.b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()
        its = self.ksp.getIterationNumber()
        reason = self.ksp.getConvergedReason()
        self.total_iters += its
        self.ksp_steps += 1
        self.last_iters = its
        self.last_reason = reason
        return its, reason

    @property
    def ksp_its(self) -> int:
        return int(self.total_iters)

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


# --- Subsolver implementations ---

class MechanicsSolver(_BaseLinearSolver):
    """Elastic equilibrium with anisotropic fabric reinforcement."""

    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        A_dir: fem.Function,
        config: Config,
        dirichlet_bcs: list[fem.DirichletBC],
        neumann_bcs: list[tuple[fem.Function, int]],
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho
        self.A_dir = A_dir

        # Neumann RHS template
        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_form = ufl.inner(zero_vec, self.test) * self.ds
        for t, tag in self.neumann_bcs:
            L_form = L_form + ufl.inner(t, self.test) * self.ds(tag)
        self.L_ufl = L_form
        self.L_form = fem.form(L_form)

    def build_lhs_form(self):
        return ufl.inner(self.sigma(self.trial, self.rho, self.A_dir), self.eps(self.test)) * self.dx

    def eps(self, u):
        """Symmetric gradient ε(u)."""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho, A_dir):
        """Cauchy stress σ(u) [MPa].

        E(ρ) = E0 * ρ^n(ρ), where n(ρ) transitions from n_trab to n_cort.
        """
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)

        # Smooth transition factor w(ρ) in [0, 1]
        rho1, rho2 = float(self.cfg.rho_trab_max), float(self.cfg.rho_cort_min)
        if rho2 <= rho1:
            n_eff = self.cfg.n_trab
        else:
            s_raw = (rho_eff - rho1) / (rho2 - rho1)
            s0 = smooth_max(s_raw, 0.0, self.smooth_eps)
            s1 = 1.0 - smooth_max(1.0 - s0, 0.0, self.smooth_eps)
            w = 3.0 * s1**2 - 2.0 * s1**3
            n_eff = (1.0 - w) * self.cfg.n_trab + w * self.cfg.n_cort

        E = self.cfg.E0 * (rho_eff ** n_eff)

        eps_ten = self.eps(u)
        I = ufl.Identity(self.gdim)
        nu = self.cfg.nu
        lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        Asym = 0.5 * (A_dir + ufl.transpose(A_dir))
        Ahat = unittrace_psd_from_any(Asym, self.gdim, self.smooth_eps)

        sigma_aniso = (self.cfg.xi_aniso * E) * ufl.inner(Ahat, eps_ten) * Ahat
        return 2 * mu * eps_ten + lmbda * ufl.tr(eps_ten) * I + sigma_aniso

    def get_strain_tensor(self, u=None):
        return self.eps(self.u if u is None else u)

    def setup(self):
        self.assemble_lhs()

        ns = build_nullspace(self.function_space)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(ns)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.dirichlet_bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.dirichlet_bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return its, reason

    def energy_balance(self):
        """Compute internal and external work and their relative error."""
        # Internal work: a(u,u)
        a_uu_local = fem.assemble_scalar(fem.form(ufl.inner(self.sigma(self.u, self.rho, self.A_dir), self.eps(self.u)) * self.dx))
        W_int = self.comm.allreduce(a_uu_local, op=MPI.SUM)
        
        # External work: l(u)
        # We need l(u). L_form is defined as inner(t, v) * ds.
        # We can replace v with u.
        L_u_form = fem.form(ufl.replace(self.L_ufl, {self.test: self.u}))
        l_u_local = fem.assemble_scalar(L_u_form)
        W_ext = self.comm.allreduce(l_u_local, op=MPI.SUM)
        
        denom = max(abs(W_int), abs(W_ext), 1e-300)
        rel_error = abs(W_int - W_ext) / denom
        return W_int, W_ext, rel_error


class StimulusSolver(_BaseLinearSolver):
    """Reaction-diffusion stimulus S driven by mechanical driver ψ(x)."""

    def __init__(
        self,
        S: fem.Function,
        S_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, S, [], [])
        self.S = self.state
        self.S_old = S_old
        self._rhs_form = None
        self.z_min = None

    def _compute_z_min(self):
        if self.z_min is None:
            z_coords = self.mesh.geometry.x[:, 2]
            local_min = z_coords.min() if z_coords.size > 0 else 1e30
            self.z_min = self.comm.allreduce(local_min, op=MPI.MIN)
        return self.z_min

    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        return (
            (self.cfg.cS / dt + self.cfg.tauS) * self.trial * self.test * self.dx
            + self.cfg.kappaS * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self, psi_expr):
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        self.S_old.x.scatter_forward()
        dt = self.cfg.dt

        # Apply distal damping to psi to avoid artifacts at u=0 boundary
        z_min = self._compute_z_min()
        mask = get_distal_damping_mask(self.mesh, z_min, height=self.cfg.distal_damping_height, 
                                       transition=self.cfg.distal_damping_transition)
        psi_effective = psi_expr * mask

        rhs = (self.cfg.cS / dt) * self.S_old + self.cfg.rS_gain * (psi_effective - 1.0)
        self._rhs_form = fem.form(rhs * self.test * self.dx)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Stimulus")
        return its, reason

    def power_balance_residual(self, psi_expr):
        """Compute power balance residual: Storage + Decay - Source."""
        dt = self.cfg.dt
        S, S_old = self.S, self.S_old
        
        # Terms
        storage = (self.cfg.cS / dt) * (S - S_old)
        decay = self.cfg.tauS * S
        
        # Source
        z_min = self._compute_z_min()
        mask = get_distal_damping_mask(self.mesh, z_min, height=self.cfg.distal_damping_height, 
                                       transition=self.cfg.distal_damping_transition)
        psi_eff = psi_expr * mask
        source = self.cfg.rS_gain * (psi_eff - 1.0)
        
        integrand = storage + decay - source
        res_local = fem.assemble_scalar(fem.form(integrand * self.dx))
        res_abs = self.comm.allreduce(res_local, op=MPI.SUM)
        
        # Relative to source magnitude
        src_mag_local = fem.assemble_scalar(fem.form(abs(source) * self.dx))
        src_mag = self.comm.allreduce(src_mag_local, op=MPI.SUM)
        
        return res_abs, abs(res_abs) / max(src_mag, 1e-300)


class DensitySolver(_BaseLinearSolver):
    """Density evolution ρ: anisotropic diffusion with soft mechanostat ρ_eq(S)."""

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        A_dir: fem.Function,
        S: fem.Function,
        config: Config,
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.A_dir = A_dir
        self.S = S
        self.L_form_template = None
        self.a_form = fem.form(self.build_lhs_form())

    
    def _get_reaction_terms(self):
        """Compute lazy-zone factor f(|S|), effective reaction rate λ_eff(S), and equilibrium density ρ_eq(S)
        for a Frost-like two-threshold mechanostat with smooth transitions.

        Returns
        -------
        lam_eff : UFL expression
            Effective reaction rate [1/day].
        rho_eq : UFL expression
            Target density given the current zone (form/resorb/neutral).
        """
        Sabs = smooth_abs(self.S, self.smooth_eps)
        S_lazy = float(self.cfg.S_lazy)
        fS = Sabs / (Sabs + S_lazy) if S_lazy > 0.0 else 1.0

        # Smooth Heavisides
        k_step = float(self.cfg.k_step)
        S_form = float(self.cfg.S_form_th)
        S_resorb = float(self.cfg.S_resorb_th)

        H_form = 1.0/(1.0 + ufl.exp(-k_step*(self.S - S_form)))
        H_resorb = 1.0/(1.0 + ufl.exp(-k_step*(S_resorb - self.S)))

        lam_form = float(self.cfg.lambda_form)
        lam_resorb = float(self.cfg.lambda_resorb)
        lam_eff = fS * (lam_form*H_form + lam_resorb*H_resorb)

        # Target density: rho_max in formation, rho_min in resorption, otherwise hold current rho
        rho_eq = self.cfg.rho_max*H_form + self.cfg.rho_min*H_resorb + (1.0 - H_form - H_resorb)*self.rho

        return lam_eff, rho_eq


    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        d = self.mesh.geometry.dim
        I = ufl.Identity(d)

        Asym = 0.5 * (self.A_dir + ufl.transpose(self.A_dir))
        Ahat = unittrace_psd_from_any(Asym, d, eps=self.smooth_eps)
        Bten = self.cfg.beta_perp * I + (self.cfg.beta_par - self.cfg.beta_perp) * Ahat

        lam_eff, _ = self._get_reaction_terms()

        return (
            (self.trial / dt) * self.test * self.dx
            + ufl.inner(Bten * ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + lam_eff * self.trial * self.test * self.dx
        )

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self):
        with self.b.localForm() as b_local:
            b_local.set(0.0)

        dt = self.cfg.dt
        lam_eff, rho_eq = self._get_reaction_terms()

        rhs_expr = (self.rho_old / dt) + lam_eff * rho_eq
        self.L_form_template = fem.form(rhs_expr * self.test * self.dx)

        assemble_vector(self.b, self.L_form_template)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Density")
        return its, reason

    def mass_balance_residual(self):
        """Compute mass balance residual."""
        dt = self.cfg.dt
        rho, rho_old = self.rho, self.rho_old
        lam_eff, rho_eq = self._get_reaction_terms()
        
        lhs = (rho - rho_old) / dt
        rhs = lam_eff * (rho_eq - rho)
        
        res_local = fem.assemble_scalar(fem.form((lhs - rhs) * self.dx))
        res_abs = self.comm.allreduce(res_local, op=MPI.SUM)
        
        rhs_mag_local = fem.assemble_scalar(fem.form(abs(rhs) * self.dx))
        rhs_mag = self.comm.allreduce(rhs_mag_local, op=MPI.SUM)
        
        return res_abs, abs(res_abs) / max(rhs_mag, 1e-300)


class DirectionSolver(_BaseLinearSolver):
    """Fabric tensor A: reaction-diffusion relaxing to strain-aligned target."""

    def __init__(
        self,
        A_dir: fem.Function,
        A_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, A_dir, [], [])
        self.A_dir = self.state
        self.A_old = A_old
        self._rhs_form = None

    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        tauA = self.cfg.tauA
        if tauA <= 0:
            raise ValueError(f"tauA must be positive, got {tauA}")
        ell2 = self.cfg.ell ** 2
        
        return (
            (self.cfg.cA / dt + self.cfg.cA / tauA) * ufl.inner(self.trial, self.test) * self.dx
            + self.cfg.cA * (ell2 / tauA) * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="direction", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self, B_sum_expr):
        B_hat = unittrace_psd(B_sum_expr, self.gdim, eps=self.smooth_eps)
        dt = self.cfg.dt
        tauA = self.cfg.tauA
        
        rhs_ten = (self.cfg.cA / dt) * self.A_old + (self.cfg.cA / tauA) * B_hat
        self._rhs_form = fem.form(ufl.inner(rhs_ten, self.test) * self.dx)

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Direction")
        return its, reason

    def trace_balance_residual(self, Mhat_expr):
        """Compute domain-averaged trace of A and Mhat, and L2 residual of trace(A)-1."""
        trA = ufl.tr(self.A_dir)
        trM = ufl.tr(Mhat_expr)
        
        vol = self.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * self.dx)), op=MPI.SUM)
        trA_avg = self.comm.allreduce(fem.assemble_scalar(fem.form(trA * self.dx)), op=MPI.SUM) / vol
        trM_avg = self.comm.allreduce(fem.assemble_scalar(fem.form(trM * self.dx)), op=MPI.SUM) / vol
        
        res_sq = self.comm.allreduce(fem.assemble_scalar(fem.form((trA - 1.0)**2 * self.dx)), op=MPI.SUM)
        import numpy as np
        return trA_avg, trM_avg, np.sqrt(res_sq)
