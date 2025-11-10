from __future__ import annotations

from typing import Tuple, List, Optional, Dict

from mpi4py import MPI
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
from dolfinx import default_scalar_type
import ufl

from simulation.utils import build_nullspace
from simulation.config import Config
from simulation.logger import get_logger

Scalar = PETSc.ScalarType


# --------------------------- Smooth helpers (C∞-ish) -------------------------
def smooth_abs(x, eps: float):
    """C∞ approximation of |x| with smoothing width ~ eps."""
    return ufl.sqrt(x * x + eps * eps)


def smooth_plus(x, eps: float):
    """Smooth approximation of max(x, 0)."""
    sabs = smooth_abs(x, eps)
    return 0.5 * (x + sabs)


def smooth_minus(x, eps: float):
    """Smooth approximation of max(-x, 0)."""
    sabs = smooth_abs(x, eps)
    return 0.5 * (sabs - x)


def smooth_max(x, xmin, eps: float):
    """Smooth clamp: approx max(x, xmin)."""
    dx = x - xmin
    return xmin + 0.5 * (dx + ufl.sqrt(dx * dx + eps * eps))


def smooth_heaviside(x, eps: float):
    """Smooth Heaviside H(x) ~ 0.5 * (1 + x/sqrt(x^2 + eps^2)).
    Useful for robust volume-fraction style QoIs."""
    return 0.5 * (1.0 + x / ufl.sqrt(x * x + eps * eps))


# --------------------------- PSD + unit-trace helpers ------------------------
def unittrace_psd_from_any(T, dim: int, eps: float):
    """
    Smooth PSD enforcement + unit-trace normalisation for arbitrary T.
    Returns: Ahat = (T^T T + eps I) / tr(T^T T + eps I)
    Properties: symmetric, SPD (eps>0), unit trace, C∞ in T.
    """
    I = ufl.Identity(dim)
    M = ufl.dot(ufl.transpose(T), T) + eps * I   # Gram + shift: SPD
    return M / ufl.tr(M)                         # unit trace


def unittrace_psd(B, dim: int, eps: float):
    """
    Unit-trace normalisation for a PSD input B (e.g. B = E^T E).
    Returns: Bhat = (B + eps I) / tr(B + eps I)
    Also SPD and C∞ in B (for eps>0).
    """
    I = ufl.Identity(dim)
    M = B + eps * I
    return M / ufl.tr(M)


# --------------------------- Base KSP wrapper --------------------------------
class _BaseLinearSolver:
    """Common PETSc KSP wrapper + stats for repeatedly-solved linear systems."""

    def __init__(self, comm: MPI.Intracomm, cfg: Config):
        self.comm = comm
        self.A: Optional[PETSc.Mat] = None
        self.b: Optional[PETSc.Vec] = None
        self.ksp: Optional[PETSc.KSP] = None
        self.cfg = cfg
        # Logger obeys cfg.verbose for INFO/DEBUG; warnings/errors always shown
        self.logger = get_logger(self.comm, verbose=bool(getattr(cfg, "verbose", True)), name=self.__class__.__name__)

        # Unified smoothness parameters for the whole model
        self.smooth_eps: float = self.cfg.smooth_eps

        self.total_iters = 0
        self.ksp_steps = 0
        self.precond_updates = 0
        self.last_reason: Optional[int] = None
        self.last_iters: Optional[int] = None

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
        self.total_iters = 0
        self.ksp_steps = 0
        self.precond_updates = 0
        self.last_reason = None
        self.last_iters = None

    def _solve(self, x: fem.Function) -> Tuple[int, int]:
        self.ksp.solve(self.b, x.x.petsc_vec)
        x.x.scatter_forward()
        its = self.ksp.getIterationNumber()
        reason = self.ksp.getConvergedReason()
        self.total_iters += its
        self.ksp_steps += 1
        self.last_iters = its
        self.last_reason = reason
        return its, reason

    # Convenience properties for external reporting (tests, summaries)
    @property
    def ksp_its(self) -> int:
        """Total KSP iterations accumulated across solves."""
        return int(self.total_iters)

    def _maybe_warn(self, reason: int, label: str):
        if reason < 0:
            self.logger.warning(f"{label} solver failed to converge (reason: {reason})")

    def create_ksp(self, prefix: str, ksp_options: Dict[str, object]) -> PETSc.KSP:
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOptionsPrefix(prefix + "_")

        opts = PETSc.Options()
        for k, v in ksp_options.items():
            opts[f"{prefix}_{k}"] = v
        opts[f"{prefix}_ksp_rtol"] = self.cfg.ksp_rtol
        opts[f"{prefix}_ksp_atol"] = self.cfg.ksp_atol
        opts[f"{prefix}_ksp_max_it"] = self.cfg.ksp_max_it

        self.ksp.setInitialGuessNonzero(True)
        self.ksp.setOperators(self.A)

        self.ksp.setFromOptions()
        self.ksp.setUp()
        return self.ksp


# --------------------------- Mechanics ---------------------------------------
class MechanicsSolver(_BaseLinearSolver):
    def __init__(
        self,
        V: fem.FunctionSpace,
        rho: fem.Function,
        A_dir: fem.Function,
        dirichlets: List[fem.DirichletBC],
        neumanns: List[Tuple[fem.Function, int]],
        config: Config,
    ):
        super().__init__(V.mesh.comm, config)
        self.V = V
        self.mesh = V.mesh
        self.rank = self.comm.rank
        self.gdim = self.mesh.geometry.dim
        self.dx = self.cfg.dx
        self.ds = self.cfg.ds

        self.rho = rho
        self.A_dir = A_dir
        self.bcs = dirichlets
        self.neumanns = neumanns

        vol_local = fem.assemble_scalar(fem.form(1 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)

        self.a = ufl.inner(self.sigma(self.u, self.rho), self.eps(self.v)) * self.dx
        self.a_form = fem.form(self.a)

        L_neu = ufl.inner(fem.Constant(self.mesh, (0., 0., 0.)), self.v) * self.ds
        for t, tag in self.neumanns:
            L_neu += ufl.inner(t, self.v) * self.ds(tag)
        self.L = L_neu    
        self.L_form = fem.form(self.L)

    def eps(self, u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho):
        # Smooth clamp to avoid non-differentiability and log(0)
        rho_eff = smooth_max(rho, self.cfg.rho_min_nd, self.smooth_eps)

        # Elastic modulus as a power-law of density (avoids ln/exp pathologies)
        E_nd = self.cfg.E0_nd * (rho_eff ** self.cfg.n_power_c)

        eps_ten = self.eps(u)
        I = ufl.Identity(self.gdim)
        nu = self.cfg.nu_c
        lmbda = E_nd * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E_nd / (2 * (1 + nu))

        # Smooth, normalized direction tensor (unit trace), PSD by construction
        Asym = 0.5 * (self.A_dir + ufl.transpose(self.A_dir))
        Ahat = unittrace_psd_from_any(Asym, self.gdim, self.smooth_eps)

        sigma_aniso = (self.cfg.xi_aniso_c * E_nd) * ufl.inner(Ahat, eps_ten) * Ahat
        return 2 * mu * eps_ten + lmbda * ufl.tr(eps_ten) * I + sigma_aniso

    def solver_setup(self):
        self.rho.x.scatter_forward()
        self.A_dir.x.scatter_forward()

        self.A = create_matrix(self.a_form)
        assemble_matrix(self.A, self.a_form, bcs=self.bcs)
        self.A.assemble()


        self.b = assemble_vector(self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        # Validate Dirichlet boundary conditions
        # In some contexts (e.g., matrix-only tests), it's valid to proceed
        # without Dirichlet DOFs. Emit a warning instead of raising.
        from simulation.utils import collect_dirichlet_dofs
        owned_fix = collect_dirichlet_dofs(self.bcs, self.V.dofmap.index_map.size_local)
        n_fix_local = int(owned_fix.size)
        n_fix = self.comm.allreduce(n_fix_local, op=MPI.SUM)
        if n_fix == 0:
            if self.comm.rank == 0:
                self.logger.warning(
                    "No mechanics Dirichlet DOFs found (facet tag 'fixed' missing/empty). Proceeding with free system."
                )

        ns = build_nullspace(self.V)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(ns)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)
        self._reset_stats()

    def update_stiffness(self):
        self.rho.x.scatter_forward()
        self.A_dir.x.scatter_forward()
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.bcs)
        self.A.assemble()
        if self.ksp is not None:
            self.ksp.setOperators(self.A)

    def update_rhs(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self, u_func: fem.Function):
        its, reason = self._solve(u_func)
        self._maybe_warn(reason, "Mechanics")
        return its, reason

    def average_strain_energy(self, u_func: fem.Function) -> float:
        """Average strain-energy density in physical units."""
        u_func.x.scatter_forward()
        self.rho.x.scatter_forward()
        self.A_dir.x.scatter_forward()

        strain_energy_nd = 0.5 * ufl.inner(self.sigma(u_func, self.rho), self.eps(u_func))
        dx = self.dx
        vol_local = fem.assemble_scalar(fem.form(1.0 * dx))
        vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        psi_local_nd = fem.assemble_scalar(fem.form(strain_energy_nd * dx))
        psi_nd = self.comm.allreduce(psi_local_nd, op=MPI.SUM)
        # Scale to physical units [J/m^3] once
        return float((psi_nd / max(vol, 1e-300)) * self.cfg.psi_c)
    
    def energy_balance_nd(self, u_func: fem.Function) -> tuple[float, float, float]:
        u_func.x.scatter_forward()
        self.rho.x.scatter_forward()

        sigma_u = self.sigma(u_func, self.rho)
        eps_u = self.eps(u_func)

        Wint_local = fem.assemble_scalar(fem.form(ufl.inner(sigma_u, eps_u) * self.dx))
        W_int_nd = float(self.comm.allreduce(Wint_local, op=MPI.SUM))

        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        Wext_form = ufl.inner(zero_vec, u_func) * self.ds
        if hasattr(self, "neumanns") and self.neumanns:
            for t, tag in self.neumanns:
                Wext_form = Wext_form + ufl.inner(t, u_func) * self.ds(tag)
        Wext_local = fem.assemble_scalar(fem.form(Wext_form))
        W_ext_nd = float(self.comm.allreduce(Wext_local, op=MPI.SUM))

        rel_err = abs(W_ext_nd - W_int_nd) / max(W_ext_nd, W_int_nd, 1e-30)
        return W_int_nd, W_ext_nd, rel_err



# --------------------------- Stimulus S --------------------------------------
class StimulusSolver(_BaseLinearSolver):
    def __init__(self, Q: fem.FunctionSpace, S_old: fem.Function, config: Config):
        super().__init__(Q.mesh.comm, config)
        self.mesh = Q.mesh
        self.rank = self.comm.rank
        self.Q = Q
        self.S_old = S_old
        self.dx = self.cfg.dx

        # domain volume for normalized QoIs
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        self.Str = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        a = (
            (self.cfg.cS_c / self.cfg.dt_nd + self.cfg.tauS_c) * self.Str * self.q * self.dx
            + self.cfg.kappaS_c * ufl.inner(ufl.grad(self.Str), ufl.grad(self.q)) * self.dx
        )
        self.a_form = fem.form(a)
        self._rhs_form: Optional[fem.Form] = None

    def solver_setup(self):
        self.A = create_matrix(self.a_form)
        assemble_matrix(self.A, self.a_form)
        self.A.assemble()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        self.b = create_vector(self.Q)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)
        self._reset_stats()

    def update_lhs(self):
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form)
        self.A.assemble()
        self.ksp.setOperators(self.A)
        self.ksp.setUp()
        self._reset_stats()

    def update_rhs(self, psi_expr):
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        self.S_old.x.scatter_forward()
        rhs = (self.cfg.cS_c / self.cfg.dt_nd) * self.S_old + self.cfg.rS_gain_c * (
            psi_expr - self.cfg.psi_ref_nd
        )
        self._rhs_form = fem.form(rhs * self.q * self.dx)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self, S_func: fem.Function):
        S_func.x.scatter_forward()
        its, reason = self._solve(S_func)
        self._maybe_warn(reason, "Stimulus")
        return its, reason

    def power_balance_residual(self, S_current: fem.Function, psi_expr) -> tuple[float, float]:
        """Power balance check: storage + decay ≈ source + diffusion flux."""
        S_current.x.scatter_forward()
        self.S_old.x.scatter_forward()
        one = fem.Constant(self.mesh, default_scalar_type(1.0))
        dt_val = float(self.cfg.dt_nd.value)
        # Storage term: c_S * (S - S_old)/dt
        storage_loc = fem.assemble_scalar(fem.form((self.cfg.cS_c / dt_val) * (S_current - self.S_old) * one * self.dx))
        storage = float(self.comm.allreduce(storage_loc, op=MPI.SUM))
        # Decay term: tau_S * S
        decay_loc = fem.assemble_scalar(fem.form(self.cfg.tauS_c * S_current * one * self.dx))
        decay = float(self.comm.allreduce(decay_loc, op=MPI.SUM))
        # Source term: r_S * (psi - psi_ref)
        source_loc = fem.assemble_scalar(fem.form(self.cfg.rS_gain_c * (psi_expr - self.cfg.psi_ref_nd) * one * self.dx))
        source = float(self.comm.allreduce(source_loc, op=MPI.SUM))
        # Diffusion flux through domain (should be ~0 with Neumann BCs)
        flux_loc = fem.assemble_scalar(fem.form(self.cfg.kappaS_c * ufl.inner(ufl.grad(S_current), ufl.grad(S_current)) * self.dx))
        flux = float(self.comm.allreduce(flux_loc, op=MPI.SUM))
        # Residual: (storage + decay) - (source + 0) since ∫∇·(κ∇S) = 0 with Neumann BC
        R = (storage + decay) - source
        denom = abs(storage) + abs(decay) + abs(source) + 1e-30
        return abs(R), abs(R) / denom


# --------------------------- Density rho -------------------------------------
class DensitySolver(_BaseLinearSolver):
    def __init__(
        self,
        Q: fem.FunctionSpace,
        rho_old: fem.Function,
        A_field: fem.Function,
        S_func: fem.Function,
        config: Config,
    ):
        super().__init__(Q.mesh.comm, config)
        self.mesh = Q.mesh
        self.rank = self.comm.rank
        self.Q = Q
        self.rho_old = rho_old
        self.A_field = A_field
        self.S = S_func
        self.dx = self.cfg.dx

        # domain volume for normalized QoIs
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        self.rho = ufl.TrialFunction(self.Q)
        self.q = ufl.TestFunction(self.Q)

        d = self.Q.mesh.geometry.dim
        I = ufl.Identity(d)

        # Smooth, normalized direction tensor (unit trace) for transport tensor (PSD by construction)
        Asym = 0.5 * (self.A_field + ufl.transpose(self.A_field))
        Ahat = unittrace_psd_from_any(Asym, d, eps=self.smooth_eps)

        Bten = self.cfg.beta_perp_nd * I + (self.cfg.beta_par_nd - self.cfg.beta_perp_nd) * Ahat

        # --- Smooth S+, S-, |S| (unified epsilon) ---
        S = self.S
        Sabs_smooth = smooth_abs(S, self.smooth_eps)
        Splus_smooth = 0.5 * (S + Sabs_smooth)
        Sminus_smooth = 0.5 * (Sabs_smooth - S)

        a = (
            (self.rho / self.cfg.dt_nd) * self.q * self.dx
            + ufl.inner(Bten * ufl.grad(self.rho), ufl.grad(self.q)) * self.dx
            + Sabs_smooth * self.rho * self.q * self.dx
        )
        self.a_form = fem.form(a)

        rhs_expr = (
            (self.rho_old / self.cfg.dt_nd)
            + Splus_smooth * self.cfg.rho_max_nd
            + Sminus_smooth * self.cfg.rho_min_nd
        )
        self.L_form_template = fem.form(rhs_expr * self.q * self.dx)

    def solver_setup(self):
        self.A = create_matrix(self.a_form)
        assemble_matrix(self.A, self.a_form)
        self.A.assemble()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        self.b = create_vector(self.Q)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)
        self._reset_stats()

    def update_system(self):
        self.rho_old.x.scatter_forward()
        self.A_field.x.scatter_forward()
        self.S.x.scatter_forward()

        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form)
        self.A.assemble()
        self.ksp.setOperators(self.A)
        self.ksp.setUp()

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form_template)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self, rho_func: fem.Function):
        rho_func.x.scatter_forward()
        its, reason = self._solve(rho_func)
        self._maybe_warn(reason, "Density")
        rho_func.x.scatter_forward()
        return its, reason
    
    def mass_balance_residual(self, rho_current: fem.Function) -> tuple[float, float]:
        """Mass balance check for density: returns (abs_residual, relative_error) in ND units."""
        rho_current.x.scatter_forward()
        self.rho_old.x.scatter_forward()
        self.S.x.scatter_forward()
        self.A_field.x.scatter_forward()
        one = fem.Constant(self.mesh, default_scalar_type(1.0))
        M_new_loc = fem.assemble_scalar(fem.form(rho_current * one * self.dx))
        M_old_loc = fem.assemble_scalar(fem.form(self.rho_old * one * self.dx))
        M_new = float(self.comm.allreduce(M_new_loc, op=MPI.SUM))
        M_old = float(self.comm.allreduce(M_old_loc, op=MPI.SUM))
        S = self.S
        Sabs = smooth_abs(S, self.smooth_eps)
        Splus = 0.5 * (S + Sabs)
        Sminus = 0.5 * (Sabs - S)
        decay_loc = fem.assemble_scalar(fem.form(Sabs * rho_current * self.dx))
        src_loc = fem.assemble_scalar(fem.form((Splus * self.cfg.rho_max_nd + Sminus * self.cfg.rho_min_nd) * self.dx))
        decay = float(self.comm.allreduce(decay_loc, op=MPI.SUM))
        src = float(self.comm.allreduce(src_loc, op=MPI.SUM))
        dt_val = float(self.cfg.dt_nd.value)
        R = (M_new - M_old) / max(dt_val, 1e-30) + decay - src
        denom = abs((M_new - M_old) / max(dt_val, 1e-30)) + abs(decay) + abs(src) + 1e-30
        return abs(R), abs(R) / denom



# --------------------------- Direction tensor A -------------------------------
class DirectionSolver(_BaseLinearSolver):
    def __init__(self, T: fem.FunctionSpace, A_old: fem.Function, config: Config):
        super().__init__(T.mesh.comm, config)
        self.T = T
        self.A_old = A_old
        self.dx = self.cfg.dx
        self.gdim = T.mesh.geometry.dim

        # domain volume for normalized QoIs
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        self.Atr = ufl.TrialFunction(self.T)
        self.Q = ufl.TestFunction(self.T)

        ell2 = self.cfg.ell_c ** 2
        a = (
            (self.cfg.cA_c / self.cfg.dt_nd + self.cfg.tauA_c) * ufl.inner(self.Atr, self.Q) * self.dx
            + self.cfg.cA_c * ell2 * ufl.inner(ufl.grad(self.Atr), ufl.grad(self.Q)) * self.dx
        )
        self.a_form = fem.form(a)
        self._rhs_form: Optional[fem.Form] = None

    def solver_setup(self):
        self.A = create_matrix(self.a_form)
        assemble_matrix(self.A, self.a_form)
        self.A.assemble()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        self.b = create_vector(self.T)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="direction", ksp_options=ksp_options)
        self._reset_stats()

    def update_lhs(self):
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form)
        self.A.assemble()
        self.ksp.setOperators(self.A)
        self.ksp.setUp()
        self._reset_stats()

    def update_rhs(self, mech: MechanicsSolver, u_func: fem.Function):
        u_func.x.scatter_forward()
        self.A_old.x.scatter_forward()

        eps_ten = mech.eps(u_func)
        B = ufl.dot(ufl.transpose(eps_ten), eps_ten)     # symmetric PSD
        B_hat = unittrace_psd(B, self.gdim, eps=self.smooth_eps)

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        rhs_ten = (self.cfg.cA_c / self.cfg.dt_nd) * self.A_old + self.cfg.tauA_c * B_hat
        self._rhs_form = fem.form(ufl.inner(rhs_ten, self.Q) * self.dx)

        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self, A_func: fem.Function):
        A_func.x.scatter_forward()
        its, reason = self._solve(A_func)
        self._maybe_warn(reason, "Direction")
        return its, reason

    def trace_balance_residual(self, A_current: fem.Function, Mhat_expr) -> tuple[float, float, float]:
        """Trace conservation check: tr(A) should relax to tr(M̂)=1."""
        A_current.x.scatter_forward()
        mesh = A_current.function_space.mesh
        one = fem.Constant(mesh, default_scalar_type(1.0))
        # Volume-averaged trace of A
        trA_loc = fem.assemble_scalar(fem.form(ufl.tr(A_current) * one * self.dx))
        trA_vol = float(self.comm.allreduce(trA_loc, op=MPI.SUM)) / self.total_vol
        # Volume-averaged trace of M̂ (should be 1 by construction)
        trMhat_loc = fem.assemble_scalar(fem.form(ufl.tr(Mhat_expr) * one * self.dx))
        trMhat_vol = float(self.comm.allreduce(trMhat_loc, op=MPI.SUM)) / self.total_vol
        # Residual: how far tr(A) deviates from target tr(M̂)
        R = trA_vol - trMhat_vol
        return trA_vol, trMhat_vol, abs(R)
    
    def update_rhs_from_Bexpr(self, B_sum_expr):
        B_hat = unittrace_psd(B_sum_expr, self.gdim, eps=self.smooth_eps)
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        rhs_ten = (self.cfg.cA_c / self.cfg.dt_nd) * self.A_old + self.cfg.tauA_c * B_hat
        self._rhs_form = fem.form(ufl.inner(rhs_ten, self.Q) * self.dx)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
