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


# --------------------------- Smooth helpers (C∞-ish) -------------------------
def smooth_abs(x, eps: float):
    return ufl.sqrt(x * x + eps * eps)


def smooth_plus(x, eps: float):
    sabs = smooth_abs(x, eps)
    return 0.5 * (x + sabs)


def smooth_minus(x, eps: float):
    sabs = smooth_abs(x, eps)
    return 0.5 * (sabs - x)


def smooth_max(x, xmin, eps: float):
    dx = x - xmin
    return xmin + 0.5 * (dx + ufl.sqrt(dx * dx + eps * eps))


def smooth_heaviside(x, eps: float):
    return 0.5 * (1.0 + x / ufl.sqrt(x * x + eps * eps))


# --------------------------- PSD + unit-trace helpers ------------------------
def unittrace_psd_from_any(T, dim: int, eps: float):
    I = ufl.Identity(dim)
    M = ufl.dot(ufl.transpose(T), T) + eps * I
    return M / ufl.tr(M)


def unittrace_psd(B, dim: int, eps: float):
    I = ufl.Identity(dim)
    M = B + eps * I
    return M / ufl.tr(M)


# --------------------------- Helpers: robust value extraction -----------------
def _val(x):
    try:
        return float(x.value)
    except AttributeError:
        return float(x)


# --------------------------- Base KSP wrapper --------------------------------
class _BaseLinearSolver:
    def __init__(
        self,
        cfg: Config,
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
        self.logger = get_logger(self.comm, verbose=self.cfg.verbose, name=self.__class__.__name__)
        self.smooth_eps: float = self.cfg.smooth_eps

        self.trial = ufl.TrialFunction(self.function_space)
        self.test = ufl.TestFunction(self.function_space)

        self.total_iters = 0
        self.ksp_steps = 0
        self.last_reason: Optional[int] = None
        self.last_iters: Optional[int] = None

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: Optional[PETSc.KSP] = None
        self.A: Optional[PETSc.Mat] = None
        self.b: Optional[PETSc.Vec] = create_vector(self.function_space)
        self.a_form: Optional[ufl.Form] = None

    def destroy(self):
        if self.ksp is not None:
            self.ksp.destroy(); self.ksp = None
        if self.A is not None:
            self.A.destroy(); self.A = None
        if self.b is not None:
            self.b.destroy(); self.b = None

    def _reset_stats(self):
        self.total_iters = 0
        self.ksp_steps = 0
        self.last_reason = None
        self.last_iters = None

    def assemble_lhs(self):
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()
        if self.ksp is not None:
            self.ksp.setOperators(self.A)
            self.ksp.setUp()
        self._reset_stats()

    def _solve(self) -> Tuple[int, int]:
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

    def create_ksp(self, prefix: str, ksp_options: Dict[str, object]) -> PETSc.KSP:
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


# --------------------------- Mechanics ---------------------------------------
class MechanicsSolver(_BaseLinearSolver):
    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        A_dir: fem.Function,
        config: Config,
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho
        self.A_dir = A_dir
        
        vol_local = fem.assemble_scalar(fem.form(1 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        self.a_form = fem.form(ufl.inner(self.sigma(self.trial, self.rho, self.A_dir), self.eps(self.test)) * self.dx)

        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_form = ufl.inner(zero_vec, self.test) * self.ds
        for t, tag in self.neumann_bcs:
            L_form = L_form + ufl.inner(t, self.test) * self.ds(tag)
        self.L_form = fem.form(L_form)

    def eps(self, u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho, A_dir):
        rho_eff = smooth_max(rho, self.cfg.rho_min_nd, self.smooth_eps)
        E_nd = self.cfg.E0_nd * (rho_eff ** self.cfg.n_power_c)

        eps_ten = self.eps(u)
        I = ufl.Identity(self.gdim)
        nu = self.cfg.nu_c
        lmbda = E_nd * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E_nd / (2 * (1 + nu))

        Asym = 0.5 * (A_dir + ufl.transpose(A_dir))
        Ahat = unittrace_psd_from_any(Asym, self.gdim, self.smooth_eps)

        sigma_aniso = (self.cfg.xi_aniso_c * E_nd) * ufl.inner(Ahat, eps_ten) * Ahat
        return 2 * mu * eps_ten + lmbda * ufl.tr(eps_ten) * I + sigma_aniso

    def setup(self):
        self.rho.x.scatter_forward()
        self.A_dir.x.scatter_forward()

        self.A = create_matrix(self.a_form)
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

    def average_strain_energy(self) -> float:
        self.u.x.scatter_forward()
        self.rho.x.scatter_forward()
        self.A_dir.x.scatter_forward()

        strain_energy_nd = 0.5 * ufl.inner(self.sigma(self.u, self.rho, self.A_dir), self.eps(self.u))
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        vol = self.comm.allreduce(vol_local, op=MPI.SUM)
        psi_local_nd = fem.assemble_scalar(fem.form(strain_energy_nd * self.dx))
        psi_nd = self.comm.allreduce(psi_local_nd, op=MPI.SUM)
        return float((psi_nd / max(vol, 1e-300)) * self.cfg.psi_c)
    
    def energy_balance_nd(self) -> tuple[float, float, float]:
        self.u.x.scatter_forward()
        self.rho.x.scatter_forward()

        sigma_u = self.sigma(self.u, self.rho, self.A_dir)
        eps_u = self.eps(self.u)
        Wint_local = fem.assemble_scalar(fem.form(ufl.inner(sigma_u, eps_u) * self.dx))
        W_int_nd = float(self.comm.allreduce(Wint_local, op=MPI.SUM))

        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        Wext_form = ufl.inner(zero_vec, self.u) * self.ds
        if self.neumann_bcs:
            for t, tag in self.neumann_bcs:
                Wext_form = Wext_form + ufl.inner(t, self.u) * self.ds(tag)
        Wext_local = fem.assemble_scalar(fem.form(Wext_form))
        W_ext_nd = float(self.comm.allreduce(Wext_local, op=MPI.SUM))

        rel_err = abs(W_ext_nd - W_int_nd) / max(W_ext_nd, W_int_nd, 1e-30)
        return W_int_nd, W_ext_nd, rel_err
    
    def accumulated_strain_energy_gait(self, gait_loader) -> float:
        """Compute daily accumulated strain energy over gait cycle.
        
        Uses bone adaptation convention: accumulate strain energy density
        over gait cycle using trapezoid quadrature. This represents the
        daily mechanical stimulus that drives remodeling.
        
        Parameters
        ----------
        gait_loader : FemurGaitLoader
            Gait loading object with get_quadrature() and update_loads()
        
        Returns
        -------
        float
            Accumulated average strain energy density [Pa] over one day
        """
        quadrature = gait_loader.get_quadrature()
        accumulated_psi = 0.0
        
        for phase, weight in quadrature:
            # Update loads for this gait phase
            gait_loader.update_loads(phase)
            
            # Reassemble RHS with updated loads
            self.assemble_rhs()
            
            # Solve mechanics
            self.solve()
            
            # Compute average strain energy density for this phase
            psi_phase = self.average_strain_energy()
            
            # Accumulate weighted contribution
            accumulated_psi += weight * psi_phase
        
        return accumulated_psi
    

# --------------------------- Stimulus S --------------------------------------
class StimulusSolver(_BaseLinearSolver):
    def __init__(
        self,
        S: fem.Function,
        S_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, S, [], [])
        self.S = self.state
        self.S_old = S_old

        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        a = (
            (self.cfg.cS_c / self.cfg.dt_nd + self.cfg.tauS_c) * self.trial * self.test * self.dx
            + self.cfg.kappaS_c * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )
        self.a_form = fem.form(a)
        self._rhs_form = None

    def setup(self):
        self.A = create_matrix(self.a_form)
        self.assemble_lhs()

        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self, psi_expr):
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        self.S_old.x.scatter_forward()
        rhs = (self.cfg.cS_c / self.cfg.dt_nd) * self.S_old + self.cfg.rS_gain_c * (
            psi_expr - self.cfg.psi_ref_nd
        )
        self._rhs_form = fem.form(rhs * self.test * self.dx)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        self.S.x.scatter_forward()
        its, reason = self._solve()
        self._maybe_warn(reason, "Stimulus")
        return its, reason

    def power_balance_residual(self, psi_expr) -> tuple[float, float]:
        self.S.x.scatter_forward()
        self.S_old.x.scatter_forward()
        one = fem.Constant(self.mesh, default_scalar_type(1.0))
        dt_val = _val(self.cfg.dt_nd)
        storage_loc = fem.assemble_scalar(fem.form((self.cfg.cS_c / dt_val) * (self.S - self.S_old) * one * self.dx))
        storage = float(self.comm.allreduce(storage_loc, op=MPI.SUM))
        decay_loc = fem.assemble_scalar(fem.form(self.cfg.tauS_c * self.S * one * self.dx))
        decay = float(self.comm.allreduce(decay_loc, op=MPI.SUM))
        source_loc = fem.assemble_scalar(fem.form(self.cfg.rS_gain_c * (psi_expr - self.cfg.psi_ref_nd) * one * self.dx))
        source = float(self.comm.allreduce(source_loc, op=MPI.SUM))
        n = ufl.FacetNormal(self.mesh)
        flux_loc = fem.assemble_scalar(fem.form(self.cfg.kappaS_c * ufl.dot(ufl.grad(self.S), n) * self.ds))
        flux = float(self.comm.allreduce(flux_loc, op=MPI.SUM))
        R = (storage + decay) - (source + flux)
        denom = abs(storage) + abs(decay) + abs(source) + 1e-30
        return abs(R), abs(R) / denom


# --------------------------- Density rho -------------------------------------
class DensitySolver(_BaseLinearSolver):
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

        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        d = self.mesh.geometry.dim  # fixed: self.Q not defined
        I = ufl.Identity(d)

        Asym = 0.5 * (self.A_dir + ufl.transpose(self.A_dir))
        Ahat = unittrace_psd_from_any(Asym, d, eps=self.smooth_eps)

        Bten = self.cfg.beta_perp_nd * I + (self.cfg.beta_par_nd - self.cfg.beta_perp_nd) * Ahat

        Sabs_smooth = smooth_abs(self.S, self.smooth_eps)
        Splus_smooth = 0.5 * (self.S + Sabs_smooth)
        Sminus_smooth = 0.5 * (Sabs_smooth - self.S)

        a = (
            (self.trial / self.cfg.dt_nd) * self.test * self.dx
            + ufl.inner(Bten * ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + Sabs_smooth * self.trial * self.test * self.dx
        )
        self.a_form = fem.form(a)

        rhs_expr = (
            (self.rho_old / self.cfg.dt_nd)
            + Splus_smooth * self.cfg.rho_max_nd
            + Sminus_smooth * self.cfg.rho_min_nd
        )
        self.L_form_template = fem.form(rhs_expr * self.test * self.dx)

    def setup(self):
        self.A = create_matrix(self.a_form)
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self):
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form_template)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        self.rho.x.scatter_forward()
        its, reason = self._solve()
        self._maybe_warn(reason, "Density")
        self.rho.x.scatter_forward()
        return its, reason
    
    def mass_balance_residual(self) -> tuple[float, float]:
        self.rho.x.scatter_forward()
        self.rho_old.x.scatter_forward()
        self.S.x.scatter_forward()
        self.A_dir.x.scatter_forward()
        one = fem.Constant(self.mesh, default_scalar_type(1.0))
        M_new_loc = fem.assemble_scalar(fem.form(self.rho * one * self.dx))
        M_old_loc = fem.assemble_scalar(fem.form(self.rho_old * one * self.dx))
        M_new = float(self.comm.allreduce(M_new_loc, op=MPI.SUM))
        M_old = float(self.comm.allreduce(M_old_loc, op=MPI.SUM))
        Sabs = smooth_abs(self.S, self.smooth_eps)
        Splus = 0.5 * (self.S + Sabs)
        Sminus = 0.5 * (Sabs - self.S)
        decay_loc = fem.assemble_scalar(fem.form(Sabs * self.rho * self.dx))
        src_loc = fem.assemble_scalar(fem.form((Splus * self.cfg.rho_max_nd + Sminus * self.cfg.rho_min_nd) * self.dx))
        decay = float(self.comm.allreduce(decay_loc, op=MPI.SUM))
        src = float(self.comm.allreduce(src_loc, op=MPI.SUM))
        dt_val = _val(self.cfg.dt_nd)
        R = (M_new - M_old) / max(dt_val, 1e-30) + decay - src
        denom = abs((M_new - M_old) / max(dt_val, 1e-30)) + abs(decay) + abs(src) + 1e-30
        return abs(R), abs(R) / denom


# --------------------------- Direction tensor A -------------------------------
class DirectionSolver(_BaseLinearSolver):
    def __init__(
        self,
        A_dir: fem.Function,
        A_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, A_dir, [], [])
        self.A_dir = self.state
        self.A_old = A_old

        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        self.total_vol = self.comm.allreduce(vol_local, op=MPI.SUM)

        ell2 = self.cfg.ell_c ** 2
        a = (
            (self.cfg.cA_c / self.cfg.dt_nd + self.cfg.tauA_c) * ufl.inner(self.trial, self.test) * self.dx
            + self.cfg.cA_c * ell2 * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )
        self.a_form = fem.form(a)
        self._rhs_form = None

    def setup(self):
        self.A = create_matrix(self.a_form)
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="direction", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self, B_sum_expr):
        B_hat = unittrace_psd(B_sum_expr, self.gdim, eps=self.smooth_eps)
        rhs_ten = (self.cfg.cA_c / self.cfg.dt_nd) * self.A_old + self.cfg.tauA_c * B_hat
        self._rhs_form = fem.form(ufl.inner(rhs_ten, self.test) * self.dx)

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        self.A_dir.x.scatter_forward()
        its, reason = self._solve()
        self._maybe_warn(reason, "Direction")
        return its, reason

    def trace_balance_residual(self, Mhat_expr) -> tuple[float, float, float]:
        self.A_dir.x.scatter_forward()
        one = fem.Constant(self.mesh, default_scalar_type(1.0))
        trA_loc = fem.assemble_scalar(fem.form(ufl.tr(self.A_dir) * one * self.dx))
        trA_vol = float(self.comm.allreduce(trA_loc, op=MPI.SUM)) / self.total_vol
        trMhat_loc = fem.assemble_scalar(fem.form(ufl.tr(Mhat_expr) * one * self.dx))
        trMhat_vol = float(self.comm.allreduce(trMhat_loc, op=MPI.SUM)) / self.total_vol
        R = trA_vol - trMhat_vol
        return trA_vol, trMhat_vol, abs(R)
