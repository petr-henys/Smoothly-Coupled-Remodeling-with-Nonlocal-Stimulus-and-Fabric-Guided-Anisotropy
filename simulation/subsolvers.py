"""PDE subsolvers: mechanics (u) and density (ρ)."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Optional

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
import ufl

from simulation.utils import build_nullspace, smooth_plus, smooth_max
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config

class _BaseLinearSolver:
    """Base linear solver with assembly, KSP solve, and iteration tracking."""

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
        self.logger = get_logger(self.comm, name=self.__class__.__name__)
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
        self.b: Optional[PETSc.Vec] = None
        
        self.a_form: Optional[fem.Form] = None
        self.L_form: Optional[fem.Form] = None

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

    def _compile_forms(self):
        """Compile UFL forms. Must be implemented by subclasses."""
        raise NotImplementedError

    def setup(self):
        """Initialize forms, matrices, vectors, and KSP."""
        self._compile_forms()
        self.A = create_matrix(self.a_form)
        self.b = create_vector(self.function_space)
        
        # Assemble matrix so it has values for KSP setup (needed for GAMG etc)
        self.assemble_lhs()
        
        # Subclass specific KSP setup
        self._setup_ksp()

    def _setup_ksp(self):
        """Configure KSP solver."""
        raise NotImplementedError

    def assemble_lhs(self):
        """Assemble LHS matrix."""
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()
        if self.ksp is not None:
            self.ksp.setOperators(self.A)

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
    """Isotropic elastic equilibrium solver."""

    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        config: Config,
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho
        self.L_ufl = None # Store UFL for diagnostics

    def _compile_forms(self):
        # LHS: a(u, v) = inner(sigma(u), eps(v))
        a_ufl = ufl.inner(self.sigma(self.trial, self.rho), self.eps(self.test)) * self.dx
        self.a_form = fem.form(a_ufl)

        # RHS: L(v) = inner(t, v) * ds
        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_ufl = ufl.inner(zero_vec, self.test) * self.ds
        for t, tag in self.neumann_bcs:
            L_ufl = L_ufl + ufl.inner(t, self.test) * self.ds(tag)
        
        self.L_ufl = L_ufl
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        ns = build_nullspace(self.function_space)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(ns)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)

    def eps(self, u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho):
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)
        
        def smoothstep(x, edge0, edge1):
            t = ufl.max_value(0.0, ufl.min_value(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        w = smoothstep(rho_eff, self.cfg.rho_trab_max, self.cfg.rho_cort_min)
        k_var = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w
        
        # Normalize density by rho_max for stiffness calculation
        rho_rel = rho_eff / self.cfg.rho_max
        E = self.cfg.E0 * (rho_rel**k_var)
        nu = self.cfg.nu0
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
        return 2.0 * mu * self.eps(u) + lmbda * ufl.tr(self.eps(u)) * ufl.Identity(self.gdim)

    def get_strain_tensor(self, u=None):
        return self.eps(self.u if u is None else u)

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
        a_uu_local = fem.assemble_scalar(fem.form(ufl.inner(self.sigma(self.u, self.rho), self.eps(self.u)) * self.dx))
        W_int = self.comm.allreduce(a_uu_local, op=MPI.SUM)
        
        L_u_form = fem.form(ufl.replace(self.L_ufl, {self.test: self.u}))
        l_u_local = fem.assemble_scalar(L_u_form)
        W_ext = self.comm.allreduce(l_u_local, op=MPI.SUM)
        
        denom = max(abs(W_int), abs(W_ext), 1e-300)
        rel_error = abs(W_int - W_ext) / denom
        return W_int, W_ext, rel_error

    def average_strain_energy(self):
        psi = 0.5 * ufl.inner(self.sigma(self.u, self.rho), self.eps(self.u))
        E_local = fem.assemble_scalar(fem.form(psi * self.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        
        E_total = self.comm.allreduce(E_local, op=MPI.SUM)
        vol_total = self.comm.allreduce(vol_local, op=MPI.SUM)
        
        return E_total / max(vol_total, 1e-300)


class DensitySolver(_BaseLinearSolver):
    """Density evolution with isotropic diffusion and local remodeling."""

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.z_min = None
        self.S_driving = fem.Function(self.function_space, name="S_driving")
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

    
    def update_driving_force(self, psi_expr):
        driving_force_expr = (psi_expr / self.cfg.psi_ref) - 1.0
        
        expr = fem.Expression(driving_force_expr, self.function_space.element.interpolation_points)
        self.S_driving.interpolate(expr)
        self.S_driving.x.scatter_forward()

    def _compile_forms(self):
        dt = self.dt_c
        S_driving = self.S_driving
        S_plus = smooth_plus(S_driving, self.smooth_eps)
        S_minus = smooth_plus(-S_driving, self.smooth_eps)
        
        # LHS: (rho/dt)*v + D_rho*grad(rho)*grad(v) + k_rho*(S_plus + S_minus)*rho*v
        reaction_coeff = self.cfg.k_rho * (S_plus + S_minus)
        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction_coeff * self.trial * self.test * self.dx
        )
        self.a_form = fem.form(a_ufl)

        # RHS: (rho_old/dt)*v + k_rho*(S_plus*rho_max + S_minus*rho_min)*v
        source_term = self.cfg.k_rho * (S_plus * self.cfg.rho_max + S_minus * self.cfg.rho_min)
        L_ufl = ((self.rho_old / dt) + source_term) * self.test * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)

    def assemble_lhs(self):
        self.dt_c.value = float(self.cfg.dt)
        super().assemble_lhs()

    def assemble_rhs(self):
        self.dt_c.value = float(self.cfg.dt)
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Density")
        return its, reason

    def mass_balance_residual(self):
        dt = self.dt_c
        rho, rho_old = self.rho, self.rho_old
        S_driving = self.S_driving

        S_plus = smooth_plus(S_driving, self.smooth_eps)
        S_minus = smooth_plus(-S_driving, self.smooth_eps)
        
        rate = self.cfg.k_rho * (S_plus * (self.cfg.rho_max - rho) + S_minus * (self.cfg.rho_min - rho))
        
        lhs = (rho - rho_old) / dt
        rhs = rate
        
        res_local = fem.assemble_scalar(fem.form((lhs - rhs) * self.dx))
        res_abs = self.comm.allreduce(res_local, op=MPI.SUM)
        
        rhs_mag_local = fem.assemble_scalar(fem.form(abs(rhs) * self.dx))
        rhs_mag = self.comm.allreduce(rhs_mag_local, op=MPI.SUM)
        
        return res_abs, abs(res_abs) / max(rhs_mag, 1e-300)
