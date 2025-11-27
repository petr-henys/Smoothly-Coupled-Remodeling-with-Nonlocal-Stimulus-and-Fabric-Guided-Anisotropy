"""Linear subsolvers: mechanics (elastic equilibrium) and density (reaction-diffusion)."""

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
    """Base class: assembly, KSP solve, iteration tracking."""
    # ... (No changes to _BaseLinearSolver logic, keep as is) ...
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
        self.logger = get_logger(self.comm, name=self.__class__.__name__, log_file=self.cfg.log_file)
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

    def setup(self):
        self._compile_forms()
        self.A = create_matrix(self.a_form)
        self.b = create_vector(self.function_space)
        self.assemble_lhs()
        self._setup_ksp()

    def assemble_lhs(self):
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
    """Isotropic elasticity with density-dependent stiffness E(rho)."""

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
        self.L_ufl = None 

    def _compile_forms(self):
        # LHS: a(u, v) = inner(sigma(u), eps(v))
        a_ufl = ufl.inner(self.sigma(self.trial, self.rho), self.eps(self.test)) * self.dx
        self.a_form = fem.form(a_ufl)

        # RHS
        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_ufl = ufl.inner(zero_vec, self.test) * self.ds
        n = ufl.FacetNormal(self.mesh)
        
        for t, tag in self.neumann_bcs:
            if len(t.ufl_shape) == 0:
                L_ufl = L_ufl + ufl.inner(-t * n, self.test) * self.ds(tag)
            else:
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
        
        # Smoothstep interpolation for material regimes
        def smoothstep(x, edge0, edge1):
            t = ufl.max_value(0.0, ufl.min_value(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        w = smoothstep(rho_eff, self.cfg.rho_trab_max, self.cfg.rho_cort_min)
        k_var = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w
        
        # Stiffness E = E0 * (rho / rho_ref)^k
        rho_rel = rho_eff / self.cfg.rho_ref
        E = self.cfg.E0 * (rho_rel**k_var)
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
        its, reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return its, reason


class DensitySolver(_BaseLinearSolver):
    """Density evolution: diffusion + stimulus-driven relaxation to bounds."""

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        psi_field: fem.Function,  # NEW: Accept the stimulus function directly
        config: Config,
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.psi_field = psi_field # Store reference to stimulus function
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

    # REMOVED: update_driving_force (redundant interpolation)

    def _compile_forms(self):
        """
        Compile Reaction-Diffusion forms.
        Based on Article Eq. 9: rho_dot = c * (Psi - Psi_ref)[cite: 153].
        Rearranged for Implicit Euler:
        (rho - rho_old)/dt = Diffusion + k_rho * S_driving
        """
        dt = self.dt_c
        
        # Calculate driving force directly from the passed-in Psi function
        # S_driving = Psi - Psi_ref (dimensional difference)
        # Positive if Psi > Psi_ref (Bone formation), Negative if Psi < Psi_ref (Resorption)
        S_driving = self.psi_field - self.cfg.psi_ref
        
        # Use smooth_plus for a "lazy zone" or pure one-sided reaction if desired.
        # Here we follow the provided logic of splitting into formation (+) and resorption (-)
        S_plus = smooth_plus(S_driving, self.smooth_eps)
        S_minus = smooth_plus(-S_driving, self.smooth_eps)
        
        # LHS: (rho/dt)*v + D_rho*grad(rho)*grad(v) + k_rho*(S_plus + S_minus)*rho*v
        # Note: The reaction term *rho*v implies a proportional rate. 
        # If strict Eq (9) is needed (constant rate independent of rho), remove rho from reaction term.
        # However, usually remodeling is density dependent. We keep the semi-implicit structure.
        reaction_coeff = self.cfg.k_rho * (S_plus + S_minus)
        
        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction_coeff * self.trial * self.test * self.dx
        )
        self.a_form = fem.form(a_ufl)

        # RHS: (rho_old/dt)*v + k_rho*(S_plus*rho_max + S_minus*rho_min)*v
        # This drives rho towards rho_max if S_plus > 0, and rho_min if S_minus > 0
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