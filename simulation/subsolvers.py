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

    def assemble_lhs(self, *, scatter_state: bool = False):
        """Assemble LHS matrix.
        
        Args:
            scatter_state: If True, scatter state before assembly.
                          Default False - caller is responsible for ensuring
                          state is synced before calling.
        """
        if scatter_state:
            self.state.x.scatter_forward()
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
    """Density evolution: diffusion + stimulus-driven update without soft bounds."""

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        psi_field: fem.Function,
        config: Config,
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.psi_field = psi_field
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

    def _compile_forms(self):
        """
        Compile forms for density evolution *without* soft bounds.

        PDE (semi-discrete, pointwise form):

            ∂ρ/∂t - D_rho Δρ = k_rho * S,

        where
            S_raw   = (Psi - psi_ref) / psi_ref  (dimensionless stimulus),
            S_plus  = smooth_plus(S_raw, smooth_eps),
            S_minus = smooth_plus(-S_raw, smooth_eps),
            S       = S_plus - S_minus.

        ŽÁDNÉ relaxování k rho_min / rho_max v samotné PDE.
        Fyzikální meze se vynucují až po solve natvrdo clampem.
        """
        dt = self.dt_c

        # Dimensionless mechanical stimulus (hladké kolem nuly, žádná dead zone)
        S_raw = (self.psi_field - self.cfg.psi_ref) / self.cfg.psi_ref
        S_plus = smooth_plus(S_raw, self.smooth_eps)
        S_minus = smooth_plus(-S_raw, self.smooth_eps)
        S = S_plus - S_minus  # signovaný stimul ≈ S_raw

        # LHS: (rho^{n+1}/dt)*v + D_rho * grad(rho^{n+1})·grad(v)
        # → žádný reakční člen s (S_plus + S_minus), jen mass + diffusion
        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )
        self.a_form = fem.form(a_ufl)

        # RHS: (rho^n/dt)*v + k_rho * S * v
        source_term = self.cfg.k_rho * S
        L_ufl = ((self.rho_old / dt) + source_term) * self.test * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)

    def assemble_lhs(self, *, scatter_psi: bool = False):
        """Assemble LHS matrix.
        
        Args:
            scatter_psi: If True, scatter psi_field before assembly.
                         Default False - caller ensures psi is already synced.
        """
        if scatter_psi:
            self.psi_field.x.scatter_forward()
        self.dt_c.value = float(self.cfg.dt)
        super().assemble_lhs(scatter_state=False)

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

        # Hard clamp: rho ∈ [rho_min, rho_max] až po solve
        rho_min = self.cfg.rho_min
        rho_max = self.cfg.rho_max

        # Stejný styl jako u self.b.localForm() výše – pracujeme s PETSc Vec
        with self.rho.x.petsc_vec.localForm() as loc:
            local = loc.array
            local[local < rho_min] = rho_min
            local[local > rho_max] = rho_max

        # Update ghost dofů po úpravě lokálních hodnot
        self.rho.x.scatter_forward()

        return its, reason
