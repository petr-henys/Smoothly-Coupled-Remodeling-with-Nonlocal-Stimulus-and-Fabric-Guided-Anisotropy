"""Linear subsolvers for mechanics and density evolution with Helmholtz filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Optional

import numpy as np
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

from simulation.utils import build_nullspace, smooth_max
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config


class _BaseLinearSolver:
    """Base class for PETSc KSP linear solvers."""

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
        self.last_iters: int = 0
        self.last_reason: Optional[int] = None

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: Optional[PETSc.KSP] = None
        self.A: Optional[PETSc.Mat] = None
        self.b: Optional[PETSc.Vec] = None

        self.a_form: Optional[fem.Form] = None
        self.L_form: Optional[fem.Form] = None

    # --------------------------- lifecycle ---------------------------------

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
    """Elasticity solver with density-dependent stiffness E = E0*(ρ/ρ_ref)^n."""

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

    def _setup_ksp(self):
        self._nullspace = build_nullspace(self.function_space)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(self._nullspace)
        self.A.setNullSpace(self._nullspace)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": "cg", "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)

    @staticmethod
    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho):
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)
        rho_rel = rho_eff / self.cfg.rho_ref
        E = self.cfg.E0 * (rho_rel ** self.cfg.n)
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
        if self._nullspace is not None:
            self._nullspace.remove(self.b)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return its, reason


class DensitySolver(_BaseLinearSolver):
    """Density evolution: dρ/dt = D∇²ρ + k_rho*S with optional Helmholtz filtering."""

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

        # Helmholtz filter length from config (physical parameter, ~0.3mm for bone)
        self.helmholtz_L = self.cfg.helmholtz_L
        self._use_helmholtz = self.helmholtz_L > 0.0
        
        if self.rank == 0 and self._use_helmholtz:
            self.logger.info(f"Helmholtz filter: L={self.helmholtz_L:.4f} mm")

        # Helmholtz filter resources (lazy init)
        self._helm_tr: Optional[ufl.TrialFunction] = None
        self._helm_ts: Optional[ufl.TestFunction] = None
        self._a_helm_form: Optional[fem.Form] = None
        self._L_helm_form: Optional[fem.Form] = None
        self._A_helm: Optional[PETSc.Mat] = None
        self._b_helm: Optional[PETSc.Vec] = None
        self._ksp_helm: Optional[PETSc.KSP] = None

        if self._use_helmholtz:
            self._setup_helmholtz_filter()

    def _compile_forms(self):
        dt = self.dt_c
        rho_safe = smooth_max(self.rho, self.cfg.rho_min, self.smooth_eps)

        # Specific energy stimulus: S = (Ψ/ρ - Ψ_ref/ρ_ref) / (Ψ_ref/ρ_ref)
        S_specific = self.psi_field / rho_safe
        S_ref_specific = self.cfg.psi_ref / self.cfg.rho_ref
        S_linear = (S_specific - S_ref_specific) / S_ref_specific

        # Saturated stimulus with tanh
        saturation_limit = 1.0
        S_saturated = saturation_limit * ufl.tanh(S_linear / saturation_limit)

        # Implicit Euler: (ρ^{n+1}/dt, v) + D(∇ρ^{n+1}, ∇v) = (ρ^n/dt, v) + (k*S, v)
        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )
        self.a_form = fem.form(a_ufl)

        source_term = self.cfg.k_rho * S_saturated
        L_ufl = ((self.rho_old / dt) + source_term) * self.test * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)

    def assemble_lhs(self, *, scatter_psi: bool = False):
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

    def _setup_helmholtz_filter(self):
        """Assemble Helmholtz filter: (ρ_filt, v) + L²(∇ρ_filt, ∇v) = (ρ_raw, v)."""
        if not self._use_helmholtz:
            return

        self._helm_tr = ufl.TrialFunction(self.function_space)
        self._helm_ts = ufl.TestFunction(self.function_space)
        L2 = self.helmholtz_L ** 2

        a_h = (
            self._helm_tr * self._helm_ts * self.dx
            + L2 * ufl.inner(ufl.grad(self._helm_tr), ufl.grad(self._helm_ts)) * self.dx
        )
        self._a_helm_form = fem.form(a_h)

        L_h = self.rho * self._helm_ts * self.dx
        self._L_helm_form = fem.form(L_h)

        self._A_helm = create_matrix(self._a_helm_form)
        assemble_matrix(self._A_helm, self._a_helm_form, bcs=[])
        self._A_helm.assemble()
        self._A_helm.setOption(PETSc.Mat.Option.SPD, True)
        self._b_helm = create_vector(self.function_space)

        self._ksp_helm = PETSc.KSP().create(self.comm)
        self._ksp_helm.setOptionsPrefix("helmholtz_")
        opts = PETSc.Options()
        opts["helmholtz_ksp_type"] = "cg"
        opts["helmholtz_pc_type"] = self.cfg.pc_type
        opts["helmholtz_ksp_rtol"] = self.cfg.ksp_rtol
        opts["helmholtz_ksp_atol"] = self.cfg.ksp_atol
        opts["helmholtz_ksp_max_it"] = self.cfg.ksp_max_it
        self._ksp_helm.setOperators(self._A_helm)
        self._ksp_helm.setFromOptions()
        self._ksp_helm.setUp()

    def _apply_helmholtz_filter(self):
        """Apply Helmholtz filter to ρ in-place."""
        if not self._use_helmholtz:
            return
        with self._b_helm.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self._b_helm, self._L_helm_form)
        self._b_helm.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self._b_helm.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        self._ksp_helm.solve(self._b_helm, self.rho.x.petsc_vec)
        self.rho.x.scatter_forward()

    def destroy(self):
        super().destroy()
        if self._ksp_helm is not None:
            self._ksp_helm.destroy()
            self._ksp_helm = None
        if self._A_helm is not None:
            self._A_helm.destroy()
            self._A_helm = None
        if self._b_helm is not None:
            self._b_helm.destroy()
            self._b_helm = None

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Density")
        self._apply_helmholtz_filter()

        # Clamp to physical bounds
        rho_min, rho_max = self.cfg.rho_min, self.cfg.rho_max
        with self.rho.x.petsc_vec.localForm() as loc:
            arr = loc.array
            arr[arr < rho_min] = rho_min
            arr[arr > rho_max] = rho_max
        self.rho.x.scatter_forward()
        return its, reason
