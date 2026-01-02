"""Base class for PDE subsolvers with PETSc KSP backend."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
)
import ufl

from simulation.logger import get_logger
from simulation.stats import SweepStats

if TYPE_CHECKING:
    from simulation.config import Config


class BaseLinearSolver:
    """Base class for implicit PDE solvers using PETSc KSP.

    Provides:
    - PETSc matrix/vector management
    - KSP setup with configurable tolerances
    - CouplingBlock protocol default implementations
    - Solve timing and statistics
    - Optional Helmholtz smoothing (smoothing_length > 0)
    """

    _label: str = "base"

    def __init__(
        self,
        cfg: "Config",
        state_function: fem.Function,
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
        smoothing_length: float = 0.0,
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
        self.smooth_eps: float = self.cfg.numerics.smooth_eps

        self.trial = ufl.TrialFunction(self.function_space)
        self.test = ufl.TestFunction(self.function_space)

        # Helmholtz filter (internal, invisible to user)
        self._smoothing_length: float = float(smoothing_length)
        self._filter_ksp: PETSc.KSP | None = None
        self._filter_A: PETSc.Mat | None = None
        self._filter_b: PETSc.Vec | None = None
        self._filter_a_form: fem.Form | None = None
        self._filter_L_form: fem.Form | None = None
        self._filter_length_c: fem.Constant | None = None

        self.last_reason: int = 0

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: PETSc.KSP = None
        self.A: PETSc.Mat = None
        self.b: PETSc.Vec = None

        self.a_form: fem.Form = None
        self.L_form: fem.Form = None

    def destroy(self):
        # Destroy Helmholtz filter resources
        if self._filter_ksp is not None:
            self._filter_ksp.destroy()
            self._filter_ksp = None
        if self._filter_A is not None:
            self._filter_A.destroy()
            self._filter_A = None
        if self._filter_b is not None:
            self._filter_b.destroy()
            self._filter_b = None
        # Main solver resources
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None
        if self.A is not None:
            self.A.destroy()
            self.A = None
        if self.b is not None:
            self.b.destroy()
            self.b = None

    # -------------------------------------------------------------------------
    # CouplingBlock protocol defaults
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        return ()

    def post_step_update(self) -> None:
        pass

    def setup(self):
        self._compile_forms()
        self.A = create_matrix(self.a_form)
        self.b = create_vector(self.function_space)
        self.assemble_lhs()
        self._setup_ksp()
        self._setup_helmholtz_filter()

    def assemble_lhs(self) -> None:
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()
        if self.ksp is not None:
            self.ksp.setOperators(self.A)

    def assemble_rhs(self) -> None:
        raise NotImplementedError

    def _compile_forms(self) -> None:
        raise NotImplementedError

    def _setup_ksp(self) -> None:
        raise NotImplementedError

    def _compute_extra_stats(self) -> Dict[str, Any]:
        return {}

    # -------------------------------------------------------------------------
    # Helmholtz filter (internal implementation)
    # -------------------------------------------------------------------------

    def _setup_helmholtz_filter(self) -> None:
        """Initialize Helmholtz filter if smoothing_length > 0."""
        if self._smoothing_length <= 0.0:
            return

        # Length constant
        self._filter_length_c = fem.Constant(self.mesh, self._smoothing_length)

        # Compile filter forms: (I - ℓ² Δ) u = u_rhs
        # Use self.state directly as RHS source (no copy needed)
        u = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)
        ell = self._filter_length_c

        a_ufl = ufl.inner(u, v) * self.dx + (ell * ell) * ufl.inner(ufl.grad(u), ufl.grad(v)) * self.dx
        self._filter_a_form = fem.form(a_ufl)

        L_ufl = ufl.inner(self.state, v) * self.dx
        self._filter_L_form = fem.form(L_ufl)

        # Create matrix/vector
        self._filter_A = create_matrix(self._filter_a_form)
        self._filter_b = create_vector(self.function_space)

        # Assemble LHS
        self._filter_A.zeroEntries()
        assemble_matrix(self._filter_A, self._filter_a_form, bcs=[])
        self._filter_A.assemble()
        self._filter_A.setOption(PETSc.Mat.Option.SPD, True)

        # Setup KSP (CG + AMG for SPD Helmholtz)
        self._filter_ksp = PETSc.KSP().create(self.comm)
        self._filter_ksp.setOptionsPrefix(f"{self._label}_filter_")
        
        opts = PETSc.Options()
        opts[f"{self._label}_filter_ksp_type"] = "cg"
        opts[f"{self._label}_filter_pc_type"] = "gamg"
        opts[f"{self._label}_filter_ksp_rtol"] = 1e-10
        opts[f"{self._label}_filter_ksp_atol"] = 1e-14
        opts[f"{self._label}_filter_ksp_max_it"] = 200

        self._filter_ksp.setOperators(self._filter_A)
        self._filter_ksp.setInitialGuessNonzero(True)
        self._filter_ksp.setFromOptions()
        self._filter_ksp.setUp()

    def _apply_helmholtz_filter(self) -> None:
        """Apply Helmholtz smoothing to self.state in-place."""
        if self._filter_ksp is None:
            return

        # Assemble RHS from current state
        with self._filter_b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self._filter_b, self._filter_L_form)
        self._filter_b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self._filter_b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        # Solve filter directly into state
        self._filter_ksp.solve(self._filter_b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()

    # -------------------------------------------------------------------------
    # Main solve
    # -------------------------------------------------------------------------

    def _solve(self) -> SweepStats:
        t0 = time.perf_counter()
        self.ksp.solve(self.b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()
        t1 = time.perf_counter()

        reason = int(self.ksp.getConvergedReason())
        iters = int(self.ksp.getIterationNumber())
        self.last_reason = reason

        if reason < 0:
            self.logger.warning(f"{self._label} solver failed to converge (reason: {reason})")

        # Apply Helmholtz filter if enabled (purely internal)
        self._apply_helmholtz_filter()

        return SweepStats(
            label=self._label,
            ksp_iters=iters,
            ksp_reason=reason,
            solve_time=t1 - t0,
            extra=self._compute_extra_stats(),
        )

    def create_ksp(self, prefix: str, ksp_options: dict[str, object]) -> PETSc.KSP:
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOptionsPrefix(prefix + "_")

        opts = PETSc.Options()
        for k, v in ksp_options.items():
            if v is not None:
                opts[f"{prefix}_{k}"] = v

        opts[f"{prefix}_ksp_rtol"] = self.cfg.solver.ksp_rtol
        opts[f"{prefix}_ksp_atol"] = self.cfg.solver.ksp_atol
        opts[f"{prefix}_ksp_max_it"] = self.cfg.solver.ksp_max_it

        self.ksp.setInitialGuessNonzero(True)
        if self.A is not None:
            self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()
        self.ksp.setUp()
        return self.ksp
    