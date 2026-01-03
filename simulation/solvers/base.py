"""Base class for PDE subsolvers with PETSc KSP backend."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_matrix,
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
    """

    _label: str = "base"

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
        self.smooth_eps: float = self.cfg.numerics.smooth_eps

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

        extra = self._compute_extra_stats()

        return SweepStats(
            label=self._label,
            ksp_iters=iters,
            ksp_reason=reason,
            solve_time=t1 - t0,
            extra=extra,
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
