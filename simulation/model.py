"""Top-level remodeling loop: couples mechanics, stimulus, and density.

Orchestrates:
- mechanics solve(s) via `GaitDriver` (multi-load SED averaging)
- density solve via `DensitySolver`
- fixed-point coupling with optional Anderson acceleration
- adaptive time stepping via `TimeIntegrator`
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import basix.ufl
from dolfinx import fem
from dolfinx.fem import Function, functionspace

from simulation.config import Config
from simulation.drivers import GaitDriver
from simulation.fixedsolver import FixedPointSolver
from simulation.loader import Loader, LoadingCase
from simulation.logger import get_logger
from simulation.progress import ProgressReporter
from simulation.registry import BlockRegistry
from simulation.storage import UnifiedStorage
from simulation.subsolvers import DensitySolver, FabricSolver, MechanicsSolver, StimulusSolver
from simulation.timeintegrator import TimeIntegrator
from simulation.utils import assign, build_dirichlet_bcs

if TYPE_CHECKING:
    from mpi4py import MPI


class Remodeller:
    """Orchestrates coupled mechanics↔density remodeling (MPI-parallel)."""

    def __init__(
        self,
        cfg: Config,
        loader: Loader,
        loading_cases: List[LoadingCase],
    ):
        """Initialize coupled solvers, I/O, and precomputed loading cases."""
        self.cfg = cfg
        self.domain = cfg.domain
        self.comm: MPI.Comm = self.domain.comm
        self.rank = self.comm.rank
        self.closed = False

        self.loader = loader
        self.loading_cases = loading_cases

        # Ensure log directory exists (rank 0 creates, all ranks wait)
        self._ensure_log_dir()

        self.logger = get_logger(self.comm, name="Remodeller", log_file=self.cfg.log_file)
        self.logger.debug("Initializing Remodeller...")

        self.storage = UnifiedStorage(cfg)

        # Build function spaces and state fields, then wire solvers
        self._build_spaces_and_fields()
        self._build_solvers()

        self.solvers_initialized = False
        self._current_dt: float = 0.0

        # Persist initial configuration
        self.cfg.update_config_json()

    def _ensure_log_dir(self) -> None:
        """Create log directory on rank 0."""
        if self.rank == 0:
            log_path = Path(self.cfg.log_file)
            if log_path.parent:
                log_path.parent.mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()

    def _build_spaces_and_fields(self) -> None:
        """Create function spaces and state fields."""
        gdim = self.domain.geometry.dim
        cell = self.domain.basix_cell()

        # P1 spaces: vector (u), scalar (rho, S), tensor (L)
        P1_vec = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
        P1 = basix.ufl.element("Lagrange", cell, 1)
        P1_ten = basix.ufl.element("Lagrange", cell, 1, shape=(gdim, gdim))

        self.V = functionspace(self.domain, P1_vec)
        self.Q = functionspace(self.domain, P1)
        self.T = functionspace(self.domain, P1_ten)

        # Total DOFs (for diagnostics)
        dofs_V = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        dofs_Q = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
        dofs_T = self.T.dofmap.index_map.size_global * self.T.dofmap.index_map_bs
        self.num_dofs_total = int(dofs_V + dofs_Q + dofs_T)

        # State fields
        self.u = Function(self.V, name="u")
        self.rho = Function(self.Q, name="rho")
        self.rho_old = Function(self.Q, name="rho_old")
        assign(self.rho, self.cfg.density.rho0)

        self.L = Function(self.T, name="L")
        self.L_old = Function(self.T, name="L_old")
        assign(self.L, 0.0)
        assign(self.L_old, 0.0)

        self.S = Function(self.Q, name="S")
        self.S_old = Function(self.Q, name="S_old")
        assign(self.S, 0.0)
        assign(self.S_old, 0.0)

    def _build_solvers(self) -> None:
        """Wire up the solver graph: mechanics → fabric → stimulus → density."""
        # Boundary conditions
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(self.loader.traction, self.loader.load_tag)]

        # =====================================================================
        # Block registration - add/remove blocks here
        # =====================================================================
        self.registry = BlockRegistry(self.comm, self.cfg)

        # 1. Mechanics solver (wrapped by GaitDriver)
        mechsolver = MechanicsSolver(
            self.u, self.rho, self.cfg, bc_mech, neumann_bcs, L=self.L
        )

        # 2. GaitDriver: mechanics + multi-load SED averaging
        self.driver = GaitDriver(
            mechsolver, self.cfg,
            loader=self.loader,
            loading_cases=self.loading_cases,
        )
        self.registry.register(self.driver)

        # 3. Fabric solver: log-fabric evolution L -> L_target(Qbar)
        self.fabricsolver = FabricSolver(
            self.L, self.L_old, self.driver.Qbar_field(), self.cfg
        )
        self.registry.register(self.fabricsolver)

        # 4. Stimulus solver: S(psi, rho)
        self.stimsolver = StimulusSolver(
            self.S, self.S_old, self.driver.stimulus_field(), self.rho, self.cfg
        )
        self.registry.register(self.stimsolver)

        # 5. Density solver: rho(S)
        self.densolver = DensitySolver(self.rho, self.rho_old, self.S, self.cfg)
        self.registry.register(self.densolver)

        # =====================================================================
        # Auto-discover fields from blocks
        # =====================================================================
        self.state_fields = self.registry.state_fields
        self.state_fields_old = self.registry.state_fields_old

        # Register output fields for VTX storage (auto-collected from blocks)
        output_fields = self.registry.output_fields
        self.storage.fields.register("fields", output_fields, filename="fields.bp")

        # Fixed-point solver (uses blocks from registry)
        self.fixedsolver = FixedPointSolver(
            self.comm, self.cfg, blocks=self.registry.blocks,
        )

        # Time integrator (uses auto-discovered state fields)
        self.integrator = TimeIntegrator(self.comm, self.cfg, self.state_fields)

    def _setup_blocks(self) -> None:
        """Call setup() on all coupling blocks."""
        self.registry.setup_all()

    def _assemble_blocks_lhs(self) -> None:
        """Reassemble LHS matrices for all coupling blocks."""
        self.registry.assemble_lhs_all()

    def _ensure_solvers_initialized(self, dt: float) -> None:
        """Initialize solver objects and (re)assemble dt-dependent operators."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got dt={dt}.")

        if not self.solvers_initialized:
            self.cfg.set_dt(dt)
            self._current_dt = dt
            self._setup_blocks()
            self.solvers_initialized = True
            return

        if abs(dt - self._current_dt) > 1e-12:
            self.cfg.set_dt(dt)
            self._current_dt = dt
            self._assemble_blocks_lhs()

    def close(self) -> None:
        """Release PETSc resources and close I/O."""
        if self.closed:
            return

        self.comm.Barrier()

        # Destroy all blocks via registry
        if hasattr(self, "registry") and self.registry is not None:
            self.registry.destroy_all()

        if self.storage is not None:
            self.storage.close()

        self.comm.Barrier()
        self.closed = True

    def _output(self, t: float) -> None:
        """Write fields to storage."""
        self.registry.post_step_update_all()
        self.storage.fields.write("fields", float(t))

    def step(self, dt: float, reporter: ProgressReporter | None = None) -> Tuple[float, Dict]:
        """Execute a single timestep attempt.

        Args:
            dt: Timestep size [days].
            reporter: Optional progress reporter for subiteration display.

        Returns:
            Tuple of (error_norm, metrics_dict).
        """
        for name, f in self.state_fields.items():
            assign(self.state_fields_old[name], f, scatter=True)

        self._ensure_solvers_initialized(dt)

        x_pred = self.integrator.predict(dt)
        for name, pred in x_pred.items():
            assign(self.state_fields[name], pred, scatter=True)

        # Pass reporter for subiteration progress (rank 0 only)
        converged = self.fixedsolver.run(
            reporter.progress if reporter and reporter.rank == 0 else None,
            reporter.sub_task_id if reporter and reporter.rank == 0 else None,
        )

        error_norm = self.integrator.compute_wrms_error(x_pred)
        metrics = list(self.fixedsolver.subiter_metrics)
        used_subiters = len(metrics)

        return error_norm, {"converged": converged, "iters": used_subiters}

    def simulate(self, reporter: ProgressReporter | None = None) -> None:
        """Run remodeling loop using dt_initial and total_time from Config.

        Args:
            reporter: Optional progress reporter. If None, a default reporter
                     is created on rank 0.
        """
        t = 0.0
        dt = float(self.cfg.time.dt_initial)
        total_time = float(self.cfg.time.total_time)
        step_idx = 0

        self.comm.Barrier()

        # Create default reporter if none provided
        owns_reporter = False
        if reporter is None and self.rank == 0:
            reporter = ProgressReporter(
                self.comm, total_time, self.cfg.solver.max_subiters
            )
            reporter.start()
            owns_reporter = True

        try:
            while t < total_time:
                if t + dt > total_time:
                    dt = total_time - t

                error, metrics = self.step(dt, reporter)

                if self.cfg.time.adaptive_dt:
                    accepted, next_dt, reason = self.integrator.suggest_dt(
                        dt, metrics["converged"], error
                    )
                else:
                    accepted = True
                    next_dt = dt

                if accepted:
                    self.integrator.commit_step(dt, self.state_fields, self.state_fields_old)
                    t += dt
                    step_idx += 1

                    if step_idx % self.cfg.output.saving_interval == 0:
                        self._output(t)

                    if reporter is not None:
                        reporter.update_main(t, next_dt, error)

                    dt = next_dt
                else:
                    for name in self.state_fields:
                        assign(self.state_fields[name], self.state_fields_old[name])
                    if not metrics["converged"]:
                        self.integrator.reset_history()
                    dt = next_dt

            if reporter is not None:
                reporter.update_main(t, dt, 0.0, done=True)

        finally:
            if owns_reporter and reporter is not None:
                reporter.stop()

    def __enter__(self) -> "Remodeller":
        """Context manager entry."""
        return self

    def __exit__(self, *_) -> None:
        """Context manager exit."""
        self.close()
