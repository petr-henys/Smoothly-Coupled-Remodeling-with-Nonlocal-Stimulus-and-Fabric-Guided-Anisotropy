"""Remodeller: orchestrates coupled mechanics, fabric, stimulus, and density."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import basix.ufl
from dolfinx import fem
from dolfinx.fem import Function, functionspace

from simulation.config import Config
from simulation.conservation import ConservationMonitor
from simulation.fixedsolver import FixedPointSolver
from femur.loader import Loader
from simulation.logger import get_logger
from simulation.progress import ProgressReporter, SweepProgressReporter
from simulation.registry import BlockRegistry
from simulation.storage import UnifiedStorage
from simulation.timeintegrator import TimeIntegrator
from simulation.utils import assign
from simulation.factory import SolverFactory, DefaultSolverFactory

if TYPE_CHECKING:
    from mpi4py import MPI


class Remodeller:
    """Orchestrates the coupled remodeling loop (Mechanics ↔ Fabric ↔ Stimulus ↔ Density)."""

    def __init__(
        self,
        cfg: Config,
        loader: Loader,
        factory: SolverFactory | None = None,
    ):
        """Initialize simulation environment, storage, and solvers.
        
        Parameters
        ----------
        cfg
            Simulation configuration.
        loader
            Traction loader with precomputed loading cases.
        factory
            Solver factory for creating solvers (defaults to DefaultSolverFactory).
        """
        self.cfg = cfg
        self.domain = cfg.domain
        self.comm: MPI.Comm = self.domain.comm
        self.rank = self.comm.rank
        self.closed = False

        self.loader = loader
        self.factory = factory or DefaultSolverFactory(cfg)

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

        # Create P1 function spaces for displacement (vector), density/stimulus (scalar), and fabric (tensor)
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
        assign(self.rho_old, self.cfg.density.rho0)  # Must match rho initial value

        self.L = Function(self.T, name="L")
        self.L_old = Function(self.T, name="L_old")
        assign(self.L, 0.0)
        assign(self.L_old, 0.0)

        self.S = Function(self.Q, name="S")
        self.S_old = Function(self.Q, name="S_old")
        assign(self.S, 0.0)
        assign(self.S_old, 0.0)

    def _build_solvers(self) -> None:
        """Initialize and register solver blocks (Mechanics, Fabric, Stimulus, Density)."""
        # =====================================================================
        # Block registration - add/remove blocks here
        # =====================================================================
        self.registry = BlockRegistry(self.comm, self.cfg)

        # 1. Create mechanics solver (used by GaitDriver)
        mechsolver = self.factory.create_mechanics_solver(
            self.u, self.rho, self.L, self.loader
        )

        # 2. Create GaitDriver (handles mechanics and SED averaging)
        self.driver = self.factory.create_driver(mechsolver, self.loader)
        self.registry.register(self.driver)

        # 3. Create FabricSolver (evolves log-fabric tensor)
        self.fabricsolver = self.factory.create_fabric_solver(
            self.L, self.L_old, self.driver.Qbar_field()
        )
        self.registry.register(self.fabricsolver)

        # 4. Create StimulusSolver (computes stimulus field)
        self.stimsolver = self.factory.create_stimulus_solver(
            self.S, self.S_old, self.driver.stimulus_field(), self.rho
        )
        self.registry.register(self.stimsolver)

        # 5. Create DensitySolver (evolves density)
        self.densolver = self.factory.create_density_solver(
            self.rho, self.rho_old, self.S
        )
        self.registry.register(self.densolver)

        # =====================================================================
        # Retrieve state fields from registered blocks
        # =====================================================================
        self.state_fields = self.registry.state_fields
        self.state_fields_old = self.registry.state_fields_old

        # Register output fields for VTX storage
        # Note: registry.output_fields is already sorted (CG first, then DG)
        # to ensure VTXWriter initializes with correct topology.
        output_fields = self.registry.output_fields
        self.storage.fields.register("fields", output_fields, filename="fields.bp")

        # Fixed-point solver (uses blocks from registry)
        self.fixedsolver = FixedPointSolver(
            self.comm, self.cfg, blocks=self.registry.blocks,
        )

        # Time integrator (uses auto-discovered state fields)
        self.integrator = TimeIntegrator(
            self.comm,
            self.state_fields,
            time_params=self.cfg.time,
            log_file=self.cfg.log_file,
        )

        # Conservation monitor (tracks mass/energy balance)
        self.conservation = ConservationMonitor(
            self.cfg,
            rho=self.rho,
            rho_old=self.rho_old,
            S=self.S,
            psi=self.driver.psi,  # Cycle-averaged SED from GaitDriver
        )

    def _setup_blocks(self) -> None:
        """Call setup() on all coupling blocks."""
        self.registry.setup_all()

    def _assemble_blocks_lhs(self) -> None:
        """Reassemble LHS matrices for all coupling blocks."""
        self.registry.assemble_lhs_all()

    def _ensure_solvers_initialized(self, dt: float) -> None:
        """Initialize solvers or reassemble LHS if time step changed."""
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

    def step(
        self,
        dt: float,
        reporter: ProgressReporter | SweepProgressReporter | None = None,
        step_index: int = 0,
        sim_time: float = 0.0,
    ) -> Tuple[float, Dict]:
        """Execute one time step using fixed-point iteration."""
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
            step_index=step_index,
            sim_time=sim_time,
        )
        fp_stop_reason = str(getattr(self.fixedsolver, "stop_reason", "") or "")

        error_norm = self.integrator.compute_wrms_error(x_pred)
        subiter_metrics = list(self.fixedsolver.subiter_metrics)
        used_subiters = len(subiter_metrics)

        return error_norm, {
            "converged": converged,
            "iters": used_subiters,
            "fp_stop_reason": fp_stop_reason,
            "subiter_metrics": subiter_metrics,
        }

    def simulate(self, reporter: ProgressReporter | SweepProgressReporter | None = None) -> None:
        """Run main simulation loop with time adaptivity."""
        t = 0.0
        dt = float(self.cfg.time.dt_initial)
        total_time = float(self.cfg.time.total_time)
        step_idx = 0
        attempt = 1  # Track attempt number for current step

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

                error, metrics = self.step(dt, reporter, step_index=step_idx + 1, sim_time=t + dt)

                if self.cfg.time.adaptive_dt:
                    accepted, next_dt, reason = self.integrator.suggest_dt(
                        dt,
                        metrics["converged"],
                        error,
                        coupling_reason=str(metrics.get("fp_stop_reason", "") or ""),
                    )
                else:
                    accepted = True
                    next_dt = dt
                    reason = "accepted"

                # Compute conservation metrics (after step, before potential rejection)
                cons_metrics = self.conservation.compute(dt).to_dict()

                # Write metrics to CSV for ALL steps (accepted and rejected)
                self.storage.metrics.write_step(
                    step=step_idx + 1,  # Use next step number for this attempt
                    attempt=attempt,
                    time_days=t + dt,   # Would-be time if accepted
                    dt_days=dt,
                    converged=metrics["converged"],
                    accepted=accepted,
                    reject_reason=str(reason),
                    error_norm=error,
                    subiter_metrics=metrics.get("subiter_metrics", []),
                    conservation_metrics=cons_metrics,
                )

                if accepted:
                    self.integrator.commit_step(dt, self.state_fields, self.state_fields_old)
                    t += dt
                    step_idx += 1
                    attempt = 1  # Reset attempt counter for next step

                    if step_idx % self.cfg.output.saving_interval == 0:
                        self._output(t)

                    if reporter is not None:
                        reporter.update_main(t, dt, error)

                    dt = next_dt
                else:
                    attempt += 1  # Increment attempt for retry
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
