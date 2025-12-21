"""Top-level remodeling loop: couples mechanics, stimulus, and density.

Orchestrates:
- mechanics solve(s) via `GaitDriver` (multi-load SED averaging)
- density solve via `DensitySolver`
- fixed-point coupling with optional Anderson acceleration
- adaptive time stepping via `TimeIntegrator`
"""

from __future__ import annotations

from typing import Dict, Tuple, List
from pathlib import Path

import basix.ufl
from dolfinx import fem
from dolfinx.fem import Function, functionspace

from simulation.storage import UnifiedStorage
from simulation.logger import get_logger
from simulation.utils import build_dirichlet_bcs, assign
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, FabricSolver, StimulusSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver
from simulation.drivers import GaitDriver
from simulation.timeintegrator import TimeIntegrator
from simulation.registry import BlockRegistry
from simulation.loader import Loader, LoadingCase


class Remodeller:
    """Orchestrates coupled mechanics↔density remodeling (MPI-parallel)."""

    def __init__(self, cfg: Config, loader: Loader, loading_cases: List[LoadingCase]):
        """Initialize coupled solvers, I/O, and precomputed loading cases."""
        self.cfg = cfg
        self.domain = self.cfg.domain
        self.closed = False
        self.progress = None
        self.main_task_id = None
        self.loader = loader
        self.loading_cases = loading_cases
        self.comm = self.domain.comm
        self.rank = self.comm.rank
        
        # Ensure log directory exists (rank 0 creates, all ranks wait)
        if self.rank == 0:
            log_path = Path(self.cfg.log_file)
            if log_path.parent:
                log_path.parent.mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()

        self.logger = get_logger(self.comm, name="Remodeller", log_file=self.cfg.log_file)
        self.logger.debug("Initializing Remodeller...")

        self.storage = UnifiedStorage(cfg)

        self.dx = self.cfg.dx
        self.ds = self.cfg.ds
        self.gdim = self.domain.geometry.dim

        # Build function spaces
        P1_vec = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim,))
        P1 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim, self.gdim))

        self.V = functionspace(self.domain, P1_vec)
        self.Q = functionspace(self.domain, P1)
        self.T = functionspace(self.domain, P1_ten)

        # Total DOFs (for diagnostics)
        dofs_V = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        dofs_Q = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
        dofs_T = self.T.dofmap.index_map.size_global * self.T.dofmap.index_map_bs
        self.num_dofs_total = int(dofs_V + dofs_Q + dofs_T)

        # Create state fields
        u = Function(self.V, name="u")
        rho = Function(self.Q, name="rho")
        rho_old = Function(self.Q, name="rho_old")
        assign(rho, self.cfg.rho0)

        L = Function(self.T, name="L")
        L_old = Function(self.T, name="L_old")
        assign(L, 0.0)
        assign(L_old, 0.0)

        S = Function(self.Q, name="S")
        S_old = Function(self.Q, name="S_old")
        assign(S, 0.0)
        assign(S_old, 0.0)

        # Boundary conditions
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(self.loader.traction, self.loader.load_tag)]

        # =====================================================================
        # Block registration - add/remove blocks here
        # =====================================================================
        self.registry = BlockRegistry(self.comm, self.cfg)

        # 1. Mechanics solver (wrapped by GaitDriver)
        mechsolver = MechanicsSolver(u, rho, self.cfg, bc_mech, neumann_bcs, L=L)

        # 2. GaitDriver: mechanics + multi-load SED averaging
        self.driver = GaitDriver(
            mechsolver, self.cfg,
            loader=self.loader,
            loading_cases=self.loading_cases,
        )
        self.registry.register(self.driver)

        # 3. Fabric solver: log-fabric evolution L -> L_target(Qbar)
        self.fabricsolver = FabricSolver(L, L_old, self.driver.Qbar_field(), self.cfg)
        self.registry.register(self.fabricsolver)

        # 4. Stimulus solver: S(psi, rho)
        self.stimsolver = StimulusSolver(S, S_old, self.driver.stimulus_field(), rho, self.cfg)
        self.registry.register(self.stimsolver)

        # 5. Density solver: rho(S)
        self.densolver = DensitySolver(rho, rho_old, S, self.cfg)
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
            self.comm,
            self.cfg,
            blocks=self.registry.blocks,
        )

        # Time integrator (uses auto-discovered state fields)
        self.integrator = TimeIntegrator(self.comm, self.cfg, self.state_fields)

        self.solvers_initialized = False
        self._current_dt: float = 0.0

        # Persist initial configuration
        self.cfg.update_config_json()

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

    def close(self):
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
        # Call post_step_update on all blocks (e.g., compute eigenvectors)
        self.registry.post_step_update_all()
        self.storage.fields.write("fields", float(t))

    def step(self, dt: float) -> Tuple[float, Dict]:
        """Single timestep attempt."""
        for name, f in self.state_fields.items():
            assign(self.state_fields_old[name], f, scatter=True)

        self._ensure_solvers_initialized(dt)

        x_pred = self.integrator.predict(dt)
        for name, pred in x_pred.items():
            assign(self.state_fields[name], pred, scatter=True)

        converged = self.fixedsolver.run(
            self.progress if self.rank == 0 else None,
            getattr(self, 'sub_task_id', None) if self.rank == 0 else None,
        )

        error_norm = self.integrator.compute_wrms_error(x_pred)
        metrics = list(self.fixedsolver.subiter_metrics)
        used_subiters = len(metrics)

        return error_norm, {"converged": converged, "iters": used_subiters}

    def simulate(self) -> None:
        """Run remodeling loop using dt_initial and total_time from Config."""
        t = 0.0
        dt = float(self.cfg.dt_initial)
        total_time = float(self.cfg.total_time)
        step_idx = 0

        self.comm.Barrier()

        if self.rank == 0:
            from rich.progress import (
                Progress,
                TextColumn,
                BarColumn,
                SpinnerColumn,
                TimeRemainingColumn,
                TimeElapsedColumn,
            )
            from rich.console import Console

            console = Console(stderr=True, force_terminal=True)
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=60),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(compact=True),
                TextColumn("{task.fields[info]}"),
                console=console,
                transient=False,
            )
            self.main_task_id = self.progress.add_task("Remodeling", total=total_time, info=" " * 35)
            self.sub_task_id = self.progress.add_task("  Coupling", total=self.cfg.max_subiters, info=" " * 35)
            self.progress.start()

        while t < total_time:
            if t + dt > total_time:
                dt = total_time - t
            
            error, metrics = self.step(dt)
            
            if self.cfg.adaptive_dt:
                # Adaptive time stepping with PI controller
                accepted, next_dt, reason = self.integrator.suggest_dt(dt, metrics["converged"], error)
            else:
                # Fixed time stepping - always accept, keep dt constant
                accepted = True
                next_dt = dt
            
            if accepted:
                self.integrator.commit_step(dt, self.state_fields, self.state_fields_old)
                t += dt
                step_idx += 1
                
                if step_idx % self.cfg.saving_interval == 0:
                    self._output(t)
                
                if self.progress is not None and self.main_task_id is not None:
                    info_str = f"t={t:5.1f}d dt={next_dt:5.1f} err={error:.1e}"
                    self.progress.update(self.main_task_id, completed=t, info=f"{info_str:<35}")
                
                dt = next_dt
            else:
                for name in self.state_fields:
                    assign(self.state_fields[name], self.state_fields_old[name])
                if not metrics["converged"]:
                     self.integrator.reset_history()
                dt = next_dt

        if self.progress is not None:
            info_str = f"t={t:5.1f}d dt={dt:5.1f} done"
            self.progress.update(self.main_task_id, completed=t, info=f"{info_str:<35}")
            self.progress.stop()
            self.progress = None

    def __enter__(self) -> "Remodeller":
        return self

    def __exit__(self, *_) -> None:
        self.close()
