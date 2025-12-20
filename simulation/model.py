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
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver
from simulation.drivers import GaitDriver
from simulation.timeintegrator import TimeIntegrator
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

        P1_vec = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim,))
        P1 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)

        self.V = functionspace(self.domain, P1_vec)
        self.Q = functionspace(self.domain, P1)

        # Total DOFs
        dofs_V = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        dofs_Q = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
        self.num_dofs_total = int(dofs_V + dofs_Q)

        u = Function(self.V, name="u")
        self.rho = Function(self.Q, name="rho")
        self.rho_old = Function(self.Q, name="rho_old")

        assign(self.rho, self.cfg.rho0)

        # Dirichlet BC: clamp cut surface (u=0)
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)

        # Neumann BC: traction on proximal surface (hip + muscles)
        neumann_bcs = [(self.loader.traction, self.loader.load_tag)]

        # 1. Mechanics Solver
        mechsolver = MechanicsSolver(u, self.rho, self.cfg, bc_mech, neumann_bcs)

        # 2. Driver with loading cases
        self.driver = GaitDriver(
            mechsolver, self.cfg, 
            loader=self.loader,
            loading_cases=self.loading_cases,
        )

        # 3. Stimulus field and solver (implicit Euler diffusion/decay + explicit drive).
        self.S = fem.Function(self.Q, name="S")
        self.S_old = fem.Function(self.Q, name="S_old")
        assign(self.S, 0.0)
        assign(self.S_old, 0.0)

        # Register all fields for output in a single VTX file
        self.storage.fields.register(
            "fields",
            [self.rho, self.S, self.driver.psi],
            filename="fields.bp",
        )

        self.stimsolver = StimulusSolver(
            self.S,
            self.S_old,
            self.driver.stimulus_field(),
            self.rho,
            self.cfg,
        )

        # 4. Density Solver
        self.densolver = DensitySolver(
            self.rho, 
            self.rho_old, 
            self.S,
            self.cfg
        )
        self.fixedsolver = FixedPointSolver(
            self.comm,
            self.cfg,
            # Only (S, rho) are in the coupled state vector (see each block's `state_fields`):
            # - driver: updates quasi-static mechanics and recomputes psi, but contributes no state fields
            # - stimsolver: updates S
            # - densolver: updates rho
            blocks=(self.driver, self.stimsolver, self.densolver),
        )

        # State fields for time integration (S, rho) with their old-step counterparts.
        self.state_fields = {
            "S": self.S,
            "rho": self.rho,
        }
        self.state_fields_old = {
            "S": self.S_old,
            "rho": self.rho_old,
        }
        self.integrator = TimeIntegrator(self.comm, self.cfg, self.state_fields)

        self.solvers_initialized = False
        self._current_dt: float = 0.0

        # Persist initial configuration
        self.cfg.update_config_json()

    def _setup_blocks(self) -> None:
        """Call setup() on all coupling blocks."""
        for blk in self.fixedsolver.blocks:
            blk.setup()

    def _assemble_blocks_lhs(self) -> None:
        """Reassemble LHS matrices for all coupling blocks."""
        for blk in self.fixedsolver.blocks:
            blk.assemble_lhs()

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

        if hasattr(self, "driver") and self.driver is not None:
            self.driver.destroy()

        for attr in ("stimsolver", "densolver"):
            solver = getattr(self, attr, None)
            if solver is not None:
                solver.destroy()

        if self.storage is not None:
            self.storage.close()

        self.comm.Barrier()
        self.closed = True

    def _output(self, t: float) -> None:
        """Write fields to storage."""
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
            )
            from rich.console import Console

            console = Console(stderr=True, force_terminal=True)
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=60),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
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
