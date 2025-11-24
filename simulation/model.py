"""Core remodeling model: orchestration of coupled subsolvers.

This module exposes a single high-level entry point, :class:`Remodeller`,
which wires the mechanics, stimulus, density and direction solvers together
for a given finite-element domain and :class:`simulation.config.Config`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
from mpi4py import MPI
import basix.ufl
from dolfinx import fem
from dolfinx.fem import Function, functionspace

from simulation.storage import UnifiedStorage
from simulation.logger import get_logger, Level
from simulation.utils import build_dirichlet_bcs, assign, current_memory_mb
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver
from simulation.drivers import SimplifiedGaitDriver
from simulation.traction_utils import create_traction_function, create_pressure_function


class Remodeller:
    """High-level bone remodeling orchestrator.
    
    Owns FE fields, subsolvers, and storage. Requires fully specified Config.
    """

    def __init__(self, cfg: Config, stages: List[Dict]):
        """Initialize remodeler with configuration.
        
        Parameters
        ----------
        cfg : Config
            Complete simulation configuration with domain and facet_tags.
        stages : List[Dict]
            List of load stages for the simplified gait driver.
        """
        self.cfg = cfg
        self.domain = self.cfg.domain
        self.closed = False
        self.progress = None
        self.main_task_id = None
        self.stages = stages

        if self.cfg.verbose == "progressbar":
            self.verbose = False
        else:
            self.verbose = bool(self.cfg.verbose)

        self.comm = self.domain.comm
        self.rank = self.comm.rank
        
        # Initialize log file (create directory and clear file) on rank 0
        if self.rank == 0:
            try:
                log_path = Path(self.cfg.log_file)
                if log_path.parent:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cfg.log_file, "w", encoding="utf-8") as f:
                    pass
            except IOError:
                pass

        self.logger = get_logger(self.comm, verbose=self.verbose, name="Remodeller", log_file=self.cfg.log_file)
        self.logger.info("Initializing Remodeller...")

        self.storage = UnifiedStorage(cfg)
        self.telemetry = self.cfg.telemetry

        if self.telemetry is not None:
            self.telemetry.register_csv(
                "steps",
                [
                    "step",
                    "time_days",
                    "dt_days",
                    "tol",
                    "used_subiters",
                    "mech_time_s",
                    "dens_time_s",
                    "solve_time_s_total",
                    "proj_res_last",
                    "num_dofs_total",
                    "rss_mem_mb",
                    "mech_iters",
                    "dens_iters",
                ],
                filename="steps.csv",
            )

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

        # Predictor state (Adams-Bashforth)
        self.step_count = 0
        self.dt_prev: Optional[float] = None
        
        self.rho_rate_last = Function(self.Q, name="rho_rate_last")
        self.rho_rate_last2 = Function(self.Q, name="rho_rate_last2")
        
        assign(self.rho_rate_last, 0.0)
        assign(self.rho_rate_last2, 0.0)

        assign(self.rho, self.cfg.rho0)

        # Register fields
        self.storage.fields.register("scalars", [self.rho], filename="scalars.bp")

        # Boundary conditions
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)

        # --- Setup Simplified Driver ---
        # Create placeholder functions for tractions
        t_hip = fem.Function(self.V, name="t_hip")
        t_glmed = fem.Function(self.V, name="t_glmed")
        
        # Collect all unique tags used in stages
        hip_tags = set()
        gl_tags = set()
        for s in self.stages:
            hip_tags.add(s.get("hip_tag", 3))
            gl_tags.add(s.get("gl_tag", 4))
            
        neumann_bcs = []
        for tag in hip_tags:
            neumann_bcs.append((t_hip, tag))
        for tag in gl_tags:
            neumann_bcs.append((t_glmed, tag))

        # Subsolvers
        mechsolver = MechanicsSolver(u, self.rho, self.cfg, bc_mech, neumann_bcs)
        self.densolver = DensitySolver(self.rho, self.rho_old, self.cfg)

        # Driver
        self.driver = SimplifiedGaitDriver(mechsolver, t_hip, t_glmed, self.stages, self.cfg)

        self.fixedsolver = FixedPointSolver(
            self.comm,
            self.cfg,
            self.driver,
            self.densolver,
            self.rho,
            self.rho_old,
        )

        self.solvers_initialized = False
        self._current_dt: Optional[float] = None

        # Persist initial configuration
        self.cfg.update_config_json()

    def close(self):
        """Release PETSc resources and close I/O."""
        if self.closed:
            return

        self.comm.Barrier()

        if hasattr(self, "driver") and self.driver is not None:
            self.driver.destroy()

        for attr in ("densolver",):
            solver = getattr(self, attr, None)
            if solver is not None:
                solver.destroy()

        if self.storage is not None:
            self.storage.close()

        if self.telemetry is not None:
            self.telemetry.close()

        self.comm.Barrier()
        self.closed = True

    def _field_stats(self, field: fem.Function) -> Tuple[float, float, float, float]:
        """MPI global min/max/mean/median."""
        if len(field.x.array) > 0:
            field_min_local = field.x.array.min()
            field_max_local = field.x.array.max()
        else:
            field_min_local = float("inf")
            field_max_local = float("-inf")
        field_min = self.comm.allreduce(field_min_local, op=MPI.MIN)
        field_max = self.comm.allreduce(field_max_local, op=MPI.MAX)

        bs = field.function_space.dofmap.index_map_bs
        local_size = field.x.index_map.size_local * bs
        local_data = field.x.array[:local_size]
        local_sum = np.sum(local_data)
        local_count = local_size

        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        global_count = self.comm.allreduce(local_count, op=MPI.SUM)
        field_mean = global_sum / global_count if global_count > 0 else 0.0

        # Median (approximate via gather to rank 0)
        all_data = self.comm.gather(local_data, root=0)
        field_median = 0.0
        if self.comm.rank == 0:
            full_data = np.concatenate(all_data)
            if full_data.size > 0:
                field_median = float(np.median(full_data))
        field_median = self.comm.bcast(field_median, root=0)

        return field_min, field_max, field_mean, field_median

    def _collect_field_stats(self) -> Dict[str, float]:
        """Gather field min/max/mean and energy for reporting."""
        rho_min, rho_max, rho_mean, rho_median = self._field_stats(self.rho)

        psi_avg = 0.0
        psi_min = 0.0
        psi_max = 0.0
        psi_median = 0.0
        
        if hasattr(self.driver, "get_stimulus_stats"):
            psi_stats = self.driver.get_stimulus_stats()
            psi_avg = psi_stats.get("psi_avg", 0.0)
            psi_min = psi_stats.get("psi_min", 0.0)
            psi_max = psi_stats.get("psi_max", 0.0)
            psi_median = psi_stats.get("psi_median", 0.0)

        return dict(
            rho_min=rho_min, rho_max=rho_max, rho_mean=rho_mean, rho_median=rho_median,
            psi_avg=psi_avg, psi_min=psi_min, psi_max=psi_max, psi_median=psi_median
        )

    def _output(self, t: float, step: int, coupling_stats: Dict[str, float]):
        """Scatter, stats, log, write."""
        fields = self._collect_field_stats()
        
        # Get solver stats from fixedsolver
        s_stats = self.fixedsolver.solver_stats
        
        # Memory
        mem_mb = self.comm.allreduce(current_memory_mb(), op=MPI.SUM)

        if self.progress is not None and self.main_task_id is not None:
            # Fixed width formatting to prevent progress bar resizing
            info_str = f"t={t:6.1f}d rho={fields['rho_mean']:4.2f} GS={coupling_stats['iters']:2d}"
            self.progress.update(self.main_task_id, info=f"{info_str:<35}")

        # Concise summary line
        self.logger.info(f"Step {step:2d} | t={t:6.1f}d | GS={coupling_stats['iters']}")

        # Detailed table
        def format_table():
            lines = []
            lines.append("-" * 96)
            lines.append(f"{'Field':>10} | {'Min':>12} | {'Max':>12} | {'Mean':>12} | {'Median':>12} |")
            lines.append("-" * 96)
            
            # Rows
            lines.append(f"{'rho':>10} | {fields['rho_min']:12.3e} | {fields['rho_max']:12.3e} | {fields['rho_mean']:12.3e} | {fields['rho_median']:12.3e} |")
            lines.append(f"{'psi':>10} | {fields['psi_min']:12.3e} | {fields['psi_max']:12.3e} | {fields['psi_avg']:12.3e} | {fields['psi_median']:12.3e} |")
            
            lines.append("-" * 96)
            lines.append(f"{'Solver':>10} | {'Tot Time':>12} | {'Avg Time':>12} | {'Tot Iters':>12} | {'Avg Iters':>12} | {'Reason':>12} |")
            lines.append("-" * 96)
            
            gs_iters = max(1, coupling_stats['iters'])
            
            for name, key in [("Mechanics", "mech"), ("Density", "dens")]:
                st = s_stats[key]
                
                # Mechanics solver runs for each gait sample in every GS iteration
                # For SimplifiedGaitDriver, it runs for each stage
                n_stages = len(self.stages)
                if key == "mech":
                    n_solves = gs_iters * n_stages
                else:
                    n_solves = gs_iters
                    
                avg_its = st['iters'] / max(1, n_solves)
                avg_time = st['time'] / max(1, n_solves)
                lines.append(f"{name:>10} | {st['time']:12.2f} | {avg_time:12.4f} | {st['iters']:12d} | {avg_its:12.1f} | {st['reason']:12d} |")
            
            lines.append("-" * 96)
            lines.append(f"Memory (RSS): {mem_mb:.1f} MB")
            lines.append("-" * 96)
            
            return "\n" + "\n".join(lines)

        self.logger.info(format_table)

        self.storage.fields.write("scalars", float(t))

    def step(self, dt: float, *, step_index: Optional[int] = None, time_days: Optional[float] = None) -> None:
        """Single timestep: fixed-point iteration until coupling tolerance met."""
        assign(self.rho_old, self.rho)

        if not self.solvers_initialized:
            self.driver.setup()
            self.densolver.setup()
            self.solvers_initialized = True

        # Predictor step (Adams-Bashforth)
        if self.step_count > 0:
            dt_curr = float(dt)
            dt_prev = self.dt_prev if self.dt_prev is not None else dt_curr
            
            # Coefficients
            if self.step_count >= 2:
                # AB2
                w1 = 1.0 + dt_curr / (2.0 * dt_prev)
                w2 = dt_curr / (2.0 * dt_prev)
                
                self.rho.x.array[:] = self.rho_old.x.array + dt_curr * (
                    w1 * self.rho_rate_last.x.array - w2 * self.rho_rate_last2.x.array
                )
            else:
                # AB1 (Forward Euler)
                self.rho.x.array[:] = self.rho_old.x.array + dt_curr * self.rho_rate_last.x.array
            
            self.rho.x.scatter_forward()

        # Update dt [days] and reassemble LHS for time-dependent solvers if dt changed
        if self._current_dt is None or abs(float(dt) - float(self._current_dt)) > 1e-12:
            self.cfg.set_dt(float(dt))
            # Note: densolver.assemble_lhs() is called inside fixedsolver.run() loop
            # because the matrix depends on the non-linear driving force S.
            # We don't need to call it here unless we want to ensure it's ready for the first iteration,
            # but fixedsolver handles that.
            self._current_dt = float(dt)
        
        self.fixedsolver.run(
            progress=self.progress if self.rank == 0 else None,
            task_id=getattr(self, 'sub_task_id', None) if self.rank == 0 else None
        )

        metrics = list(self.fixedsolver.subiter_metrics)
        used_subiters = len(metrics)
        last_rec = metrics[-1] if metrics else None

        total_time = (
            float(self.fixedsolver.mech_time_total)
            + float(self.fixedsolver.dens_time_total)
        )

        if self.telemetry is not None:
            rss_mb_local = current_memory_mb()
            rss_mb_total = self.comm.allreduce(float(rss_mb_local), op=MPI.SUM)

            payload = {
                "step": step_index,
                "dt_days": float(dt),
                "tol": float(self.cfg.coupling_tol),
                "used_subiters": used_subiters,
                "mech_time_s": float(self.fixedsolver.mech_time_total),
                "dens_time_s": float(self.fixedsolver.dens_time_total),
                "solve_time_s_total": total_time,
                "num_dofs_total": self.num_dofs_total,
                "rss_mem_mb": rss_mb_total,
                "mech_iters": self.fixedsolver.mech_iters_total,
                "dens_iters": 1,
            }
            if time_days is not None:
                payload["time_days"] = float(time_days)
            if last_rec is not None:
                proj_val = last_rec.get("proj_res")
                if proj_val is not None:
                    payload["proj_res_last"] = float(proj_val)
            self.telemetry.record("steps", payload, csv_event=True)

        # Update rates for next step (Adams-Bashforth history)
        dt_curr = float(dt)
        
        # Shift history
        assign(self.rho_rate_last2, self.rho_rate_last)
        
        # Calculate new rates: (x_new - x_old) / dt
        self.rho_rate_last.x.array[:] = (self.rho.x.array - self.rho_old.x.array) / dt_curr
        
        self.rho_rate_last.x.scatter_forward()
        
        self.dt_prev = dt_curr
        self.step_count += 1

    def simulate(self, dt: float, total_time: float) -> None:
        """Run remodeling loop for ``total_time`` [days] with fixed ``dt`` [days]."""
        t = 0.0
        n_steps = int(np.ceil(total_time / dt))

        self.cfg.set_dt(float(dt))

        self.driver.setup()
        self.densolver.setup()
        self.solvers_initialized = True

        self.comm.Barrier()
        overall_start = MPI.Wtime()

        if self.rank == 0 and self.cfg.verbose == "progressbar":
            try:
                from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
                from rich.console import Console
                
                # Force terminal output to stderr to ensure visibility even if stdout is buffered/redirected
                console = Console(stderr=True, force_terminal=True)
                
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=60),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("{task.fields[info]}"),
                    console=console,
                    transient=False, # Keep the main progress bar visible
                )
                # Initialize with spaces to reserve width and prevent resizing
                self.main_task_id = self.progress.add_task("Remodeling", total=n_steps, info=" " * 35)
                self.sub_task_id = self.progress.add_task("  Coupling", total=self.cfg.max_subiters, info=" " * 35)
                self.progress.start()
            except ImportError:
                self.logger.warning("rich not installed, falling back to standard logging")
                self.logger.level = Level.INFO

        for step in range(n_steps):
            step_time = t + dt
            self.step(dt, step_index=step, time_days=step_time)
            
            t = step_time
            if (step + 1) % self.cfg.saving_interval == 0:
                coupling_stats = {
                    "iters": len(self.fixedsolver.subiter_metrics),
                    "time": float(self.fixedsolver.mech_time_total + self.fixedsolver.dens_time_total)
                }
                self._output(t, step, coupling_stats)
            
            if self.progress is not None and self.main_task_id is not None:
                self.progress.update(self.main_task_id, advance=1)

        if self.progress is not None:
            self.progress.stop()
            self.progress = None
            self.main_task_id = None
            self.sub_task_id = None

        self.comm.Barrier()
        overall_elapsed = MPI.Wtime() - overall_start
        overall_elapsed = self.comm.allreduce(float(overall_elapsed), op=MPI.MAX)

        if self.telemetry is not None:
            summary = {
                "status": "completed",
                "total_time_days": float(t),
                "wall_time_seconds": overall_elapsed,
                "steps_completed": n_steps,
            }
            self.telemetry.write_metadata(summary, filename="run_summary.json")

        if self.logger.is_enabled_for(Level.INFO):
            self.logger.info(f"Simulation completed in {overall_elapsed:.2f} s")

    def __enter__(self) -> "Remodeller":
        return self

    def __exit__(self, *_) -> None:
        self.close()
