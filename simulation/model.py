"""Core remodeling model: orchestration of coupled subsolvers.

This module exposes a single high-level entry point, :class:`Remodeller`,
which wires the mechanics, stimulus, density and direction solvers together
for a given finite-element domain and :class:`simulation.config.Config`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from mpi4py import MPI
import basix
from dolfinx import fem
from dolfinx.fem import Function, functionspace

from simulation.storage import UnifiedStorage
from simulation.logger import get_logger, Level
from simulation.utils import build_dirichlet_bcs, assign, current_memory_mb
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.fixedsolver import FixedPointSolver
from simulation.drivers import GaitDriver


class Remodeller:
    """High-level bone remodeling orchestrator.
    
    Owns FE fields, subsolvers, and storage. Requires fully specified Config.
    """

    def __init__(self, cfg: Config):
        """Initialize remodeler with configuration.
        
        Parameters
        ----------
        cfg : Config
            Complete simulation configuration with domain and facet_tags.
        """
        self.cfg = cfg
        self.domain = self.cfg.domain
        self.closed = False

        self.verbose = bool(self.cfg.verbose)
        self.comm = self.domain.comm
        self.rank = self.comm.rank
        self.logger = get_logger(self.comm, verbose=self.verbose, name="Remodeller")

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
                    "stim_time_s",
                    "dens_time_s",
                    "dir_time_s",
                    "solve_time_s_total",
                    "proj_res_last",
                    "num_dofs_total",
                    "rss_mem_mb",
                    "mech_iters",
                    "stim_iters",
                    "dens_iters",
                    "dir_iters",
                ],
                filename="steps.csv",
            )

        self.dx = self.cfg.dx
        self.ds = self.cfg.ds
        self.gdim = self.domain.geometry.dim

        P1_vec = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim,))
        P1 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim, self.gdim))

        self.V = functionspace(self.domain, P1_vec)
        self.Q = functionspace(self.domain, P1)
        self.T = functionspace(self.domain, P1_ten)

        # Total DOFs
        dofs_V = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        dofs_Q = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
        dofs_T = self.T.dofmap.index_map.size_global * self.T.dofmap.index_map_bs
        self.num_dofs_total = int(dofs_V + dofs_Q + dofs_T)

        u = Function(self.V, name="u")
        self.rho = Function(self.Q, name="rho")
        self.rho_old = Function(self.Q, name="rho_old")

        self.L = Function(self.T, name="log_dir_tensor")
        self.L_old = Function(self.T, name="log_dir_tensor_old")

        self.S = Function(self.Q, name="stimulus")
        self.S_old = Function(self.Q, name="stimulus_old")

        # Predictor state (Adams-Bashforth)
        self.step_count = 0
        self.dt_prev: Optional[float] = None
        
        self.rho_rate_last = Function(self.Q, name="rho_rate_last")
        self.rho_rate_last2 = Function(self.Q, name="rho_rate_last2")
        
        self.S_rate_last = Function(self.Q, name="S_rate_last")
        self.S_rate_last2 = Function(self.Q, name="S_rate_last2")
        
        self.L_rate_last = Function(self.T, name="L_rate_last")
        self.L_rate_last2 = Function(self.T, name="L_rate_last2")
        
        assign(self.rho_rate_last, 0.0)
        assign(self.rho_rate_last2, 0.0)
        assign(self.S_rate_last, 0.0)
        assign(self.S_rate_last2, 0.0)
        assign(self.L_rate_last, 0.0)
        assign(self.L_rate_last2, 0.0)

        assign(self.rho, self.cfg.rho0)

        d = self.gdim

        def _L_init(x):
            # Initialize L = log(A). A is approx I/3.
            # We perturb eigenvalues to avoid singularity in spectral decomposition.
            # A ~ diag(1/3, 1/3, 1/3) -> L ~ diag(ln(1/3), ln(1/3), ln(1/3))
            # Perturbation: 1e-4
            n = x.shape[1]
            
            # Base value: ln(1/3)
            val = np.log(1.0/3.0)
            
            # Perturbations
            eps = 1e-4
            # diag = [val - eps, val, val + eps]
            # This ensures distinct eigenvalues for L, and thus for A = exp(L).
            
            vals = np.zeros((d*d, n), dtype=np.float64)
            
            # 3D case assumed
            if d == 3:
                # L11
                vals[0, :] = val - eps
                # L22
                vals[4, :] = val
                # L33
                vals[8, :] = val + eps
            elif d == 2:
                vals[0, :] = val - eps
                vals[3, :] = val + eps
                
            return vals

        self.L.interpolate(_L_init)
        self.L.x.scatter_forward()

        assign(self.S, 0.0)

        # Register fields
        self.storage.fields.register("scalars", [self.rho, self.S], filename="scalars.bp")
        self.storage.fields.register("L", [self.L], filename="L.bp")

        # Boundary conditions
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)

        # Create gait loader
        from simulation.femur_gait import setup_femur_gait_loading
        gait_loader = setup_femur_gait_loading(
            self.V,
            mass_tonnes=float(self.cfg.body_mass_tonnes),
            n_samples=int(self.cfg.gait_samples),
            load_scale=float(self.cfg.load_scale),
        )
        
        neumann_bcs = [
            (gait_loader.t_hip, 2),
            (gait_loader.t_glmed, 2),
            (gait_loader.t_glmax, 2),
        ]

        # Subsolvers
        mechsolver = MechanicsSolver(u, self.rho, self.L, self.cfg, bc_mech, neumann_bcs)
        self.stimsolver = StimulusSolver(self.S, self.S_old, self.cfg)
        self.densolver = DensitySolver(self.rho, self.rho_old, self.L, self.S, self.cfg)
        self.dirsolver = DirectionSolver(self.L, self.L_old, self.cfg)

        # Driver
        self.driver = GaitDriver(mechsolver, gait_loader, self.cfg)

        self.fixedsolver = FixedPointSolver(
            self.comm,
            self.cfg,
            self.driver,
            self.stimsolver,
            self.densolver,
            self.dirsolver,
            self.rho,
            self.rho_old,
            self.L,
            self.L_old,
            self.S,
            self.S_old,
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

        for attr in ("stimsolver", "densolver", "dirsolver"):
            solver = getattr(self, attr, None)
            if solver is not None:
                solver.destroy()

        if self.storage is not None:
            self.storage.close()

        if self.telemetry is not None:
            self.telemetry.close()

        self.comm.Barrier()
        self.closed = True

    def _field_stats(self, field: fem.Function) -> Tuple[float, float, float]:
        """MPI global min/max/mean."""
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
        local_sum = np.sum(field.x.array[:local_size])
        local_count = local_size

        global_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        global_count = self.comm.allreduce(local_count, op=MPI.SUM)
        field_mean = global_sum / global_count if global_count > 0 else 0.0

        return field_min, field_max, field_mean

    def _collect_field_stats(self) -> Dict[str, float]:
        """Gather field min/max/mean and energy for reporting."""
        rho_min, rho_max, rho_mean = self._field_stats(self.rho)
        S_min, S_max, S_mean = self._field_stats(self.S)

        psi_avg = 0.0
        if hasattr(self.driver, "_last_stats") and self.driver._last_stats:
            psi_avg = self.driver._last_stats.get("psi_avg", 0.0)

        return dict(
            rho_min=rho_min, rho_max=rho_max, rho_mean=rho_mean,
            S_min=S_min, S_max=S_max, S_mean=S_mean,
            psi_avg=psi_avg
        )

    def _output(self, t: float, step: int, coupling_stats: Dict[str, float]):
        """Scatter, stats, log, write."""
        fields = self._collect_field_stats()

        self.logger.info(
            lambda: (
                f"Step {step:2d} | t={t:6.1f}d | "
                f"ρ=[{fields['rho_min']:.3f},{fields['rho_max']:.3f}] (μ={fields['rho_mean']:.3f}) | "
                f"S=[{fields['S_min']:.2e},{fields['S_max']:.2e}] (μ={fields['S_mean']:.2e}) | "
                f"ψ_avg={fields['psi_avg']:.3e} | "
                f"GS={coupling_stats['iters']}"
            )
        )

        self.storage.write_fields("scalars", float(t))
        self.storage.write_fields("L", float(t))

    def step(self, dt: float, *, step_index: Optional[int] = None, time_days: Optional[float] = None) -> None:
        """Single timestep: fixed-point iteration until coupling tolerance met."""
        assign(self.rho_old, self.rho)
        assign(self.L_old, self.L)
        assign(self.S_old, self.S)

        if not self.solvers_initialized:
            self.driver.setup()
            self.stimsolver.setup()
            self.densolver.setup()
            self.dirsolver.setup()
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
                self.S.x.array[:] = self.S_old.x.array + dt_curr * (
                    w1 * self.S_rate_last.x.array - w2 * self.S_rate_last2.x.array
                )
                self.L.x.array[:] = self.L_old.x.array + dt_curr * (
                    w1 * self.L_rate_last.x.array - w2 * self.L_rate_last2.x.array
                )
            else:
                # AB1 (Forward Euler)
                self.rho.x.array[:] = self.rho_old.x.array + dt_curr * self.rho_rate_last.x.array
                self.S.x.array[:] = self.S_old.x.array + dt_curr * self.S_rate_last.x.array
                self.L.x.array[:] = self.L_old.x.array + dt_curr * self.L_rate_last.x.array
            
            self.rho.x.scatter_forward()
            self.S.x.scatter_forward()
            self.L.x.scatter_forward()

        # Update dt [days] and reassemble LHS for time-dependent solvers if dt changed
        if self._current_dt is None or abs(float(dt) - float(self._current_dt)) > 1e-12:
            self.cfg.set_dt(float(dt))
            if self.solvers_initialized:
                self.stimsolver.assemble_lhs()
                self.densolver.assemble_lhs()
                self.dirsolver.assemble_lhs()
            self._current_dt = float(dt)
        
        self.fixedsolver.run(time_days=time_days, step_index=step_index)

        metrics = list(self.fixedsolver.subiter_metrics)
        used_subiters = len(metrics)
        last_rec = metrics[-1] if metrics else None

        total_time = (
            float(self.fixedsolver.mech_time_total)
            + float(self.fixedsolver.stim_time_total)
            + float(self.fixedsolver.dens_time_total)
            + float(self.fixedsolver.dir_time_total)
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
                "stim_time_s": float(self.fixedsolver.stim_time_total),
                "dens_time_s": float(self.fixedsolver.dens_time_total),
                "dir_time_s": float(self.fixedsolver.dir_time_total),
                "solve_time_s_total": total_time,
                "num_dofs_total": self.num_dofs_total,
                "rss_mem_mb": rss_mb_total,
                "mech_iters": self.fixedsolver.mech_iters_total,
                "stim_iters": 1,
                "dens_iters": 1,
                "dir_iters": 1,
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
        assign(self.S_rate_last2, self.S_rate_last)
        assign(self.L_rate_last2, self.L_rate_last)
        
        # Calculate new rates: (x_new - x_old) / dt
        self.rho_rate_last.x.array[:] = (self.rho.x.array - self.rho_old.x.array) / dt_curr
        self.S_rate_last.x.array[:] = (self.S.x.array - self.S_old.x.array) / dt_curr
        self.L_rate_last.x.array[:] = (self.L.x.array - self.L_old.x.array) / dt_curr
        
        self.rho_rate_last.x.scatter_forward()
        self.S_rate_last.x.scatter_forward()
        self.L_rate_last.x.scatter_forward()
        
        self.dt_prev = dt_curr
        self.step_count += 1

    def simulate(self, dt: float, total_time: float) -> None:
        """Run remodeling loop for ``total_time`` [days] with fixed ``dt`` [days]."""
        t = 0.0
        n_steps = int(np.ceil(total_time / dt))

        self.cfg.set_dt(float(dt))

        self.driver.setup()
        self.stimsolver.setup()
        self.densolver.setup()
        self.dirsolver.setup()
        self.solvers_initialized = True

        self.comm.Barrier()
        overall_start = MPI.Wtime()

        for step in range(n_steps):
            step_time = t + dt
            self.step(dt, step_index=step, time_days=step_time)
            
            t = step_time
            if (step + 1) % self.cfg.saving_interval == 0:
                coupling_stats = {
                    "iters": len(self.fixedsolver.subiter_metrics),
                    "time": float(self.fixedsolver.mech_time_total + self.fixedsolver.stim_time_total + self.fixedsolver.dens_time_total + self.fixedsolver.dir_time_total)
                }
                self._output(t, step, coupling_stats)

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
