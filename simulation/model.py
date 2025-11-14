"""Core remodeling model: orchestration of coupled subsolvers.

This module exposes a single high-level entry point, :class:`Remodeller`,
which wires the mechanics, stimulus, density and direction solvers together
for a given finite-element domain and :class:`simulation.config.Config`.

The module intentionally contains **no** executable CLI entry point and
performs no path hacks or implicit defaults. Callers are expected to:

- construct the mesh and associated facet tags explicitly,
- build a :class:`Config` with fully-specified parameters, and
- drive the time integration via :meth:`Remodeller.step` or
    :meth:`Remodeller.simulate`.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import numpy as np
from mpi4py import MPI
import basix
from dolfinx import fem
from dolfinx.fem import Function, functionspace

from simulation.storage import UnifiedStorage
from simulation.logger import get_logger, Level
from simulation.utils import build_dirichlet_bcs, assign, current_memory_mb
from simulation.config import Config
from simulation.febio_parser import FEBio2Dolfinx
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.fixedsolver import FixedPointSolver
from simulation.drivers import GaitEnergyDriver
from simulation.projector import L2Projector



class Remodeller:
    """High-level remodeling driver.

    The class is intentionally thin: it owns FE fields, subsolvers and
    storage/telemetry, but does not hide configuration or provide fallback
    behaviour. All required parameters must be supplied via :class:`Config`.
    """

    def __init__(self, cfg: Config):
        """Bind configuration, allocate fields and construct subsolvers.

        Parameters
        ----------
        cfg:
            Fully specified simulation configuration; in particular,
            ``cfg.domain`` and ``cfg.facet_tags`` must already be set and
            consistent.
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
                    "rhoJ_last",
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

        u = Function(self.V, name="u")
        self.rho = Function(self.Q, name="rho")
        self.rho_old = Function(self.Q, name="rho_old")

        self.A = Function(self.T, name="dir_tensor")
        self.A_old = Function(self.T, name="dir_tensor_old")

        self.S = Function(self.Q, name="stimulus")
        self.S_old = Function(self.Q, name="stimulus_old")

        self.scatter_fields: Tuple[fem.Function, ...] = ()

        assign(self.rho, self.cfg.rho0)

        d = self.gdim

        def _A_const(x):
            n = x.shape[1]
            vals = (np.eye(d, dtype=np.float64) / d).reshape(d * d, 1)
            return np.tile(vals, (1, n))

        self.A.interpolate(_A_const)
        self.A.x.scatter_forward()

        assign(self.S, 0.0)
        # Do not track or output displacement 'u' anymore
        self.scatter_fields = (self.rho, self.A, self.S)

        # Projector and field for strain–energy density (DG0 scalar)
        DG0 = basix.ufl.element("DG", self.domain.basix_cell(), 0)
        self.Q_energy = functionspace(self.domain, DG0)
        self.psi = Function(self.Q_energy, name="psi")
        self.energy_projector = L2Projector(self.Q_energy)

        # Register fields (omit 'u', add energy density)
        self.storage.fields.register("scalars", [self.rho, self.S, self.psi], filename="scalars.bp")
        self.storage.fields.register("A", [self.A], filename="A.bp")

        # Boundary conditions
        # Dirichlet BCs: fix distal end (tag 1)
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)

        # Create gait loader (None means use internal dummy loader)
        # This must happen before MechanicsSolver to provide traction functions
        gait_loader = self._create_gait_loader(None)
        
        # Neumann BCs: traction from gait loader on tags 2, 3, 4
        neumann_bcs = [
            (gait_loader.t_hip, 2),
            (gait_loader.t_glmed, 3),
            (gait_loader.t_glmax, 4),
        ]

        # Mechanics: boundary conditions and external loading
        self.mechsolver = MechanicsSolver(u, self.rho, self.A, self.cfg, bc_mech, neumann_bcs)

        self.stimsolver = StimulusSolver(self.S, self.S_old, self.cfg)
        self.densolver = DensitySolver(self.rho, self.rho_old, self.A, self.S, self.cfg)
        self.dirsolver = DirectionSolver(self.A, self.A_old, self.cfg)

        # Energy-driven driver with gait loader
        self.driver = GaitEnergyDriver(self.mechsolver, gait_loader, self.cfg)

        self.fixedsolver = FixedPointSolver(
            self.comm,
            self.cfg,
            self.driver,
            self.stimsolver,
            self.densolver,
            self.dirsolver,
            self.rho,
            self.rho_old,
            self.A,
            self.A_old,
            self.S,
            self.S_old,
        )


        # Iteration accounting window
        self.acc_steps = 0
        self.iters_snap = {
            "stim": {"iters": 0},
            "dens": {"iters": 0},
            "dir":  {"iters": 0},
            "gs":   {"iters": 0},
        }

        self.solvers_initialized = False

        # Storage bookkeeping
        self.last_solver_stats: Dict[str, int] = {"stim": 0, "dens": 0, "dir": 0}
        self.last_coupling_stats: Dict[str, float] = {"iters": 0, "time": 0.0}
        self.last_dt: Optional[float] = None
        self.last_step_index: Optional[int] = None

        # Persist initial configuration once.
        self.cfg.update_config_json()

        self._current_dt: Optional[float] = None

    def _create_gait_loader(self, gait_loader):
        """Create or wrap gait loader, ensuring it has required interface."""
        if gait_loader is None:
            from simulation.drivers import _DummyGaitLoader
            return _DummyGaitLoader(self.V)
        return gait_loader
        
    def close(self):
        """Release PETSc resources and close I/O."""
        if self.closed:
            return

        self.comm.Barrier()

        for attr in ("mechsolver", "stimsolver", "densolver", "dirsolver"):
            solver = getattr(self, attr, None)
            if solver is not None:
                solver.destroy()

        if self.storage is not None:
            self.storage.close()

        if self.telemetry is not None:
            self.telemetry.close()

        self.comm.Barrier()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _scatter_forward(self):
        """Ghost update: scatter all fields."""
        for f in self.scatter_fields:
            f.x.scatter_forward()

    def _field_minmax(self, field: fem.Function) -> Tuple[float, float]:
        """MPI global min/max."""
        if len(field.x.array) > 0:
            field_min_local = field.x.array.min()
            field_max_local = field.x.array.max()
        else:
            field_min_local = float("inf")
            field_max_local = float("-inf")
        field_min = self.comm.allreduce(field_min_local, op=MPI.MIN)
        field_max = self.comm.allreduce(field_max_local, op=MPI.MAX)
        return field_min, field_max

    def _reset_iters_window(self) -> None:
        """Snapshot cumulative iteration counters for statistics window."""
        self.iters_snap["stim"]["iters"] = getattr(self.stimsolver, "total_iters", 0)
        self.iters_snap["dens"]["iters"] = getattr(self.densolver,  "total_iters", 0)
        self.iters_snap["dir"]["iters"]  = getattr(self.dirsolver,  "total_iters", 0)
        self.iters_snap["gs"]["iters"]   = getattr(self.fixedsolver, "total_gs_iters", 0)
        self.acc_steps = 0

    def _iters_window_stats(self) -> Dict[str, float]:
        """Avg KSP iterations per GS iteration in current window."""
        steps = max(self.acc_steps, 1)
        d_gs = getattr(self.fixedsolver, "total_gs_iters", 0) - self.iters_snap["gs"]["iters"]

        def per_gs(solver, key: str) -> float:
            d_iters = getattr(solver, "total_iters", 0) - self.iters_snap[key]["iters"]
            return (d_iters / d_gs) if d_gs > 0 else 0.0

        stim = per_gs(self.stimsolver, "stim")
        dens = per_gs(self.densolver,  "dens")
        ddir = per_gs(self.dirsolver,  "dir")
        gs_per_step = (d_gs / steps) if steps > 0 else 0.0
        return dict(stim_gs=stim, dens_gs=dens, dir_gs=ddir, gs_per_step=gs_per_step)

    def _collect_field_stats(self) -> Dict[str, float]:
        """Gather field min/max and energy for reporting."""
        rho_min, rho_max = self._field_minmax(self.rho)
        S_min, S_max = self._field_minmax(self.S)
        return dict(rho_min=rho_min, rho_max=rho_max, S_min=S_min, S_max=S_max)

    def _is_output_step(self, step: int) -> bool:
        return (step + 1) % self.cfg.saving_interval == 0

    def _output(self, t: float, step: int):
        """Scatter, stats, log, write (saving_interval steps)."""
        self._scatter_forward()

        # Update projected strain–energy density field before output
        psi_expr = self.driver.energy_expr()
        psi_proj = self.energy_projector.project(psi_expr)
        self.psi.x.array[:] = psi_proj.x.array
        self.psi.x.scatter_forward()

        iters = self._iters_window_stats()
        fields = self._collect_field_stats()

        self.logger.info(
            lambda: (
                f"Step {step:2d} | t={t:6.1f}d | "
                f"ρ=[{fields['rho_min']:.0f},{fields['rho_max']:.0f}] | "
                f"S=[{fields['S_min']:.2e},{fields['S_max']:.2e}] | "
                f"stim={iters['stim_gs']:.1f} | "
                f"dens={iters['dens_gs']:.1f} | dir={iters['dir_gs']:.1f} | "
                f"GS={iters['gs_per_step']:.1f}"
            )
        )

        solver_stats = {
            "stim": int(self.last_solver_stats.get("stim", 0)),
            "dens": int(self.last_solver_stats.get("dens", 0)),
            "dir": int(self.last_solver_stats.get("dir", 0)),
        }

        coupling_stats = {
            "iters": int(self.last_coupling_stats.get("iters", 0)),
            "time": float(self.last_coupling_stats.get("time", 0.0)),
        }

        dt_days = float(self.last_dt) if self.last_dt is not None else 0.0

        # Total DOFs and memory
        dofs_V = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
        dofs_Q = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
        dofs_T = self.T.dofmap.index_map.size_global * self.T.dofmap.index_map_bs
        num_dofs_total = int(dofs_V + dofs_Q + dofs_T)

        rss_mb_local = current_memory_mb()
        rss_mb_total = self.comm.allreduce(float(rss_mb_local), op=MPI.SUM)

        self.storage.write_step(
            step=step,
            time_days=float(t),
            dt_days=dt_days,
            num_dofs_total=num_dofs_total,
            rss_mem_mb=rss_mb_total,
            solver_stats=solver_stats,
            coupling_stats=coupling_stats,
        )

        self._reset_iters_window()

    def step(self, dt: float, *, step_index: Optional[int] = None, time_days: Optional[float] = None) -> None:
        """Single timestep: fixed-point iteration until coupling tolerance met."""
        assign(self.rho_old, self.rho)
        assign(self.A_old, self.A)
        assign(self.S_old, self.S)

        self.last_dt = float(dt)
        self.last_step_index = step_index

        solver_totals_before = {
            "stim": getattr(self.stimsolver, "total_iters", 0),
            "dens": getattr(self.densolver, "total_iters", 0),
            "dir": getattr(self.dirsolver, "total_iters", 0),
        }
        coupling_iters_before = getattr(self.fixedsolver, "total_gs_iters", 0)

        if not self.solvers_initialized:
            self.mechsolver.setup()
            self.stimsolver.setup()
            self.densolver.setup()
            self.dirsolver.setup()
            self.solvers_initialized = True


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

        solver_stats = {
            "stim": max(int(getattr(self.stimsolver, "total_iters", 0) - solver_totals_before["stim"]), 0),
            "dens": max(int(getattr(self.densolver, "total_iters", 0) - solver_totals_before["dens"]), 0),
            "dir": max(int(getattr(self.dirsolver, "total_iters", 0) - solver_totals_before["dir"]), 0),
        }
        self.last_solver_stats = solver_stats

        coupling_iters = int(getattr(self.fixedsolver, "total_gs_iters", 0) - coupling_iters_before)
        if coupling_iters <= 0:
            coupling_iters = used_subiters

        total_time = (
            float(self.fixedsolver.mech_time_total)
            + float(self.fixedsolver.stim_time_total)
            + float(self.fixedsolver.dens_time_total)
            + float(self.fixedsolver.dir_time_total)
        )

        self.last_coupling_stats = {"iters": coupling_iters, "time": total_time}

        if self.telemetry is not None:
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
            }
            if time_days is not None:
                payload["time_days"] = float(time_days)
            if last_rec is not None:
                proj_val = last_rec.get("proj_res")
                if proj_val is not None:
                    payload["proj_res_last"] = float(proj_val)
                rhoj_val = last_rec.get("rhoJ")
                if rhoj_val is not None:
                    payload["rhoJ_last"] = float(rhoj_val)
            self.telemetry.record("steps", payload, csv_event=True)

    def _print_final_summary(self, num_steps: int, overall_elapsed: float,
                            mech_times: List[float], stim_times: List[float],
                            dens_times: List[float], dir_times: List[float]):
        """Log median timing per step and write run_summary.json."""
        avg_mech = float(np.median(mech_times)) if mech_times else 0.0
        avg_stim = float(np.median(stim_times)) if stim_times else 0.0
        avg_dens = float(np.median(dens_times)) if dens_times else 0.0
        avg_dir  = float(np.median(dir_times))  if dir_times  else 0.0
        avg_total = avg_mech + avg_stim + avg_dens + avg_dir

        if self.logger.is_enabled_for(Level.INFO):
            lines = [
                f"Total steps completed    : {num_steps:6d}",
                f"Overall wall time        : {overall_elapsed:10.3f} s",
                "-" * 60,
                "Typical timing per step (median):",
                f"  Mechanics solver       : {avg_mech:10.6f} s",
                f"  Stimulus solver        : {avg_stim:10.6f} s",
                f"  Density solver         : {avg_dens:10.6f} s",
                f"  Direction solver       : {avg_dir:10.6f} s",
                f"  Total per step         : {avg_total:10.6f} s",
                "-" * 60,
            ]
            for line in lines:
                self.logger.info(line)

        if self.telemetry is not None:
            fields = self._collect_field_stats()
            dofs_V = self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs
            dofs_Q = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
            dofs_T = self.T.dofmap.index_map.size_global * self.T.dofmap.index_map_bs

            num_dofs_total = int(dofs_V + dofs_Q + dofs_T)

            rss_mb_local = current_memory_mb()
            rss_mb_sum = self.comm.allreduce(float(rss_mb_local), op=MPI.SUM)
            rss_mb_max = self.comm.allreduce(float(rss_mb_local), op=MPI.MAX)
            data = {
                "num_steps": int(num_steps),
                "overall_wall_time_s": float(overall_elapsed),
                "median_step_times_s": {
                    "mech": avg_mech,
                    "stim": avg_stim,
                    "dens": avg_dens,
                    "dir": avg_dir,
                    "total": avg_total,
                },
                "dofs": {
                    "V": int(dofs_V),
                    "Q": int(dofs_Q),
                    "T": int(dofs_T),
                    "total": int(num_dofs_total),
                },
                "memory_mb": {
                    "rss_sum": float(rss_mb_sum),
                    "rss_max": float(rss_mb_max),
                },
                "final_field_stats": {
                    "rho_min": float(fields.get("rho_min", 0.0)),
                    "rho_max": float(fields.get("rho_max", 0.0)),
                    "S_min": float(fields.get("S_min", 0.0)),
                    "S_max": float(fields.get("S_max", 0.0)),
                },
            }
            self.telemetry.write_metadata(data, filename="run_summary.json", overwrite=True)

    def simulate(self, dt: float, total_time: float) -> None:
        """Run remodeling loop for ``total_time`` [days] with fixed ``dt`` [days]."""
        t = 0.0
        n_steps = int(np.ceil(total_time / dt))

        # Set timestep in days once and assemble static operators
        self.cfg.set_dt(float(dt))

        self.mechsolver.setup()
        self.stimsolver.setup()
        self.densolver.setup()
        self.dirsolver.setup()
        self.solvers_initialized = True

        self._reset_iters_window()

        self.comm.Barrier()
        overall_start = MPI.Wtime()
        mech_times: List[float] = []
        stim_times: List[float] = []
        dens_times: List[float] = []
        dir_times: List[float] = []

        for step in range(n_steps):
            step_time = t + dt
            self.step(dt, step_index=step, time_days=step_time)
            self.acc_steps += 1

            mech_times.append(float(self.fixedsolver.mech_time_total))
            stim_times.append(float(self.fixedsolver.stim_time_total))
            dens_times.append(float(self.fixedsolver.dens_time_total))
            dir_times.append(float(self.fixedsolver.dir_time_total))

            t = step_time
            if self._is_output_step(step):
                self._output(t, step)

        self.comm.Barrier()
        overall_elapsed = MPI.Wtime() - overall_start
        overall_elapsed = self.comm.allreduce(float(overall_elapsed), op=MPI.MAX)

        self._print_final_summary(n_steps, overall_elapsed, mech_times, stim_times, dens_times, dir_times)
