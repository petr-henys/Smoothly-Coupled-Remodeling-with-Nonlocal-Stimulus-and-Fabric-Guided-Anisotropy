import sys
from pathlib import Path

# Add parent directory to sys.path to enable absolute imports when running as script
# This allows: cd simulation && python3 model.py
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from mpi4py import MPI
import basix

from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
from typing import Dict, Tuple, List, Optional

from simulation.storage import UnifiedStorage
from simulation.logger import get_logger, Level

from simulation.utils import build_dirichlet_bcs, build_facetag, assign, current_memory_mb
from simulation.config import Config
from simulation.subsolvers import (MechanicsSolver, StimulusSolver,
                        DensitySolver, DirectionSolver)

from simulation.fixedsolver import FixedPointSolver

# Track meshes already scaled by L_c (per-process)
_SCALED_MESH_IDS: set[int] = set()


class Remodeller:
    def __init__(self, cfg: Config):
        """Driver class orchestrating the coupled bone remodeling simulation."""
        self.cfg = cfg
        self.domain = self.cfg.domain
        # Idempotent close guard
        self._closed = False

        # Verbosity / logging
        self.verbose = bool(getattr(self.cfg, "verbose", True))
        self.comm = self.domain.comm
        self.rank = self.comm.rank
        self.logger = get_logger(self.comm, verbose=self.verbose, name="Remodeller")

        # Unified storage system
        self.storage = UnifiedStorage(cfg)

        self.telemetry = getattr(self.cfg, "telemetry", None)
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
                    "rel_change_last",
                    "proj_res_last",
                    "rhoJ_last",
                ],
                gz=False,
                filename="steps.csv",
            )

        # ND geometry scaling (scale once per mesh object)
        if float(self.cfg.L_c) != 1.0:
            mid = id(self.domain)
            if mid not in _SCALED_MESH_IDS:
                self.domain.geometry.x[:] /= self.cfg.L_c
                _SCALED_MESH_IDS.add(mid)

        self.dx = self.cfg.dx
        self.ds = self.cfg.ds
        self.gdim = self.domain.geometry.dim

        P1_vec = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim,))
        P1 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1, shape=(self.gdim, self.gdim))

        self.V = functionspace(self.domain, P1_vec)
        self.Q = functionspace(self.domain, P1)
        self.T = functionspace(self.domain, P1_ten)

        self.u = Function(self.V, name="u")
        self.rho = Function(self.Q, name="rho")
        self.rho_old = Function(self.Q, name="rho_old")

        self.A = Function(self.T, name="dir_tensor")
        self.A_old = Function(self.T, name="dir_tensor_old")

        self.S = Function(self.Q, name="stimulus")
        self.S_old = Function(self.Q, name="stimulus_old")

        # Keep references of fields needing scatter to avoid manual repetition
        self._scatter_fields: Tuple[fem.Function, ...] = ()

        assign(self.rho, self.cfg.rho0 / self.cfg.rho_c)

        d = self.gdim

        def _A_const(x):
            n = x.shape[1]
            vals = (np.eye(d, dtype=np.float64) / d).reshape(d * d, 1)
            return np.tile(vals, (1, n))

        self.A.interpolate(_A_const)
        self.A.x.scatter_forward()

        assign(self.S, 0.0)
        # Initialize scatter field tuple once all functions are created
        self._scatter_fields = (self.u, self.rho, self.A, self.S)

        # Register field output groups with unified storage
        self.storage.fields.register_group("u", [self.u], filename="u.bp")
        self.storage.fields.register_group("scalars", [self.rho, self.S], filename="scalars.bp")
        self.storage.fields.register_group("A", [self.A], filename="A.bp")

        # left fixed
        bc_mech = build_dirichlet_bcs(self.V, self.cfg.facet_tags, id_tag=1, value=0.0)

        # right compression load (dim-agnostic)
        t_vec = np.zeros(self.gdim, dtype=np.float64)
        t_vec[0] = -self.cfg.t_p / self.cfg.sigma_c
        traction = (fem.Constant(self.domain, t_vec), 2)

        self.mechsolver = MechanicsSolver(self.V, self.rho, self.A, bc_mech, [traction], self.cfg)
        self.stimsolver = StimulusSolver(self.Q, self.S_old, self.cfg)
        self.densolver = DensitySolver(self.Q, self.rho_old, self.A, self.S, self.cfg)
        self.dirsolver = DirectionSolver(self.T, self.A_old, self.cfg)

        self.fixedsolver = FixedPointSolver(
            self.comm, self.cfg,
            self.mechsolver, self.stimsolver, self.densolver, self.dirsolver,
            self.u, self.rho, self.rho_old, self.A, self.A_old, self.S, self.S_old
        )

        # --- iteration accounting window ---
        # KSP per-step numbers are averaged by the number of GS iterations in the window.
        self._acc_steps = 0
        self._iters_snap = {
            "mech": {"iters": 0},
            "stim": {"iters": 0},
            "dens": {"iters": 0},
            "dir":  {"iters": 0},
            "gs":   {"iters": 0},  # cumulative GS sweeps
        }

        # Flag to ensure solvers are initialized when using step() directly
        self._solvers_initialized = False

        # Storage bookkeeping for the last completed external step
        self._last_solver_stats: Dict[str, int] = {"mech": 0, "stim": 0, "dens": 0, "dir": 0}
        self._last_coupling_stats: Dict[str, float] = {"iters": 0, "time": 0.0}
        self._last_dt: Optional[float] = None
        self._last_step_index: Optional[int] = None

        # Update config.json to reflect any parameter changes made before Remodeller creation
        self.cfg.update_config_json()

    # ---------- lifecycle/context management ----------
    def close(self):
        """Deterministically free PETSc resources and close I/O."""
        if getattr(self, "_closed", False):
            return

        self.comm.Barrier()

        # Destroy PETSc/KSP/Mat/Vec held by linear solvers
        for attr in ("mechsolver", "stimsolver", "densolver", "dirsolver"):
            solver = getattr(self, attr, None)
            if solver is not None and hasattr(solver, "destroy"):
                solver.destroy()

        # Close unified storage (collective operation)
        if hasattr(self, "storage") and self.storage is not None:
            self.storage.close()

        # Close telemetry resources
        if self.telemetry is not None:
            self.telemetry.close()

        self.comm.Barrier()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ---------- internal helpers ----------
    def _scatter_forward(self):
        """Scatter all registered field vectors forward (halo update)."""
        for f in self._scatter_fields:
            f.x.scatter_forward()

    def _field_minmax(self, field: fem.Function) -> Tuple[float, float]:
        """Parallel min/max using MPI reductions."""
        if len(field.x.array) > 0:
            field_min_local = field.x.array.min()
            field_max_local = field.x.array.max()
        else:
            field_min_local = float("inf")
            field_max_local = float("-inf")
        field_min = self.comm.allreduce(field_min_local, op=MPI.MIN)
        field_max = self.comm.allreduce(field_max_local, op=MPI.MAX)
        return field_min, field_max

    # ---- iteration window tracking ----
    def _reset_iters_window(self) -> None:
        """Start a new accounting window from current cumulative counters."""
        self._iters_snap["mech"]["iters"] = getattr(self.mechsolver, "total_iters", 0)
        self._iters_snap["stim"]["iters"] = getattr(self.stimsolver, "total_iters", 0)
        self._iters_snap["dens"]["iters"] = getattr(self.densolver,  "total_iters", 0)
        self._iters_snap["dir"]["iters"]  = getattr(self.dirsolver,  "total_iters", 0)
        self._iters_snap["gs"]["iters"]   = getattr(self.fixedsolver, "total_gs_iters", 0)
        self._acc_steps = 0

    def _iters_window_stats(self) -> Dict[str, float]:
        """
        KSP metrics: average KSP iterations **per one GS iteration** in the current window.
        GS metric : average number of GS iterations **per one external time step** in the window.
        """
        steps = max(self._acc_steps, 1)
        d_gs = getattr(self.fixedsolver, "total_gs_iters", 0) - self._iters_snap["gs"]["iters"]

        def per_gs(solver, key: str) -> float:
            d_iters = getattr(solver, "total_iters", 0) - self._iters_snap[key]["iters"]
            return (d_iters / d_gs) if d_gs > 0 else 0.0

        mech = per_gs(self.mechsolver, "mech")
        stim = per_gs(self.stimsolver, "stim")
        dens = per_gs(self.densolver,  "dens")
        ddir = per_gs(self.dirsolver,  "dir")
        gs_per_step = (d_gs / steps) if steps > 0 else 0.0

        return dict(mech_gs=mech, stim_gs=stim, dens_gs=dens, dir_gs=ddir, gs_per_step=gs_per_step)

    # ---- stats collection & output ----
    def _collect_field_stats(self) -> Dict[str, float]:
        """Gather frequently reported field statistics (no iteration math here)."""
        rho_min, rho_max = self._field_minmax(self.rho)
        u_min, u_max = self._field_minmax(self.u)
        S_min, S_max = self._field_minmax(self.S)
        psi_avg_dim = self.mechsolver.average_strain_energy(self.u)

        return dict(
            rho_min=rho_min * self.cfg.rho_c,
            rho_max=rho_max * self.cfg.rho_c,
            u_min=u_min * self.cfg.u_c * 1e3,
            u_max=u_max * self.cfg.u_c * 1e3,
            S_min=S_min,
            S_max=S_max,
            psi=psi_avg_dim,
        )

    def _is_output_step(self, step: int) -> bool:
        return (step + 1) % self.cfg.saving_interval == 0

    def _output(self, t: float, step: int):
        """Single place to scatter, collect, print and write (called only on output steps)."""
        # One scatter for both print and write
        self._scatter_forward()

        # Collect stats
        iters = self._iters_window_stats()
        fields = self._collect_field_stats()

        self.logger.info(
            lambda: (
                f"Step {step:2d} | t={t:6.1f}d | "
                f"ρ=[{fields['rho_min']:.0f},{fields['rho_max']:.0f}] | "
                f"u=[{fields['u_min']:.2e},{fields['u_max']:.2e}] | "
                f"S=[{fields['S_min']:.2e},{fields['S_max']:.2e}] | "
                f"ψ={fields['psi']:.1f} | "
                f"mech={iters['mech_gs']:.1f} | stim={iters['stim_gs']:.1f} | "
                f"dens={iters['dens_gs']:.1f} | dir={iters['dir_gs']:.1f} | "
                f"GS={iters['gs_per_step']:.1f}"
            )
        )


        solver_stats = {
            "mech": int(self._last_solver_stats.get("mech", 0)),
            "stim": int(self._last_solver_stats.get("stim", 0)),
            "dens": int(self._last_solver_stats.get("dens", 0)),
            "dir": int(self._last_solver_stats.get("dir", 0)),
        }

        coupling_stats = {
            "iters": int(self._last_coupling_stats.get("iters", 0)),
            "time": float(self._last_coupling_stats.get("time", 0.0)),
        }

        dt_days = float(self._last_dt) if self._last_dt is not None else 0.0

        # Per-step num DOFs (constant per run) and RSS memory (rank-summed MB)
        # Report total scalar DOFs (account for block size of vector/tensor spaces)
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
            u=self.u,
            rho=self.rho,
            S=self.S,
            A=self.A,
            num_dofs_total=num_dofs_total,
            rss_mem_mb=rss_mb_total,
            solver_stats=solver_stats,
            coupling_stats=coupling_stats,
        )

        # Start a fresh accounting window
        self._reset_iters_window()

    def step(
        self,
        dt: float,
        *,
        step_index: Optional[int] = None,
        time_days: Optional[float] = None,
    ) -> None:
        """Single external time step delegating inner coupling iterations."""
        assign(self.rho_old, self.rho)
        assign(self.A_old, self.A)
        assign(self.S_old, self.S)

        self._last_dt = float(dt)
        self._last_step_index = step_index

        solver_totals_before = {
            "mech": getattr(self.mechsolver, "total_iters", 0),
            "stim": getattr(self.stimsolver, "total_iters", 0),
            "dens": getattr(self.densolver, "total_iters", 0),
            "dir": getattr(self.dirsolver, "total_iters", 0),
        }
        coupling_iters_before = getattr(self.fixedsolver, "total_gs_iters", 0)

        if not self._solvers_initialized:
            self.mechsolver.solver_setup()
            self.stimsolver.solver_setup()
            self.densolver.solver_setup()
            self.dirsolver.solver_setup()
            self._solvers_initialized = True

        # Propagate external step index so subiteration telemetry uses the same label
        self.fixedsolver.run(time_days=time_days, step_index=step_index)

        metrics = list(self.fixedsolver.subiter_metrics)
        used_subiters = len(metrics)
        last_rec = metrics[-1] if metrics else None

        solver_stats = {
            "mech": max(int(getattr(self.mechsolver, "total_iters", 0) - solver_totals_before["mech"]), 0),
            "stim": max(int(getattr(self.stimsolver, "total_iters", 0) - solver_totals_before["stim"]), 0),
            "dens": max(int(getattr(self.densolver, "total_iters", 0) - solver_totals_before["dens"]), 0),
            "dir": max(int(getattr(self.dirsolver, "total_iters", 0) - solver_totals_before["dir"]), 0),
        }
        self._last_solver_stats = solver_stats

        coupling_iters = int(getattr(self.fixedsolver, "total_gs_iters", 0) - coupling_iters_before)
        if coupling_iters <= 0:
            coupling_iters = used_subiters

        total_time = (
            float(self.fixedsolver.mech_time_total)
            + float(self.fixedsolver.stim_time_total)
            + float(self.fixedsolver.dens_time_total)
            + float(self.fixedsolver.dir_time_total)
        )

        self._last_coupling_stats = {"iters": coupling_iters, "time": total_time}

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
                rel_val = last_rec.get("rel_change")
                if rel_val is not None:
                    payload["rel_change_last"] = float(rel_val)
                proj_val = last_rec.get("proj_res")
                if proj_val is not None:
                    payload["proj_res_last"] = float(proj_val)
                rhoj_val = last_rec.get("rhoJ")
                if rhoj_val is not None:
                    payload["rhoJ_last"] = float(rhoj_val)
            self.telemetry.record("steps", payload, csv_event=True)

    def _print_final_summary(
        self,
        num_steps: int,
        overall_elapsed: float,
        mech_times: List[float],
        stim_times: List[float],
        dens_times: List[float],
        dir_times: List[float],
    ):
        """Print final timing summary and record telemetry.

        Logging to console follows the configured verbosity, but telemetry
        should always be recorded regardless of the logger level.
        """
        # Medians are stabler against outliers in HPC runs
        avg_mech = float(np.median(mech_times)) if mech_times else 0.0
        avg_stim = float(np.median(stim_times)) if stim_times else 0.0
        avg_dens = float(np.median(dens_times)) if dens_times else 0.0
        avg_dir  = float(np.median(dir_times))  if dir_times  else 0.0
        avg_total = avg_mech + avg_stim + avg_dens + avg_dir

        # Emit console summary only if INFO logging is enabled
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

        # Always record a JSON run summary (merged with final field stats and performance)
        if self.telemetry is not None:
            # Collect final field stats at end-of-run
            fields = self._collect_field_stats()
            # DOFs and memory at end-of-run
            dofs_V = self.V.dofmap.index_map.size_global
            dofs_Q = self.Q.dofmap.index_map.size_global
            dofs_T = self.T.dofmap.index_map.size_global
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
                    "u_min": float(fields.get("u_min", 0.0)),
                    "u_max": float(fields.get("u_max", 0.0)),
                    "S_min": float(fields.get("S_min", 0.0)),
                    "S_max": float(fields.get("S_max", 0.0)),
                    "psi": float(fields.get("psi", 0.0)),
                },
            }
            # Persist as JSON under results_dir
            self.telemetry.write_metadata(data, filename="run_summary.json", overwrite=True)

    # ---------- main run ----------
    def simulate(self, dt: float, total_time: float):
        """Run the remodeling simulation for the specified total physical time."""
        t = 0.0
        num_steps = int(total_time / dt)

        # Consistent API for setting nondimensional Δt
        self.cfg.set_dt_dim(dt)

        self.mechsolver.solver_setup()
        self.stimsolver.solver_setup()
        self.densolver.solver_setup()
        self.dirsolver.solver_setup()
        self._solvers_initialized = True

        # Start a fresh accounting window
        self._reset_iters_window()

        self.comm.Barrier()
        overall_start = MPI.Wtime()
        mech_times, stim_times, dens_times, dir_times = [], [], [], []

        for step in range(num_steps):
            step_time = t + dt
            self.step(dt, step_index=step, time_days=step_time)
            # mark a finished external step in the current print window
            self._acc_steps += 1

            # Store timings directly (fixedsolver resets them each step in run())
            mech_times.append(self.fixedsolver.mech_time_total)
            stim_times.append(self.fixedsolver.stim_time_total)
            dens_times.append(self.fixedsolver.dens_time_total)
            dir_times.append(self.fixedsolver.dir_time_total)

            t = step_time
            if self._is_output_step(step):
                self._output(t, step)

        self.comm.Barrier()
        overall_elapsed = MPI.Wtime() - overall_start
        overall_elapsed = self.comm.allreduce(overall_elapsed, op=MPI.MAX)

        self._print_final_summary(num_steps, overall_elapsed, mech_times, stim_times, dens_times, dir_times)

if __name__ == "__main__":
    # Programmatic demonstration run (no CLI)
    comm = MPI.COMM_WORLD
    m = mesh.create_unit_cube(
        comm, 33, 33, 33,
        cell_type=mesh.CellType.tetrahedron,
        ghost_mode=mesh.GhostMode.shared_facet
    )

    # Build config with default parameters
    facet_tags = build_facetag(m)
    config = Config(facet_tags=facet_tags, domain=m, verbose=True)
    with Remodeller(config) as remodeller:
        remodeller.simulate(dt=50, total_time=2000)
