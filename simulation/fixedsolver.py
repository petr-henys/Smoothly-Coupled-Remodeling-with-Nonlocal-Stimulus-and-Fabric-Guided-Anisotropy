from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.subsolvers import DensitySolver
from simulation.utils import assign, get_owned_size, current_memory_mb
from simulation.logger import get_logger
from simulation.anderson import _Anderson
from simulation.drivers import RemodelingDriver


class FixedPointSolver:
    """Gauss-Seidel coupling with Anderson/Picard acceleration."""

    def __init__(self, comm: MPI.Comm, cfg: Config,
                 driver: RemodelingDriver,
                 densolver: DensitySolver,
                 rho: fem.Function, rho_old: fem.Function):
        self.comm = comm
        self.cfg = cfg
        self.driver = driver
        self.den = densolver

        self.rho = rho
        self.rho_old = rho_old

        self.total_gs_iters = 0
        self.subiter_metrics: List[Dict[str, Any]] = []

        # Performance tracking
        self.mech_time_total = 0.0
        self.dens_time_total = 0.0

        # Local DOF counts
        self.n_rho = get_owned_size(rho)

        self._build_state_slices()
        self.state_buffer = np.zeros(self.state_size, dtype=self.rho.x.array.dtype)

        # Only enable console logging if verbose is strictly True (not "progressbar")
        self.logger = get_logger(self.comm, verbose=(self.cfg.verbose is True), name="FixedPoint")
        self.telemetry = self.cfg.telemetry
        
        if self.telemetry:
            self.telemetry.register_csv(
                "subiterations",
                [
                    "step", "iter", "time_days", "proj_res", "r_norm",
                    "aa_hist", "accepted", "backtracks", "restart",
                    "mech_time", "dens_time",
                    "memory_mb",
                ],
                filename="subiterations.csv",
            )

    def _build_state_slices(self) -> None:
        """Slice indices for flattened state (ρ)."""
        n_rho = self.n_rho
        self.state_size = n_rho
        self.state_slices = (
            slice(0, n_rho),
        )

    def _flatten_state(self, copy: bool = True) -> np.ndarray:
        """Flatten current fields (ρ) to 1D array (local DOFs)."""
        s_rho, = self.state_slices
        buf = self.state_buffer
        buf[s_rho] = self.rho.x.array[:self.n_rho]
        return buf.copy() if copy else buf

    def _restore_state(self, flat: np.ndarray) -> None:
        """Unpack flattened state back to field functions."""
        s_rho, = self.state_slices
        assign(self.rho, flat[s_rho])

    def _elapsed_max(self, t0: float) -> float:
        """Max wall time across ranks since t0."""
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)

    def _gauss_seidel_sweep(self) -> Dict[str, Dict[str, Any]]:
        """One GS sweep: mechanics → ρ."""
        # Mechanics
        t0 = MPI.Wtime()
        self.driver.update_stiffness()
        stats = self.driver.update_snapshots() or {}
        mech_time = self._elapsed_max(t0) + float(stats.get("total_time", 0.0))
        mech_iters = int(sum(stats.get("phase_iters", [])))

        # Density
        t0 = MPI.Wtime()
        psi_expr = self.driver.stimulus_expr()
        self.den.update_driving_force(psi_expr)
        self.den.assemble_lhs()
        self.den.assemble_rhs()
        its_d, reason_d = self.den.solve()
        dens_time = self._elapsed_max(t0)

        return {
            "mech": {"time": mech_time, "iters": mech_iters, "reason": 0},
            "dens": {"time": dens_time, "iters": its_d, "reason": reason_d},
        }

    def _weighted_dist(self, x_a: np.ndarray, x_b: np.ndarray, weights: Tuple[float]) -> float:
        """Robust weighted distance ||x_a - x_b||_W."""
        local_dots = np.zeros(1, dtype=float)
        
        for i, s in enumerate(self.state_slices):
            diff = x_a[s] - x_b[s]
            local_dots[i] = float(diff @ diff)
            
        global_dots = np.zeros_like(local_dots)
        self.comm.Allreduce(local_dots, global_dots, op=MPI.SUM)
        
        total = sum(w * d for w, d in zip(weights, global_dots))
        return float(max(total, 0.0) ** 0.5)

    def run(self, *, time_days: Optional[float] = None, step_index: Optional[int] = None, progress: Any = None) -> None:
        """Inner fixed-point loop: GS + Anderson acceleration until coupling_tol met."""
        # Weights for norm
        n_rho_g = self.comm.allreduce(self.n_rho, op=MPI.SUM)
        weights = (
            1.0 / max(int(n_rho_g), 1),
        )

        accelerator = None
        if self.cfg.accel_type == "anderson":
            accelerator = _Anderson(
                self.comm, m=self.cfg.m, beta=self.cfg.beta, lam=self.cfg.lam,
                restart_on_reject_k=self.cfg.restart_on_reject_k,
                restart_on_stall=self.cfg.restart_on_stall,
                restart_on_cond=self.cfg.restart_on_cond,
                step_limit_factor=self.cfg.step_limit_factor,
                verbose=(self.cfg.verbose is True)
            )

        self.driver.invalidate()
        x_k = self._flatten_state(copy=True)
        
        # Enable logging on rank 0 regardless of verbose setting (Logger handles console/file split)
        log_enabled = (self.comm.rank == 0)
        self.subiter_metrics = []
        
        self.mech_time_total = 0.0
        self.mech_iters_total = 0
        self.dens_time_total = 0.0

        # Initialize detailed stats accumulator
        self.solver_stats = {
            "mech": {"time": 0.0, "iters": 0, "reason": 0},
            "dens": {"time": 0.0, "iters": 0, "reason": 0},
        }

        inner_task_id = None
        if progress is not None:
            inner_task_id = progress.add_task(f"  Coupling", total=self.cfg.max_subiters, info=" " * 15)

        for itr in range(1, self.cfg.max_subiters + 1):
            sweep_stats = self._gauss_seidel_sweep()
            
            # Accumulate stats
            for key in ["mech", "dens"]:
                self.solver_stats[key]["time"] += sweep_stats[key]["time"]
                self.solver_stats[key]["iters"] += sweep_stats[key]["iters"]
                self.solver_stats[key]["reason"] = sweep_stats[key]["reason"]

            tm = sweep_stats["mech"]["time"]
            td = sweep_stats["dens"]["time"]
            m_iters = sweep_stats["mech"]["iters"]

            self.mech_time_total += tm
            self.mech_iters_total += int(m_iters)
            self.dens_time_total += td

            x_raw = self._flatten_state(copy=True)
            
            # Acceleration
            info: Dict[str, Any] = {
                "accepted": True, "backtracks": 0, "aa_hist": 0,
                "r_norm": self._weighted_dist(x_raw, x_k, weights),
                "restart_reason": ""
            }
            
            x_next = x_raw
            if accelerator:
                x_next, info = accelerator.mix(
                    x_old=x_k, x_raw=x_raw, mask_fixed=None,
                    norm_func=lambda a, b: self._weighted_dist(a, b, weights),
                    gamma=self.cfg.gamma, use_safeguard=self.cfg.safeguard, 
                    backtrack_max=self.cfg.backtrack_max
                )

            self._restore_state(x_next)
            
            # This is the actual residual of the step just taken (Picard step size)
            proj_norm = self._weighted_dist(x_raw, x_k, weights)

            if log_enabled:
                msg = f"      Substep {itr}: proj-res = {proj_norm:.3e}"
                if accelerator:
                    msg += f" | AA={info['aa_hist']} | acc={'Y' if info['accepted'] else 'N'}"
                    if info['backtracks'] > 0:
                        msg += f" | bt={info['backtracks']}"
                    if info['restart_reason']:
                        msg += f" | R: {info['restart_reason']}"
                self.logger.info(msg)

            if progress is not None and inner_task_id is not None:
                res_str = f"res={proj_norm:.2e}"
                progress.update(inner_task_id, advance=1, info=f"{res_str:<15}")

            mem_local = current_memory_mb()
            mem_sum = self.comm.allreduce(mem_local, op=MPI.SUM)

            rec = {
                "step": step_index if step_index is not None else 0,
                "iter": itr,
                "time_days": float(time_days) if time_days is not None else 0.0,
                "proj_res": float(proj_norm),
                "r_norm": float(info.get('r_norm', 0.0)),
                "aa_hist": int(info.get('aa_hist', 0)),
                "accepted": bool(info.get('accepted', True)),
                "backtracks": int(info.get('backtracks', 0)),
                "restart": str(info.get('restart_reason', '')),
                "mech_time": tm,
                "dens_time": td,
                "memory_mb": float(mem_sum),
            }
            
            self.subiter_metrics.append(rec)

            if self.telemetry:
                self.telemetry.record("subiterations", rec, csv_event=True)

            if proj_norm < self.cfg.coupling_tol and itr >= self.cfg.min_subiters:
                break

            x_k = x_next

        if progress is not None and inner_task_id is not None:
            progress.remove_task(inner_task_id)

        self.total_gs_iters += itr