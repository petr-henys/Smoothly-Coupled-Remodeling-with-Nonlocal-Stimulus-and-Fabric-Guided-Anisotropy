"""Fixed-point solver orchestrating coupled PDE subsolvers with Anderson acceleration."""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.subsolvers import StimulusSolver, DensitySolver, DirectionSolver
from simulation.utils import assign, get_owned_size, current_memory_mb
from simulation.logger import get_logger
from simulation.anderson import _Anderson
from simulation.drivers import RemodelingDriver


class FixedPointSolver:
    """Orchestrate Gauss-Seidel iteration over four coupled PDEs with Anderson or Picard."""

    def __init__(self, comm: MPI.Comm, cfg: Config,
                 driver: RemodelingDriver,
                 stimsolver: StimulusSolver,
                 densolver: DensitySolver,
                 dirsolver: DirectionSolver,
                 rho: fem.Function, rho_old: fem.Function,
                 A: fem.Function, A_old: fem.Function,
                 S: fem.Function, S_old: fem.Function):
        self.comm = comm
        self.cfg = cfg
        self.driver = driver
        self.stim = stimsolver
        self.den = densolver
        self.dir = dirsolver
        # self.mech removed - use self.driver.update_stiffness()

        self.rho = rho
        self.rho_old = rho_old
        self.A = A
        self.A_old = A_old
        self.S = S
        self.S_old = S_old

        self.total_gs_iters = 0
        self.subiter_metrics: List[Dict[str, Any]] = []

        # Performance tracking
        self.mech_time_total = 0.0
        self.stim_time_total = 0.0
        self.dens_time_total = 0.0
        self.dir_time_total = 0.0

        # Local DOF counts
        self.n_rho = get_owned_size(rho)
        self.n_A = get_owned_size(A)
        self.n_S = get_owned_size(S)

        self._build_state_slices()
        self.state_buffer = np.empty(self.state_size, dtype=self.rho.x.array.dtype)

        self.logger = get_logger(self.comm, verbose=bool(self.cfg.verbose), name="FixedPoint")
        self.telemetry = self.cfg.telemetry
        
        if self.telemetry:
            self.telemetry.register_csv(
                "subiterations",
                [
                    "step", "iter", "time_days", "proj_res", "r_norm",
                    "aa_hist", "accepted", "backtracks", "restart",
                    "mech_time", "stim_time", "dens_time", "dir_time",
                    "memory_mb",
                ],
                filename="subiterations.csv",
            )

    def _build_state_slices(self) -> None:
        """Build slice indices for flattened (ρ, A, S) state vector."""
        n_rho, n_A, n_S = self.n_rho, self.n_A, self.n_S
        self.state_size = n_rho + n_A + n_S
        self.state_slices = (
            slice(0, n_rho),
            slice(n_rho, n_rho + n_A),
            slice(n_rho + n_A, self.state_size)
        )

    def _flatten_state(self, copy: bool = True) -> np.ndarray:
        """Flatten current fields (ρ, A, S) to 1D array (local DOFs)."""
        s_rho, s_A, s_S = self.state_slices
        buf = self.state_buffer
        buf[s_rho] = self.rho.x.array[:self.n_rho]
        buf[s_A] = self.A.x.array[:self.n_A]
        buf[s_S] = self.S.x.array[:self.n_S]
        return buf.copy() if copy else buf

    def _restore_state(self, flat: np.ndarray) -> None:
        """Unpack flattened state back to field functions."""
        s_rho, s_A, s_S = self.state_slices
        assign(self.rho, flat[s_rho])
        assign(self.A, flat[s_A])
        assign(self.S, flat[s_S])

    def _elapsed_max(self, t0: float) -> float:
        """Max wall time across ranks since t0."""
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)

    def _gauss_seidel_sweep(self) -> Tuple[float, float, float, float, int]:
        """One GS sweep: mechanics + S → ρ → A."""
        # Mechanics
        t0 = MPI.Wtime()
        self.driver.update_stiffness()
        stats = self.driver.update_snapshots() or {}
        mech_time = self._elapsed_max(t0) + float(stats.get("total_time", 0.0))
        mech_iters = int(sum(stats.get("phase_iters", [])))

        # Stimulus
        t0 = MPI.Wtime()
        psi_expr = self.driver.stimulus_expr()
        self.stim.assemble_rhs(psi_expr)
        self.stim.solve()
        stim_time = self._elapsed_max(t0)

        # Density
        t0 = MPI.Wtime()
        self.den.assemble_lhs()
        self.den.assemble_rhs()
        self.den.solve()
        dens_time = self._elapsed_max(t0)

        # Direction
        t0 = MPI.Wtime()
        M_expr = self.driver.structure_expr()
        self.dir.assemble_rhs(M_expr)
        self.dir.solve()
        dir_time = self._elapsed_max(t0)

        return mech_time, stim_time, dens_time, dir_time, mech_iters

    def _proj_residual_norm(self, x_old: np.ndarray, x_test: np.ndarray,
                           x_raw: np.ndarray, weights: Tuple[float, float, float]) -> float:
        """Weighted residual ||x_test - x_base||_W with per-block weights."""
        is_picard = (x_test is x_raw)
        local_dots = np.zeros(3, dtype=float)
        
        for i, s in enumerate(self.state_slices):
            base = x_old[s] if is_picard else x_raw[s]
            diff = x_test[s] - base
            local_dots[i] = float(diff @ diff)
            
        global_dots = np.zeros_like(local_dots)
        self.comm.Allreduce(local_dots, global_dots, op=MPI.SUM)
        
        total = sum(w * d for w, d in zip(weights, global_dots))
        return float(max(total, 0.0) ** 0.5)

    def run(self, *, time_days: Optional[float] = None, step_index: Optional[int] = None) -> None:
        """Inner fixed-point loop: GS + Anderson acceleration until coupling_tol met."""
        # Weights for norm
        n_rho_g = self.comm.allreduce(self.n_rho, op=MPI.SUM)
        n_A_g = self.comm.allreduce(self.n_A, op=MPI.SUM)
        n_S_g = self.comm.allreduce(self.n_S, op=MPI.SUM)
        weights = (
            1.0 / max(int(n_rho_g), 1),
            1.0 / max(int(n_A_g), 1),
            1.0 / max(int(n_S_g), 1),
        )

        accelerator = None
        if self.cfg.accel_type == "anderson":
            accelerator = _Anderson(
                self.comm, m=self.cfg.m, beta=self.cfg.beta, lam=self.cfg.lam,
                restart_on_reject_k=self.cfg.restart_on_reject_k,
                restart_on_stall=self.cfg.restart_on_stall,
                restart_on_cond=self.cfg.restart_on_cond,
                step_limit_factor=self.cfg.step_limit_factor,
                verbose=self.cfg.verbose
            )

        self.driver.invalidate()
        x_k = self._flatten_state(copy=True)
        
        log_enabled = (self.comm.rank == 0) and self.cfg.verbose
        self.subiter_metrics = []
        
        self.mech_time_total = 0.0
        self.mech_iters_total = 0
        self.stim_time_total = 0.0
        self.dens_time_total = 0.0
        self.dir_time_total = 0.0

        for itr in range(1, self.cfg.max_subiters + 1):
            tm, ts, td, tdir, m_iters = self._gauss_seidel_sweep()
            self.mech_time_total += tm
            self.mech_iters_total += int(m_iters)
            self.stim_time_total += ts
            self.dens_time_total += td
            self.dir_time_total += tdir

            x_raw = self._flatten_state(copy=True)
            
            # Acceleration
            info: Dict[str, Any] = {
                "accepted": True, "backtracks": 0, "aa_hist": 0,
                "r_norm": self._proj_residual_norm(x_k, x_raw, x_raw, weights),
                "restart_reason": ""
            }
            
            x_next = x_raw
            if accelerator:
                x_next, info = accelerator.mix(
                    x_old=x_k, x_raw=x_raw, mask_fixed=None,
                    proj_residual_norm=lambda x_ref, x_test, xR: self._proj_residual_norm(x_ref, x_test, xR, weights),
                    gamma=self.cfg.gamma, use_safeguard=self.cfg.safeguard, backtrack_max=self.cfg.backtrack_max
                )

            self._restore_state(x_next)
            proj_norm = self._proj_residual_norm(x_k, x_raw, x_raw, weights)

            if log_enabled:
                msg = f"      Substep {itr}: proj-res = {proj_norm:.3e}"
                if accelerator:
                    msg += f" | AA_hist={info['aa_hist']} | accepted={'Y' if info['accepted'] else 'N'}"
                self.logger.info(msg)

            # Record metrics
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
                "stim_time": ts,
                "dens_time": td,
                "dir_time": tdir,
                "memory_mb": float(mem_sum),
            }
            
            self.subiter_metrics.append(rec)

            if self.telemetry:
                self.telemetry.record("subiterations", rec, csv_event=True)

            if proj_norm < self.cfg.coupling_tol and itr >= self.cfg.min_subiters:
                break

            x_k = x_next

        self.total_gs_iters += itr
