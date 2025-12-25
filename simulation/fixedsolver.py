"""Block Gauss-Seidel fixed-point solver with optional Anderson acceleration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import resource

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.protocols import CouplingBlock
from simulation.stats import SweepStats, StepSummary
from simulation.utils import get_owned_size
from simulation.logger import get_logger
from simulation.anderson import Anderson


@dataclass(frozen=True)
class _FieldSpec:
    f: fem.Function
    n_owned: int
    N_global: int
    inv_sqrt_N: float
    offset: int
    name: str


class FixedPointSolver:
    """Block Gauss-Seidel iteration with optional Anderson acceleration."""

    def __init__(self, comm: MPI.Comm, cfg: Config, blocks: Sequence[CouplingBlock]):
        self.comm = comm
        self.cfg = cfg
        self.blocks: Tuple[CouplingBlock, ...] = tuple(blocks)

        self.logger = get_logger(self.comm, name="FixedPoint", log_file=self.cfg.log_file)

        self.subiter_metrics: List[Dict[str, Any]] = []

        # Collect coupled state fields (deduplicate by object identity, preserve order)
        fields: List[fem.Function] = []
        seen = set()
        for blk in self.blocks:
            # Protocol check at runtime (optional, for debugging)
            if not isinstance(blk, CouplingBlock):
                raise TypeError(
                    f"Block {type(blk).__name__} does not implement CouplingBlock protocol."
                )
            for f in tuple(blk.state_fields):
                if id(f) not in seen:
                    seen.add(id(f))
                    fields.append(f)

        if len(fields) == 0:
            raise ValueError("No coupled state fields provided by blocks (empty state_fields).")

        self._specs: List[_FieldSpec] = []
        off = 0
        for f in fields:
            n_owned = int(get_owned_size(f))
            N_global = int(self.comm.allreduce(n_owned, op=MPI.SUM))
            inv_sqrt_N = 1.0 / max(np.sqrt(float(N_global)), 1e-30)
            name = f.name
            self._specs.append(_FieldSpec(f=f, n_owned=n_owned, N_global=N_global, inv_sqrt_N=inv_sqrt_N, offset=off, name=name))
            off += n_owned

        self._n_state = int(off)

        # Per-field normalization scales (set on the first sweep each timestep)
        self._scales: np.ndarray | None = None

        # Anderson accelerator (optional)
        self.anderson: Anderson | None = None
        if self.cfg.solver.accel_type == "anderson":
            self.anderson = Anderson(
                comm=self.comm,
                m=int(self.cfg.solver.m),
                beta=float(self.cfg.solver.beta),
                lam=float(self.cfg.solver.lam),
                gamma=float(self.cfg.solver.gamma),
                safeguard=bool(self.cfg.solver.safeguard),
                backtrack_max=int(self.cfg.solver.backtrack_max),
                restart_on_reject_k=int(self.cfg.solver.restart_on_reject_k),
                restart_on_stall=float(self.cfg.solver.restart_on_stall),
                restart_on_cond=float(self.cfg.solver.restart_on_cond),
                step_limit_factor=float(self.cfg.solver.step_limit_factor),
                verbose=False,
            )

    # ------------------------------- packing --------------------------------

    def _pack_unscaled(self) -> np.ndarray:
        x = np.empty(self._n_state, dtype=float)
        for sp in self._specs:
            x[sp.offset : sp.offset + sp.n_owned] = sp.f.x.array[: sp.n_owned]
        return x

    def _compute_field_rms(self, values_owned: np.ndarray, N_global: int) -> float:
        loc = float(np.dot(values_owned, values_owned))
        glob = float(self.comm.allreduce(loc, op=MPI.SUM))
        if N_global <= 0:
            return 0.0
        return float(np.sqrt(glob / float(N_global)))

    def _init_scales(self, x_old: np.ndarray, x_raw: np.ndarray) -> None:
        scales = np.empty(len(self._specs), dtype=float)
        tiny = 1e-30
        for i, sp in enumerate(self._specs):
            sl = slice(sp.offset, sp.offset + sp.n_owned)
            rms_old = self._compute_field_rms(x_old[sl], sp.N_global)
            rms_raw = self._compute_field_rms(x_raw[sl], sp.N_global)
            scales[i] = max(rms_old, rms_raw, tiny)
        self._scales = scales

    def _pack_scaled(self, x_unscaled: np.ndarray) -> np.ndarray:
        if self._scales is None:
            raise RuntimeError("Internal error: scales are not initialized.")
        y = np.empty_like(x_unscaled)
        for i, sp in enumerate(self._specs):
            sl = slice(sp.offset, sp.offset + sp.n_owned)
            y[sl] = (x_unscaled[sl] / self._scales[i]) * sp.inv_sqrt_N
        return y

    def _unpack_scaled_into_fields(self, x_scaled: np.ndarray) -> None:
        if self._scales is None:
            raise RuntimeError("Internal error: scales are not initialized.")
        for i, sp in enumerate(self._specs):
            sl = slice(sp.offset, sp.offset + sp.n_owned)
            sp.f.x.array[: sp.n_owned] = (x_scaled[sl] / sp.inv_sqrt_N) * self._scales[i]
            sp.f.x.scatter_forward()

    # ----------------------------- norms/metrics ----------------------------

    def _gdot(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self.comm.allreduce(float(np.dot(a, b)), op=MPI.SUM))

    def _proj_step(self, x_old: np.ndarray, x_trial: np.ndarray, x_ref: np.ndarray) -> float:
        """Relative step: ||x_trial-x_old|| / ||x_ref|| in global L2 (all inputs scaled)."""
        d = x_trial - x_old
        d2 = self._gdot(d, d)
        r2 = self._gdot(x_ref, x_ref)
        if r2 <= 1e-300:
            return float(np.sqrt(d2))
        return float(np.sqrt(d2 / r2))

    # ------------------------------- main loop ------------------------------

    def _format_iteration_log(
        self,
        itr: int,
        picard_res: float,
        aa_step_res: float,
        aa_info: Dict[str, Any],
        mem_mb: float,
        block_stats: List[SweepStats],
    ) -> str:
        """Format one Picard iteration as multi-line hierarchical output.

        Option D format:
            Picard 5: res=5.63e-02 | step=2.85e-02 (cond=2.1e+03, m=5, ACC)
                mech  81it  0.572s │ fab  6it  0.272s │ stim  6it  0.002s │ dens  4it  0.003s
                └─ aniso: a=[0.56, 1.82]  p2=[0.01, 0.09]
        """
        cond_val = aa_info.get("condH")
        cond_str = f"{cond_val:.1e}" if cond_val is not None else "N/A"
        acc_str = "ACC" if aa_info.get("accepted", True) else "REJ"
        rst_str = f", RST:{aa_info['restart_reason']}" if aa_info.get("restart_reason") else ""

        # Line 1: Picard header
        line1 = f"Picard {itr:>2}: res={picard_res:.2e} | step={aa_step_res:.2e} (cond={cond_str}, m={aa_info.get('aa_hist', 0)}, {acc_str}{rst_str})"

        # Line 2: Block performance (compact, aligned)
        block_parts = [s.format_short(width=4) for s in block_stats]
        line2 = "    " + " │ ".join(block_parts)

        # Line 3: Physics-specific extras (if any block has them)
        extras = []
        for s in block_stats:
            extra_str = s.format_extra()
            if extra_str:
                extras.append(f"{s.label}: {extra_str}")
        
        lines = [line1, line2]
        if extras:
            line3 = "    └─ " + "  |  ".join(extras)
            lines.append(line3)

        return "\n".join(lines)

    def run(
        self,
        progress,
        task_id,
        step_index: int = 0,
        sim_time: float = 0.0,
    ) -> bool:
        """Run fixed-point iteration until convergence.

        Args:
            progress: Progress reporter (or None).
            task_id: Task ID for progress updates (or None).
            step_index: Current timestep index (1-based), for logging.
            sim_time: Current simulation time [days], for logging.

        Returns:
            True if converged within tolerance.
        """
        tol = float(self.cfg.solver.coupling_tol)
        max_subiters = int(self.cfg.solver.max_subiters)
        min_subiters = int(self.cfg.solver.min_subiters)

        self.subiter_metrics = []

        self._scales = None
        if self.anderson is not None:
            self.anderson.reset()

        if progress is not None and task_id is not None:
            progress.reset(task_id, total=max_subiters)
            progress.start_task(task_id)

        converged = False

        for itr in range(1, max_subiters + 1):
            # Snapshot old iterate (unscaled, owned DOFs)
            x_old = self._pack_unscaled()

            # One Gauss-Seidel sweep of all blocks (each returns SweepStats)
            sweep_stats: List[SweepStats] = []
            for blk in self.blocks:
                stats = blk.sweep()
                sweep_stats.append(stats)

            # Raw iterate after the sweep
            x_raw = self._pack_unscaled()

            if self._scales is None:
                self._init_scales(x_old, x_raw)

            x_old_s = self._pack_scaled(x_old)
            x_raw_s = self._pack_scaled(x_raw)

            # Picard residual: ||x_raw - x_old|| / ||x_raw|| (scaled global L2)
            picard_res = self._proj_step(x_old_s, x_raw_s, x_raw_s)

            if self.anderson is not None:
                x_new_s, aa = self.anderson.mix(x_old_s, x_raw_s)
            else:
                beta = float(self.cfg.solver.beta)
                x_new_s = x_old_s + beta * (x_raw_s - x_old_s)
                aa = {"aa_hist": 0, "accepted": True, "restart_reason": "", "condH": 1.0}

            aa_step_res = self._proj_step(x_old_s, x_new_s, x_raw_s)

            self._unpack_scaled_into_fields(x_new_s)

            # Memory usage (Max RSS in KB on Linux -> MB)
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

            # Log per-Picard step info (file only via DEBUG level)
            cond_val = aa.get("condH")
            log_line = self._format_iteration_log(itr, picard_res, aa_step_res, aa, mem_mb, sweep_stats)
            self.logger.debug(log_line)

            rec = {
                "iter": int(itr),
                "proj_res": float(picard_res),
                "picard_res": float(picard_res),
                "aa_step_res": float(aa_step_res),
                "aa_hist": int(aa.get("aa_hist", 0)),
                "aa_accepted": bool(aa.get("accepted", True)),
                "aa_restart": str(aa.get("restart_reason", "")),
                "condH": float(cond_val) if cond_val is not None else 0.0,
                "mem_mb": float(mem_mb),
                "block_stats": sweep_stats,
            }
            self.subiter_metrics.append(rec)

            if progress is not None and task_id is not None:
                info_str = f"res={picard_res:.1e} m={rec['aa_hist']}"
                if not rec["aa_accepted"]:
                    info_str += " REJ"
                if rec["aa_restart"]:
                    info_str += " RST"
                progress.update(task_id, advance=1, info=f"{info_str:<35}")

            if itr >= min_subiters and picard_res <= tol:
                converged = True
                break

        # Summary stats for the time step using StepSummary
        summary = StepSummary.from_iteration_records(self.subiter_metrics)
        self.logger.info(summary.format_summary(step_index=step_index, sim_time=sim_time))

        if progress is not None and task_id is not None:
            progress.update(task_id, completed=True)
            progress.stop_task(task_id)

        return converged
