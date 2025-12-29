"""Block Gauss-Seidel fixed-point solver with optional Anderson acceleration."""

from __future__ import annotations

import resource
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from dolfinx import fem
from mpi4py import MPI

from simulation.anderson import Anderson
from simulation.config import Config
from simulation.logger import get_logger
from simulation.protocols import CouplingBlock
from simulation.stats import StepSummary, SweepStats
from simulation.utils import get_owned_size


@dataclass(frozen=True)
class _FieldSpec:
    """Metadata for one field: sizes, offset in packed vector, scaling factor."""
    field: fem.Function
    n_owned: int
    n_global: int
    inv_sqrt_n: float
    offset: int
    name: str


class FixedPointSolver:
    """Block Gauss-Seidel solver with optional Anderson acceleration."""

    _TINY: float = 1e-300  # Guard against division by zero

    def __init__(self, comm: MPI.Comm, cfg: Config, blocks: Sequence[CouplingBlock]):
        self.comm = comm
        self.cfg = cfg
        self.blocks: Tuple[CouplingBlock, ...] = tuple(blocks)
        self.logger = get_logger(self.comm, name="FixedPoint", log_file=self.cfg.log_file)

        self.subiter_metrics: List[Dict[str, Any]] = []
        self.stop_reason: str = ""

        self._specs = self._build_field_specs()
        self._n_state = sum(sp.n_owned for sp in self._specs)

        self._scales: np.ndarray | None = None  # Per-field RMS for normalization
        self.anderson = self._create_anderson() if cfg.solver.accel_type == "anderson" else None

    def _build_field_specs(self) -> List[_FieldSpec]:
        """Collect state fields from blocks (deduplicated, order-preserving)."""
        fields: List[fem.Function] = []
        seen_ids = set()
        
        for blk in self.blocks:
            for f in blk.state_fields:
                if id(f) not in seen_ids:
                    seen_ids.add(id(f))
                    fields.append(f)

        if not fields:
            raise ValueError("No coupled state fields provided by blocks.")

        specs: List[_FieldSpec] = []
        offset = 0
        for f in fields:
            n_owned = get_owned_size(f)
            n_global = self.comm.allreduce(n_owned, op=MPI.SUM)
            inv_sqrt_n = 1.0 / max(np.sqrt(n_global), self._TINY)
            specs.append(_FieldSpec(
                field=f,
                n_owned=n_owned,
                n_global=n_global,
                inv_sqrt_n=inv_sqrt_n,
                offset=offset,
                name=f.name,
            ))
            offset += n_owned
        return specs

    def _create_anderson(self) -> Anderson:
        """Create Anderson accelerator from config."""
        s = self.cfg.solver
        return Anderson(
            comm=self.comm,
            m=s.m,
            beta=s.beta,
            lam=s.lam,
            restart_on_stall=s.restart_on_stall,
            restart_on_cond=s.restart_on_cond,
            step_limit_factor=s.step_limit_factor,
            restart_stall_window=s.restart_stall_window,
            restart_stall_patience=s.restart_stall_patience,
        )

    # ------------------------------- Packing --------------------------------

    def _pack_unscaled(self) -> np.ndarray:
        """Pack owned DOFs from all fields into flat array."""
        x = np.empty(self._n_state, dtype=float)
        for sp in self._specs:
            x[sp.offset:sp.offset + sp.n_owned] = sp.field.x.array[:sp.n_owned]
        return x

    def _compute_field_rms(self, values: np.ndarray, n_global: int) -> float:
        """Global RMS of local values."""
        local_sum_sq = np.dot(values, values)
        global_sum_sq = self.comm.allreduce(local_sum_sq, op=MPI.SUM)
        if n_global <= 0:
            return 0.0
        return np.sqrt(global_sum_sq / n_global)

    def _init_scales(self, x_old: np.ndarray, x_raw: np.ndarray) -> None:
        """Set per-field scales from max(RMS_old, RMS_raw) for normalization."""
        scales = np.empty(len(self._specs), dtype=float)
        for i, sp in enumerate(self._specs):
            sl = slice(sp.offset, sp.offset + sp.n_owned)
            rms_old = self._compute_field_rms(x_old[sl], sp.n_global)
            rms_raw = self._compute_field_rms(x_raw[sl], sp.n_global)
            scales[i] = max(rms_old, rms_raw, self._TINY)
        self._scales = scales

    def _pack_scaled(self, x_unscaled: np.ndarray) -> np.ndarray:
        """Scale and normalize: y = (x / scale) * inv_sqrt_n per field."""
        if self._scales is None:
            raise RuntimeError("Scales not initialized.")
        y = np.empty_like(x_unscaled)
        for i, sp in enumerate(self._specs):
            sl = slice(sp.offset, sp.offset + sp.n_owned)
            y[sl] = (x_unscaled[sl] / self._scales[i]) * sp.inv_sqrt_n
        return y

    def _unpack_scaled_to_fields(self, x_scaled: np.ndarray) -> None:
        """Inverse of _pack_scaled: write back to field DOFs and scatter."""
        if self._scales is None:
            raise RuntimeError("Scales not initialized.")
        for i, sp in enumerate(self._specs):
            sl = slice(sp.offset, sp.offset + sp.n_owned)
            sp.field.x.array[:sp.n_owned] = (x_scaled[sl] / sp.inv_sqrt_n) * self._scales[i]
            sp.field.x.scatter_forward()

    # ----------------------------- Norms ------------------------------------

    def _gdot(self, a: np.ndarray, b: np.ndarray) -> float:
        """Global dot product via MPI allreduce."""
        local_dot = np.dot(a, b)
        return self.comm.allreduce(local_dot, op=MPI.SUM)

    def _relative_step(self, x_old: np.ndarray, x_new: np.ndarray, x_ref: np.ndarray) -> float:
        """Compute ||x_new - x_old|| / ||x_ref|| globally. Must match Anderson._rel_step."""
        diff = x_new - x_old
        diff_norm_sq = self._gdot(diff, diff)
        ref_norm_sq = self._gdot(x_ref, x_ref)
        if ref_norm_sq <= self._TINY:
            return np.sqrt(diff_norm_sq)
        return np.sqrt(diff_norm_sq / ref_norm_sq)

    # ------------------------------- Logging --------------------------------

    def _format_iteration_log(
        self,
        itr: int,
        picard_res: float,
        aa_step_res: float,
        aa_info: Dict[str, Any],
        block_stats: List[SweepStats],
        contraction: float | None = None,
    ) -> str:
        """Format iteration as two-line log: residuals + per-block stats."""
        cond_val = aa_info.get("condH")
        cond_str = f"{cond_val:.1e}" if cond_val is not None else "N/A"
        flags = ""
        if aa_info.get("restart_reason"):
            flags += " RST"
        if aa_info.get("limited"):
            flags += " LIM"
        if aa_info.get("aa_off"):
            flags += " [Picard]"

        contraction_str = f"ρ={contraction:.2f}" if contraction is not None else "ρ=N/A"

        line1 = (
            f"Picard {itr:>2}: res={picard_res:.2e} {contraction_str} | "
            f"step={aa_step_res:.2e} (cond={cond_str}, m={aa_info.get('aa_hist', 0)}{flags})"
        )
        block_parts = [s.format_short(width=4) for s in block_stats]
        line2 = "    " + " │ ".join(block_parts)

        return f"{line1}\n{line2}"

    # ------------------------------- Main loop ------------------------------

    def run(
        self,
        progress,
        task_id,
        step_index: int = 0,
        sim_time: float = 0.0,
    ) -> bool:
        """Run fixed-point iteration until convergence or max iterations.

        Args:
            progress: Progress reporter (or None).
            task_id: Task ID for progress updates (or None).
            step_index: Timestep index for logging.
            sim_time: Simulation time [days] for logging.

        Returns:
            True if converged within tolerance.
        """
        tol = self.cfg.solver.coupling_tol
        max_subiters = self.cfg.solver.max_subiters

        stall_window = self.cfg.solver.outer_stall_window
        stall_min_drop = self.cfg.solver.outer_stall_min_rel_drop
        stall_patience = self.cfg.solver.outer_stall_patience
        res_history: deque = deque(maxlen=stall_window)
        stall_count = 0
        stop_reason = ""

        self.subiter_metrics = []
        self.stop_reason = ""
        self._scales = None
        if self.anderson is not None:
            self.anderson.reset()

        # Counter for consecutive contractive iterations (ρ < threshold)
        contractive_streak = 0
        # Track current mode: True = using Anderson, False = using Picard
        was_using_anderson = True

        if progress is not None and task_id is not None:
            progress.reset(task_id, total=max_subiters)
            progress.start_task(task_id)

        converged = False

        for itr in range(1, max_subiters + 1):
            x_old = self._pack_unscaled()

            sweep_stats: List[SweepStats] = []
            for blk in self.blocks:
                stats = blk.sweep()
                sweep_stats.append(stats)

            x_raw = self._pack_unscaled()

            if self._scales is None:
                self._init_scales(x_old, x_raw)

            x_old_s = self._pack_scaled(x_old)
            x_raw_s = self._pack_scaled(x_raw)
            picard_res = self._relative_step(x_old_s, x_raw_s, x_raw_s)
            res_history.append(picard_res)

            # Contraction ratio ρ = r_k / r_{k-1} from history
            contraction: float | None = None
            if len(res_history) >= 2:
                prev_res = res_history[-2]
                if prev_res > self._TINY:
                    contraction = picard_res / prev_res

            # Track consecutive contractive iterations for Picard switch
            if contraction is not None and contraction < self.cfg.solver.rho_anderson_off:
                contractive_streak += 1
            else:
                contractive_streak = 0

            # Hysteresis logic for mode switching:
            # - Switch to Picard: need rho_anderson_patience consecutive ρ < rho_anderson_off
            # - Switch back to Anderson: need ρ >= rho_anderson_on (higher threshold)
            if self.anderson is not None:
                if was_using_anderson:
                    # Currently in Anderson mode: switch to Picard if strongly contractive
                    use_anderson = contractive_streak < self.cfg.solver.rho_anderson_patience
                else:
                    # Currently in Picard mode: switch back to Anderson only if ρ >= rho_on
                    if contraction is not None and contraction >= self.cfg.solver.rho_anderson_on:
                        use_anderson = True
                        # Reset Anderson history when switching back from Picard
                        self.anderson.reset()
                    else:
                        use_anderson = False
            else:
                use_anderson = False
            
            # Update mode tracking for next iteration
            was_using_anderson = use_anderson

            # Skip acceleration if already converged
            if picard_res <= tol:
                x_new_s = x_raw_s
                aa = {
                    "aa_hist": 0,
                    "accepted": True,
                    "restart_reason": "",
                    "condH": 1.0,
                    "limited": False,
                    "aa_off": False,
                }
            elif use_anderson:
                x_new_s, aa = self.anderson.mix(x_old_s, x_raw_s)
                aa["aa_off"] = False
            else:
                # Pure damped Picard
                beta = self.cfg.solver.beta
                x_new_s = x_old_s + beta * (x_raw_s - x_old_s)
                aa = {
                    "aa_hist": 0,
                    "accepted": True,
                    "restart_reason": "",
                    "condH": 1.0,
                    "limited": False,
                    "aa_off": True,
                }

            aa_step_res = self._relative_step(x_old_s, x_new_s, x_raw_s)

            self._unpack_scaled_to_fields(x_new_s)

            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            log_line = self._format_iteration_log(
                itr, picard_res, aa_step_res, aa, sweep_stats, contraction
            )
            self.logger.debug(log_line)

            rec = {
                "iter": itr,
                "proj_res": picard_res,
                "picard_res": picard_res,
                "aa_step_res": aa_step_res,
                "aa_hist": aa.get("aa_hist", 0),
                "aa_accepted": aa.get("accepted", True),
                "aa_restart": aa.get("restart_reason", ""),
                "aa_limited": aa.get("limited", False),
                "aa_off": aa.get("aa_off", False),
                "condH": aa.get("condH", 0.0),
                "contraction": contraction,  # ρ_k = r_k / r_{k-1}
                "mem_mb": mem_mb,
                "block_stats": sweep_stats,
            }
            self.subiter_metrics.append(rec)

            if progress is not None and task_id is not None:
                rho_str = f"ρ={contraction:.2f}" if contraction is not None else ""
                mode_str = "P" if aa.get("aa_off") else f"m={rec['aa_hist']}"
                info_str = f"res={picard_res:.1e} {rho_str} {mode_str}"
                if aa.get('limited'):
                    info_str += " LIM"
                if rec["aa_restart"]:
                    info_str += " RST"
                progress.update(task_id, advance=1, info=f"{info_str:<30}")

            if picard_res <= tol:
                converged = True
                break

            if stop_reason == "" and len(res_history) == stall_window:
                r_first = res_history[0]
                r_best = min(res_history)
                if np.isfinite(r_first) and np.isfinite(r_best) and r_first > 0.0:
                    rel_drop = (r_first - r_best) / r_first
                    if rel_drop < stall_min_drop:
                        stall_count += 1
                    else:
                        stall_count = 0

                    if stall_count >= stall_patience:
                        stop_reason = "no_progress"
                        self.subiter_metrics[-1]["fp_stop_reason"] = stop_reason
                        self.logger.file_only(
                            f"Early abort fixed-point at itr={itr}: no progress "
                            f"(rel_drop={rel_drop:.2%} < {stall_min_drop:.2%} "
                            f"for {stall_patience} windows of {stall_window} iters)."
                        )
                        break
                else:
                    stall_count = 0

        if converged:
            stop_reason = "converged"
        elif stop_reason == "" and max_subiters > 0:
            stop_reason = "max_subiters"

        self.stop_reason = stop_reason
        if self.subiter_metrics:
            self.subiter_metrics[-1].setdefault("fp_stop_reason", stop_reason)

        summary = StepSummary.from_iteration_records(self.subiter_metrics)
        self.logger.info(summary.format_summary(step_index=step_index, sim_time=sim_time))

        if progress is not None and task_id is not None:
            progress.update(task_id, completed=True)
            progress.stop_task(task_id)

        return converged
