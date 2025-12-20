"""Fixed-point coupling via block Gauss–Seidel + optional Anderson acceleration.

Each block must provide:
- `state_fields`: tuple of fields that form the coupled state
- `sweep()`: perform one block update

Blocks with `state_fields == ()` may still run each sweep (side effects), but do
not contribute entries to the Anderson-mixed state vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
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
    """Block Gauss–Seidel + optional Anderson acceleration."""

    def __init__(self, comm: MPI.Comm, cfg: Config, blocks: Sequence[object]):
        self.comm = comm
        self.cfg = cfg
        self.blocks = tuple(blocks)

        self.logger = get_logger(self.comm, name="FixedPoint", log_file=self.cfg.log_file)

        self.subiter_metrics: List[Dict] = []

        # Collect coupled state fields (deduplicate by object identity, preserve order)
        fields: List[fem.Function] = []
        seen = set()
        for blk in self.blocks:
            if not hasattr(blk, "state_fields"):
                raise AttributeError(f"Block {type(blk).__name__} missing required attribute `state_fields`.")
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
            name = getattr(f, "name", "field")
            self._specs.append(_FieldSpec(f=f, n_owned=n_owned, N_global=N_global, inv_sqrt_N=inv_sqrt_N, offset=off, name=name))
            off += n_owned

        self._n_state = int(off)

        # Per-field normalization scales (set on the first sweep each timestep)
        self._scales: np.ndarray | None = None

        # Anderson accelerator (optional)
        self.anderson: Anderson | None = None
        if self.cfg.accel_type == "anderson":
            self.anderson = Anderson(
                comm=self.comm,
                m=int(self.cfg.m),
                beta=float(self.cfg.beta),
                lam=float(self.cfg.lam),
                gamma=float(self.cfg.gamma),
                safeguard=bool(self.cfg.safeguard),
                backtrack_max=int(self.cfg.backtrack_max),
                restart_on_reject_k=int(self.cfg.restart_on_reject_k),
                restart_on_stall=float(self.cfg.restart_on_stall),
                restart_on_cond=float(self.cfg.restart_on_cond),
                step_limit_factor=float(self.cfg.step_limit_factor),
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

    def run(self, progress, task_id) -> bool:
        tol = float(self.cfg.coupling_tol)
        max_subiters = int(self.cfg.max_subiters)
        min_subiters = int(self.cfg.min_subiters)

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

            # One Gauss–Seidel sweep of all blocks
            for blk in self.blocks:
                blk.sweep()

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
                beta = float(self.cfg.beta)
                x_new_s = x_old_s + beta * (x_raw_s - x_old_s)
                aa = {"aa_hist": 0, "accepted": True, "backtracks": 0, "restart_reason": ""}

            aa_step_res = self._proj_step(x_old_s, x_new_s, x_raw_s)

            self._unpack_scaled_into_fields(x_new_s)

            rec = {
                "iter": int(itr),
                # Keep `proj_res` as the convergence residual used for stopping/postprocessing.
                "proj_res": float(picard_res),
                "picard_res": float(picard_res),
                "aa_step_res": float(aa_step_res),
                "aa_hist": int(aa.get("aa_hist", 0)),
                "aa_accepted": bool(aa.get("accepted", True)),
                "aa_backtracks": int(aa.get("backtracks", 0)),
                "aa_restart": str(aa.get("restart_reason", "")),
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

        if progress is not None and task_id is not None:
            progress.update(task_id, completed=True)
            progress.stop_task(task_id)

        return converged
