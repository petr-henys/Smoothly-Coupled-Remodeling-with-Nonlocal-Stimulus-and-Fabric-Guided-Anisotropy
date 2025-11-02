from collections import deque
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np
from mpi4py import MPI

from simulation.utils import _global_norm
from simulation.logger import get_logger

class _Anderson:
    """
    Type-I Anderson (Pulay) acceleration with equality-constrained LS
    (sum of coefficients = 1), global Gram construction, and Tikhonov
    regularization scaled by residual energy.

    Includes restart heuristics driven by acceptance history, residual stalling,
    and Gram conditioning. No coupling to preconditioner updates.
    """

    def __init__(self, comm: MPI.Comm, m: int = 8, beta: float = 1.0, lam: float = 1e-10,
                 # restart policy (overridable via FixedPointSolver.run / cfg)
                 restart_on_reject_k: int = 2,
                 restart_on_stall: float = 1.10,
                 restart_on_cond: float = 1e12,
                 step_limit_factor: float = 2.0,
                 safeguard_abs_floor: float = 1e-10,
                 gamma_decay_p: float = 0.5,
                 verbose: bool = True):
        self.comm = comm
        self.m = int(m)
        self.beta = float(beta)   # mixing for the *newest* Picard residual
        self.lam = float(lam)
        self.logger = get_logger(self.comm, verbose=verbose, name="Anderson")

        self._x_hist: deque[np.ndarray] = deque(maxlen=self.m + 1)
        self._r_hist: deque[np.ndarray] = deque(maxlen=self.m + 1)

        # restart / health bookkeeping
        self._reject_streak = 0
        self._best_picard_res = np.inf
        self._pending_reset = False
        self.restart_on_reject_k = int(restart_on_reject_k)
        self.restart_on_stall = float(restart_on_stall)
        self.restart_on_cond = float(restart_on_cond)
        self.step_limit_factor = float(step_limit_factor)
        self.safeguard_abs_floor = float(safeguard_abs_floor)
        self.gamma_decay_p = float(gamma_decay_p)

    def reset(self) -> None:
        self._x_hist.clear()
        self._r_hist.clear()
        self._reject_streak = 0
        self._best_picard_res = np.inf
        self._pending_reset = False

    # -- core linear algebra --

    def _build_gram_global(self, r_list: Sequence[np.ndarray]) -> np.ndarray:
        """Global Gram matrix H = R R^T with R = [r_i]^T stacked locally then MPI-reduced."""
        if len(r_list) == 0:
            return np.zeros((0, 0), dtype=float)
        R_loc = np.vstack(r_list)         # (p, n_loc)
        H_loc = R_loc @ R_loc.T           # (p, p)
        return self.comm.allreduce(H_loc, op=MPI.SUM)

    def _solve_kkt_weights(self, H: np.ndarray, lam_eff: float) -> np.ndarray:
        """
        Solve min ||alpha||_{H+lam_eff I}  s.t. 1^T alpha = 1 via a (p+1)x(p+1) KKT system.
        """
        p = H.shape[0]
        alpha = np.zeros(p, dtype=float)
        if p == 0:
            return alpha

        if self.comm.rank == 0:
            K = np.zeros((p + 1, p + 1), dtype=float)
            K[:p, :p] = H + lam_eff * np.eye(p)
            K[:p, p] = 1.0
            K[p, :p] = 1.0
            rhs = np.zeros(p + 1, dtype=float)
            rhs[p] = 1.0
            sol = np.linalg.solve(K, rhs)
            alpha = sol[:p]
        self.comm.Bcast(alpha, root=0)
        return alpha

    def _cond_number(self, H: np.ndarray, lam_eff: float) -> float:
        """Cheap condition estimate of (H + lam_eff I)."""
        p = H.shape[0]
        if p == 0:
            return 1.0
        w = np.linalg.eigvalsh(H + lam_eff * np.eye(p))
        w = np.clip(w, 0.0, None)
        wmax = float(np.max(w))
        wmin = float(np.min(w))
        eps = 1e-30
        return wmax / max(wmin, eps)

    # -- main entry point --

    def mix(self,
            x_old: np.ndarray,
            x_raw: np.ndarray,
            mask_fixed: Optional[np.ndarray] = None,
            # Stopping / safeguard hooks use *weighted projected residual* supplied by caller
            proj_residual_norm: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
            gamma: float = 0.05,
            use_safeguard: bool = True,
            backtrack_max: int = 6) -> Tuple[np.ndarray, Dict]:
        """
        Produce a mixed iterate x_new given a Picard/GS output x_raw.

        Returns
        -------
        x_new, info_dict
        """
        # Honor pending soft-reset (scheduled on previous call)
        if self._pending_reset:
            self.reset()

        # Fixed-point residual (masked)
        r = x_raw - x_old
        if mask_fixed is not None:
            r = r.copy()
            r[mask_fixed] = 0.0

        # Seed histories
        self._x_hist.append(x_old.copy())
        self._r_hist.append(r)

        p = len(self._r_hist)
        info: Dict[str, object] = {"aa_hist": p, "accepted": True, "backtracks": 0, "r_norm": None, "r_proxy_norm": None,
                                   "restart_reason": ""}

        # If history not long enough, use damped Picard
        if p == 1:
            x_new = x_old + self.beta * r
            if mask_fixed is not None and mask_fixed.any():
                x_new[mask_fixed] = x_raw[mask_fixed]
            # Metrics
            if proj_residual_norm is not None:
                r_norm = proj_residual_norm(x_old, x_raw, x_raw)
                rp_norm = proj_residual_norm(x_old, x_new, x_raw)
                info["r_norm"] = r_norm
                info["r_proxy_norm"] = rp_norm
                self._best_picard_res = min(self._best_picard_res, r_norm)
                                # Smoothly decay gamma near the absolute floor to avoid over-rejecting tiny steps
                gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
                # Only apply safeguard when we're above the absolute floor
                if use_safeguard and (r_norm > self.safeguard_abs_floor) and rp_norm > (1.0 - gamma_eff) * r_norm:
# fall back to pure Picard (beta=1) if proxy fails badly
                    x_new = x_raw.copy()
                    info["accepted"] = False
                    self._reject_streak += 1
                else:
                    self._reject_streak = 0
            return x_new, info

        # Build weights
        H = self._build_gram_global(list(self._r_hist))
        lam_eff = self.lam * max(float(np.trace(H)) / max(p, 1), 1.0)
        condH = self._cond_number(H, lam_eff)
        alpha = self._solve_kkt_weights(H, lam_eff)

        # Candidate AA step with current beta — accumulation avoids building a (p,n) stack
        y = np.zeros_like(x_old)
        for a_i, xi, ri in zip(alpha, self._x_hist, self._r_hist):
            # y += a_i * (xi + beta * ri)
            y += a_i * xi
            y += (a_i * self.beta) * ri
        s = y - x_old

        # Step limiter: keep ||s|| = O(||r||) to avoid wild excursions
        s_norm = _global_norm(self.comm, s)
        r_norm_unweighted = _global_norm(self.comm, r) + 1e-300
        if s_norm > self.step_limit_factor * r_norm_unweighted:
            s *= (self.step_limit_factor * r_norm_unweighted) / s_norm

        # Compose candidate
        x_cand = x_old + s

        # Safeguard / backtracking using *weighted projected-residual proxy*
        if proj_residual_norm is not None:
            r_norm = proj_residual_norm(x_old, x_raw, x_raw)
            rp_norm = proj_residual_norm(x_old, x_cand, x_raw)
            info["r_norm"] = r_norm
            info["r_proxy_norm"] = rp_norm
            info["condH"] = condH

            # Update best Picard residual
            self._best_picard_res = min(self._best_picard_res, r_norm)

            gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
            if use_safeguard and (r_norm > self.safeguard_abs_floor) and rp_norm > (1.0 - gamma_eff) * r_norm:
                # backtrack along s: x = x_old + theta * s, theta in (0,1]
                theta = 0.5
                bt = 0
                accepted = False
                while bt < backtrack_max:
                    x_try = x_old + theta * s
                    rp_try = proj_residual_norm(x_old, x_try, x_raw)
                    if rp_try <= (1.0 - gamma_eff) * r_norm:
                        x_cand = x_try
                        accepted = True
                        break
                    theta *= 0.5
                    bt += 1
                if not accepted:
                    x_cand = x_raw.copy()  # fallback to Picard
                    info["accepted"] = False
                    self._reject_streak += 1
                else:
                    self._reject_streak = 0
                info["backtracks"] = bt
            else:
                self._reject_streak = 0

            # --- Restart heuristics (schedule for next call) ---
            if self._reject_streak >= self.restart_on_reject_k:
                self._pending_reset = True
                info["restart_reason"] = f"reject_streak>={self.restart_on_reject_k}"
            elif r_norm > self.restart_on_stall * self._best_picard_res:
                self._pending_reset = True
                info["restart_reason"] = f"stall>(x{self.restart_on_stall:.2f})"
            elif condH > self.restart_on_cond:
                self._pending_reset = True
                info["restart_reason"] = f"illcond~{condH:.1e}"

        # Enforce Dirichlet DOFs from *fresh* GS result
        if mask_fixed is not None and mask_fixed.any():
            x_cand = x_cand.copy()
            x_cand[mask_fixed] = x_raw[mask_fixed]

        return x_cand, info
