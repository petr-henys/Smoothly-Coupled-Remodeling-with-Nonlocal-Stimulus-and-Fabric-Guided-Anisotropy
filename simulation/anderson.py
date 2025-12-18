"""Anderson acceleration with safeguard and restart."""

from collections import deque
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np
from mpi4py import MPI

from simulation.logger import get_logger


class _Anderson:
    """Anderson mixing for fixed-point iterations (MPI-aware).

    Builds a global Gram matrix from residual history, computes mixing weights,
    and applies safeguards (step limiting/backtracking/restart).
    """

    def __init__(
        self,
        comm: MPI.Comm,
        m: int = 8,
        beta: float = 1.0,
        lam: float = 1e-10,
        restart_on_reject_k: int = 2,
        restart_on_stall: float = 1.10,
        restart_on_cond: float = 1e12,
        step_limit_factor: float = 2.0,
        safeguard_abs_floor: float = 1e-10,
        gamma_decay_p: float = 0.5,
    ):
        self.comm = comm
        self.m = m
        self.beta = beta
        self.lam = lam
        self.logger = get_logger(comm, name="Anderson")

        # History buffers
        self.x_hist: deque[np.ndarray] = deque(maxlen=m + 1)
        self.r_hist: deque[np.ndarray] = deque(maxlen=m + 1)

        # Restart tracking
        self.reject_streak = 0
        self.best_picard_res = np.inf
        self.pending_reset = False
        
        # Restart thresholds
        self.restart_on_reject_k = restart_on_reject_k
        self.restart_on_stall = restart_on_stall
        self.restart_on_cond = restart_on_cond
        
        # Step control
        self.step_limit_factor = step_limit_factor
        self.safeguard_abs_floor = safeguard_abs_floor
        self.gamma_decay_p = gamma_decay_p

    def reset(self) -> None:
        """Clear history."""
        self.x_hist.clear()
        self.r_hist.clear()
        self.reject_streak = 0
        self.best_picard_res = np.inf
        self.pending_reset = False

    def _build_gram(self, r_list: Sequence[np.ndarray]) -> np.ndarray:
        """Global Gram matrix H = R R^T."""
        if len(r_list) == 0:
            return np.zeros((0, 0), dtype=float)
        R_loc = np.vstack(r_list)
        H_loc = R_loc @ R_loc.T
        return self.comm.allreduce(H_loc, op=MPI.SUM)

    def _solve_kkt(self, H: np.ndarray, lam_eff: float) -> np.ndarray:
        """Solve min ||alpha||_H s.t. 1^T alpha = 1."""
        p = H.shape[0]
        if p == 0:
            return np.zeros(0, dtype=float)
        Hp = H + lam_eff * np.eye(p)
        one = np.ones(p, dtype=float)
        try:
            y = np.linalg.solve(Hp, one)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Hp + 1e-15 * np.eye(p))
            w = np.clip(w, 1e-15, None)
            y = V @ (V.T @ one / w)
        denom = float(one @ y)
        if abs(denom) < 1e-30:
            return np.full(p, 1.0 / p, dtype=float)
        return y / denom

    def _condition_number(self, H: np.ndarray, lam_eff: float) -> float:
        """Condition number of H + lam*I."""
        p = H.shape[0]
        if p == 0:
            return 1.0
        w = np.linalg.eigvalsh(H + lam_eff * np.eye(p))
        w = np.clip(w, 0.0, None)
        return float(np.max(w) / max(np.min(w), 1e-30))

    def mix(
        self,
        x_old: np.ndarray,
        x_raw: np.ndarray,
        mask_fixed: Optional[np.ndarray] = None,
        proj_residual_norm: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
        gamma: float = 0.05,
        use_safeguard: bool = True,
        backtrack_max: int = 6,
    ) -> Tuple[np.ndarray, Dict]:
        
        if self.pending_reset:
            self.reset()

        r = x_raw - x_old
        if mask_fixed is not None:
            r = r.copy()
            r[mask_fixed] = 0.0

        self.x_hist.append(x_old.copy())
        self.r_hist.append(r)
        p = len(self.r_hist)

        info: Dict = {
            "aa_hist": p,
            "accepted": True,
            "backtracks": 0,
            "r_norm": None,
            "r_proxy_norm": None,
            "restart_reason": "",
        }

        # First iterate: fall back to a Picard step (no safeguard).
        if p == 1:
            return self._picard_step(x_old, x_raw, r, mask_fixed, proj_residual_norm, 
                                     gamma, False, info)

        # Anderson acceleration
        H = self._build_gram(list(self.r_hist))
        lam_eff = self.lam * max(float(np.trace(H)) / p, 1.0)
        alpha = self._solve_kkt(H, lam_eff)

        y = np.zeros_like(x_old)
        for a_i, xi, ri in zip(alpha, self.x_hist, self.r_hist):
            y += a_i * (xi + self.beta * ri)
        s = y - x_old

        # Compute r_norm once (used for step limiting and safeguard)
        r_norm = None
        if proj_residual_norm is not None:
            r_norm = proj_residual_norm(x_old, x_raw, x_raw)
            s_proxy = proj_residual_norm(x_old, x_old + s, x_raw)
            r_proxy = r_norm + 1e-300  # Reuse r_norm instead of recomputing
            if s_proxy > self.step_limit_factor * r_proxy:
                s *= (self.step_limit_factor * r_proxy) / max(s_proxy, 1e-300)

        x_cand = x_old + s

        if proj_residual_norm is not None:
            rp_norm = proj_residual_norm(x_old, x_cand, x_raw)
            info["r_norm"] = r_norm
            info["r_proxy_norm"] = rp_norm
            info["condH"] = self._condition_number(H, lam_eff)

            self.best_picard_res = min(self.best_picard_res, r_norm)

            gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
            
            if use_safeguard and r_norm > self.safeguard_abs_floor and rp_norm > (1.0 - gamma_eff) * r_norm:
                x_cand, accepted, bt = self._backtrack(
                    x_old, s, x_raw, r_norm, gamma_eff, proj_residual_norm, backtrack_max
                )
                info["backtracks"] = bt
                if not accepted:
                    info["accepted"] = False
                    self.reject_streak += 1
                else:
                    self.reject_streak = 0
            else:
                self.reject_streak = 0

            self._check_restart(r_norm, info["condH"], info)

        if mask_fixed is not None and mask_fixed.any():
            x_cand[mask_fixed] = x_raw[mask_fixed]

        return x_cand, info

    def _picard_step(
        self,
        x_old: np.ndarray,
        x_raw: np.ndarray,
        r: np.ndarray,
        mask_fixed: Optional[np.ndarray],
        proj_residual_norm: Optional[Callable],
        gamma: float,
        use_safeguard: bool,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Damped Picard: x_new = x_old + beta*r."""
        x_new = x_old + self.beta * r
        
        if mask_fixed is not None and mask_fixed.any():
            x_new[mask_fixed] = x_raw[mask_fixed]

        if proj_residual_norm is not None:
            r_norm = proj_residual_norm(x_old, x_raw, x_raw)
            rp_norm = proj_residual_norm(x_old, x_new, x_raw)
            info["r_norm"] = r_norm
            info["r_proxy_norm"] = rp_norm
            self.best_picard_res = min(self.best_picard_res, r_norm)

            gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
            if use_safeguard and r_norm > self.safeguard_abs_floor and rp_norm > (1.0 - gamma_eff) * r_norm:
                x_new = x_raw.copy()
                info["accepted"] = False
                self.reject_streak += 1
            else:
                self.reject_streak = 0

        return x_new, info

    def _backtrack(
        self,
        x_old: np.ndarray,
        s: np.ndarray,
        x_raw: np.ndarray,
        r_norm: float,
        gamma_eff: float,
        proj_residual_norm: Callable,
        backtrack_max: int,
    ) -> Tuple[np.ndarray, bool, int]:
        theta = 0.5
        for bt in range(backtrack_max):
            x_try = x_old + theta * s
            rp_try = proj_residual_norm(x_old, x_try, x_raw)
            if rp_try <= (1.0 - gamma_eff) * r_norm:
                return x_try, True, bt
            theta *= 0.5
        return x_raw.copy(), False, backtrack_max

    def _check_restart(self, r_norm: float, condH: float, info: Dict) -> None:
        if self.reject_streak >= self.restart_on_reject_k:
            self.pending_reset = True
            info["restart_reason"] = f"reject_streak>={self.restart_on_reject_k}"
        elif r_norm > self.restart_on_stall * self.best_picard_res:
            self.pending_reset = True
            info["restart_reason"] = f"stall>(x{self.restart_on_stall:.2f})"
        elif condH > self.restart_on_cond:
            self.pending_reset = True
            info["restart_reason"] = f"illcond~{condH:.1e}"