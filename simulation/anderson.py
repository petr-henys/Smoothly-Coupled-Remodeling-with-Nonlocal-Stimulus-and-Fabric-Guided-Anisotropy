from collections import deque
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np
from mpi4py import MPI

from simulation.logger import get_logger


class _Anderson:
    """Anderson acceleration with backtracking and restart heuristics."""

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
        verbose: bool = True,
    ):
        self.comm = comm
        self.m = m
        self.beta = beta
        self.lam = lam
        self.logger = get_logger(comm, verbose=verbose, name="Anderson")

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
        """Clear history and reset tracking."""
        self.x_hist.clear()
        self.r_hist.clear()
        self.reject_streak = 0
        self.best_picard_res = np.inf
        self.pending_reset = False

    def _build_gram(self, r_list: Sequence[np.ndarray]) -> np.ndarray:
        """MPI-global Gram matrix H = R R^T."""
        if len(r_list) == 0:
            return np.zeros((0, 0), dtype=float)
        R_loc = np.vstack(r_list)  # (p, n_loc)
        H_loc = R_loc @ R_loc.T     # (p, p)
        return self.comm.allreduce(H_loc, op=MPI.SUM)
    
    def _solve_kkt(self, H: np.ndarray, lam_eff: float) -> np.ndarray:
        """Solve min ||alpha||_{H+lam I} s.t. 1^T alpha = 1 using closed-form.
        
        alpha = (H+lam I)^{-1} 1 / (1^T (H+lam I)^{-1} 1)
        """
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
        """Condition number κ(H + λI) via eigenvalues."""
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
        norm_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        gamma: float = 0.05,
        use_safeguard: bool = True,
        backtrack_max: int = 6,
    ) -> Tuple[np.ndarray, Dict]:
        """Mix iterate via Anderson or damped Picard.
        
        Parameters
        ----------
        x_old : Previous iteration state (x_{k}).
        x_raw : Current Picard iteration result (G(x_{k})).
        norm_func : Callable(a, b) -> float returning ||a - b||.
        """
        if self.pending_reset:
            self.reset()

        # Fixed-point residual r = G(x) - x
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

        # Insufficient history → damped Picard
        if p == 1:
            return self._picard_step(x_old, x_raw, r, mask_fixed, norm_func, 
                                     gamma, use_safeguard, info)

        # Anderson acceleration
        H = self._build_gram(list(self.r_hist))
        lam_eff = self.lam * max(float(np.trace(H)) / p, 1.0)
        alpha = self._solve_kkt(H, lam_eff)

        # Build candidate: x = sum(alpha_i * (x_i + beta*r_i))
        y = np.zeros_like(x_old)
        for a_i, xi, ri in zip(alpha, self.x_hist, self.r_hist):
            y += a_i * (xi + self.beta * ri)
        s = y - x_old

        # Limit step size using r_norm (Picard step size) as the reference scale
        if norm_func is not None:
            # Measure Picard step size: ||G(x) - x||
            r_norm = norm_func(x_raw, x_old)
            
            # Measure Anderson step size: ||x_anderson - x_old||
            s_norm = norm_func(y, x_old)
            
            # Prevent Anderson from taking a step wildly larger than the physics suggests
            if s_norm > self.step_limit_factor * (r_norm + 1e-300):
                scale = (self.step_limit_factor * (r_norm + 1e-300)) / max(s_norm, 1e-300)
                s *= scale

        x_cand = x_old + s

        # Safeguard with backtracking
        if norm_func is not None:
            r_norm = norm_func(x_raw, x_old)       # Reference: Picard step size
            rp_norm = norm_func(x_cand, x_raw)     # Deviation: ||Anderson - Picard||
            
            info["r_norm"] = r_norm
            info["r_proxy_norm"] = rp_norm
            info["condH"] = self._condition_number(H, lam_eff)

            self.best_picard_res = min(self.best_picard_res, r_norm)

            # Adaptive gamma near convergence
            gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
            
            # Condition: If Anderson prediction deviates significantly from Picard prediction relative to the step size
            if use_safeguard and r_norm > self.safeguard_abs_floor and rp_norm > (1.0 - gamma_eff) * r_norm:
                # Backtrack from x_cand towards x_raw
                x_cand, accepted, bt = self._backtrack(
                    x_cand, x_raw, r_norm, gamma_eff, norm_func, backtrack_max
                )
                info["backtracks"] = bt
                if not accepted:
                    info["accepted"] = False
                    self.reject_streak += 1
                else:
                    self.reject_streak = 0
            else:
                self.reject_streak = 0

            # Check restart conditions
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
        norm_func: Optional[Callable],
        gamma: float,
        use_safeguard: bool,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Damped Picard step when history is insufficient."""
        x_new = x_old + self.beta * r
        if mask_fixed is not None and mask_fixed.any():
            x_new[mask_fixed] = x_raw[mask_fixed]

        if norm_func is not None:
            r_norm = norm_func(x_raw, x_old)
            rp_norm = norm_func(x_new, x_raw)
            info["r_norm"] = r_norm
            info["r_proxy_norm"] = rp_norm
            self.best_picard_res = min(self.best_picard_res, r_norm)

            gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
            
            if use_safeguard and r_norm > self.safeguard_abs_floor and rp_norm > (1.0 - gamma_eff) * r_norm:
                # Reject, fallback to pure Picard
                x_new = x_raw.copy()
                info["accepted"] = False
                self.reject_streak += 1
            else:
                self.reject_streak = 0

        return x_new, info

    def _backtrack(
        self,
        x_cand: np.ndarray,
        x_raw: np.ndarray,
        r_norm: float,
        gamma_eff: float,
        norm_func: Callable,
        backtrack_max: int,
    ) -> Tuple[np.ndarray, bool, int]:
        """Backtrack from x_cand towards x_raw."""
        diff = x_cand - x_raw
        theta = 0.5
        
        # Start backtracking. If we enter this loop, we have rejected the full step (theta=1.0)
        # so we count at least 1 backtrack.
        for bt in range(1, backtrack_max + 1):
            x_try = x_raw + theta * diff
            rp_try = norm_func(x_try, x_raw)
            
            if rp_try <= (1.0 - gamma_eff) * r_norm:
                return x_try, True, bt
            theta *= 0.5
            
        # Backtracking failed → fallback to x_raw (Picard)
        return x_raw.copy(), False, backtrack_max

    def _check_restart(self, r_norm: float, condH: float, info: Dict) -> None:
        """Check restart heuristics (rejection streak, stalling, conditioning)."""
        if self.reject_streak >= self.restart_on_reject_k:
            self.pending_reset = True
            info["restart_reason"] = f"reject_streak>={self.restart_on_reject_k}"
        elif r_norm > self.restart_on_stall * self.best_picard_res:
            self.pending_reset = True
            info["restart_reason"] = f"stall>(x{self.restart_on_stall:.2f})"
        elif condH > self.restart_on_cond:
            self.pending_reset = True
            info["restart_reason"] = f"illcond~{condH:.1e}"
