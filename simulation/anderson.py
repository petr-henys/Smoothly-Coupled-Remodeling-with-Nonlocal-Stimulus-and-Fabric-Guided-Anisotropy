"""Anderson acceleration with MPI-collective Gram matrix and adaptive restart."""

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
        """Solve equality-constrained LS: min ||α||²_{H+λI} s.t. 1ᵀα=1."""
        p = H.shape[0]
        if p == 0:
            return np.zeros(0, dtype=float)
        Hp = H + lam_eff * np.eye(p)
        one = np.ones(p, dtype=float)
        
        # Use lstsq for robustness against singular matrices
        y, _, _, _ = np.linalg.lstsq(Hp, one, rcond=None)
        
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
        proj_residual_norm: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
        gamma: float = 0.05,
        use_safeguard: bool = True,
        backtrack_max: int = 6,
    ) -> Tuple[np.ndarray, Dict]:
        """Mix iterate via Anderson or damped Picard.
        
        Returns (x_new, info_dict) with acceptance, backtracking, and restart info.
        """
        # Execute pending reset
        if self.pending_reset:
            self.reset()

        # Fixed-point residual (mask Dirichlet DOFs)
        r = x_raw - x_old
        if mask_fixed is not None:
            r = r.copy()
            r[mask_fixed] = 0.0

        # Update history
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
            return self._picard_step(x_old, x_raw, r, mask_fixed, proj_residual_norm, 
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

        # Limit step size using the same metric as safeguarding (explicit, no fallbacks)
        if proj_residual_norm is not None:
            s_proxy = proj_residual_norm(x_old, x_old + s, x_raw)
            r_proxy = proj_residual_norm(x_old, x_raw, x_raw) + 1e-300
            if s_proxy > self.step_limit_factor * r_proxy:
                s *= (self.step_limit_factor * r_proxy) / max(s_proxy, 1e-300)

        x_cand = x_old + s

        # Safeguard with backtracking
        if proj_residual_norm is not None:
            r_norm = proj_residual_norm(x_old, x_raw, x_raw)
            rp_norm = proj_residual_norm(x_old, x_cand, x_raw)
            info["r_norm"] = r_norm
            info["r_proxy_norm"] = rp_norm
            info["condH"] = self._condition_number(H, lam_eff)

            self.best_picard_res = min(self.best_picard_res, r_norm)

            # Adaptive gamma near convergence
            gamma_eff = gamma * (r_norm / (r_norm + self.safeguard_abs_floor)) ** self.gamma_decay_p
            
            if use_safeguard and r_norm > self.safeguard_abs_floor and rp_norm > (1.0 - gamma_eff) * r_norm:
                # Backtrack from x_cand towards x_raw (Picard step)
                x_cand, accepted, bt = self._backtrack(
                    x_cand, x_raw, r_norm, gamma_eff, proj_residual_norm, backtrack_max, x_old
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

        # Enforce Dirichlet DOFs
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
        """Damped Picard step when history is insufficient."""
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
        proj_residual_norm: Callable,
        backtrack_max: int,
        x_old: np.ndarray,
    ) -> Tuple[np.ndarray, bool, int]:
        """Backtrack from x_cand towards x_raw (Picard step)."""
        # We interpolate x_try = x_raw + theta * (x_cand - x_raw)
        # theta=1 => x_cand (failed), theta=0 => x_raw (safe)
        diff = x_cand - x_raw
        theta = 0.5
        
        for bt in range(backtrack_max):
            x_try = x_raw + theta * diff
            # Note: proj_residual_norm(x_old, x_test, x_raw) uses x_raw as base if x_test != x_raw
            rp_try = proj_residual_norm(x_old, x_try, x_raw)
            
            if rp_try <= (1.0 - gamma_eff) * r_norm:
                return x_try, True, bt
            theta *= 0.5
            
        # Backtracking failed → fallback to Picard
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