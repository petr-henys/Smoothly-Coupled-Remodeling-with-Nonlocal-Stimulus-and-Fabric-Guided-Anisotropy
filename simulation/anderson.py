"""MPI-aware Anderson acceleration with restart and step limiting."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
from mpi4py import MPI

from simulation.logger import get_logger


class Anderson:
    """Implements Anderson acceleration (Pulay mixing) with adaptive restart and step limiting.
    
    Pulay (type-II/DIIS) residual-minimization form with Tikhonov regularization.
    Restarts on ill-conditioning or stall. Limits step size to prevent divergence.
    """

    # Numerical constants
    _TINY: float = 1e-8
    _EIG_REL: float = 1e-12  # Relative eigenvalue floor for fallback solver

    def __init__(
        self,
        comm: MPI.Comm,
        m: int,
        beta: float,
        lam: float,
        restart_on_stall: float,
        restart_on_cond: float,
        step_limit_factor: float,
        restart_stall_window: int,
        restart_stall_patience: int,
    ):
        """Initialize Anderson accelerator.
        
        Args:
            comm: MPI communicator for global reductions.
            m: History depth (max previous iterates stored).
            beta: Relaxation factor for mixing.
            lam: Tikhonov regularization (relative to Gram diagonal).
            restart_on_stall: Trigger restart if residual > this × recent best.
            restart_on_cond: Trigger restart if Gram condition number exceeds this.
            step_limit_factor: Max step size as multiple of Picard step.
            restart_stall_window: Window size for tracking recent-best residual.
            restart_stall_patience: Consecutive stall iters before restart.
        """
        self.comm = comm
        self.m = m
        self.beta = beta
        self.lam = lam
        self.restart_on_stall = restart_on_stall
        self.restart_on_cond = restart_on_cond
        self.step_limit_factor = step_limit_factor
        self.restart_stall_window = restart_stall_window
        self.restart_stall_patience = restart_stall_patience

        self.logger = get_logger(self.comm, name="Anderson")

        # History buffers (maxlen = m+1 to store current + m previous)
        self.x_hist: Deque[np.ndarray] = deque(maxlen=self.m + 1)
        self.r_hist: Deque[np.ndarray] = deque(maxlen=self.m + 1)

        # Restart state
        self._stall_streak = 0
        self._recent_res: Deque[float] = deque(maxlen=max(self.restart_stall_window, 2))
        self._pending_reset = False

    def reset(self) -> None:
        """Clear history and restart state."""
        self.x_hist.clear()
        self.r_hist.clear()
        self._stall_streak = 0
        self._recent_res.clear()
        self._pending_reset = False

    # ----------------------------- Linear algebra -----------------------------

    def _gdot(self, a: np.ndarray, b: np.ndarray) -> float:
        """Global dot product across MPI ranks."""
        local_dot = np.dot(a, b)
        return self.comm.allreduce(local_dot, op=MPI.SUM)

    def _rel_step(self, x_old: np.ndarray, x_new: np.ndarray, x_ref: np.ndarray) -> float:
        """Compute ||x_new - x_old|| / ||x_ref|| globally. Returns absolute norm if ref ≈ 0."""
        diff = x_new - x_old
        diff_norm_sq = self._gdot(diff, diff)
        ref_norm_sq = self._gdot(x_ref, x_ref)
        if ref_norm_sq <= self._TINY:
            return np.sqrt(diff_norm_sq)
        return np.sqrt(diff_norm_sq / ref_norm_sq)

    def _build_gram(self, r_list: List[np.ndarray]) -> np.ndarray:
        """Build Gram matrix H[i,j] = <r_i, r_j> with MPI reduction."""
        p = len(r_list)
        if p == 0:
            return np.zeros((0, 0), dtype=float)

        # Build local upper triangle, then symmetrize
        H_local = np.empty((p, p), dtype=float)
        for i in range(p):
            H_local[i, i] = np.dot(r_list[i], r_list[i])
            for j in range(i + 1, p):
                dot_ij = np.dot(r_list[i], r_list[j])
                H_local[i, j] = dot_ij
                H_local[j, i] = dot_ij

        return self.comm.allreduce(H_local, op=MPI.SUM)

    def _solve_weights(self, H: np.ndarray, lam_eff: float) -> Tuple[np.ndarray, str]:
        """Solve (H + λI)α = 1 with normalization sum(α) = 1.
        
        Returns (alpha, method) where method is 'solve', 'eigh', or 'uniform'.
        """
        p = H.shape[0]
        if p == 0:
            return np.zeros(0, dtype=float), "solve"

        H_reg = H + lam_eff * np.eye(p)
        ones = np.ones(p, dtype=float)
        method = "solve"

        try:
            y = np.linalg.solve(H_reg, ones)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition for singular/near-singular case
            method = "eigh"
            eigvals, eigvecs = np.linalg.eigh(H_reg)
            max_eig = np.max(np.abs(eigvals)) if eigvals.size else 0.0
            floor = self._EIG_REL * max(max_eig, abs(lam_eff), self._TINY)
            eigvals_safe = np.clip(eigvals, floor, None)
            y = eigvecs @ (eigvecs.T @ ones / eigvals_safe)

        denom = ones @ y
        if not np.isfinite(denom) or abs(denom) <= self._TINY:
            return np.full(p, 1.0 / p, dtype=float), "uniform"
        return y / denom, method

    def _cond_number(self, H: np.ndarray, lam_eff: float) -> float:
        """Condition number of (H + λI)."""
        p = H.shape[0]
        if p == 0:
            return 1.0
        eigvals = np.linalg.eigvalsh(H + lam_eff * np.eye(p))
        eigvals = np.clip(eigvals, 0.0, None)
        return np.max(eigvals) / max(np.min(eigvals), self._TINY)

    # ----------------------------- Main update --------------------------------

    def mix(self, x_old: np.ndarray, x_raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compute accelerated iterate from current state and Picard image.
        
        Args:
            x_old: Current iterate (owned DOFs only).
            x_raw: Result of one Picard sweep g(x_old).
            
        Returns:
            (x_new, info) with accelerated iterate and diagnostics dict.
        """
        if self._pending_reset:
            self.reset()

        # Picard residual
        residual = x_raw - x_old

        # Update history
        self.x_hist.append(x_old.copy())
        self.r_hist.append(residual.copy())
        p = len(self.r_hist)

        # Build Gram matrix
        r_list = list(self.r_hist)
        H = self._build_gram(r_list)
        
        # Prune stale history if newest residual is orders of magnitude smaller
        diag = np.diag(H)
        if diag.size > 0:
            r2_last = diag[-1]
            r2_max = np.max(diag)
            if r2_last > 0.0 and r2_max > 0.0 and r2_last < 1e-8 * r2_max:
                self._prune_history_to_last()
                p = 1
                H = self._build_gram(list(self.r_hist))

        # Scale regularization to Gram matrix magnitude
        avg_diag = np.trace(H) / max(p, 1)
        lam_eff = self.lam * max(avg_diag, self._TINY)

        cond_H = self._cond_number(H, lam_eff)

        # Compute residual norm for diagnostics and restart logic
        r_norm = self._rel_step(x_old, x_raw, x_raw)
        self._recent_res.append(r_norm)

        # Check for restart conditions *before* forming the AA step to avoid
        # the "last toxic Anderson step" right before a restart.
        restart_reason = self._check_restart(r_norm, cond_H)
        if restart_reason:
            x_picard = x_old + self.beta * residual
            step_norm = self._rel_step(x_old, x_picard, x_raw)
            info: Dict = {
                "aa_hist": p - 1,
                "condH": cond_H,
                "r_norm": r_norm,
                "step_norm": step_norm,
                "alpha_method": "skipped",
                "limited": False,
                "restart_reason": restart_reason,
                "accepted": True,  # For API compatibility
                "aa_off": True,  # This iteration used damped Picard
            }
            return x_picard, info

        # Solve for mixing weights (safe: restart not triggered)
        alpha, alpha_method = self._solve_weights(H, lam_eff)

        # Compute accelerated iterate (Pulay/DIIS form)
        x_aa = self._compute_accelerated_iterate(x_old, residual, alpha, p)

        # Step limiting
        step_norm = self._rel_step(x_old, x_aa, x_raw)
        limited = False
        max_step = self.step_limit_factor * max(r_norm, self._TINY)

        if step_norm > max_step:
            limited = True
            scale_factor = max_step / max(step_norm, self._TINY)
            x_aa = x_old + scale_factor * (x_aa - x_old)
            step_norm = self._rel_step(x_old, x_aa, x_raw)

        info: Dict = {
            "aa_hist": p - 1,
            "condH": cond_H,
            "r_norm": r_norm,
            "step_norm": step_norm,
            "alpha_method": alpha_method,
            "limited": limited,
            "restart_reason": restart_reason,
            "accepted": True,  # For API compatibility
            "aa_off": False,
        }

        return x_aa, info

    def _prune_history_to_last(self) -> None:
        """Keep only the most recent entry in history buffers."""
        self.x_hist = deque([self.x_hist[-1]], maxlen=self.m + 1)
        self.r_hist = deque([self.r_hist[-1]], maxlen=self.m + 1)

    def _compute_accelerated_iterate(
        self,
        x_old: np.ndarray,
        residual: np.ndarray,
        alpha: np.ndarray,
        p: int,
    ) -> np.ndarray:
        """Compute x_aa = Σ α_i (x_i + β r_i). Falls back to Picard if p < 2."""
        if p < 2:
            # Fall back to damped Picard for insufficient history
            return x_old + self.beta * residual

        x_aa = np.zeros_like(x_old)
        for a_i, x_i, r_i in zip(alpha, self.x_hist, self.r_hist):
            x_aa += a_i * (x_i + self.beta * r_i)
        return x_aa

    def _check_restart(self, r_norm: float, cond_H: float) -> str:
        """Check restart conditions. Returns reason string or empty if no restart."""
        CONVERGED_THRESHOLD = 1e-12
        
        if r_norm <= CONVERGED_THRESHOLD:
            self._stall_streak = 0
            return ""

        if cond_H > self.restart_on_cond:
            self._pending_reset = True
            self._stall_streak = 0
            return f"cond={cond_H:.1e}"

        # Stall: residual not improving vs recent best
        best_recent = min(self._recent_res) if self._recent_res else r_norm
        if best_recent > 0.0 and r_norm > self.restart_on_stall * best_recent:
            self._stall_streak += 1
        else:
            self._stall_streak = 0

        if self._stall_streak >= self.restart_stall_patience:
            self._pending_reset = True
            self._stall_streak = 0
            return f"stall>{self.restart_on_stall:.2f}x"

        return ""
