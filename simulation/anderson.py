"""MPI-aware Anderson acceleration with restart and step limiting."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Sequence, Tuple

import numpy as np
from mpi4py import MPI

from simulation.logger import get_logger


class Anderson:
    """Anderson mixing with history restart and step limiting.
    
    Implements Type-I Anderson acceleration (Walker-Ni formulation) with:
    - Tikhonov regularization scaled to residual magnitude
    - Step limiting to prevent overshooting
    - History restart on ill-conditioning or stall detection
    """

    # Numerical constants (internal, not configurable)
    _TINY: float = 1e-300
    _EIG_REL: float = 1e-12  # Relative eigen floor for fallback solver

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
            comm: MPI communicator.
            m: History depth (number of previous iterates to use).
            beta: Mixing parameter (relaxation factor).
            lam: Relative Tikhonov regularization (scaled by Gram diagonal).
            restart_on_stall: Restart threshold (ratio to recent-best residual).
            restart_on_cond: Restart threshold for Gram matrix condition number.
            step_limit_factor: Maximum step size relative to Picard residual.
            restart_stall_window: Window size for recent-best residual tracking.
            restart_stall_patience: Consecutive stall iterations before restart.
        """
        self.comm = comm
        self.m = int(m)
        self.beta = float(beta)
        self.lam = float(lam)
        self.restart_on_stall = float(restart_on_stall)
        self.restart_on_cond = float(restart_on_cond)
        self.step_limit_factor = float(step_limit_factor)
        self.restart_stall_window = int(restart_stall_window)
        self.restart_stall_patience = int(restart_stall_patience)

        self.logger = get_logger(self.comm, name="Anderson")

        # History buffers
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
        return float(self.comm.allreduce(float(np.dot(a, b)), op=MPI.SUM))

    def _rel_step(self, x_old: np.ndarray, x_new: np.ndarray, x_ref: np.ndarray) -> float:
        """Relative step: ||x_new - x_old|| / ||x_ref|| in global L2."""
        d = x_new - x_old
        d2 = self._gdot(d, d)
        r2 = self._gdot(x_ref, x_ref)
        if r2 <= self._TINY:
            return float(np.sqrt(d2))
        return float(np.sqrt(d2 / r2))

    def _build_gram(self, r_list: Sequence[np.ndarray]) -> np.ndarray:
        """Build global Gram matrix H_ij = <r_i, r_j> via MPI allreduce."""
        p = len(r_list)
        if p == 0:
            return np.zeros((0, 0), dtype=float)

        H_loc = np.empty((p, p), dtype=float)
        for i in range(p):
            for j in range(i, p):
                val = float(np.dot(r_list[i], r_list[j]))
                H_loc[i, j] = val
                H_loc[j, i] = val

        return self.comm.allreduce(H_loc, op=MPI.SUM)

    def _solve_weights(self, H: np.ndarray, lam_eff: float) -> Tuple[np.ndarray, str]:
        """Solve for optimal mixing weights: min ||alpha||_H s.t. sum(alpha) = 1.
        
        Returns:
            (alpha, method) where method is 'solve', 'eigh', or 'uniform'.
        """
        p = int(H.shape[0])
        if p == 0:
            return np.zeros(0, dtype=float), "solve"

        Hp = H + lam_eff * np.eye(p)
        one = np.ones(p, dtype=float)
        method = "solve"

        try:
            y = np.linalg.solve(Hp, one)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition
            method = "eigh"
            w, V = np.linalg.eigh(Hp)
            wmax = float(np.max(np.abs(w))) if w.size else 0.0
            scale = max(wmax, abs(lam_eff), self._TINY)
            w = np.clip(w, self._EIG_REL * scale, None)
            y = V @ (V.T @ one / w)

        denom = float(one @ y)
        if (not np.isfinite(denom)) or abs(denom) <= 1e-30:
            return np.full(p, 1.0 / p, dtype=float), "uniform"
        return y / denom, method

    def _cond_number(self, H: np.ndarray, lam_eff: float) -> float:
        """Estimate condition number of regularized Gram matrix."""
        p = int(H.shape[0])
        if p == 0:
            return 1.0
        w = np.linalg.eigvalsh(H + lam_eff * np.eye(p))
        w = np.clip(w, 0.0, None)
        return float(np.max(w) / max(float(np.min(w)), 1e-30))

    # ----------------------------- Main update --------------------------------

    def mix(self, x_old: np.ndarray, x_raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compute accelerated iterate from current state and Picard image.
        
        Args:
            x_old: Current iterate (scaled, owned DOFs).
            x_raw: Picard image g(x_old) after one block sweep (scaled, owned DOFs).
            
        Returns:
            (x_new, info) where x_new is the mixed iterate and info holds diagnostics.
        """
        if self._pending_reset:
            self.reset()

        # Picard residual
        r = x_raw - x_old

        # Update history
        self.x_hist.append(x_old.copy())
        self.r_hist.append(r.copy())
        p = len(self.r_hist)

        # Build Gram matrix with scale-aware regularization
        H = self._build_gram(list(self.r_hist))
        
        # Prune stale history if newest residual is orders of magnitude smaller
        diag = np.diag(H)
        r2_last = float(diag[-1]) if diag.size else 0.0
        r2_max = float(np.max(diag)) if diag.size else 0.0
        if r2_last > 0.0 and r2_max > 0.0 and r2_last < 1e-8 * r2_max:
            self.x_hist = deque([self.x_hist[-1]], maxlen=self.m + 1)
            self.r_hist = deque([self.r_hist[-1]], maxlen=self.m + 1)
            p = 1
            H = self._build_gram(list(self.r_hist))

        # Scale regularization to Gram matrix magnitude
        avg_diag = float(np.trace(H)) / max(p, 1)
        lam_eff = self.lam * max(avg_diag, self._TINY)

        # Solve for mixing weights
        alpha, alpha_method = self._solve_weights(H, lam_eff)
        condH = self._cond_number(H, lam_eff)

        # Compute residual norm for diagnostics and restart logic
        r_norm = self._rel_step(x_old, x_raw, x_raw)
        self._recent_res.append(r_norm)

        # Compute accelerated iterate (Walker-Ni Type-I form)
        if p >= 2:
            x_aa = np.zeros_like(x_old)
            for a_i, x_i, r_i in zip(alpha, self.x_hist, self.r_hist):
                x_aa += float(a_i) * (x_i + self.beta * r_i)
        else:
            # Fall back to damped Picard
            x_aa = x_old + self.beta * r

        # Step limiting
        step = x_aa - x_old
        step_norm = self._rel_step(x_old, x_aa, x_raw)
        limited = False

        if step_norm > self.step_limit_factor * max(r_norm, self._TINY):
            limited = True
            scale = (self.step_limit_factor * max(r_norm, self._TINY)) / max(step_norm, self._TINY)
            step *= scale
            x_aa = x_old + step
            step_norm = self._rel_step(x_old, x_aa, x_raw)

        # Check for restart conditions
        restart_reason = self._check_restart(r_norm, condH)

        # If restart triggered, reject AA step *immediately* and fall back to damped Picard.
        #
        # Rationale: if the Gram matrix is ill-conditioned or residual is stalling, using the
        # just-computed AA combination can be actively harmful. Resetting on the *next* call
        # leaves one unstable step in the iterate history.
        accepted = True
        if restart_reason:
            accepted = False
            x_aa = x_old + self.beta * r
            step_norm = self._rel_step(x_old, x_aa, x_raw)
            limited = False
            # Clear history and stall tracking right away.
            self.reset()

        info: Dict = {
            "aa_hist": int(p - 1),
            "accepted": accepted,
            "condH": float(condH),
            "r_norm": float(r_norm),
            "step_norm": float(step_norm),
            "alpha_method": str(alpha_method),
            "limited": limited,
            "restart_reason": restart_reason,
        }

        return x_aa, info

    def _check_restart(self, r_norm: float, condH: float) -> str:
        """Check restart conditions and update internal state.
        
        Returns:
            Restart reason string (empty if no restart triggered).
        """
        # Skip restart near convergence
        if r_norm <= 1e-12:
            self._stall_streak = 0
            return ""

        # Restart on ill-conditioning
        if condH > self.restart_on_cond:
            self._pending_reset = True
            self._stall_streak = 0
            return f"cond={condH:.1e}"

        # Restart on stall (residual not improving relative to recent best)
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
