"""MPI-aware Anderson acceleration with safeguard and restart."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Sequence, Tuple

import numpy as np
from mpi4py import MPI

from simulation.logger import get_logger


class Anderson:
    """Anderson mixing with backtracking safeguard and history restart."""

    def __init__(
        self,
        comm: MPI.Comm,
        m: int,
        beta: float,
        lam: float,
        gamma: float,
        safeguard: bool,
        backtrack_max: int,
        restart_on_reject_k: int,
        restart_on_cond: float,
        step_limit_factor: float,
        verbose: bool,
    ):
        self.comm = comm
        self.m = int(m)
        self.beta = float(beta)
        self.lam = float(lam)

        self.gamma = float(gamma)
        self.safeguard = bool(safeguard)
        self.backtrack_max = int(backtrack_max)

        self.restart_on_reject_k = int(restart_on_reject_k)
        self.restart_on_cond = float(restart_on_cond)

        self.step_limit_factor = float(step_limit_factor)
        self.verbose = bool(verbose)

        self.logger = get_logger(self.comm, name="Anderson")

        self.x_hist: Deque[np.ndarray] = deque(maxlen=self.m + 1)
        self.r_hist: Deque[np.ndarray] = deque(maxlen=self.m + 1)

        self.reject_streak = 0
        self.pending_reset = False

        # Numerical floors (internal constants; not exposed as knobs)
        self._tiny = 1e-300
        self._eig_floor = 1e-15

    def reset(self) -> None:
        self.x_hist.clear()
        self.r_hist.clear()
        self.reject_streak = 0
        self.pending_reset = False

    # ------------------------- linear algebra helpers -------------------------

    def _gdot(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self.comm.allreduce(float(np.dot(a, b)), op=MPI.SUM))

    def _rel_step(self, x_old: np.ndarray, x_trial: np.ndarray, x_ref: np.ndarray) -> float:
        """Relative step: ||x_trial-x_old|| / (||x_ref|| + eps) globally."""
        d = x_trial - x_old
        d2 = self._gdot(d, d)
        r2 = self._gdot(x_ref, x_ref)

        # Guard against division by ~0 for vanishing reference fields.
        epsilon = 1e-20

        return float(np.sqrt(d2 / (r2 + epsilon)))

    def _build_gram(self, r_list: Sequence[np.ndarray]) -> np.ndarray:
        """Build the global Gram matrix H_ij = <r_i, r_j>.

        This avoids stacking residuals into a (m × n) dense matrix (which can
        double peak memory for large state vectors). We compute the local Gram
        entries directly and perform a single MPI allreduce on the (m × m) matrix.
        """
        p = len(r_list)
        if p == 0:
            return np.zeros((0, 0), dtype=float)

        H_loc = np.empty((p, p), dtype=float)
        for i in range(p):
            ri = r_list[i]
            # symmetry
            for j in range(i, p):
                val = float(np.dot(ri, r_list[j]))
                H_loc[i, j] = val
                H_loc[j, i] = val

        return self.comm.allreduce(H_loc, op=MPI.SUM)

    def _solve_kkt(self, H: np.ndarray, lam_eff: float) -> tuple[np.ndarray, str]:
        """Solve min ||alpha||_H s.t. 1^T alpha = 1.

        Returns (alpha, method) where method is one of:
        - "solve": direct solve succeeded
        - "eigh": eigen-based fallback used (regularized)
        - "uniform": degenerate constraint normalization -> uniform weights
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
            method = "eigh"
            w, V = np.linalg.eigh(Hp + self._eig_floor * np.eye(p))
            w = np.clip(w, self._eig_floor, None)
            y = V @ (V.T @ one / w)

        denom = float(one @ y)
        if abs(denom) <= 1e-30:
            return np.full(p, 1.0 / p, dtype=float), "uniform"
        return y / denom, method

    def _cond_number(self, H: np.ndarray, lam_eff: float) -> float:
        p = int(H.shape[0])
        if p == 0:
            return 1.0
        w = np.linalg.eigvalsh(H + lam_eff * np.eye(p))
        w = np.clip(w, 0.0, None)
        return float(np.max(w) / max(float(np.min(w)), 1e-30))

    # ------------------------------ main update ------------------------------

    def mix(self, x_old: np.ndarray, x_raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Return accelerated iterate based on stored history.

        Args:
            x_old: Current iterate (scaled global vector).
            x_raw: Picard image g(x_old) from one block sweep (scaled global vector).

        Returns:
            (x_new, info) where x_new is the mixed iterate and info holds diagnostics.
        """
        if self.pending_reset:
            self.reset()

        # Picard residual at the *current* iterate: r_k = g(x_k) - x_k
        r = x_raw - x_old

        # Store current iterate/residual pair in history (bounded by maxlen)
        self.x_hist.append(x_old.copy())
        self.r_hist.append(r.copy())
        p = len(self.r_hist)

        info: Dict = {
            "aa_hist": int(p),
            "accepted": True,
            "restart_reason": "",
            "alpha_method": None,
            "condH": None,
            "r_norm": None,
            "r_proxy_norm": None,
            "r2_curr": None,
            "r2_pred": None,
        }

        # Build Gram matrix of residuals and a scale-aware regularization.
        H = self._build_gram(list(self.r_hist))
        tr = float(np.trace(H)) if p > 0 else 0.0
        avg_diag = tr / max(p, 1)

        # IMPORTANT: regularize *relative* to the scale of H, otherwise the
        # regularization dominates near convergence and AA degenerates into
        # history-averaging (often slower than plain Picard).
        lam_eff = self.lam * max(avg_diag, self._eig_floor)

        alpha, alpha_method = self._solve_kkt(H, lam_eff)
        info["alpha_method"] = str(alpha_method)

        # Diagnostics: current residual norm and predicted residual proxy.
        r_norm = self._rel_step(x_old, x_raw, x_raw)
        r2_curr = float(H[-1, -1]) if p > 0 else 0.0
        r2_curr = max(r2_curr, 0.0)

        r2_pred = float(alpha @ H @ alpha) if p > 0 else 0.0
        if not np.isfinite(r2_pred):
            r2_pred = float("inf")
        else:
            r2_pred = max(r2_pred, 0.0)

        info["r_norm"] = float(r_norm)
        info["condH"] = float(self._cond_number(H, lam_eff))
        info["r2_curr"] = float(r2_curr)
        info["r2_pred"] = float(r2_pred)

        # Baseline: (possibly damped) Picard iterate.
        x_pic = x_old + self.beta * r

        # Anderson iterate (Walker–Ni / Type-I form) using under-relaxed images.
        x_cand = x_pic
        if p >= 2:
            y = np.zeros_like(x_old)
            for a_i, x_i, r_i in zip(alpha, self.x_hist, self.r_hist):
                y += float(a_i) * (x_i + self.beta * r_i)
            x_cand = y

        # Safeguard (residual-based): if AA does not predict improvement, fall back
        # to Picard.
        if self.safeguard and p >= 2 and r2_curr > self._tiny:
            thresh = ((1.0 - self.gamma) ** 2) * r2_curr
            if (not np.isfinite(r2_pred)) or (r2_pred > thresh):
                x_cand = x_pic
                info["accepted"] = False
                self.reject_streak += 1
            else:
                self.reject_streak = 0
        else:
            # If safeguard is disabled, we do not treat any step as a "rejection".
            self.reject_streak = 0

        # Step limiting relative to the Picard step length (robustness).
        s = x_cand - x_old
        s_norm = self._rel_step(x_old, x_old + s, x_raw)
        info["r_proxy_norm"] = float(s_norm)

        if s_norm > self.step_limit_factor * max(r_norm, self._tiny):
            s *= (self.step_limit_factor * max(r_norm, self._tiny)) / max(s_norm, self._tiny)
            x_cand = x_old + s
            info["r_proxy_norm"] = float(self._rel_step(x_old, x_cand, x_raw))

        # History restart logic (cheap, improves robustness in practice).
        self._check_restart(float(r_norm), float(info["condH"]), info)

        return x_cand, info

    def _check_restart(self, r_norm: float, condH: float, info: Dict) -> None:
        if self.reject_streak >= self.restart_on_reject_k:
            self.pending_reset = True
            info["restart_reason"] = f"reject_streak>={self.restart_on_reject_k}"
        elif condH > self.restart_on_cond:
            self.pending_reset = True
            info["restart_reason"] = f"illcond~{condH:.1e}"
