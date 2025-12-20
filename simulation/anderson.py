"""MPI-aware Anderson acceleration with a fixed interface.

Exposes `reset()` and `mix(x_old, x_raw) -> (x_new, info)`. Inner products are
global (MPI reductions). The caller is responsible for scaling/normalizing the
state vector before mixing.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Sequence, Tuple

import numpy as np
from mpi4py import MPI

from simulation.logger import get_logger


class Anderson:
    """MPI-aware Anderson acceleration with safeguard + restart.

    Constructor arguments are intentionally explicit: no defaults.
    """

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
        restart_on_stall: float,
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
        self.restart_on_stall = float(restart_on_stall)
        self.restart_on_cond = float(restart_on_cond)

        self.step_limit_factor = float(step_limit_factor)
        self.verbose = bool(verbose)

        self.logger = get_logger(self.comm, name="Anderson")

        self.x_hist: Deque[np.ndarray] = deque(maxlen=self.m + 1)
        self.r_hist: Deque[np.ndarray] = deque(maxlen=self.m + 1)

        self.reject_streak = 0
        self.best_picard = np.inf
        self.pending_reset = False

        # Numerical floors (internal constants; not exposed as knobs)
        self._tiny = 1e-300
        self._eig_floor = 1e-15

    def reset(self) -> None:
        self.x_hist.clear()
        self.r_hist.clear()
        self.reject_streak = 0
        self.best_picard = np.inf
        self.pending_reset = False

    # ------------------------- linear algebra helpers -------------------------

    def _gdot(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self.comm.allreduce(float(np.dot(a, b)), op=MPI.SUM))

    def _rel_step(self, x_old: np.ndarray, x_trial: np.ndarray, x_ref: np.ndarray) -> float:
            """Relative step: ||x_trial-x_old|| / (||x_ref|| + eps) globally."""
            d = x_trial - x_old
            d2 = self._gdot(d, d)
            r2 = self._gdot(x_ref, x_ref)
            
            # Ochrana proti dělení nulou u vyhasínajících polí
            # epsilon 1e-10 je dostatečně malé pro přesnost, ale brání explozi chyby
            epsilon = 1e-20 
            
            return float(np.sqrt(d2 / (r2 + epsilon)))

    def _build_gram(self, r_list: Sequence[np.ndarray]) -> np.ndarray:
        if len(r_list) == 0:
            return np.zeros((0, 0), dtype=float)
        R_loc = np.vstack(r_list)
        H_loc = R_loc @ R_loc.T
        return self.comm.allreduce(H_loc, op=MPI.SUM)

    def _solve_kkt(self, H: np.ndarray, lam_eff: float) -> np.ndarray:
        """Solve min ||alpha||_H s.t. 1^T alpha = 1."""
        p = int(H.shape[0])
        if p == 0:
            return np.zeros(0, dtype=float)

        Hp = H + lam_eff * np.eye(p)
        one = np.ones(p, dtype=float)

        try:
            y = np.linalg.solve(Hp, one)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Hp + self._eig_floor * np.eye(p))
            w = np.clip(w, self._eig_floor, None)
            y = V @ (V.T @ one / w)

        denom = float(one @ y)
        if abs(denom) <= 1e-30:
            return np.full(p, 1.0 / p, dtype=float)
        return y / denom

    def _cond_number(self, H: np.ndarray, lam_eff: float) -> float:
        p = int(H.shape[0])
        if p == 0:
            return 1.0
        w = np.linalg.eigvalsh(H + lam_eff * np.eye(p))
        w = np.clip(w, 0.0, None)
        return float(np.max(w) / max(float(np.min(w)), 1e-30))

    # ------------------------------ main update ------------------------------

    def mix(self, x_old: np.ndarray, x_raw: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Return accelerated iterate based on stored history."""
        if self.pending_reset:
            self.reset()

        r = x_raw - x_old

        self.x_hist.append(x_old.copy())
        self.r_hist.append(r.copy())
        p = len(self.r_hist)

        info: Dict = {
            "aa_hist": int(p),
            "accepted": True,
            "backtracks": 0,
            "restart_reason": "",
            "condH": None,
            "r_norm": None,
            "r_proxy_norm": None,
        }

        H = self._build_gram(list(self.r_hist))
        tr = float(np.trace(H)) if p > 0 else 0.0
        lam_eff = self.lam * max(tr / max(p, 1), 1.0)

        alpha = self._solve_kkt(H, lam_eff)

        # Current residual (Picard step) norm and Gram-matrix residual proxies.
        r_norm = self._rel_step(x_old, x_raw, x_raw)
        r2_curr = float(H[-1, -1]) if p > 0 else 0.0
        r2_pred = float(alpha @ H @ alpha) if p > 0 else 0.0

        # Safeguard (residual-based): accept acceleration only if predicted residual
        # is sufficiently smaller than the current residual.
        info["r_norm"] = float(r_norm)
        info["condH"] = float(self._cond_number(H, lam_eff))
        info["r2_curr"] = float(r2_curr)
        info["r2_pred"] = float(r2_pred)
        self.best_picard = min(self.best_picard, float(r_norm))

        if self.safeguard and p >= 2 and r2_curr > self._tiny:
            if (not np.isfinite(r2_pred)) or (r2_pred > ((1.0 - self.gamma) ** 2) * r2_curr):
                x_pic = x_old + self.beta * r
                info["accepted"] = False
                info["r_proxy_norm"] = float(self._rel_step(x_old, x_pic, x_raw))
                self.reject_streak += 1
                self._check_restart(float(r_norm), float(info["condH"]), info)
                return x_pic, info

        # Anderson combination (Walker-Ni form)
        y = np.zeros_like(x_old)
        for a_i, x_i, r_i in zip(alpha, self.x_hist, self.r_hist):
            y += float(a_i) * (x_i + self.beta * r_i)

        s = y - x_old

        # Step limiting relative to Picard step length
        s_norm = self._rel_step(x_old, x_old + s, x_raw)
        if s_norm > self.step_limit_factor * max(r_norm, self._tiny):
            s *= (self.step_limit_factor * max(r_norm, self._tiny)) / max(s_norm, self._tiny)

        x_cand = x_old + s

        # Safeguard (step-proxy): accept if proxy step is sufficiently smaller than Picard step
        rp_norm = self._rel_step(x_old, x_cand, x_raw)
        info["r_proxy_norm"] = float(rp_norm)

        if self.safeguard and r_norm > self._tiny:
            if rp_norm > (1.0 - self.gamma) * r_norm:
                x_cand, accepted, bt = self._backtrack(x_old, s, x_raw, r_norm)
                info["backtracks"] = int(bt)
                if not accepted:
                    info["accepted"] = False
                    self.reject_streak += 1
                else:
                    self.reject_streak = 0
            else:
                self.reject_streak = 0

            self._check_restart(float(r_norm), float(info["condH"]), info)

        return x_cand, info

    def _backtrack(
        self,
        x_old: np.ndarray,
        s: np.ndarray,
        x_raw: np.ndarray,
        r_norm: float,
    ) -> Tuple[np.ndarray, bool, int]:
        theta = 0.5
        for bt in range(self.backtrack_max):
            x_try = x_old + theta * s
            rp_try = self._rel_step(x_old, x_try, x_raw)
            if rp_try <= (1.0 - self.gamma) * r_norm:
                return x_try, True, bt
            theta *= 0.5
        # Fall back to (possibly damped) Picard step to keep beta-consistent meaning.
        return x_old + self.beta * (x_raw - x_old), False, self.backtrack_max

    def _check_restart(self, r_norm: float, condH: float, info: Dict) -> None:
        if self.reject_streak >= self.restart_on_reject_k:
            self.pending_reset = True
            info["restart_reason"] = f"reject_streak>={self.restart_on_reject_k}"
        elif r_norm > self.restart_on_stall * self.best_picard:
            self.pending_reset = True
            info["restart_reason"] = f"stall>x{self.restart_on_stall:.2f}"
        elif condH > self.restart_on_cond:
            self.pending_reset = True
            info["restart_reason"] = f"illcond~{condH:.1e}"
