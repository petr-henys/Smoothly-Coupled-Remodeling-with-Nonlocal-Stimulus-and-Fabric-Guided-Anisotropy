"""Adaptive AB2 predictor with Gustafsson PI control."""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.utils import assign, get_owned_size


class TimeIntegrator:
    """
    AB2 predictor for error estimation, PI controller for dt adaptation.
    WRMS error: sqrt(mean((ρ_corr - ρ_pred)² / (atol + rtol·|ρ|)²)).
    Step accepted if WRMS ≤ 1.0.
    """

    def __init__(self, comm: MPI.Intracomm, cfg: Config, Q: fem.FunctionSpace):
        """Initialize with MPI comm, config, and density function space."""
        self.comm = comm
        self.cfg = cfg
        self.Q = Q

        # History for AB2 predictor
        self.rho_rate_last = fem.Function(Q, name="rho_rate_last")
        self.rho_rate_last2 = fem.Function(Q, name="rho_rate_last2")

        # Controller state
        self.step_count = 0
        self.dt_prev: Optional[float] = None
        self.error_prev = 1.0  # Initialize with 1.0

        # Controller parameters
        self.safety = 0.9
        self.growth_factor = 5.0
        self.shrink_factor = 0.1
        self.k_exp = 1.5
        self.kp = 0.20
        self.ki = 0.40

        self.reset_history()

    def reset_history(self):
        """Reset predictor history."""
        assign(self.rho_rate_last, 0.0)
        assign(self.rho_rate_last2, 0.0)
        self.step_count = 0
        self.dt_prev = None
        self.error_prev = 1.0

    def predict(self, dt: float, rho_current: fem.Function) -> np.ndarray:
        """AB2 (or AB1) prediction for owned DOFs."""
        dt_curr = float(dt)

        # Get owned DOF count
        n_owned = get_owned_size(rho_current)

        # Current values (owned)
        rho_vals = rho_current.x.array[:n_owned]
        rate_last = self.rho_rate_last.x.array[:n_owned]
        rate_last2 = self.rho_rate_last2.x.array[:n_owned]

        if self.step_count >= 2 and self.dt_prev is not None:
            # Variable-step AB2 Predictor
            r = dt_curr / self.dt_prev
            w1 = 1.0 + 0.5 * r
            w2 = 0.5 * r
            pred = rho_vals + dt_curr * (w1 * rate_last - w2 * rate_last2)
        else:
            # Forward Euler (AB1) for first steps
            pred = rho_vals + dt_curr * rate_last

        return pred

    def compute_wrms_error(self, x_pred: np.ndarray, x_corr: fem.Function) -> float:
        """WRMS error between prediction and correction."""
        n_owned = get_owned_size(x_corr)

        # Get owned arrays
        val_corr = x_corr.x.array[:n_owned]
        val_pred = x_pred[:n_owned]

        # Calculate weights
        atol = self.cfg.adaptive_atol
        rtol = self.cfg.adaptive_rtol

        scale = rtol * np.abs(val_corr) + atol

        diff = val_corr - val_pred
        sq_error_local = np.sum((diff / scale) ** 2)

        # Global sum
        sq_error_global = self.comm.allreduce(sq_error_local, op=MPI.SUM)

        # Total number of DOFs
        N_global = self.Q.dofmap.index_map.size_global * self.Q.dofmap.index_map_bs
        
        # Avoid division by zero if empty mesh (unlikely)
        if N_global == 0: return 0.0

        return np.sqrt(sq_error_global / N_global)

    def suggest_dt(self, dt: float, converged: bool, error_norm: float) -> Tuple[bool, float, str]:
        """PI controller: returns (accepted, next_dt, reason)."""
        if not converged:
            # Divergence: Cut aggressively
            next_dt = max(self.cfg.dt_min, dt * 0.5)
            return False, next_dt, "diverged"
        
        if self.step_count == 0:
            # zapamatuj si chybu jako historii, ale neodmítej
            self.error_prev = max(error_norm, 1.0)
            return True, dt, "first step accepted (no rejection)"

        # Rejection (Error too high)
        if error_norm > 1.0:
            # Standard rejection: h_new = h * safety * (1/err)^(1/k)
            factor = self.safety * (1.0 / error_norm) ** (1.0 / self.k_exp)
            factor = max(self.shrink_factor, min(0.9, factor))  # Ensure reduction
            
            next_dt = max(self.cfg.dt_min, dt * factor)
            
            # Reset PI history on rejection to avoid bad memory
            self.error_prev = error_norm
            return False, next_dt, f"error {error_norm:.2f} > 1.0"

        # Acceptance (Error OK)
        safe_error = max(1e-10, error_norm)

        # Gustafsson PI controller
        # Factor = S * (err_n)^(-kI) * (err_n / err_{n-1})^(-kP)
        # Equivalently: S * (err_n)^(-kI - kP) * (err_{n-1})^(kP)
        
        if self.step_count > 1:
            # PI Control
            factor = self.safety * (safe_error ** (-self.ki)) * ((self.error_prev / safe_error) ** self.kp)
        else:
            # I Control (First step, no history)
            # Use same gain as PI integral part for consistency
            factor = self.safety * (safe_error ** (-self.ki))

        # Apply limiters
        factor = min(self.growth_factor, max(self.shrink_factor, factor))
        
        # Don't shrink if error is good (unless factor < 1.0 due to noise, but we clamp at 1.0 for very good steps usually)
        if safe_error < 0.5:
             factor = max(1.0, factor)

        next_dt = dt * factor
        next_dt = max(self.cfg.dt_min, min(self.cfg.dt_max, next_dt))

        # Store error for next step
        self.error_prev = safe_error

        return True, next_dt, "accepted"

    def commit_step(self, dt: float, rho_new: fem.Function, rho_old: fem.Function):
        """Update AB2 history after accepted step."""
        dt_curr = float(dt)

        # Shift history (internal fields, only owned DOFs matter)
        assign(self.rho_rate_last2, self.rho_rate_last, scatter=False)

        # Calculate new rate from owned DOFs only
        n_owned = get_owned_size(rho_new)
        rate_data = (rho_new.x.array[:n_owned] - rho_old.x.array[:n_owned]) / dt_curr
        self.rho_rate_last.x.array[:n_owned] = rate_data
        # Skip scatter - rate history is only used for owned DOF prediction

        self.dt_prev = dt_curr
        self.step_count += 1