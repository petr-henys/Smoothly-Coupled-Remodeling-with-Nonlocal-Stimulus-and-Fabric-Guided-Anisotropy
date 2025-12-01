"""Adaptive time stepping with AB2 predictor and PI error control."""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.utils import assign, get_owned_size


class TimeIntegrator:
    """AB2 predictor, WRMS error estimation, Gustafsson PI step control."""

    def __init__(self, comm: MPI.Intracomm, cfg: Config, Q: fem.FunctionSpace):
        """Initialize time integrator.

        Parameters
        ----------
        comm : MPI.Intracomm
            MPI communicator.
        cfg : Config
            Simulation configuration.
        Q : fem.FunctionSpace
            Function space for density (scalar).
        """
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

        # --- CONTROLLER TUNING (Aggressive) ---
        self.safety = 0.9
        self.growth_factor = 5.0   # Allow step to grow up to 5x in one go
        self.shrink_factor = 0.1   # Limit shrinking
        
        # PID parameters adjusted for aggressive growth
        # Standard theory suggests kI ~ 1/k. For AB2 (k=2), limit is 0.5.
        # Previous values (0.08) were too conservative.
        self.k_exp = 1.5   # Exponent for rejection (lower = safer restart)
        self.kp = 0.20     # Proportional gain (damping)
        self.ki = 0.40     # Integral gain (Main driver for speed!)

        self.reset_history()

    def reset_history(self):
        """Reset history for restart or initial step."""
        assign(self.rho_rate_last, 0.0)
        assign(self.rho_rate_last2, 0.0)
        self.step_count = 0
        self.dt_prev = None
        self.error_prev = 1.0

    def predict(self, dt: float, rho_current: fem.Function) -> np.ndarray:
        """Calculate predictor step using AB2 (or Forward Euler).

        Returns
        -------
        np.ndarray
            Predicted values for the owned DOFs.
        """
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
        """Compute Weighted RMS error between prediction and correction."""
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
        """Suggest next timestep based on Gustafsson PI controller."""
        if not converged:
            # Divergence: Cut aggressively
            next_dt = max(self.cfg.dt_min, dt * 0.5)
            return False, next_dt, "diverged"

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
        """Update history with accepted step."""
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