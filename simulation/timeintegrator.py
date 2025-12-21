"""Adaptive AB2 predictor with Gustafsson PI control.

The integrator is field-agnostic: it can predict and estimate error for an
arbitrary set of coupled state fields (e.g., ``S`` and ``rho``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.utils import assign, get_owned_size


@dataclass
class _FieldHistory:
    f: fem.Function
    n_owned: int
    N_global: int
    rate_last: fem.Function
    rate_last2: fem.Function


class TimeIntegrator:
    """Adaptive time stepping for state-field updates.

    Uses an AB2 predictor for error estimation and a Gustafsson PI controller.
    Step is accepted if WRMS ≤ 1.
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        state_fields: Mapping[str, fem.Function],
        dt_min: float = 1e-3,
        dt_max: float = 10.0,
        adaptive_rtol: float = 1e-3,
        adaptive_atol: float = 1e-4,
    ):
        """Initialize with MPI comm, parameters, and a mapping of state fields."""
        self.comm = comm
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.rtol = adaptive_rtol
        self.atol = adaptive_atol

        # Controller state
        self.step_count = 0
        self.dt_prev: float = 0.0
        self.error_prev = 1.0  # Initialize with 1.0

        # Controller parameters
        self.safety = 0.9
        self.growth_factor = 5.0
        self.shrink_factor = 0.1
        self.k_exp = 1.5
        self.kp = 0.20
        self.ki = 0.40

        self._fields: dict[str, _FieldHistory] = {}
        self._N_total: int = 0

        self.set_state_fields(state_fields)

    # ---------------------------- field management ----------------------------

    def set_state_fields(self, state_fields: Mapping[str, fem.Function]) -> None:
        """(Re)configure which fields are integrated.

        This resets the predictor history, since adding/removing fields makes
        the AB history inconsistent.
        """
        if not state_fields:
            raise ValueError("TimeIntegrator requires at least one state field.")

        fields: dict[str, _FieldHistory] = {}
        N_total = 0
        for name, f in state_fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"Invalid field name {name!r}; expected non-empty str.")

            n_owned = int(get_owned_size(f))
            N_global = int(self.comm.allreduce(n_owned, op=MPI.SUM))

            rate_last = fem.Function(f.function_space, name=f"{name}_rate_last")
            rate_last2 = fem.Function(f.function_space, name=f"{name}_rate_last2")

            fields[name] = _FieldHistory(
                f=f,
                n_owned=n_owned,
                N_global=N_global,
                rate_last=rate_last,
                rate_last2=rate_last2,
            )
            N_total += N_global

        self._fields = fields
        self._N_total = int(N_total)
        self.reset_history()

    # ---------------------------- AB predictor/error ---------------------------

    def reset_history(self) -> None:
        """Reset predictor history for all configured fields."""
        for hist in self._fields.values():
            assign(hist.rate_last, 0.0)
            assign(hist.rate_last2, 0.0)
        self.step_count = 0
        self.dt_prev = 0.0
        self.error_prev = 1.0

    def predict(self, dt: float) -> dict[str, np.ndarray]:
        """AB2 (or AB1) prediction for owned DOFs of all state fields."""
        dt_curr = float(dt)
        out: dict[str, np.ndarray] = {}

        for name, hist in self._fields.items():
            n_owned = hist.n_owned
            vals = hist.f.x.array[:n_owned]
            rate_last = hist.rate_last.x.array[:n_owned]
            rate_last2 = hist.rate_last2.x.array[:n_owned]

            if self.step_count >= 2 and self.dt_prev > 0.0:
                r = dt_curr / self.dt_prev
                w1 = 1.0 + 0.5 * r
                w2 = 0.5 * r
                pred = vals + dt_curr * (w1 * rate_last - w2 * rate_last2)
            else:
                pred = vals + dt_curr * rate_last

            out[name] = np.asarray(pred, dtype=hist.f.x.array.dtype).copy()

        return out

    def compute_wrms_error(self, x_pred: Mapping[str, np.ndarray]) -> float:
        """WRMS error between prediction and corrected state (current field values)."""
        if self._N_total <= 0:
            return 0.0

        atol = float(self.atol)
        rtol = float(self.rtol)

        sq_error_local = 0.0
        for name, hist in self._fields.items():
            pred = x_pred.get(name, None)
            if pred is None:
                raise KeyError(f"Missing prediction for field {name!r}.")

            n_owned = hist.n_owned
            val_corr = hist.f.x.array[:n_owned]
            val_pred = np.asarray(pred, dtype=val_corr.dtype)[:n_owned]

            scale = rtol * np.abs(val_corr) + atol
            diff = val_corr - val_pred
            sq_error_local += float(np.sum(np.abs(diff / scale) ** 2))

        sq_error_global = float(self.comm.allreduce(sq_error_local, op=MPI.SUM))
        return float(np.sqrt(sq_error_global / float(self._N_total)))

    # ----------------------------- time-step control ---------------------------

    def suggest_dt(self, dt: float, converged: bool, error_norm: float) -> Tuple[bool, float, str]:
        """PI controller: returns (accepted, next_dt, reason)."""
        if not converged:
            # Divergence: Cut aggressively
            next_dt = max(self.dt_min, dt * 0.5)
            return False, next_dt, "diverged"
        
        if self.step_count == 0:
            # Initialize PI history; do not reject the very first step.
            self.error_prev = max(error_norm, 1.0)
            return True, dt, "first step accepted (no rejection)"

        # Rejection (Error too high)
        if error_norm > 1.0:
            # Standard rejection: h_new = h * safety * (1/err)^(1/k)
            factor = self.safety * (1.0 / error_norm) ** (1.0 / self.k_exp)
            factor = max(self.shrink_factor, min(0.9, factor))  # Ensure reduction
            
            next_dt = max(self.dt_min, dt * factor)
            
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
        next_dt = max(self.dt_min, min(self.dt_max, next_dt))

        # Store error for next step
        self.error_prev = safe_error

        return True, next_dt, "accepted"

    def commit_step(self, dt: float, new_fields: Mapping[str, fem.Function], old_fields: Mapping[str, fem.Function]) -> None:
        """Update AB2 history after an accepted step."""
        dt_curr = float(dt)

        for name, hist in self._fields.items():
            f_new = new_fields.get(name, None)
            f_old = old_fields.get(name, None)
            if f_new is None or f_old is None:
                raise KeyError(f"commit_step requires both new and old fields for {name!r}.")

            assign(hist.rate_last2, hist.rate_last, scatter=False)

            n_owned = hist.n_owned
            rate_data = (f_new.x.array[:n_owned] - f_old.x.array[:n_owned]) / dt_curr
            assign(hist.rate_last, rate_data, scatter=False)

        self.dt_prev = dt_curr
        self.step_count += 1
