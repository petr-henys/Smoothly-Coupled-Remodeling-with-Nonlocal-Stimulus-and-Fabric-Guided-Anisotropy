"""Analytic demonstration of the adaptive timestep controller.

Unlike the performance diagnostics (which read `steps.csv`), this figure is
generated from a *synthetic* predictor--corrector experiment. The goal is to
explain how the AB2 predictor, BE corrector, WRMS error estimate, and the
Gustafsson PI controller interact, without relying on any remodeling run.

We use a scalar linear ODE with a time-dependent forcing and a closed-form
Backward--Euler (BE) update:
    x'(t) = -λ x(t) + f(t),
    x^{n+1} = (x^n + Δt^n f(t_{n+1})) / (1 + λ Δt^n).

The forcing includes a smooth "fast transient" burst to intentionally trigger
controller rejections and timestep cuts, while remaining fully analytic.

The AB2 predictor and the PI controller follow the implementation in
`simulation/timeintegrator.py` (special-case first-step acceptance, rejection on
e>1, and Gustafsson PI update on accepted steps).

Output:
    manuscript/images/time_adaptivity_demo.png

Usage:
    python3 analysis/time_adaptivity_plot.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import COLORS, apply_style, save_manuscript_figure


OUTPUT_BASENAME = "time_adaptivity_demo"


@dataclass(frozen=True)
class ToyProblem:
    """Scalar ODE used for demonstration."""

    lam: float = 0.20  # decay rate
    x0: float = 0.0

    # Slow forcing component (sets the "easy" regime where dt can grow).
    amp: float = 1.0
    omega: float = 2.0 * np.pi / 1000.0  # period 1000 days

    # Transient "burst" component (forces dt cuts in the middle of the run).
    amp_burst: float = 3.0
    omega_burst: float = 2.0 * np.pi / 200.0  # period 200 days
    burst_center: float = 300.0  # days
    burst_width: float = 60.0    # days (Gaussian envelope)


@dataclass(frozen=True)
class ControllerParams:
    """Adaptive controller parameters (matches config.json keys)."""

    # Error norm
    rtol: float = 1e-2
    atol: float = 1e-3

    # dt bounds
    dt_min: float = 1e-4
    dt_max: float = 100.0

    # Gustafsson PI controller parameters
    safety: float = 0.9
    g_max: float = 5.0
    g_min: float = 0.1
    k_exp: float = 1.5
    k_p: float = 0.20
    k_i: float = 0.40


class ToyAdaptiveStepper:
    """Minimal scalar replica of AB predictor + PI dt controller.

    This mirrors the state machine in `simulation/timeintegrator.py`:
      - AB2 predictor based on stored rates from accepted steps
      - WRMS error from predictor--corrector difference
      - PI controller updates dt using error memory from accepted steps only
      - First accepted step is always accepted; if e>1, only next dt is reduced
      - Time-error rejections keep PI memory; coupling failures reset history
    """

    def __init__(self, params: ControllerParams):
        self.p = params

        # Predictor history (rates)
        self.rate_last = 0.0
        self.rate_last2 = 0.0

        # Controller memory
        self.step_count = 0
        self.dt_prev = 0.0
        self.error_prev = 1.0

    def reset_history(self) -> None:
        self.rate_last = 0.0
        self.rate_last2 = 0.0
        self.step_count = 0
        self.dt_prev = 0.0
        self.error_prev = 1.0

    def predict(self, x: float, dt: float) -> Tuple[float, bool]:
        dt = float(dt)
        if self.step_count >= 2 and self.dt_prev > 0.0:
            r = dt / self.dt_prev
            w1 = 1.0 + 0.5 * r
            w2 = 0.5 * r
            pred = x + dt * (w1 * self.rate_last - w2 * self.rate_last2)
            return float(pred), True
        pred = x + dt * self.rate_last
        return float(pred), False

    def wrms_error(self, x_corr: float, x_pred: float) -> float:
        scale = self.p.rtol * max(abs(x_corr), abs(x_pred)) + self.p.atol
        return float(abs(x_corr - x_pred) / max(scale, 1e-30))

    def _clamp_dt(self, dt: float) -> float:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            return float(self.p.dt_min)
        return float(max(self.p.dt_min, min(self.p.dt_max, dt)))

    def suggest_dt(self, dt: float, converged: bool, error_norm: float) -> Tuple[bool, float, str]:
        """Return (accepted, next_dt, reason)."""
        dt = float(dt)

        if not converged:
            next_dt = self._clamp_dt(dt * float(self.p.g_min))
            self.error_prev = 1.0
            return False, next_dt, "reject:coupling_nonconverged"

        # First accepted step: accept regardless; if err>1 reduce *next* dt.
        if self.step_count == 0:
            safe_error = max(1e-10, float(error_norm))
            self.error_prev = safe_error
            if safe_error > 1.0:
                factor = self.p.safety * (1.0 / safe_error) ** (1.0 / self.p.k_exp)
                factor = max(self.p.g_min, min(0.9, factor))
                return True, self._clamp_dt(dt * factor), "first step accepted (reduce next dt)"
            return True, self._clamp_dt(dt), "first step accepted"

        # Reject on large error (time-error controller)
        if error_norm > 1.0:
            safe_error = max(1e-10, float(error_norm))
            factor = self.p.safety * (1.0 / safe_error) ** (1.0 / self.p.k_exp)
            factor = max(self.p.g_min, min(0.9, factor))
            return False, self._clamp_dt(dt * factor), "reject:time_error"

        # Accepted step: Gustafsson PI for the *next* dt.
        safe_error = max(1e-10, float(error_norm))
        if self.step_count > 1:
            factor = self.p.safety * (safe_error ** (-self.p.k_i)) * ((self.error_prev / safe_error) ** self.p.k_p)
        else:
            factor = self.p.safety * (safe_error ** (-self.p.k_i))

        factor = min(self.p.g_max, max(self.p.g_min, factor))
        if safe_error < 0.5:
            factor = max(1.0, factor)

        self.error_prev = safe_error
        return True, self._clamp_dt(dt * factor), "accepted"

    def commit_step(self, x_new: float, x_old: float, dt: float) -> None:
        dt = float(dt)
        self.rate_last2 = self.rate_last
        self.rate_last = (float(x_new) - float(x_old)) / max(dt, 1e-30)
        self.dt_prev = dt
        self.step_count += 1


def be_correct(problem: ToyProblem, x_old: float, t_new: float, dt: float) -> float:
    """Backward--Euler update for x' = -λ x + f(t)."""
    lam = float(problem.lam)
    rhs = float(x_old) + float(dt) * float(forcing(problem, float(t_new)))
    return rhs / (1.0 + lam * float(dt))


def forcing(problem: ToyProblem, t: float) -> float:
    """Analytic forcing f(t) with a smooth transient burst."""
    slow = float(problem.amp) * float(np.sin(float(problem.omega) * float(t)))
    z = (float(t) - float(problem.burst_center)) / max(float(problem.burst_width), 1e-12)
    envelope = float(np.exp(-(z * z)))
    burst = float(problem.amp_burst) * envelope * float(np.sin(float(problem.omega_burst) * float(t)))
    return slow + burst


def simulate_toy(
    problem: ToyProblem,
    ctrl_params: ControllerParams,
    *,
    total_time: float,
    dt_initial: float,
    toy_coupling_dt_limit: float | None = None,
    max_attempts: int = 10_000,
) -> Dict[str, np.ndarray]:
    """Run the synthetic predictor--corrector + PI loop and return a trace."""
    stepper = ToyAdaptiveStepper(ctrl_params)

    t = 0.0
    x = float(problem.x0)
    dt = float(dt_initial)

    accepted_times: List[float] = [t]
    accepted_states: List[float] = [x]

    rec: Dict[str, List] = {
        "attempt": [],
        "t": [],
        "dt": [],
        "dt_prev": [],
        "step_count": [],
        "used_ab2": [],
        "x_old": [],
        "x_pred": [],
        "x_corr": [],
        "err": [],
        "converged": [],
        "accepted": [],
        "reason": [],
        "dt_next": [],
        "factor": [],
        "trunc_to_T": [],
    }

    attempt = 0
    while t < total_time and attempt < max_attempts:
        attempt += 1

        # Truncate to hit T exactly (as in simulation/model.py)
        trunc = False
        if t + dt > total_time:
            dt = total_time - t
            trunc = True

        x_old = x
        used_ab2 = False
        x_pred, used_ab2 = stepper.predict(x_old, dt)
        x_corr = be_correct(problem, x_old, t + dt, dt)

        converged = True
        if toy_coupling_dt_limit is not None and dt > float(toy_coupling_dt_limit):
            converged = False

        err = stepper.wrms_error(x_corr, x_pred)
        accepted, dt_next, reason = stepper.suggest_dt(dt, converged, err)

        rec["attempt"].append(attempt)
        rec["t"].append(t)
        rec["dt"].append(dt)
        rec["dt_prev"].append(stepper.dt_prev)
        rec["step_count"].append(stepper.step_count)
        rec["used_ab2"].append(int(used_ab2))
        rec["x_old"].append(x_old)
        rec["x_pred"].append(x_pred)
        rec["x_corr"].append(x_corr)
        rec["err"].append(err)
        rec["converged"].append(int(converged))
        rec["accepted"].append(int(accepted))
        rec["reason"].append(str(reason))
        rec["dt_next"].append(dt_next)
        rec["factor"].append(dt_next / max(dt, 1e-30))
        rec["trunc_to_T"].append(int(trunc))

        if accepted:
            stepper.commit_step(x_corr, x_old, dt)
            x = x_corr
            t = t + dt
            accepted_times.append(t)
            accepted_states.append(x)
        else:
            # Roll back; reset AB history only for coupling failures (matches model.py)
            if not converged:
                stepper.reset_history()

        dt = dt_next

    rec_np = {k: np.asarray(v) for k, v in rec.items()}
    rec_np["accepted_times"] = np.asarray(accepted_times, dtype=float)
    rec_np["accepted_states"] = np.asarray(accepted_states, dtype=float)
    return rec_np


def _pick_demo_attempt(trace: Dict[str, np.ndarray]) -> int:
    """Pick a representative accepted AB2 attempt near the acceptance boundary."""
    used_ab2 = trace["used_ab2"].astype(bool)
    accepted = trace["accepted"].astype(bool)
    err = trace["err"].astype(float)

    mask = used_ab2 & accepted & np.isfinite(err)
    if not np.any(mask):
        if np.any(accepted):
            return int(np.where(accepted)[0][0])
        return 0

    idx = np.where(mask)[0]

    # Prefer a step where the controller decision is "interesting": e close to 1.
    target = 0.8
    dist = np.abs(np.log10(np.maximum(err[idx], 1e-12)) - np.log10(target))
    return int(idx[int(np.argmin(dist))])


def _controller_factor_curves(
    params: ControllerParams,
    e_grid: np.ndarray,
    *,
    e_prev: float,
    use_pi: bool,
) -> np.ndarray:
    """Compute unclipped+clipped controller factor f(e) for accepted regime."""
    e = np.asarray(e_grid, dtype=float)
    safe_e = np.maximum(e, 1e-10)
    if use_pi:
        fac = params.safety * (safe_e ** (-params.k_i)) * ((float(e_prev) / safe_e) ** params.k_p)
    else:
        fac = params.safety * (safe_e ** (-params.k_i))
    fac = np.clip(fac, params.g_min, params.g_max)
    fac = np.where(safe_e < 0.5, np.maximum(1.0, fac), fac)
    return fac


def generate_time_adaptivity_demo() -> Path:
    apply_style()

    problem = ToyProblem()
    params = ControllerParams()

    # Synthetic run: slow regime + transient burst -> dt grow/shrink + rejections.
    trace = simulate_toy(
        problem,
        params,
        total_time=600.0,
        dt_initial=25.0,
        toy_coupling_dt_limit=None,  # keep the demo focused on time-error control
    )

    i_demo = _pick_demo_attempt(trace)

    fig = plt.figure(figsize=(11.69, 7.0))
    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232)
    ax3 = plt.subplot(233)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235)
    ax6 = plt.subplot(236)

    # =========================================================================
    # (a) Predictor vs corrector on one representative step
    # =========================================================================
    t0 = float(trace["t"][i_demo])
    dt0 = float(trace["dt"][i_demo])
    x_old = float(trace["x_old"][i_demo])
    x_pred = float(trace["x_pred"][i_demo])
    x_corr = float(trace["x_corr"][i_demo])
    err0 = float(trace["err"][i_demo])

    # Use accepted history to show context (last two accepted states).
    acc_t = trace["accepted_times"]
    acc_x = trace["accepted_states"]
    # Find the accepted index matching t0 (may fail if i_demo is a rejected attempt).
    j = int(np.argmin(np.abs(acc_t - t0)))
    j0 = max(j - 1, 0)

    ax1.plot(acc_t[j0 : j + 1], acc_x[j0 : j + 1], color=COLORS["blue"], linewidth=1.5, label="accepted")
    ax1.plot([t0 + dt0], [x_corr], color=COLORS["blue"], marker="o", markersize=4, linestyle="None")
    ax1.plot([t0 + dt0], [x_pred], color=COLORS["orange"], marker="x", markersize=5, linestyle="None", label="predictor")
    ax1.plot([t0, t0 + dt0], [x_old, x_corr], color=COLORS["blue"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.vlines(t0 + dt0, min(x_pred, x_corr), max(x_pred, x_corr), color=COLORS["grey"], linewidth=1.0, alpha=0.8)
    ax1.text(
        0.02,
        0.98,
        rf"$e={err0:.2f}$",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    ax1.set_title("(a) Predictor--corrector step", loc="left", fontweight="bold")
    ax1.set_xlabel("Time $t$ [days]")
    ax1.set_ylabel("$x$")
    ax1.legend(loc="best", frameon=False)

    # =========================================================================
    # (b) Error estimator e(dt) for the same local history
    # =========================================================================
    dt_prev = float(trace["dt_prev"][i_demo])
    step_count = int(trace["step_count"][i_demo])
    # Reconstruct the predictor history at this attempt from the accepted states.
    # (This keeps the plot self-contained and avoids storing internal rates.)
    # Use finite differences on accepted points for the last two accepted steps.
    if len(acc_t) >= 3 and j >= 2:
        dt_n = float(acc_t[j] - acc_t[j - 1])
        dt_nm1 = float(acc_t[j - 1] - acc_t[j - 2])
        rate_last = (float(acc_x[j]) - float(acc_x[j - 1])) / max(dt_n, 1e-30)
        rate_last2 = (float(acc_x[j - 1]) - float(acc_x[j - 2])) / max(dt_nm1, 1e-30)
        dt_prev_eff = dt_n
        x_base = float(acc_x[j])
        t_base = float(acc_t[j])
    else:
        rate_last = 0.0
        rate_last2 = 0.0
        dt_prev_eff = max(dt_prev, 1.0)
        x_base = x_old
        t_base = t0

    # Focus the dt sweep around the selected dt0 for readability.
    dt_lo = max(params.dt_min, dt0 / 20.0)
    dt_hi = min(params.dt_max, dt0 * 20.0)
    if dt_hi <= dt_lo:
        dt_lo = params.dt_min
        dt_hi = params.dt_max
    dt_grid = np.logspace(np.log10(dt_lo), np.log10(dt_hi), 160)
    e_grid = np.empty_like(dt_grid)
    for k, dtk in enumerate(dt_grid):
        # Predictor (AB2 when available; otherwise AB1 as in code)
        if step_count >= 2 and dt_prev_eff > 0.0:
            r = dtk / dt_prev_eff
            w1 = 1.0 + 0.5 * r
            w2 = 0.5 * r
            x_pk = x_base + dtk * (w1 * rate_last - w2 * rate_last2)
        else:
            x_pk = x_base + dtk * rate_last
        x_ck = be_correct(problem, x_base, t_base + dtk, dtk)
        scale = params.rtol * max(abs(x_ck), abs(x_pk)) + params.atol
        e_grid[k] = abs(x_ck - x_pk) / max(scale, 1e-30)

    ax2.plot(dt_grid, e_grid, color=COLORS["teal"], linewidth=1.5)
    ax2.axhline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.axvline(dt0, color=COLORS["grey"], linestyle=":", linewidth=1.0, alpha=0.9)
    ax2.scatter([dt0], [err0], s=28, marker="o", color=COLORS["blue"], edgecolors="white", linewidths=0.5, zorder=5)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(r"(b) Error indicator $e(\Delta t)$", loc="left", fontweight="bold")
    ax2.set_xlabel(r"$\Delta t$ [days]")
    ax2.set_ylabel(r"$e$ (WRMS)")

    # =========================================================================
    # (c) Controller action f(e) = dt_next/dt (accepted vs rejected)
    # =========================================================================
    e_plot = np.logspace(-3, 2, 400)
    f_reject = params.safety * (1.0 / np.maximum(e_plot, 1e-10)) ** (1.0 / params.k_exp)
    f_reject = np.clip(f_reject, params.g_min, 0.9)

    # Use a representative previous error from the trace for the PI curve.
    acc_err = trace["err"][trace["accepted"].astype(bool)].astype(float)
    e_prev_repr = float(np.median(acc_err[1:])) if acc_err.size > 2 else 0.7
    f_I = _controller_factor_curves(params, e_plot, e_prev=1.0, use_pi=False)
    f_PI = _controller_factor_curves(params, e_plot, e_prev=e_prev_repr, use_pi=True)

    ax3.plot(e_plot, f_I, color=COLORS["blue"], linewidth=1.5, label="accept: I-only (2nd step)")
    ax3.plot(
        e_plot,
        f_PI,
        color=COLORS["teal"],
        linewidth=1.5,
        label=rf"accept: PI ($e_{{\mathrm{{prev}}}}\approx{e_prev_repr:.2f}$)",
    )
    ax3.plot(e_plot, f_reject, color=COLORS["orange"], linewidth=1.5, label="reject: time error")
    ax3.axvline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax3.axhline(1.0, color=COLORS["grey"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax3.axhline(params.g_min, color=COLORS["grey"], linestyle=":", linewidth=1.0, alpha=0.7)
    ax3.axhline(params.g_max, color=COLORS["grey"], linestyle=":", linewidth=1.0, alpha=0.7)

    # Overlay actual controller updates from the synthetic run.
    e_series = trace["err"].astype(float)
    f_series = trace["factor"].astype(float)
    reasons = trace["reason"].astype(str)
    accepted_mask = trace["accepted"].astype(bool)
    rej_time = (~accepted_mask) & np.array([r == "reject:time_error" for r in reasons])
    acc_first = accepted_mask & np.array([r.startswith("first step accepted") for r in reasons])
    acc_other = accepted_mask & ~acc_first

    ax3.scatter(
        e_series[acc_other],
        f_series[acc_other],
        s=18,
        marker="o",
        color=COLORS["teal"],
        edgecolors="white",
        linewidths=0.4,
        alpha=0.75,
        label="run: accepted",
        zorder=4,
    )
    if np.any(acc_first):
        ax3.scatter(
            e_series[acc_first],
            f_series[acc_first],
            s=26,
            marker="s",
            color=COLORS["blue"],
            edgecolors="white",
            linewidths=0.4,
            alpha=0.9,
            label="run: first step",
            zorder=5,
        )
    if np.any(rej_time):
        ax3.scatter(
            e_series[rej_time],
            f_series[rej_time],
            s=30,
            marker="x",
            color=COLORS["orange"],
            linewidths=1.2,
            alpha=0.9,
            label="run: rejected",
            zorder=5,
        )

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title(r"(c) Controller action $f(e)$", loc="left", fontweight="bold")
    ax3.set_xlabel(r"$e$")
    ax3.set_ylabel(r"$f=\Delta t_{\mathrm{next}}/\Delta t$")
    ax3.legend(loc="lower left", fontsize=6, frameon=False)

    # =========================================================================
    # (d) dt evolution over time (synthetic run)
    # =========================================================================
    t_series = trace["t"].astype(float)
    dt_series = trace["dt"]
    accepted = trace["accepted"].astype(bool)
    rejected = ~accepted

    ax4.semilogy(t_series[accepted], dt_series[accepted], color=COLORS["teal"], linewidth=1.5, label="accepted")
    ax4.scatter(
        t_series[accepted],
        dt_series[accepted],
        c=COLORS["teal"],
        s=14,
        marker="o",
        edgecolors="white",
        linewidths=0.4,
        alpha=0.8,
    )
    if np.any(rejected):
        rr = trace["reason"].astype(str)
        rej_time = rejected & np.array([r == "reject:time_error" for r in rr])
        if np.any(rej_time):
            ax4.scatter(t_series[rej_time], dt_series[rej_time], c=COLORS["orange"], s=34, marker="x", linewidths=1.3, label="rejected")

    ax4.axhline(params.dt_min, color=COLORS["grey"], linestyle="--", linewidth=1.0, alpha=0.6)
    ax4.axhline(params.dt_max, color=COLORS["grey"], linestyle="--", linewidth=1.0, alpha=0.6)
    # Highlight where the forcing "burst" is active.
    t_b0 = float(problem.burst_center - 1.5 * problem.burst_width)
    t_b1 = float(problem.burst_center + 1.5 * problem.burst_width)
    ax4.axvspan(t_b0, t_b1, color=COLORS["grey"], alpha=0.12, zorder=0)

    ax4.set_title(r"(d) Timestep evolution $\Delta t(t)$", loc="left", fontweight="bold")
    ax4.set_xlabel("Time $t$ [days]")
    ax4.set_ylabel(r"$\Delta t$ [days]")
    ax4.legend(loc="upper left", fontsize=7, frameon=False)

    # =========================================================================
    # (e) Error timeline over time
    # =========================================================================
    err_series = trace["err"]
    ax5.semilogy(t_series[accepted], err_series[accepted], color=COLORS["blue"], linewidth=1.2, label="accepted")
    ax5.scatter(t_series[accepted], err_series[accepted], c=COLORS["blue"], s=16, marker="o", edgecolors="white", linewidths=0.4)
    if np.any(rejected):
        ax5.scatter(t_series[rejected], err_series[rejected], c=COLORS["orange"], s=30, marker="x", linewidths=1.2, label="rejected")
    ax5.axhline(1.0, color=COLORS["black"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax5.axvspan(t_b0, t_b1, color=COLORS["grey"], alpha=0.12, zorder=0)
    ax5.set_title(r"(e) Error indicator $e(t)$", loc="left", fontweight="bold")
    ax5.set_xlabel("Time $t$ [days]")
    ax5.set_ylabel(r"$e$ (WRMS)")
    ax5.legend(loc="upper right", fontsize=7, frameon=False)

    # =========================================================================
    # (f) Sensitivity: attempts-to-reach-T over (kP, kI)
    # =========================================================================
    kp_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    ki_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=float)
    M = np.empty((len(ki_vals), len(kp_vals)), dtype=float)

    for iy, ki in enumerate(ki_vals):
        for ix, kp in enumerate(kp_vals):
            p2 = ControllerParams(
                rtol=params.rtol,
                atol=params.atol,
                dt_min=params.dt_min,
                dt_max=params.dt_max,
                safety=params.safety,
                g_max=params.g_max,
                g_min=params.g_min,
                k_exp=params.k_exp,
                k_p=float(kp),
                k_i=float(ki),
            )
            tr = simulate_toy(problem, p2, total_time=600.0, dt_initial=25.0)
            M[iy, ix] = float(np.max(tr["attempt"]))  # number of attempts executed

    im = ax6.imshow(M, origin="lower", aspect="auto", cmap="YlGnBu")
    cbar = plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    cbar.set_label("Attempts to reach $T$", rotation=270, labelpad=14)
    ax6.set_xticks(np.arange(len(kp_vals)))
    ax6.set_xticklabels([f"{v:.1f}" for v in kp_vals])
    ax6.set_yticks(np.arange(len(ki_vals)))
    ax6.set_yticklabels([f"{v:.1f}" for v in ki_vals])
    ax6.set_xlabel(r"$k_P$")
    ax6.set_ylabel(r"$k_I$")
    ax6.set_title(r"(f) PI-parameter sensitivity", loc="left", fontweight="bold")

    # Mark default (kP,kI) = (0.2, 0.4)
    ix0 = int(np.where(np.isclose(kp_vals, params.k_p))[0][0]) if np.any(np.isclose(kp_vals, params.k_p)) else None
    iy0 = int(np.where(np.isclose(ki_vals, params.k_i))[0][0]) if np.any(np.isclose(ki_vals, params.k_i)) else None
    if ix0 is not None and iy0 is not None:
        ax6.scatter([ix0], [iy0], s=60, marker="o", facecolors="none", edgecolors=COLORS["red"], linewidths=1.5)

    plt.tight_layout()
    return save_manuscript_figure(fig, OUTPUT_BASENAME, dpi=300)


if __name__ == "__main__":
    out = generate_time_adaptivity_demo()
    print(f"Generated {out}")
