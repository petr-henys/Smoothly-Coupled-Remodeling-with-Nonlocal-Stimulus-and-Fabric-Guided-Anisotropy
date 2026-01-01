"""Analysis of robust Sylvester projectors near eigenvalue degeneracy.

This script mirrors the intent of `simulation/utils.py:projectors_sylvester()` in a
pure NumPy setting and generates manuscript-ready figures:

  - `manuscript/images/sylvester_projectors.png` (degeneracy behavior summary)
  - `manuscript/images/sylvester_projectors_experiment.png` (Monte Carlo eigen-gap experiment)

The key numerical issue is that spectral *rank-1* projectors are not uniquely
defined when eigenvalues are repeated (isotropic / transversely isotropic), and
the classical Sylvester formula becomes ill-conditioned when eigenvalue gaps are
small. The implementation used in the solver resolves this by:

  - scale-aware denominator regularization for the Sylvester formula,
  - enforcing the partition of unity P1+P2+P3 = I,
  - smoothly blending to basis-invariant limits:
      isotropic: P1=P2=P3=I/3,
      TI (pair-degenerate): P_unique kept, degenerate plane split equally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_utils import COLORS, save_manuscript_figure, apply_style


def symm(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)


def frob(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a random proper rotation matrix (det=+1)."""
    M = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(M)
    # Enforce det=+1 (proper rotation).
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def projectors_from_eigh(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-1 projectors from an eigendecomposition (well-defined only if gaps > 0)."""
    Xs = symm(X)
    w, V = np.linalg.eigh(Xs)  # ascending
    idx = np.argsort(w)[::-1]  # descending: l1 >= l2 >= l3
    V = V[:, idx]
    P1 = np.outer(V[:, 0], V[:, 0])
    P2 = np.outer(V[:, 1], V[:, 1])
    P3 = np.outer(V[:, 2], V[:, 2])
    return symm(P1), symm(P2), symm(P3)


def projectors_sylvester_raw(X: np.ndarray, l1: float, l2: float, l3: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classical Sylvester formula (no regularization); undefined at degeneracy."""
    Xs = symm(X)
    I = np.eye(3)

    X_l2 = Xs - l2 * I
    X_l3 = Xs - l3 * I
    X_l1 = Xs - l1 * I

    P1 = symm((X_l2 @ X_l3) / ((l1 - l2) * (l1 - l3)))
    P2 = symm((X_l1 @ X_l3) / ((l2 - l1) * (l2 - l3)))
    P3 = symm((X_l1 @ X_l2) / ((l3 - l1) * (l3 - l2)))
    return P1, P2, P3


def projectors_sylvester_regularized(
    X: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    *,
    eps_d: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sylvester formula with scale-aware denom clamp + partition-of-unity fix."""
    Xs = symm(X)
    I = np.eye(3)

    q = float(np.trace(Xs) / 3.0)
    B = Xs - q * I
    p2 = float(np.trace(B @ B) / 6.0)
    scale2 = max(q * q + p2, 1.0)
    eps_d_scaled = eps_d * scale2

    def _safe_denom(a: float) -> float:
        sign = 1.0 if a >= 0.0 else -1.0
        return sign * max(abs(a), eps_d_scaled)

    X_l2 = Xs - l2 * I
    X_l3 = Xs - l3 * I
    X_l1 = Xs - l1 * I

    P1_raw = symm((X_l2 @ X_l3) / _safe_denom((l1 - l2) * (l1 - l3)))
    P2_raw = symm((X_l1 @ X_l3) / _safe_denom((l2 - l1) * (l2 - l3)))
    P3_raw = symm((X_l1 @ X_l2) / _safe_denom((l3 - l1) * (l3 - l2)))

    # Partition-of-unity correction (reduces downstream drift).
    Psum = P1_raw + P2_raw + P3_raw
    P_fix = (I - Psum) / 3.0
    P1 = symm(P1_raw + P_fix)
    P2 = symm(P2_raw + P_fix)
    P3 = symm(P3_raw + P_fix)
    return P1, P2, P3


def _projectors_sylvester_impl(
    X: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    *,
    eps_d: float = 1e-12,
    tol: float = 1e-14,
    tol_deg: float = 3e-6,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    """NumPy mirror of `simulation/utils.py:projectors_sylvester`.

    Returns (P1,P2,P3) and the blending weights used (for plotting/debug).
    """
    Xs = symm(X)
    I = np.eye(3)

    # Scale used for both denom safety and degeneracy detection (units: eigenvalue^2).
    q = float(np.trace(Xs) / 3.0)
    B = Xs - q * I
    p2 = float(np.trace(B @ B) / 6.0)
    scale2 = max(q * q + p2, 1.0)

    # First compute the "fully anisotropic" regularized projectors.
    P1_full, P2_full, P3_full = projectors_sylvester_regularized(Xs, l1, l2, l3, eps_d=eps_d)

    # Smooth degeneracy indicators s_ij ~ 1 if li≈lj, ~0 if well-separated.
    gap_eps2 = (tol_deg * tol_deg) * scale2
    d12_2 = (l1 - l2) * (l1 - l2)
    d23_2 = (l2 - l3) * (l2 - l3)
    d13_2 = (l1 - l3) * (l1 - l3)

    s12 = gap_eps2 / (d12_2 + gap_eps2)
    s23 = gap_eps2 / (d23_2 + gap_eps2)
    s13 = gap_eps2 / (d13_2 + gap_eps2)

    w_full = (1.0 - s12) * (1.0 - s23) * (1.0 - s13)
    w12 = s12 * (1.0 - s23) * (1.0 - s13)
    w23 = s23 * (1.0 - s12) * (1.0 - s13)
    w13 = s13 * (1.0 - s12) * (1.0 - s23)
    w_iso = 1.0 - (w_full + w12 + w23 + w13)

    I3 = I / 3.0

    # Pair-degenerate "TI" limits: keep the unique projector; split the orthogonal plane evenly.
    # pair (2,3): unique is 1
    P1_T23 = P1_full
    P2_T23 = 0.5 * (I - P1_full)
    P3_T23 = 0.5 * (I - P1_full)
    # pair (1,2): unique is 3
    P3_T12 = P3_full
    P1_T12 = 0.5 * (I - P3_full)
    P2_T12 = 0.5 * (I - P3_full)
    # pair (1,3): unique is 2
    P2_T13 = P2_full
    P1_T13 = 0.5 * (I - P2_full)
    P3_T13 = 0.5 * (I - P2_full)

    P1 = w_full * P1_full + w23 * P1_T23 + w12 * P1_T12 + w13 * P1_T13 + w_iso * I3
    P2 = w_full * P2_full + w23 * P2_T23 + w12 * P2_T12 + w13 * P2_T13 + w_iso * I3
    P3 = w_full * P3_full + w23 * P3_T23 + w12 * P3_T12 + w13 * P3_T13 + w_iso * I3

    weights = {
        "w_full": float(w_full),
        "w12": float(w12),
        "w23": float(w23),
        "w13": float(w13),
        "w_iso": float(w_iso),
        "scale2": float(scale2),
    }
    return (symm(P1), symm(P2), symm(P3)), weights


def projectors_sylvester(
    X: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    *,
    eps_d: float = 1e-12,
    tol: float = 1e-14,
    tol_deg: float = 3e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectral projectors via stabilized Sylvester construction (solver behavior)."""
    (P1, P2, P3), _ = _projectors_sylvester_impl(X, l1, l2, l3, eps_d=eps_d, tol=tol, tol_deg=tol_deg)
    return P1, P2, P3


def projectors_sylvester_with_weights(
    X: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    *,
    eps_d: float = 1e-12,
    tol: float = 1e-14,
    tol_deg: float = 3e-6,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]:
    return _projectors_sylvester_impl(X, l1, l2, l3, eps_d=eps_d, tol=tol, tol_deg=tol_deg)


def summarize_samples(x: np.ndarray, *, q: tuple[float, float, float] = (5.0, 50.0, 95.0)) -> tuple[float, float, float]:
    """Return (p5,p50,p95) with NaN-safe behavior."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("nan"), float("nan"), float("nan")
    p_lo, p_mid, p_hi = np.nanpercentile(x[finite], q)
    return float(p_lo), float(p_mid), float(p_hi)


def projector_metrics(P1: np.ndarray, P2: np.ndarray, P3: np.ndarray) -> tuple[float, float]:
    """Return (idempotence_error, orthogonality_error) using Frobenius norm."""
    idem = max(frob(P1 @ P1 - P1), frob(P2 @ P2 - P2), frob(P3 @ P3 - P3))
    ortho = max(frob(P1 @ P2), frob(P1 @ P3), frob(P2 @ P3))
    return float(idem), float(ortho)


def projector_error_to_ref(
    P1: np.ndarray, P2: np.ndarray, P3: np.ndarray, P1_ref: np.ndarray, P2_ref: np.ndarray, P3_ref: np.ndarray
) -> float:
    """Return max_i ||P_i - P_i_ref||_F."""
    return float(max(frob(P1 - P1_ref), frob(P2 - P2_ref), frob(P3 - P3_ref)))


def generate_sylvester_projector_figure() -> None:
    apply_style()

    fig = plt.figure(figsize=(11.69, 7.0))

    # ------------------------------------------------------------------
    # (a) TI degeneracy: rank-1 projectors depend on basis in degenerate plane
    # ------------------------------------------------------------------
    ax1 = plt.subplot(231)
    phi = np.linspace(0.0, np.pi / 2.0, 400)
    P_unique = np.diag([0.0, 0.0, 1.0])  # unique axis along z

    # Eigenvector-based split in the (x,y)-plane rotated by angle phi.
    P2_11 = np.cos(phi) ** 2
    P3_11 = np.sin(phi) ** 2
    ax1.plot(phi, P2_11, color=COLORS["blue"], linewidth=2.0, label=r"eig: $[P_2]_{11}$")
    ax1.plot(phi, P3_11, color=COLORS["magenta"], linewidth=2.0, label=r"eig: $[P_3]_{11}$")

    # Robust TI split: P2 = P3 = 0.5*(I - P_unique) so [P2]11 = [P3]11 = 0.5.
    ax1.axhline(0.5, color=COLORS["black"], linestyle="--", linewidth=1.5, label=r"robust: $[P_2]_{11}=[P_3]_{11}=1/2$")

    ax1.set_title(r"(a) TI limit: basis dependence", loc="left", fontweight="bold")
    ax1.set_xlabel(r"rotation in degenerate plane $\varphi$ [rad]")
    ax1.set_ylabel(r"projector entry")
    ax1.set_xlim(0.0, np.pi / 2.0)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper center", ncol=1)

    # ------------------------------------------------------------------
    # (b) Isotropic degeneracy: eigenvector projectors vary with arbitrary basis
    # ------------------------------------------------------------------
    ax2 = plt.subplot(232)
    rng = np.random.default_rng(123)
    n_samples = 2000
    p11 = np.empty(n_samples)
    for k in range(n_samples):
        R = random_rotation(rng)
        v1 = R[:, 0]
        p11[k] = v1[0] * v1[0]  # [v v^T]11

    ax2.hist(p11, bins=35, density=True, color=COLORS["teal"], alpha=0.6, label=r"eig: $[P_1]_{11}$ samples")
    ax2.axvline(1.0 / 3.0, color=COLORS["black"], linestyle="--", linewidth=1.5, label=r"robust: $[P_i]_{11}=1/3$")
    ax2.set_title(r"(b) Isotropic limit: rank-1 non-uniqueness", loc="left", fontweight="bold")
    ax2.set_xlabel(r"$[P_1]_{11}$")
    ax2.set_ylabel("density")
    ax2.set_xlim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # ------------------------------------------------------------------
    # Shared setup for near-degeneracy sweeps (fixed rotation)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(7)
    R = random_rotation(rng)
    I = np.eye(3)

    deltas = np.logspace(-1, -14, 120)  # l2-l3 = 2*delta
    err_plane_raw = np.empty_like(deltas)
    err_plane_reg = np.empty_like(deltas)
    err_plane_blend = np.empty_like(deltas)
    pu_err_raw = np.empty_like(deltas)
    pu_err_reg = np.empty_like(deltas)
    pu_err_blend = np.empty_like(deltas)
    w_full = np.empty_like(deltas)
    w23 = np.empty_like(deltas)
    w_iso = np.empty_like(deltas)
    p23_diff = np.empty_like(deltas)
    p23_diff_eig = np.empty_like(deltas)

    for i, d in enumerate(deltas):
        l1, l2, l3 = 2.0, 1.0 + d, 1.0 - d
        D = np.diag([l1, l2, l3])
        X = R @ D @ R.T

        P1_e, P2_e, P3_e = projectors_from_eigh(X)
        P_plane_e = P2_e + P3_e

        # Raw Sylvester (ill-conditioned as d -> 0)
        P1_r, P2_r, P3_r = projectors_sylvester_raw(X, l1, l2, l3)
        P_plane_r = P2_r + P3_r
        err_plane_raw[i] = frob(P_plane_r - P_plane_e)
        pu_err_raw[i] = frob(I - (P1_r + P2_r + P3_r))

        # Regularized (denom clamp + partition fix, no blending)
        P1_g, P2_g, P3_g = projectors_sylvester_regularized(X, l1, l2, l3)
        P_plane_g = P2_g + P3_g
        err_plane_reg[i] = frob(P_plane_g - P_plane_e)
        pu_err_reg[i] = frob(I - (P1_g + P2_g + P3_g))

        # Full blended (solver behavior)
        (P1_b, P2_b, P3_b), weights = projectors_sylvester_with_weights(X, l1, l2, l3)
        P_plane_b = P2_b + P3_b
        err_plane_blend[i] = frob(P_plane_b - P_plane_e)
        pu_err_blend[i] = frob(I - (P1_b + P2_b + P3_b))

        w_full[i] = weights["w_full"]
        w23[i] = weights["w23"]
        w_iso[i] = weights["w_iso"]

        p23_diff[i] = frob(P2_b - P3_b)
        p23_diff_eig[i] = frob(P2_e - P3_e)

    # ------------------------------------------------------------------
    # (c) Conditioning near TI: plane-projector error vs eigenvalue gap
    # ------------------------------------------------------------------
    ax3 = plt.subplot(233)
    gap = 2.0 * deltas
    ax3.loglog(gap, err_plane_raw, color=COLORS["red"], linewidth=1.6, label="Sylvester (raw)")
    ax3.loglog(gap, err_plane_reg, color=COLORS["orange"], linewidth=1.6, label="Sylvester (+denom,+sum)")
    ax3.loglog(gap, err_plane_blend, color=COLORS["blue"], linewidth=2.0, label="Sylvester (robust)")
    ax3.set_title(r"(c) Near-TI conditioning: plane error", loc="left", fontweight="bold")
    ax3.set_xlabel(r"eigenvalue gap $|\lambda_2-\lambda_3|$")
    ax3.set_ylabel(r"$\| (P_2+P_3) - (P_2^{eig}+P_3^{eig})\|_F$")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(loc="lower right")

    # ------------------------------------------------------------------
    # (d) Partition of unity: ||I - (P1+P2+P3)|| vs gap
    # ------------------------------------------------------------------
    ax4 = plt.subplot(234)
    ax4.loglog(gap, pu_err_raw, color=COLORS["red"], linewidth=1.6, label="raw")
    ax4.loglog(gap, pu_err_reg, color=COLORS["orange"], linewidth=1.6, label="+denom,+sum")
    ax4.loglog(gap, pu_err_blend, color=COLORS["blue"], linewidth=2.0, label="robust")
    ax4.set_title(r"(d) Partition of unity", loc="left", fontweight="bold")
    ax4.set_xlabel(r"eigenvalue gap $|\lambda_2-\lambda_3|$")
    ax4.set_ylabel(r"$\| I - (P_1+P_2+P_3)\|_F$")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(loc="lower right")

    # ------------------------------------------------------------------
    # (e) Blending weights across degeneracy (smooth transition)
    # ------------------------------------------------------------------
    ax5 = plt.subplot(235)
    ax5.semilogx(gap, w_full, color=COLORS["black"], linewidth=1.8, label=r"$w_{full}$")
    ax5.semilogx(gap, w23, color=COLORS["blue"], linewidth=1.8, label=r"$w_{23}$ (TI)")
    ax5.semilogx(gap, w_iso, color=COLORS["grey"], linewidth=1.8, label=r"$w_{iso}$")
    ax5.set_title(r"(e) Smooth degeneracy weights", loc="left", fontweight="bold")
    ax5.set_xlabel(r"eigenvalue gap $|\lambda_2-\lambda_3|$")
    ax5.set_ylabel("weight")
    ax5.set_ylim(-0.02, 1.02)
    ax5.grid(True, which="both", alpha=0.3)
    ax5.legend(loc="center left")

    # ------------------------------------------------------------------
    # (f) Equal split in TI: ||P2-P3|| -> 0 as gap -> 0
    # ------------------------------------------------------------------
    ax6 = plt.subplot(236)
    ax6.loglog(gap, p23_diff_eig, color=COLORS["magenta"], linewidth=1.6, label=r"eig: $\|P_2-P_3\|_F$")
    ax6.loglog(gap, p23_diff, color=COLORS["blue"], linewidth=2.0, label=r"robust: $\|P_2-P_3\|_F$")
    ax6.set_title(r"(f) Robust equalization in TI", loc="left", fontweight="bold")
    ax6.set_xlabel(r"eigenvalue gap $|\lambda_2-\lambda_3|$")
    ax6.set_ylabel(r"$\|P_2-P_3\|_F$")
    ax6.grid(True, which="both", alpha=0.3)
    ax6.legend(loc="lower right")

    plt.tight_layout()
    save_manuscript_figure(fig, "sylvester_projectors")


if __name__ == "__main__":
    generate_sylvester_projector_figure()
