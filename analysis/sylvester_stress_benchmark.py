"""Near-degeneracy stress benchmark for Sylvester-projector robustness.

This script quantifies how the constitutive stress behaves when a transversely
isotropic fabric state is perturbed by vanishing symmetric noise. It compares:

- rank-1 projectors from `eigh`,
- raw Sylvester projectors,
- regularized Sylvester projectors with partition-of-unity fix,
- the fully robust blended construction used in the solver.

Outputs:
- `results/reviewer_benchmarks/sylvester_stress/summary.json`
- `results/reviewer_benchmarks/sylvester_stress/noise_sweep.csv`
- `results/reviewer_benchmarks/sylvester_stress/tol_deg_sweep.csv`
- `manuscript/images/sylvester_stress_benchmark.png`
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

analysis_dir = Path(__file__).resolve().parent
project_root = analysis_dir.parent
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from plot_utils import (
    COLORS,
    PUBLICATION_DPI,
    apply_style,
    save_figure,
    save_manuscript_figure,
    setup_axis_style,
)
from sylvester_projector_plot import (
    projectors_from_eigh,
    projectors_sylvester_raw,
    projectors_sylvester_regularized,
    projectors_sylvester_with_weights,
)


RESULTS_DIR = project_root / "results" / "reviewer_benchmarks" / "sylvester_stress"

NOISE_LEVELS = np.logspace(-2, -10, 9)
TOL_DEG_VALUES = np.logspace(-12, -2, 11)
DEFAULT_TOL_DEG = 1e-6
FIXED_NOISE_FOR_TOL = 1e-8
N_SAMPLES = 256
FIGSIZE_COMPACT_TWO_PANEL = (5.5, 2.45)

# Mirror the constitutive choices used in `MechanicsSolver.sigma`.
E0 = 7500.0
NU0 = 0.3
P_E = 1.0
P_G = 1.0
M_MIN = 0.2
M_MAX = 5.0


def sigma_from_projectors(
    eigenvalues: np.ndarray,
    projectors: tuple[np.ndarray, np.ndarray, np.ndarray],
    strain: np.ndarray,
) -> np.ndarray:
    l1, l2, l3 = np.asarray(eigenvalues, dtype=float)
    P1, P2, P3 = projectors

    mean_l = (l1 + l2 + l3) / 3.0
    d_max = np.log(max(M_MAX, 1.0 / M_MIN))
    d_vals = np.clip(np.array([l1 - mean_l, l2 - mean_l, l3 - mean_l]), -d_max, d_max)
    a1_hat, a2_hat, a3_hat = np.exp(d_vals)

    E_iso = E0
    E1 = E_iso * a1_hat**P_E
    E2 = E_iso * a2_hat**P_E
    E3 = E_iso * a3_hat**P_E

    G_iso = E_iso / (2.0 * (1.0 + NU0))
    G12 = G_iso * (a1_hat * a2_hat) ** (0.5 * P_G)
    G23 = G_iso * (a2_hat * a3_hat) ** (0.5 * P_G)
    G31 = G_iso * (a3_hat * a1_hat) ** (0.5 * P_G)

    a_coeff = 1.0 / (1.0 + NU0)
    b_coeff = NU0 / ((1.0 + NU0) * (1.0 - 2.0 * NU0))

    e1 = float(np.sum(P1 * strain))
    e2 = float(np.sum(P2 * strain))
    e3 = float(np.sum(P3 * strain))

    sqrtE1 = np.sqrt(E1)
    sqrtE2 = np.sqrt(E2)
    sqrtE3 = np.sqrt(E3)
    sum_term = sqrtE1 * e1 + sqrtE2 * e2 + sqrtE3 * e3

    s1 = a_coeff * E1 * e1 + b_coeff * sqrtE1 * sum_term
    s2 = a_coeff * E2 * e2 + b_coeff * sqrtE2 * sum_term
    s3 = a_coeff * E3 * e3 + b_coeff * sqrtE3 * sum_term
    sigma_normal = s1 * P1 + s2 * P2 + s3 * P3

    def _P_eps_P(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ strain @ B

    sigma_shear = (
        2.0 * G12 * (_P_eps_P(P1, P2) + _P_eps_P(P2, P1))
        + 2.0 * G23 * (_P_eps_P(P2, P3) + _P_eps_P(P3, P2))
        + 2.0 * G31 * (_P_eps_P(P3, P1) + _P_eps_P(P1, P3))
    )
    sigma = sigma_normal + sigma_shear
    return 0.5 * (sigma + sigma.T)


def stress_from_method(
    tensor: np.ndarray,
    strain: np.ndarray,
    method: str,
    *,
    tol_deg: float = DEFAULT_TOL_DEG,
) -> np.ndarray:
    eigenvalues = np.sort(np.linalg.eigvalsh(tensor))[::-1]
    if method == "eigh":
        projectors = projectors_from_eigh(tensor)
    elif method == "raw":
        projectors = projectors_sylvester_raw(tensor, *eigenvalues)
    elif method == "reg":
        projectors = projectors_sylvester_regularized(tensor, *eigenvalues)
    elif method == "rob":
        projectors, _ = projectors_sylvester_with_weights(tensor, *eigenvalues, tol_deg=tol_deg)
    else:
        raise ValueError(f"Unknown method: {method}")
    return sigma_from_projectors(eigenvalues, projectors, strain)


def ti_reference_state(seed: int = 1234) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0.0:
        Q[:, 0] *= -1.0

    # Trace-free near-TI log-fabric tensor used as the constitutive reference.
    tensor_ti = Q @ np.diag([0.4, -0.2, -0.2]) @ Q.T
    unique_axis = Q[:, 0]

    strain = np.array(
        [
            [0.010, 0.003, -0.001],
            [0.003, -0.004, 0.002],
            [-0.001, 0.002, -0.006],
        ],
        dtype=float,
    )
    return tensor_ti, unique_axis, strain


def ti_reference_stress(unique_axis: np.ndarray, strain: np.ndarray) -> np.ndarray:
    P1 = np.outer(unique_axis, unique_axis)
    P23 = 0.5 * (np.eye(3) - P1)
    return sigma_from_projectors(np.array([0.4, -0.2, -0.2]), (P1, P23, P23), strain)


def summarize(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    arr = arr[finite]
    p05, median, p95 = np.percentile(arr, [5.0, 50.0, 95.0])
    return {"median": float(median), "p05": float(p05), "p95": float(p95)}


def noise_sweep(
    tensor_ti: np.ndarray,
    reference_stress: np.ndarray,
    strain: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(2025)
    reference_norm = float(np.linalg.norm(reference_stress))

    methods = ("eigh", "raw", "reg", "rob")
    records: list[dict[str, float]] = []
    summary_by_method: dict[str, dict[str, float]] = {}

    for noise_level in NOISE_LEVELS:
        method_errors = {method: [] for method in methods}
        for _ in range(N_SAMPLES):
            perturbation = rng.normal(size=(3, 3))
            perturbation = 0.5 * (perturbation + perturbation.T)
            tensor = tensor_ti + noise_level * perturbation

            for method in methods:
                stress = stress_from_method(tensor, strain, method, tol_deg=DEFAULT_TOL_DEG)
                rel_error = float(np.linalg.norm(stress - reference_stress) / reference_norm)
                method_errors[method].append(rel_error)

        for method in methods:
            stats = summarize(method_errors[method])
            record = {"noise_level": float(noise_level), "method": method, **stats}
            records.append(record)
            if np.isclose(noise_level, FIXED_NOISE_FOR_TOL):
                summary_by_method[method] = stats

    return records, summary_by_method


def tol_deg_sweep(
    tensor_ti: np.ndarray,
    reference_stress: np.ndarray,
    strain: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    rng = np.random.default_rng(77)
    reference_norm = float(np.linalg.norm(reference_stress))

    records: list[dict[str, float]] = []
    best_tol = None
    best_median = float("inf")

    for tol_deg in TOL_DEG_VALUES:
        samples = []
        for _ in range(N_SAMPLES):
            perturbation = rng.normal(size=(3, 3))
            perturbation = 0.5 * (perturbation + perturbation.T)
            tensor = tensor_ti + FIXED_NOISE_FOR_TOL * perturbation
            stress = stress_from_method(tensor, strain, "rob", tol_deg=float(tol_deg))
            rel_error = float(np.linalg.norm(stress - reference_stress) / reference_norm)
            samples.append(rel_error)

        stats = summarize(samples)
        records.append({"tol_deg": float(tol_deg), **stats})
        if stats["median"] < best_median:
            best_median = stats["median"]
            best_tol = float(tol_deg)

    summary = {
        "default_tol_deg": DEFAULT_TOL_DEG,
        "best_tol_deg": float(best_tol),
        "best_median_error": float(best_median),
    }
    return records, summary


def write_csv(path: Path, rows: list[dict[str, float]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_figure(
    noise_records: list[dict[str, float]],
    tol_records: list[dict[str, float]],
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_COMPACT_TWO_PANEL)

    methods = [
        ("eigh", "eig projectors", COLORS["red"], "o"),
        ("raw", "Sylvester (raw)", COLORS["orange"], "s"),
        ("reg", "Sylvester (+denom,+sum)", COLORS["cyan"], "^"),
        ("rob", "Sylvester (robust)", COLORS["blue"], "D"),
    ]

    for method, label, color, marker in methods:
        rows = [r for r in noise_records if r["method"] == method]
        noise = np.array([r["noise_level"] for r in rows], dtype=float)
        median = np.array([r["median"] for r in rows], dtype=float)
        p05 = np.array([r["p05"] for r in rows], dtype=float)
        p95 = np.array([r["p95"] for r in rows], dtype=float)

        axes[0].loglog(noise, median, color=color, marker=marker, linewidth=1.4, label=label)
        axes[0].fill_between(noise, p05, p95, color=color, alpha=0.12)

    setup_axis_style(
        axes[0],
        r"Perturbation amplitude $\varepsilon_{\mathrm{noise}}$",
        "Relative stress error",
        "(a) Near-TI noise sweep",
        loglog=True,
    )
    axes[0].legend(loc="lower left")

    tol_values = np.array([r["tol_deg"] for r in tol_records], dtype=float)
    median = np.array([r["median"] for r in tol_records], dtype=float)
    p95 = np.array([r["p95"] for r in tol_records], dtype=float)
    axes[1].loglog(
        tol_values,
        median,
        color=COLORS["blue"],
        marker="o",
        linewidth=1.4,
        label="median error",
    )
    axes[1].loglog(
        tol_values,
        p95,
        color=COLORS["magenta"],
        marker="s",
        linewidth=1.2,
        linestyle="--",
        label="95th percentile",
    )
    axes[1].axvline(DEFAULT_TOL_DEG, color=COLORS["black"], linestyle=":", linewidth=1.2, label="default")
    setup_axis_style(
        axes[1],
        r"Degeneracy tolerance $\mathrm{tol}_{\mathrm{deg}}$",
        "Relative stress error",
        "(b) Tolerance sweep",
        loglog=True,
    )
    axes[1].legend(loc="lower left")

    fig.subplots_adjust(wspace=0.34)
    save_figure(fig, RESULTS_DIR / "sylvester_stress_benchmark.png", dpi=PUBLICATION_DPI, close=False)
    save_manuscript_figure(fig, "sylvester_stress_benchmark", dpi=PUBLICATION_DPI, close=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tensor_ti, unique_axis, strain = ti_reference_state()
    reference_stress = ti_reference_stress(unique_axis, strain)

    noise_records, near_ti_summary = noise_sweep(tensor_ti, reference_stress, strain)
    tol_records, tol_summary = tol_deg_sweep(tensor_ti, reference_stress, strain)

    write_csv(
        RESULTS_DIR / "noise_sweep.csv",
        noise_records,
        fieldnames=["noise_level", "method", "median", "p05", "p95"],
    )
    write_csv(
        RESULTS_DIR / "tol_deg_sweep.csv",
        tol_records,
        fieldnames=["tol_deg", "median", "p05", "p95"],
    )

    summary = {
        "near_ti_noise_level": FIXED_NOISE_FOR_TOL,
        "near_ti_method_summary": near_ti_summary,
        "tol_deg_summary": tol_summary,
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    create_figure(noise_records, tol_records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
