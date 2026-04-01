"""Contextualize alpha_front sweep against physiologic frontal-angle variability.

This script performs a lightweight post-processing pass over the existing
`results/alpha_front_sweep/alpha_front_metrics.csv` table. It linearly
interpolates the RMSE and Pearson-correlation curves to quantify how much the
metrics vary over realistic inter-subject frontal-plane angle ranges.

Outputs:
- `results/reviewer_benchmarks/alpha_front_variability/summary.json`
- `results/reviewer_benchmarks/alpha_front_variability/alpha_front_variability_context.png`
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from analysis.plot_utils import (
    COLORS,
    PUBLICATION_DPI,
    apply_style,
    save_figure,
    setup_axis_style,
)


SWEEP_CSV = project_root / "results" / "alpha_front_sweep" / "alpha_front_metrics.csv"
RESULTS_DIR = project_root / "results" / "reviewer_benchmarks" / "alpha_front_variability"
MANUSCRIPT_NAME = "alpha_front_variability_context.png"

PHYSIOLOGIC_BANDS_DEG = [7.0, 13.0]
FIGSIZE_COMPACT_TWO_PANEL = (5.5, 2.45)


def load_metrics() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, float]] = []
    with SWEEP_CSV.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "alpha_front_offset_deg": float(row["alpha_front_offset_deg"]),
                    "rmse_total": float(row["rmse_total"]),
                    "corr_total": float(row["corr_total"]),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows found in {SWEEP_CSV}")

    rows.sort(key=lambda row: row["alpha_front_offset_deg"])
    offsets = np.array([row["alpha_front_offset_deg"] for row in rows], dtype=float)
    rmse = np.array([row["rmse_total"] for row in rows], dtype=float)
    corr = np.array([row["corr_total"] for row in rows], dtype=float)
    return offsets, rmse, corr


def summarize_band(
    offsets: np.ndarray, rmse: np.ndarray, corr: np.ndarray, half_width_deg: float
) -> dict[str, float]:
    grid = np.linspace(-half_width_deg, half_width_deg, 401)
    rmse_grid = np.interp(grid, offsets, rmse)
    corr_grid = np.interp(grid, offsets, corr)
    return {
        "half_width_deg": half_width_deg,
        "rmse_min": float(rmse_grid.min()),
        "rmse_max": float(rmse_grid.max()),
        "rmse_spread": float(rmse_grid.max() - rmse_grid.min()),
        "corr_min": float(corr_grid.min()),
        "corr_max": float(corr_grid.max()),
        "corr_spread": float(corr_grid.max() - corr_grid.min()),
    }


def create_plot(
    offsets: np.ndarray,
    rmse: np.ndarray,
    corr: np.ndarray,
    baseline_rmse: float,
    baseline_corr: float,
) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_COMPACT_TWO_PANEL)

    for axis, values, ylabel, title in [
        (axes[0], rmse, r"RMSE in $\rho$ [g/cm$^3$]", "(a) RMSE"),
        (axes[1], corr, "Pearson correlation [-]", "(b) Correlation"),
    ]:
        axis.axvspan(-13.0, 13.0, color=COLORS["grey"], alpha=0.12, zorder=0, label=r"$\pm 13^\circ$")
        axis.axvspan(-7.0, 7.0, color=COLORS["blue"], alpha=0.12, zorder=1, label=r"$\pm 7^\circ$")
        axis.plot(offsets, values, color=COLORS["black"], marker="o", linewidth=1.8, markersize=4.5)
        axis.axvline(0.0, color=COLORS["black"], linestyle=":", linewidth=1.0)
        setup_axis_style(axis, r"$\Delta \alpha_{\mathrm{front}}$ [deg]", ylabel, title)

    axes[0].scatter([0.0], [baseline_rmse], color=COLORS["red"], zorder=3, s=24)
    axes[1].scatter([0.0], [baseline_corr], color=COLORS["red"], zorder=3, s=24)
    axes[0].legend(frameon=False, loc="best")

    out_path = RESULTS_DIR / MANUSCRIPT_NAME
    fig.subplots_adjust(bottom=0.2, wspace=0.32)
    save_figure(fig, out_path, dpi=PUBLICATION_DPI)
    plt.close(fig)
    return out_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    offsets, rmse, corr = load_metrics()
    baseline_rmse = float(np.interp(0.0, offsets, rmse))
    baseline_corr = float(np.interp(0.0, offsets, corr))

    summary = {
        "baseline": {
            "alpha_front_offset_deg": 0.0,
            "rmse_total": baseline_rmse,
            "corr_total": baseline_corr,
        },
        "physiologic_bands": [summarize_band(offsets, rmse, corr, band) for band in PHYSIOLOGIC_BANDS_DEG],
        "full_sweep": {
            "rmse_min": float(rmse.min()),
            "rmse_max": float(rmse.max()),
            "rmse_spread": float(rmse.max() - rmse.min()),
            "corr_min": float(corr.min()),
            "corr_max": float(corr.max()),
            "corr_spread": float(corr.max() - corr.min()),
        },
    }

    plot_path = create_plot(offsets, rmse, corr, baseline_rmse, baseline_corr)
    summary["plot"] = {"path": str(plot_path.relative_to(project_root))}

    with (RESULTS_DIR / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
