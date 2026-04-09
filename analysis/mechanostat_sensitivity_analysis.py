"""Mechanostat parameter sensitivity analysis for the femur benchmark.

Post-processes the compact sweep over

    - stimulus_delta0  (lazy-zone threshold scale)
    - stimulus_kappa   (tanh saturation width)

and quantifies how the final femur density field changes relative to the
CT-derived population template.

Outputs:
    results/mechanostat_sweep/
    ├── mechanostat_metrics.csv
    ├── mechanostat_metrics.json
    ├── mechanostat_summary.json
    └── mechanostat_sensitivity.png

Additionally:
    manuscript/images/mechanostat_sensitivity.png
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mpi4py import MPI

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulation.checkpoint import load_checkpoint_mesh, load_checkpoint_function
from simulation.logger import get_logger
from simulation.params import load_default_params

from analysis.alpha_front_analysis import (
    _assemble_scalar,
    _global_rmse_interval_band,
    _global_rmse_mae_corr,
    _load_ct_density,
    _owned_dofs,
)
from analysis.plot_utils import PUBLICATION_DPI, apply_style, save_manuscript_figure


SWEEP_DIR = project_root / "results/mechanostat_sweep"
SUMMARY_JSON = SWEEP_DIR / "sweep_summary.json"
PARAMS_FILE = "stiff_params_femur.json"


@dataclass(frozen=True)
class RunMetrics:
    stimulus_delta0: float
    stimulus_kappa: float
    run_hash: str
    rmse_total: float
    mae_total: float
    corr_total: float
    rmse_total_lo: float
    rmse_total_hi: float
    corr_total_lo: float
    corr_total_hi: float
    mean_rho: float
    mean_rho_delta_ct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stimulus_delta0": self.stimulus_delta0,
            "stimulus_kappa": self.stimulus_kappa,
            "run_hash": self.run_hash,
            "rmse_total": self.rmse_total,
            "mae_total": self.mae_total,
            "corr_total": self.corr_total,
            "rmse_total_lo": self.rmse_total_lo,
            "rmse_total_hi": self.rmse_total_hi,
            "corr_total_lo": self.corr_total_lo,
            "corr_total_hi": self.corr_total_hi,
            "mean_rho": self.mean_rho,
            "mean_rho_delta_ct": self.mean_rho_delta_ct,
        }


def _load_sweep_runs(comm: MPI.Comm) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if comm.rank == 0:
        if SUMMARY_JSON.exists():
            with SUMMARY_JSON.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            runs = summary.get("runs", [])
        else:
            runs = []
            for run_dir in sorted(SWEEP_DIR.iterdir()):
                if not run_dir.is_dir():
                    continue
                cfg_path = run_dir / "config.json"
                if not cfg_path.exists():
                    continue
                with cfg_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                runs.append(
                    {
                        "output_dir": run_dir.name,
                        "stimulus.stimulus_delta0": float(cfg["stimulus"]["stimulus_delta0"]),
                        "stimulus.stimulus_kappa": float(cfg["stimulus"]["stimulus_kappa"]),
                    }
                )
            summary = {
                "description": "Recovered mechanostat sweep metadata from per-run config.json files",
                "runs": runs,
            }
        runs_sorted = sorted(
            runs,
            key=lambda row: (
                float(row["stimulus.stimulus_delta0"]),
                float(row["stimulus.stimulus_kappa"]),
            ),
        )
    else:
        summary = None
        runs_sorted = None
    return comm.bcast(runs_sorted, root=0), comm.bcast(summary, root=0)


def _build_matrix(
    metrics: list[RunMetrics],
    delta0_values: list[float],
    kappa_values: list[float],
    attr: str,
) -> np.ndarray:
    mat = np.full((len(delta0_values), len(kappa_values)), np.nan, dtype=float)
    d_index = {float(v): i for i, v in enumerate(delta0_values)}
    k_index = {float(v): j for j, v in enumerate(kappa_values)}
    for item in metrics:
        i = d_index[float(item.stimulus_delta0)]
        j = k_index[float(item.stimulus_kappa)]
        mat[i, j] = float(getattr(item, attr))
    return mat


def _annotate_heatmap(ax: plt.Axes, matrix: np.ndarray, fmt: str) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )


def _highlight_default(
    ax: plt.Axes,
    delta0_values: list[float],
    kappa_values: list[float],
    delta0_default: float,
    kappa_default: float,
) -> None:
    if delta0_default not in delta0_values or kappa_default not in kappa_values:
        return
    i = delta0_values.index(delta0_default)
    j = kappa_values.index(kappa_default)
    rect = plt.Rectangle(
        (j - 0.5, i - 0.5),
        1.0,
        1.0,
        fill=False,
        edgecolor="black",
        linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.scatter([j], [i], c="black", marker="s", s=18, zorder=3)


def _plot_heatmaps(
    metrics: list[RunMetrics],
    delta0_values: list[float],
    kappa_values: list[float],
    delta0_default: float,
    kappa_default: float,
    out_dir: Path,
) -> None:
    apply_style()

    rmse = _build_matrix(metrics, delta0_values, kappa_values, "rmse_total")
    corr = _build_matrix(metrics, delta0_values, kappa_values, "corr_total")
    mean_delta = _build_matrix(metrics, delta0_values, kappa_values, "mean_rho")

    default_item = next(
        (
            item
            for item in metrics
            if np.isclose(item.stimulus_delta0, delta0_default)
            and np.isclose(item.stimulus_kappa, kappa_default)
        ),
        None,
    )

    if default_item is not None:
        rmse = rmse - float(default_item.rmse_total)
        corr = corr - float(default_item.corr_total)
        mean_delta = mean_delta - float(default_item.mean_rho)

    def centered_norm(matrix: np.ndarray) -> Normalize:
        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        if vmin < 0.0 < vmax:
            return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(8.1, 2.7), constrained_layout=True)

    panels = [
        (
            axes[0],
            rmse,
            "RdBu_r",
            centered_norm(rmse),
            r"(a) $\Delta$RMSE to CT",
            r"$\Delta$RMSE [g/cm$^3$]",
            ".3f",
        ),
        (
            axes[1],
            corr,
            "RdBu_r",
            centered_norm(corr),
            r"(b) $\Delta$correlation to CT",
            r"$\Delta$ corr [-]",
            ".3f",
        ),
        (
            axes[2],
            mean_delta,
            "RdBu_r",
            centered_norm(mean_delta),
            r"(c) $\Delta$ final mean density",
            r"$\Delta \bar\rho(T)$ [g/cm$^3$]",
            ".3f",
        ),
    ]

    for ax, matrix, cmap, norm, title, cbar_label, fmt in panels:
        im = ax.imshow(matrix, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        ax.set_xticks(range(len(kappa_values)), [f"{v:.2f}" for v in kappa_values])
        ax.set_yticks(range(len(delta0_values)), [f"{v:.2f}" for v in delta0_values])
        ax.set_xlabel(r"$\kappa_S$")
        ax.set_ylabel(r"$\delta_0$")
        ax.set_title(title)
        _annotate_heatmap(ax, matrix, fmt)
        _highlight_default(ax, delta0_values, kappa_values, delta0_default, kappa_default)
        cbar = fig.colorbar(im, ax=ax, aspect=18, pad=0.03)
        cbar.set_label(cbar_label)

    save_manuscript_figure(fig, "mechanostat_sensitivity", dpi=PUBLICATION_DPI, close=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mechanostat_sensitivity.png", dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _cross_section(
    metrics: list[RunMetrics],
    *,
    delta0: float | None = None,
    kappa: float | None = None,
) -> list[RunMetrics]:
    out: list[RunMetrics] = []
    for item in metrics:
        if delta0 is not None and not np.isclose(item.stimulus_delta0, delta0):
            continue
        if kappa is not None and not np.isclose(item.stimulus_kappa, kappa):
            continue
        out.append(item)
    return out


def _metric_span(items: list[RunMetrics], attr: str) -> dict[str, float] | None:
    if not items:
        return None
    values = np.array([float(getattr(item, attr)) for item in items], dtype=float)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "spread": float(values.max() - values.min()),
    }


def main() -> None:
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="MechanostatSensitivity")

    runs, summary_meta = _load_sweep_runs(comm)
    if not runs:
        if comm.rank == 0:
            logger.error("No runs found in mechanostat sweep.")
        return

    params = load_default_params(PARAMS_FILE)
    rho_min = float(params["density"].rho_min)
    rho_max = float(params["density"].rho_max)
    delta0_default = float(params["stimulus"].stimulus_delta0)
    kappa_default = float(params["stimulus"].stimulus_kappa)

    first_ckpt = SWEEP_DIR / runs[0]["output_dir"] / "checkpoint.bp"
    if not first_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {first_ckpt}")

    mesh = load_checkpoint_mesh(first_ckpt, comm)

    from dolfinx import fem
    import ufl

    V = fem.functionspace(mesh, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=mesh)

    ct_func, ct_lo_func, ct_hi_func = _load_ct_density(
        mesh, rho_min=rho_min, rho_max=rho_max, comm=comm, logger=logger
    )

    volume = _assemble_scalar(fem.form(1.0 * dx), comm)
    ct_mass = _assemble_scalar(fem.form(ct_func * dx), comm)
    ct_mean_rho = float(ct_mass / volume) if volume > 0.0 else float("nan")

    n_owned = _owned_dofs(ct_func)
    ct_vals_local = ct_func.x.array[:n_owned].copy()
    ct_vals_lo_local = ct_lo_func.x.array[:n_owned].copy()
    ct_vals_hi_local = ct_hi_func.x.array[:n_owned].copy()

    metrics: list[RunMetrics] = []

    for run in runs:
        delta0 = float(run["stimulus.stimulus_delta0"])
        kappa = float(run["stimulus.stimulus_kappa"])
        run_hash = str(run["output_dir"])
        ckpt = SWEEP_DIR / run_hash / "checkpoint.bp"
        if not ckpt.exists():
            if comm.rank == 0:
                logger.warning(f"Missing checkpoint for delta0={delta0:g}, kappa={kappa:g}")
            continue

        rho = load_checkpoint_function(ckpt, "rho", V, time=None)
        rho_vals_local = rho.x.array[:n_owned].copy()

        rmse_total, mae_total, corr_total = _global_rmse_mae_corr(rho_vals_local, ct_vals_local, comm)
        rmse_band_lo, rmse_band_hi = _global_rmse_interval_band(
            rho_vals_local, ct_vals_lo_local, ct_vals_hi_local, comm
        )
        _, _, corr_lo = _global_rmse_mae_corr(rho_vals_local, ct_vals_lo_local, comm)
        _, _, corr_hi = _global_rmse_mae_corr(rho_vals_local, ct_vals_hi_local, comm)

        rho_mass = _assemble_scalar(fem.form(rho * dx), comm)
        rho_mean = float(rho_mass / volume) if volume > 0.0 else float("nan")

        metrics.append(
            RunMetrics(
                stimulus_delta0=delta0,
                stimulus_kappa=kappa,
                run_hash=run_hash,
                rmse_total=rmse_total,
                mae_total=mae_total,
                corr_total=corr_total,
                rmse_total_lo=rmse_band_lo,
                rmse_total_hi=rmse_band_hi,
                corr_total_lo=float(min(corr_lo, corr_hi)),
                corr_total_hi=float(max(corr_lo, corr_hi)),
                mean_rho=rho_mean,
                mean_rho_delta_ct=float(rho_mean - ct_mean_rho),
            )
        )

        if comm.rank == 0:
            logger.info(
                f"delta0={delta0:4.2f}  kappa={kappa:4.2f}  "
                f"RMSE={rmse_total:.4f}  corr={corr_total:.3f}  "
                f"mean_rho-CT={rho_mean - ct_mean_rho:+.4f}"
            )

    metrics = sorted(metrics, key=lambda item: (item.stimulus_delta0, item.stimulus_kappa))

    if comm.rank == 0 and metrics:
        delta0_values = sorted({float(item.stimulus_delta0) for item in metrics})
        kappa_values = sorted({float(item.stimulus_kappa) for item in metrics})

        metrics_csv = SWEEP_DIR / "mechanostat_metrics.csv"
        metrics_json = SWEEP_DIR / "mechanostat_metrics.json"
        with metrics_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].to_dict().keys()))
            writer.writeheader()
            for item in metrics:
                writer.writerow(item.to_dict())
        with metrics_json.open("w", encoding="utf-8") as f:
            json.dump([item.to_dict() for item in metrics], f, indent=2)

        default_items = _cross_section(metrics, delta0=delta0_default, kappa=kappa_default)
        default_item = default_items[0] if default_items else None
        best_rmse = min(metrics, key=lambda item: item.rmse_total)
        best_corr = max(metrics, key=lambda item: item.corr_total)

        summary = {
            "baseline": None
            if default_item is None
            else {
                "stimulus_delta0": default_item.stimulus_delta0,
                "stimulus_kappa": default_item.stimulus_kappa,
                "rmse_total": default_item.rmse_total,
                "corr_total": default_item.corr_total,
                "mean_rho": default_item.mean_rho,
                "mean_rho_delta_ct": default_item.mean_rho_delta_ct,
            },
            "best_rmse": {
                "stimulus_delta0": best_rmse.stimulus_delta0,
                "stimulus_kappa": best_rmse.stimulus_kappa,
                "rmse_total": best_rmse.rmse_total,
            },
            "best_corr": {
                "stimulus_delta0": best_corr.stimulus_delta0,
                "stimulus_kappa": best_corr.stimulus_kappa,
                "corr_total": best_corr.corr_total,
            },
            "global_ranges": {
                "rmse_total": _metric_span(metrics, "rmse_total"),
                "corr_total": _metric_span(metrics, "corr_total"),
                "mean_rho_delta_ct": _metric_span(metrics, "mean_rho_delta_ct"),
            },
            "delta0_slice_at_default_kappa": {
                "kappa": kappa_default,
                "rmse_total": _metric_span(_cross_section(metrics, kappa=kappa_default), "rmse_total"),
                "corr_total": _metric_span(_cross_section(metrics, kappa=kappa_default), "corr_total"),
                "mean_rho_delta_ct": _metric_span(
                    _cross_section(metrics, kappa=kappa_default), "mean_rho_delta_ct"
                ),
            },
            "kappa_slice_at_default_delta0": {
                "delta0": delta0_default,
                "rmse_total": _metric_span(_cross_section(metrics, delta0=delta0_default), "rmse_total"),
                "corr_total": _metric_span(_cross_section(metrics, delta0=delta0_default), "corr_total"),
                "mean_rho_delta_ct": _metric_span(
                    _cross_section(metrics, delta0=delta0_default), "mean_rho_delta_ct"
                ),
            },
            "sweep_metadata": summary_meta,
        }

        with (SWEEP_DIR / "mechanostat_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        _plot_heatmaps(
            metrics,
            delta0_values=delta0_values,
            kappa_values=kappa_values,
            delta0_default=delta0_default,
            kappa_default=kappa_default,
            out_dir=SWEEP_DIR,
        )

    comm.Barrier()


if __name__ == "__main__":
    main()
