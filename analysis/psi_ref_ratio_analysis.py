"""Psi_ref ratio sweep analysis for femur benchmark.

This script post-processes the psi_ref ratio sweep (cortical/trabecular mechanostat
set-point ratio) and compares the final predicted density field against the
CT-derived population template density mapped onto the same FE mesh.

It produces:
- A CSV/JSON table of quantitative agreement metrics vs ratio.
- Manuscript-ready figures saved to manuscript/images/.

Inputs (expected after running run_psi_ref_ratio_sweep.py):
    results/psi_ref_ratio_sweep/
    ├── sweep_summary.json
    ├── <hash>/checkpoint.bp
    └── ...

Usage:
    mpirun -n 4 python analysis/psi_ref_ratio_analysis.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import ants
from mpi4py import MPI

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Allow running from analysis/ directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from intensity_mapper import idw
from simulation.checkpoint import load_checkpoint_mesh, load_checkpoint_function
from simulation.logger import get_logger
from simulation.params import load_default_params

from analysis.plot_utils import apply_style, PUBLICATION_DPI, save_manuscript_figure


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SWEEP_DIR = project_root / "results/psi_ref_ratio_sweep"
SUMMARY_JSON = SWEEP_DIR / "sweep_summary.json"

# CT population template used elsewhere in the project
CT_IMAGE_PATH = project_root / "anatomy/raw/proximal_femur/template_new3.nii.gz"

# Parameter file (used for density bounds and trab/cort transition thresholds)
PARAMS_FILE = "stiff_params_femur.json"

# IDW interpolation parameters (keep consistent with analysis/density_comparison_plot.py)
IDW_THRESHOLD = 0.2
IDW_POWER = 1
IDW_K_NEIGHBORS = 16
CT_SMOOTHING_SIGMA_MM = 1.5

# CT-based compartment masks (using the same transition densities as the model)
USE_CT_MASKS_FROM_PARAMS = True


@dataclass(frozen=True)
class RunMetrics:
    psi_ref_ratio: float
    run_hash: str

    rmse_total: float
    mae_total: float
    corr_total: float

    rmse_trab: float
    rmse_cort: float

    mean_ct_trab: float
    mean_ct_cort: float
    mean_model_trab: float
    mean_model_cort: float

    contrast_ct: float
    contrast_model: float
    contrast_error: float

    n_total: int
    n_trab: int
    n_cort: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "psi_ref_ratio": self.psi_ref_ratio,
            "run_hash": self.run_hash,
            "rmse_total": self.rmse_total,
            "mae_total": self.mae_total,
            "corr_total": self.corr_total,
            "rmse_trab": self.rmse_trab,
            "rmse_cort": self.rmse_cort,
            "mean_ct_trab": self.mean_ct_trab,
            "mean_ct_cort": self.mean_ct_cort,
            "mean_model_trab": self.mean_model_trab,
            "mean_model_cort": self.mean_model_cort,
            "contrast_ct": self.contrast_ct,
            "contrast_model": self.contrast_model,
            "contrast_error": self.contrast_error,
            "n_total": self.n_total,
            "n_trab": self.n_trab,
            "n_cort": self.n_cort,
        }


def _owned_dofs(u) -> int:
    dofmap = u.function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def _global_stats_xy(
    x_local: np.ndarray,
    y_local: np.ndarray,
    comm: MPI.Comm,
) -> tuple[int, float, float, float, float, float]:
    """Return (n, sum_x, sum_y, sum_x2, sum_y2, sum_xy) with MPI reduction."""
    if x_local.shape != y_local.shape:
        raise ValueError("x_local and y_local must have the same shape")
    n_local = int(x_local.size)
    sums_local = np.array(
        [
            float(x_local.sum()),
            float(y_local.sum()),
            float((x_local * x_local).sum()),
            float((y_local * y_local).sum()),
            float((x_local * y_local).sum()),
        ],
        dtype=np.float64,
    )
    sums_global = np.zeros_like(sums_local)
    comm.Allreduce(sums_local, sums_global, op=MPI.SUM)
    n_global = comm.allreduce(n_local, op=MPI.SUM)
    return (
        n_global,
        float(sums_global[0]),
        float(sums_global[1]),
        float(sums_global[2]),
        float(sums_global[3]),
        float(sums_global[4]),
    )


def _global_mean(x_local: np.ndarray, comm: MPI.Comm) -> float:
    n_local = int(x_local.size)
    s_local = float(x_local.sum())
    s_global = comm.allreduce(s_local, op=MPI.SUM)
    n_global = comm.allreduce(n_local, op=MPI.SUM)
    return float(s_global / n_global) if n_global > 0 else float("nan")


def _global_rmse_mae_corr(
    x_local: np.ndarray,
    y_local: np.ndarray,
    comm: MPI.Comm,
) -> tuple[float, float, float]:
    """Compute global RMSE/MAE/corr over DOFs (unweighted)."""
    diff = x_local - y_local
    n_local = int(diff.size)
    sse_local = float((diff * diff).sum())
    sae_local = float(np.abs(diff).sum())
    sse = comm.allreduce(sse_local, op=MPI.SUM)
    sae = comm.allreduce(sae_local, op=MPI.SUM)
    n = comm.allreduce(n_local, op=MPI.SUM)
    rmse = float(np.sqrt(sse / n)) if n > 0 else float("nan")
    mae = float(sae / n) if n > 0 else float("nan")

    n, sx, sy, sx2, sy2, sxy = _global_stats_xy(x_local, y_local, comm)
    if n <= 1:
        return rmse, mae, float("nan")
    mx = sx / n
    my = sy / n
    cov = sxy - n * mx * my
    vx = sx2 - n * mx * mx
    vy = sy2 - n * my * my
    denom = float(np.sqrt(max(vx, 0.0) * max(vy, 0.0)))
    corr = float(cov / denom) if denom > 0.0 else float("nan")
    return rmse, mae, corr


def _load_ct_density(
    mesh,
    rho_min: float,
    rho_max: float,
    comm: MPI.Comm,
    logger,
) -> Any:
    if comm.rank == 0:
        logger.info(f"Loading CT template: {CT_IMAGE_PATH}")
    image = ants.image_read(str(CT_IMAGE_PATH))
    if CT_SMOOTHING_SIGMA_MM > 0:
        if comm.rank == 0:
            logger.info(f"CT smoothing sigma={CT_SMOOTHING_SIGMA_MM} mm")
        image = ants.smooth_image(image, CT_SMOOTHING_SIGMA_MM)

    ct = idw(
        image,
        mesh,
        threshold=IDW_THRESHOLD,
        power=IDW_POWER,
        k_neighbors=IDW_K_NEIGHBORS,
        method="nodes",
        verbose=(comm.rank == 0),
    )
    ct.name = "rho_ct"

    # Normalize to [rho_min, rho_max] globally (consistent across MPI ranks)
    vals = ct.x.array
    local_min = float(vals.min()) if vals.size else float("inf")
    local_max = float(vals.max()) if vals.size else float("-inf")
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)

    if comm.rank == 0:
        logger.info(f"CT raw range: [{global_min:.4f}, {global_max:.4f}]")
        logger.info(f"Rescaling CT to [{rho_min:.3f}, {rho_max:.3f}] g/cm^3")

    if global_max - global_min > 1e-12:
        normalized = (vals - global_min) / (global_max - global_min)
        ct.x.array[:] = rho_min + normalized * (rho_max - rho_min)
    else:
        ct.x.array[:] = 0.5 * (rho_min + rho_max)
    ct.x.scatter_forward()
    return ct


def _load_sweep_runs(comm: MPI.Comm, logger) -> list[dict[str, Any]]:
    if comm.rank == 0:
        if not SUMMARY_JSON.exists():
            raise FileNotFoundError(f"Sweep summary not found: {SUMMARY_JSON}")
        with open(SUMMARY_JSON, "r") as f:
            summary = json.load(f)
        runs = summary.get("runs", [])
        runs_sorted = sorted(runs, key=lambda r: float(r["psi_ref_ratio"]))
    else:
        runs_sorted = None
    return comm.bcast(runs_sorted, root=0)


def _plot_metrics(metrics: list[RunMetrics], out_dir: Path) -> None:
    apply_style()

    ratios = np.array([m.psi_ref_ratio for m in metrics], dtype=float)
    rmse_total = np.array([m.rmse_total for m in metrics], dtype=float)
    rmse_trab = np.array([m.rmse_trab for m in metrics], dtype=float)
    rmse_cort = np.array([m.rmse_cort for m in metrics], dtype=float)

    contrast_model = np.array([m.contrast_model for m in metrics], dtype=float)
    contrast_ct = np.array([m.contrast_ct for m in metrics], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    ax = axes[0]
    ax.plot(ratios, rmse_total, marker="o", label="RMSE (total)")
    ax.plot(ratios, rmse_trab, marker="s", label="RMSE (trab mask)")
    ax.plot(ratios, rmse_cort, marker="^", label="RMSE (cort mask)")
    ax.set_xlabel(r"$r=\psi_{\mathrm{ref}}^{\mathrm{cort}}/\psi_{\mathrm{ref}}^{\mathrm{trab}}$")
    ax.set_ylabel(r"RMSE in $\rho$ [g/cm$^3$]")
    ax.set_title("Spatial agreement vs CT")
    ax.legend()

    ax = axes[1]
    ax.plot(ratios, contrast_model, marker="o", label=r"Model $\bar\rho_{\mathrm{cort}}-\bar\rho_{\mathrm{trab}}$")
    ax.axhline(float(contrast_ct[0]) if contrast_ct.size else 0.0, color="black", linestyle="--",
               label=r"CT $\bar\rho_{\mathrm{cort}}-\bar\rho_{\mathrm{trab}}$")
    ax.set_xlabel(r"$r=\psi_{\mathrm{ref}}^{\mathrm{cort}}/\psi_{\mathrm{ref}}^{\mathrm{trab}}$")
    ax.set_ylabel(r"Contrast [g/cm$^3$]")
    ax.set_title("Cortical–trabecular contrast")
    ax.legend()

    plt.tight_layout()
    save_manuscript_figure(fig, "psi_ref_ratio_diagnostic", dpi=PUBLICATION_DPI, close=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "psi_ref_ratio_diagnostic.png", dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_histograms(
    bin_edges: np.ndarray,
    ct_counts: np.ndarray,
    model_counts_by_ratio: dict[float, np.ndarray],
    out_dir: Path,
    ratios_to_plot: list[float],
) -> None:
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = float(bin_edges[1] - bin_edges[0]) if bin_edges.size > 1 else 1.0

    ct_density = ct_counts / max(ct_counts.sum() * bin_width, 1e-12)
    ax.plot(bin_centers, ct_density, color="black", linewidth=1.2, label="CT")

    for r in ratios_to_plot:
        counts = model_counts_by_ratio.get(r)
        if counts is None:
            continue
        dens = counts / max(counts.sum() * bin_width, 1e-12)
        ax.plot(bin_centers, dens, linewidth=1.0, label=f"model r={r:g}")

    ax.set_xlabel(r"Density $\rho$ [g/cm$^3$]")
    ax.set_ylabel("Probability density")
    ax.set_title("Density distributions (final state)")
    ax.legend()
    plt.tight_layout()

    save_manuscript_figure(fig, "psi_ref_ratio_histograms", dpi=PUBLICATION_DPI, close=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "psi_ref_ratio_histograms.png", dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _plot_timecourses(
    mean_rho_by_ratio: dict[float, tuple[np.ndarray, np.ndarray]],
    ct_mean_rho: float,
    out_dir: Path,
) -> None:
    """Plot domain-mean density vs time for each ratio (rank 0)."""
    apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.0))

    ratios = sorted(mean_rho_by_ratio.keys())
    for r in ratios:
        t, rho_bar = mean_rho_by_ratio[r]
        ax.plot(t, rho_bar, linewidth=1.0, label=f"r={r:g}")

    ax.axhline(ct_mean_rho, color="black", linestyle="--", linewidth=1.0, label="CT mean")
    ax.set_xlabel("Time [day]")
    ax.set_ylabel(r"Mean density $\bar\rho$ [g/cm$^3$]")
    ax.set_title("Global density evolution")
    ax.legend(ncol=2)
    plt.tight_layout()

    save_manuscript_figure(fig, "psi_ref_ratio_timecourse", dpi=PUBLICATION_DPI, close=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "psi_ref_ratio_timecourse.png", dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="PsiRefRatioAnalysis")

    runs = _load_sweep_runs(comm, logger)
    if not runs:
        if comm.rank == 0:
            logger.error("No runs found in sweep summary.")
        return

    # Load parameters (bounds + transition thresholds)
    params = load_default_params(PARAMS_FILE)
    rho_min = float(params["density"].rho_min)
    rho_max = float(params["density"].rho_max)
    rho_trab_max = float(params["material"].rho_trab_max)
    rho_cort_min = float(params["material"].rho_cort_min)

    # Load mesh from first checkpoint
    first_ckpt = SWEEP_DIR / runs[0]["output_dir"] / "checkpoint.bp"
    if not first_ckpt.exists():
        if comm.rank == 0:
            raise FileNotFoundError(f"Checkpoint not found: {first_ckpt}")
        return

    mesh = load_checkpoint_mesh(first_ckpt, comm)
    V = None
    try:
        from dolfinx import fem

        V = fem.functionspace(mesh, ("Lagrange", 1))
    except Exception as e:
        if comm.rank == 0:
            raise RuntimeError(f"Failed to create CG1 space: {e}") from e

    # Load CT onto mesh once
    ct_func = _load_ct_density(mesh, rho_min=rho_min, rho_max=rho_max, comm=comm, logger=logger)

    # CT domain-mean density (volume-weighted), used for time-course comparison
    ct_mean_rho_vol = float("nan")
    volume_mm3 = float("nan")
    try:
        import ufl
        from dolfinx import fem

        dx = ufl.Measure("dx", domain=mesh)
        vol = fem.assemble_scalar(fem.form(1.0 * dx))
        vol = float(comm.allreduce(vol, op=MPI.SUM))
        volume_mm3 = vol
        ct_int = fem.assemble_scalar(fem.form(ct_func * dx))
        ct_int = float(comm.allreduce(ct_int, op=MPI.SUM))
        ct_mean_rho_vol = float(ct_int / vol) if vol > 0 else float("nan")
    except Exception as e:
        if comm.rank == 0:
            logger.warning(f"Failed to compute CT volume-mean density: {e}")

    n_owned = _owned_dofs(ct_func)
    ct_vals_local = ct_func.x.array[:n_owned].copy()

    # CT-based compartment masks (fixed across all runs)
    if USE_CT_MASKS_FROM_PARAMS:
        trab_mask_local = ct_vals_local <= rho_trab_max
        cort_mask_local = ct_vals_local >= rho_cort_min
    else:
        # Fallback: split by median (kept for robustness)
        med = float(np.median(ct_vals_local))
        trab_mask_local = ct_vals_local <= med
        cort_mask_local = ct_vals_local > med

    n_trab = comm.allreduce(int(trab_mask_local.sum()), op=MPI.SUM)
    n_cort = comm.allreduce(int(cort_mask_local.sum()), op=MPI.SUM)
    n_total = comm.allreduce(int(ct_vals_local.size), op=MPI.SUM)
    if comm.rank == 0:
        logger.info(f"CT masks: n_trab={n_trab}, n_cort={n_cort}, n_total={n_total}")
        logger.info(f"CT thresholds: rho_trab_max={rho_trab_max:g}, rho_cort_min={rho_cort_min:g}")

    mean_ct_trab = _global_mean(ct_vals_local[trab_mask_local], comm) if n_trab else float("nan")
    mean_ct_cort = _global_mean(ct_vals_local[cort_mask_local], comm) if n_cort else float("nan")
    contrast_ct = mean_ct_cort - mean_ct_trab

    # Compute metrics per run
    metrics: list[RunMetrics] = []
    model_counts_by_ratio: dict[float, np.ndarray] = {}
    mean_rho_time_by_ratio: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    # Histogram bins (global fixed range)
    bin_edges = np.linspace(rho_min, rho_max, 61)
    ct_counts_local, _ = np.histogram(ct_vals_local, bins=bin_edges)
    ct_counts = np.zeros_like(ct_counts_local)
    comm.Allreduce(ct_counts_local, ct_counts, op=MPI.SUM)

    for run in runs:
        ratio = float(run["psi_ref_ratio"])
        run_hash = str(run["output_dir"])
        ckpt = SWEEP_DIR / run_hash / "checkpoint.bp"

        if not ckpt.exists():
            if comm.rank == 0:
                logger.warning(f"Missing checkpoint for ratio={ratio:g}: {ckpt}")
            continue

        rho = load_checkpoint_function(ckpt, "rho", V, time=None)
        rho_vals_local = rho.x.array[:n_owned].copy()

        rmse_total, mae_total, corr_total = _global_rmse_mae_corr(rho_vals_local, ct_vals_local, comm)

        rmse_trab = (
            _global_rmse_mae_corr(rho_vals_local[trab_mask_local], ct_vals_local[trab_mask_local], comm)[0]
            if n_trab
            else float("nan")
        )
        rmse_cort = (
            _global_rmse_mae_corr(rho_vals_local[cort_mask_local], ct_vals_local[cort_mask_local], comm)[0]
            if n_cort
            else float("nan")
        )

        mean_model_trab = _global_mean(rho_vals_local[trab_mask_local], comm) if n_trab else float("nan")
        mean_model_cort = _global_mean(rho_vals_local[cort_mask_local], comm) if n_cort else float("nan")
        contrast_model = mean_model_cort - mean_model_trab

        metrics.append(
            RunMetrics(
                psi_ref_ratio=ratio,
                run_hash=run_hash,
                rmse_total=rmse_total,
                mae_total=mae_total,
                corr_total=corr_total,
                rmse_trab=rmse_trab,
                rmse_cort=rmse_cort,
                mean_ct_trab=mean_ct_trab,
                mean_ct_cort=mean_ct_cort,
                mean_model_trab=mean_model_trab,
                mean_model_cort=mean_model_cort,
                contrast_ct=contrast_ct,
                contrast_model=contrast_model,
                contrast_error=float(contrast_model - contrast_ct),
                n_total=n_total,
                n_trab=n_trab,
                n_cort=n_cort,
            )
        )

        # Global histogram counts for this ratio (for rank-0 plotting)
        model_counts_local, _ = np.histogram(rho_vals_local, bins=bin_edges)
        model_counts = np.zeros_like(model_counts_local)
        comm.Allreduce(model_counts_local, model_counts, op=MPI.SUM)
        model_counts_by_ratio[ratio] = model_counts

        # Time course (rank 0 only): mean density from steps.csv (volume-weighted)
        if comm.rank == 0:
            steps_csv = SWEEP_DIR / run_hash / "steps.csv"
            if steps_csv.exists():
                import csv

                times: list[float] = []
                rho_bar: list[float] = []
                with open(steps_csv, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if str(row.get("accepted", "1")) not in ("1", "True", "true"):
                            continue
                        t = float(row["time_days"])
                        mass_g = float(row["total_mass_g"])
                        if not np.isfinite(volume_mm3) or volume_mm3 <= 0.0:
                            continue
                        rho_mean = mass_g / (volume_mm3 * 1e-3)  # g / (cm^3)
                        times.append(t)
                        rho_bar.append(rho_mean)
                if times:
                    mean_rho_time_by_ratio[ratio] = (np.array(times), np.array(rho_bar))
            else:
                logger.debug(f"Missing steps.csv for ratio={ratio:g}: {steps_csv}")

        if comm.rank == 0:
            logger.info(
                f"ratio={ratio:g}  RMSE={rmse_total:.4f}  corr={corr_total:.3f}  "
                f"contrast={contrast_model:.3f} (CT {contrast_ct:.3f})"
            )

    # Sort metrics by ratio
    metrics = sorted(metrics, key=lambda m: m.psi_ref_ratio)

    if comm.rank == 0:
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        metrics_csv = SWEEP_DIR / "psi_ref_ratio_metrics.csv"
        metrics_json = SWEEP_DIR / "psi_ref_ratio_metrics.json"

        import csv

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].to_dict().keys()))
            writer.writeheader()
            for m in metrics:
                writer.writerow(m.to_dict())
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)

        logger.info(f"Wrote metrics: {metrics_csv}")

        # Plots (rank 0)
        _plot_metrics(metrics, out_dir=SWEEP_DIR)

        # Histogram plot: CT + r=1 + best-RMSE
        ratios_available = [m.psi_ref_ratio for m in metrics]
        ratios_to_plot: list[float] = []
        if 1.0 in ratios_available:
            ratios_to_plot.append(1.0)
        best = min(metrics, key=lambda m: m.rmse_total)
        if best.psi_ref_ratio not in ratios_to_plot:
            ratios_to_plot.append(best.psi_ref_ratio)
        _plot_histograms(
            bin_edges=bin_edges,
            ct_counts=ct_counts,
            model_counts_by_ratio=model_counts_by_ratio,
            out_dir=SWEEP_DIR,
            ratios_to_plot=ratios_to_plot,
        )

        if mean_rho_time_by_ratio:
            _plot_timecourses(mean_rho_time_by_ratio, ct_mean_rho=ct_mean_rho_vol, out_dir=SWEEP_DIR)

    comm.Barrier()


if __name__ == "__main__":
    main()
