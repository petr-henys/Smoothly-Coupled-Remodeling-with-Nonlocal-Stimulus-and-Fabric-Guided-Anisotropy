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
from mpi4py import MPI

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Allow running from analysis/ directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from morpho_mapper import rescale_to_density
from simulation.checkpoint import load_checkpoint_mesh, load_checkpoint_function
from simulation.logger import get_logger
from simulation.params import load_default_params

from analysis.plot_utils import (
    apply_style,
    PUBLICATION_DPI,
    save_manuscript_figure,
    PLOT_LINEWIDTH,
    PLOT_MARKERSIZE,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SWEEP_DIR = project_root / "results/psi_ref_ratio_sweep"
SUMMARY_JSON = SWEEP_DIR / "sweep_summary.json"

POP_STATS_CHECKPOINT = project_root / "results/ct_density/population_stats_checkpoint.bp"

# Parameter file (used for density bounds and trab/cort transition thresholds)
PARAMS_FILE = "stiff_params_femur.json"

# Cohort stats are precomputed by morpho_mapper.py; no CT template used here.

# CT-based compartment masks (using the same transition densities as the model)
USE_CT_MASKS_FROM_PARAMS = True


@dataclass(frozen=True)
class RunMetrics:
    psi_ref_ratio: float
    run_hash: str

    rmse_total: float
    mae_total: float
    corr_total: float

    rmse_total_lo: float
    rmse_total_hi: float
    corr_total_lo: float
    corr_total_hi: float

    rmse_trab: float
    rmse_cort: float

    mean_ct_trab: float
    mean_ct_cort: float
    mean_model_trab: float
    mean_model_cort: float

    contrast_ct: float
    contrast_ct_lo: float
    contrast_ct_hi: float
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
            "rmse_total_lo": self.rmse_total_lo,
            "rmse_total_hi": self.rmse_total_hi,
            "corr_total_lo": self.corr_total_lo,
            "corr_total_hi": self.corr_total_hi,
            "rmse_trab": self.rmse_trab,
            "rmse_cort": self.rmse_cort,
            "mean_ct_trab": self.mean_ct_trab,
            "mean_ct_cort": self.mean_ct_cort,
            "mean_model_trab": self.mean_model_trab,
            "mean_model_cort": self.mean_model_cort,
            "contrast_ct": self.contrast_ct,
            "contrast_ct_lo": self.contrast_ct_lo,
            "contrast_ct_hi": self.contrast_ct_hi,
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


def _global_rmse_interval_band(
    x_local: np.ndarray,
    y_lo_local: np.ndarray,
    y_hi_local: np.ndarray,
    comm: MPI.Comm,
) -> tuple[float, float]:
    """RMSE envelope over pointwise interval reference [y_lo, y_hi]."""
    if x_local.shape != y_lo_local.shape or x_local.shape != y_hi_local.shape:
        raise ValueError("x_local, y_lo_local, y_hi_local must have the same shape")

    lo = np.minimum(y_lo_local, y_hi_local)
    hi = np.maximum(y_lo_local, y_hi_local)

    inside = (x_local >= lo) & (x_local <= hi)
    d_lo = np.abs(x_local - lo)
    d_hi = np.abs(x_local - hi)

    d_min = np.where(inside, 0.0, np.minimum(d_lo, d_hi))
    d_max = np.maximum(d_lo, d_hi)

    sse_min_local = float(np.dot(d_min, d_min))
    sse_max_local = float(np.dot(d_max, d_max))
    n_local = int(x_local.size)

    sse_min = comm.allreduce(sse_min_local, op=MPI.SUM)
    sse_max = comm.allreduce(sse_max_local, op=MPI.SUM)
    n = comm.allreduce(n_local, op=MPI.SUM)
    if n <= 0:
        return float("nan"), float("nan")

    return float(np.sqrt(sse_min / n)), float(np.sqrt(sse_max / n))


def _load_ct_density(
    mesh,
    rho_min: float,
    rho_max: float,
    comm: MPI.Comm,
    logger,
) -> Any:
    if not POP_STATS_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Population stats checkpoint not found: {POP_STATS_CHECKPOINT}. "
            "Run morpho_mapper.py to generate cohort stats first."
        )

    if comm.rank == 0:
        logger.info(f"Loading cohort intensity stats from {POP_STATS_CHECKPOINT}")

    from dolfinx import fem

    V = fem.functionspace(mesh, ("Lagrange", 1))
    I_mean = load_checkpoint_function(POP_STATS_CHECKPOINT, "I_mean", V, time=0.0)
    I_lo = load_checkpoint_function(POP_STATS_CHECKPOINT, "I_mean_minus_2sigma", V, time=0.0)
    I_hi = load_checkpoint_function(POP_STATS_CHECKPOINT, "I_mean_plus_2sigma", V, time=0.0)

    n_owned = _owned_dofs(I_mean)
    i_local = I_mean.x.array[:n_owned]
    i_min = float(comm.allreduce(float(i_local.min()), op=MPI.MIN))
    i_max = float(comm.allreduce(float(i_local.max()), op=MPI.MAX))

    rho_ct_mean = rescale_to_density(I_mean, rho_min=rho_min, rho_max=rho_max, intensity_min=i_min, intensity_max=i_max)
    rho_ct_lo = rescale_to_density(I_lo, rho_min=rho_min, rho_max=rho_max, intensity_min=i_min, intensity_max=i_max)
    rho_ct_hi = rescale_to_density(I_hi, rho_min=rho_min, rho_max=rho_max, intensity_min=i_min, intensity_max=i_max)

    rho_ct_mean.name = "rho_ct_mean"
    rho_ct_lo.name = "rho_ct_mean_minus_2sigma"
    rho_ct_hi.name = "rho_ct_mean_plus_2sigma"
    rho_ct_mean.x.scatter_forward()
    rho_ct_lo.x.scatter_forward()
    rho_ct_hi.x.scatter_forward()
    return rho_ct_mean, rho_ct_lo, rho_ct_hi


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


def _plot_combined_analysis(
    metrics: list[RunMetrics],
    mean_rho_by_ratio: dict[float, tuple[np.ndarray, np.ndarray]],
    ct_mean_rho: float,
    ct_mean_rho_lo: float,
    ct_mean_rho_hi: float,
    out_dir: Path,
) -> None:
    apply_style()

    ratios = np.array([m.psi_ref_ratio for m in metrics], dtype=float)
    rmse_total = np.array([m.rmse_total for m in metrics], dtype=float)
    rmse_total_lo = np.array([m.rmse_total_lo for m in metrics], dtype=float)
    rmse_total_hi = np.array([m.rmse_total_hi for m in metrics], dtype=float)
    rmse_trab = np.array([m.rmse_trab for m in metrics], dtype=float)
    rmse_cort = np.array([m.rmse_cort for m in metrics], dtype=float)

    contrast_model = np.array([m.contrast_model for m in metrics], dtype=float)
    contrast_ct = np.array([m.contrast_ct for m in metrics], dtype=float)
    contrast_ct_lo = np.array([m.contrast_ct_lo for m in metrics], dtype=float)
    contrast_ct_hi = np.array([m.contrast_ct_hi for m in metrics], dtype=float)

    # Use 2 subplots layout
    fig = plt.figure(figsize=(7.5, 3.2))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # 1. RMSE (Spatial Agreement)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ratios, rmse_total, marker="o", label="RMSE (total)", linewidth=PLOT_LINEWIDTH, markersize=PLOT_MARKERSIZE)
    if rmse_total_lo.size and rmse_total_hi.size:
        lo = np.minimum(rmse_total_lo, rmse_total_hi)
        hi = np.maximum(rmse_total_lo, rmse_total_hi)
        ax1.fill_between(ratios, lo, hi, color="C0", alpha=0.15, label=r"RMSE vs CT $\pm 2\sigma$")
    ax1.plot(ratios, rmse_trab, marker="s", label="RMSE (trab mask)", linewidth=PLOT_LINEWIDTH, markersize=PLOT_MARKERSIZE)
    ax1.plot(ratios, rmse_cort, marker="^", label="RMSE (cort mask)", linewidth=PLOT_LINEWIDTH, markersize=PLOT_MARKERSIZE)
    ax1.set_xlabel(r"$r=\psi_{\mathrm{ref}}^{\mathrm{cort}}/\psi_{\mathrm{ref}}^{\mathrm{trab}}$")
    ax1.set_ylabel(r"RMSE in $\rho$ [g/cm$^3$]")
    ax1.set_title("Spatial agreement")
    ax1.legend()

    # 2. Timecourse (Global Density Evolution)
    ax3 = fig.add_subplot(gs[0, 1])
    ratios_sorted = sorted(mean_rho_by_ratio.keys())

    colors = []
    sm = None
    if ratios_sorted:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        vmin = float(min(ratios_sorted))
        vmax = float(max(ratios_sorted))
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
        sm = ScalarMappable(norm=norm, cmap=cmap)
        colors = [sm.to_rgba(float(r)) for r in ratios_sorted]

    if colors:
        for c, r in zip(colors, ratios_sorted):
            t, rho_bar = mean_rho_by_ratio[r]
            ax3.plot(t, rho_bar, color=c, linewidth=PLOT_LINEWIDTH)
    else:
        # Fallback if no ratios (should not happen)
        for r in ratios_sorted:
            t, rho_bar = mean_rho_by_ratio[r]
            ax3.plot(t, rho_bar, linewidth=PLOT_LINEWIDTH, label=f"r={r:g}")

    ax3.axhspan(ct_mean_rho_lo, ct_mean_rho_hi, color="black", alpha=0.12, label=r"CT $\pm 2\sigma$")
    ax3.axhline(ct_mean_rho, color="black", linestyle="--", linewidth=PLOT_LINEWIDTH, label="CT mean")
    ax3.set_xlabel("Time [day]")
    ax3.set_ylabel(r"Mean density $\bar\rho$ [g/cm$^3$]")
    ax3.set_title("Global density evolution")
    
    if sm:
        cbar = fig.colorbar(sm, ax=ax3, aspect=20, pad=0.05)
        cbar.set_label(r"$r$")
    
    ax3.legend(loc="best", fontsize=7)

    plt.tight_layout()
    save_manuscript_figure(fig, "psi_ref_ratio_analysis", dpi=PUBLICATION_DPI, close=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "psi_ref_ratio_analysis.png", dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.05)
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

    # Load cohort mean density (and mean±2σ bounds) once
    ct_func, ct_lo_func, ct_hi_func = _load_ct_density(mesh, rho_min=rho_min, rho_max=rho_max, comm=comm, logger=logger)

    # CT domain-mean density (volume-weighted), used for time-course comparison
    ct_mean_rho_vol = float("nan")
    ct_mean_rho_vol_lo = float("nan")
    ct_mean_rho_vol_hi = float("nan")
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
        ct_int_lo = fem.assemble_scalar(fem.form(ct_lo_func * dx))
        ct_int_lo = float(comm.allreduce(ct_int_lo, op=MPI.SUM))
        ct_int_hi = fem.assemble_scalar(fem.form(ct_hi_func * dx))
        ct_int_hi = float(comm.allreduce(ct_int_hi, op=MPI.SUM))
        ct_mean_rho_vol = float(ct_int / vol) if vol > 0 else float("nan")
        ct_mean_rho_vol_lo = float(ct_int_lo / vol) if vol > 0 else float("nan")
        ct_mean_rho_vol_hi = float(ct_int_hi / vol) if vol > 0 else float("nan")
    except Exception as e:
        if comm.rank == 0:
            logger.warning(f"Failed to compute CT volume-mean density: {e}")

    n_owned = _owned_dofs(ct_func)
    ct_vals_local = ct_func.x.array[:n_owned].copy()
    ct_vals_lo_local = ct_lo_func.x.array[:n_owned].copy()
    ct_vals_hi_local = ct_hi_func.x.array[:n_owned].copy()

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

    mean_ct_trab_lo = _global_mean(ct_vals_lo_local[trab_mask_local], comm) if n_trab else float("nan")
    mean_ct_trab_hi = _global_mean(ct_vals_hi_local[trab_mask_local], comm) if n_trab else float("nan")
    mean_ct_cort_lo = _global_mean(ct_vals_lo_local[cort_mask_local], comm) if n_cort else float("nan")
    mean_ct_cort_hi = _global_mean(ct_vals_hi_local[cort_mask_local], comm) if n_cort else float("nan")
    contrast_ct_lo = mean_ct_cort_lo - mean_ct_trab_lo
    contrast_ct_hi = mean_ct_cort_hi - mean_ct_trab_hi

    # Compute metrics per run
    metrics: list[RunMetrics] = []
    mean_rho_time_by_ratio: dict[float, tuple[np.ndarray, np.ndarray]] = {}

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
        rmse_band_lo, rmse_band_hi = _global_rmse_interval_band(
            rho_vals_local, ct_vals_lo_local, ct_vals_hi_local, comm
        )

        _, _, corr_lo = _global_rmse_mae_corr(rho_vals_local, ct_vals_lo_local, comm)
        _, _, corr_hi = _global_rmse_mae_corr(rho_vals_local, ct_vals_hi_local, comm)
        corr_band_lo = float(min(corr_lo, corr_hi))
        corr_band_hi = float(max(corr_lo, corr_hi))

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
                rmse_total_lo=rmse_band_lo,
                rmse_total_hi=rmse_band_hi,
                corr_total_lo=corr_band_lo,
                corr_total_hi=corr_band_hi,
                rmse_trab=rmse_trab,
                rmse_cort=rmse_cort,
                mean_ct_trab=mean_ct_trab,
                mean_ct_cort=mean_ct_cort,
                mean_model_trab=mean_model_trab,
                mean_model_cort=mean_model_cort,
                contrast_ct=contrast_ct,
                contrast_ct_lo=contrast_ct_lo,
                contrast_ct_hi=contrast_ct_hi,
                contrast_model=contrast_model,
                contrast_error=float(contrast_model - contrast_ct),
                n_total=n_total,
                n_trab=n_trab,
                n_cort=n_cort,
            )
        )

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
        if mean_rho_time_by_ratio:
            _plot_combined_analysis(
                metrics=metrics,
                mean_rho_by_ratio=mean_rho_time_by_ratio,
                ct_mean_rho=ct_mean_rho_vol,
                ct_mean_rho_lo=ct_mean_rho_vol_lo,
                ct_mean_rho_hi=ct_mean_rho_vol_hi,
                out_dir=SWEEP_DIR,
            )

    comm.Barrier()


if __name__ == "__main__":
    main()
