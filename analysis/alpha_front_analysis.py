"""Alpha_front sweep analysis for femur benchmark.

This script post-processes the alpha_front sweep (hip contact force frontal-plane
angle offset applied to all loading cases) and quantifies how the final density
field changes with the assumed hip force direction.

It compares each run against the CT-derived population-template density mapped
onto the same FE mesh, and also reports a medial–lateral "center-of-density"
shift in the femur coordinate system.

Outputs (rank 0):
    results/alpha_front_sweep/
    ├── alpha_front_metrics.csv
    ├── alpha_front_metrics.json
    ├── alpha_front_summary.png
    └── alpha_front_sensitivity.bp        (ParaView: d rho / d alpha_front)

Additionally, manuscript-ready figures are saved to:
    manuscript/images/

Usage:
    mpirun -n 4 python analysis/alpha_front_analysis.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from dolfinx import fem
import numpy as np
from mpi4py import MPI

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from morpho_mapper import rescale_to_density
from femur.css import FemurCSS, load_json_points
from femur.paths import FemurPaths
from simulation.checkpoint import load_checkpoint_mesh, load_checkpoint_function
from simulation.logger import get_logger
from simulation.params import load_default_params

from analysis.plot_utils import (
    PLOT_LINEWIDTH,
    PLOT_MARKERSIZE,
    REFERENCE_ALPHA,
    REFERENCE_COLOR,
    REFERENCE_LINESTYLE,
    REFERENCE_LINEWIDTH,
    COLORS as TOL_COLORS,
    apply_style,
    PUBLICATION_DPI,
    save_manuscript_figure,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SWEEP_DIR = project_root / "results/alpha_front_sweep"
SUMMARY_JSON = SWEEP_DIR / "sweep_summary.json"

POP_STATS_CHECKPOINT = project_root / "results/ct_density/population_stats_checkpoint.bp"
PARAMS_FILE = "stiff_params_femur.json"

# Cohort stats are precomputed by morpho_mapper.py; no CT template used here.


@dataclass(frozen=True)
class RunMetrics:
    alpha_front_offset_deg: float
    run_hash: str

    rmse_total: float
    mae_total: float
    corr_total: float

    rmse_total_lo: float
    rmse_total_hi: float
    corr_total_lo: float
    corr_total_hi: float

    # Medial–lateral center-of-density (CSS z coordinate, mass-weighted) [mm]
    zbar_rho: float
    zbar_ct: float
    zbar_diff: float

    zbar_diff_lo: float
    zbar_diff_hi: float

    # Medial–lateral spread (mass-weighted std of CSS z) [mm]
    zstd_rho: float
    zstd_ct: float
    zstd_diff: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha_front_offset_deg": self.alpha_front_offset_deg,
            "run_hash": self.run_hash,
            "rmse_total": self.rmse_total,
            "mae_total": self.mae_total,
            "corr_total": self.corr_total,
            "rmse_total_lo": self.rmse_total_lo,
            "rmse_total_hi": self.rmse_total_hi,
            "corr_total_lo": self.corr_total_lo,
            "corr_total_hi": self.corr_total_hi,
            "zbar_rho_mm": self.zbar_rho,
            "zbar_ct_mm": self.zbar_ct,
            "zbar_diff_mm": self.zbar_diff,
            "zbar_diff_lo_mm": self.zbar_diff_lo,
            "zbar_diff_hi_mm": self.zbar_diff_hi,
            "zstd_rho_mm": self.zstd_rho,
            "zstd_ct_mm": self.zstd_ct,
            "zstd_diff_mm": self.zstd_diff,
        }


def _owned_dofs(u) -> int:
    dofmap = u.function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def _global_stats_xy(
    x_local: np.ndarray,
    y_local: np.ndarray,
    comm: MPI.Comm,
) -> tuple[int, float, float, float, float, float]:
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


def _global_rmse_mae_corr(
    x_local: np.ndarray,
    y_local: np.ndarray,
    comm: MPI.Comm,
) -> tuple[float, float, float]:
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
    """RMSE envelope over pointwise interval reference [y_lo, y_hi].

    Returns (rmse_min, rmse_max) where for each DOF i the reference y_i can be
    chosen anywhere in [lo_i, hi_i].

    This bounds RMSE(x vs y_mean) as long as y_mean is pointwise within [lo, hi].
    """
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


def _load_sweep_runs(comm: MPI.Comm) -> list[dict[str, Any]]:
    if comm.rank == 0:
        if not SUMMARY_JSON.exists():
            raise FileNotFoundError(f"Sweep summary not found: {SUMMARY_JSON}")
        with open(SUMMARY_JSON, "r") as f:
            summary = json.load(f)
        runs = summary.get("runs", [])
        runs_sorted = sorted(runs, key=lambda r: float(r["alpha_front_offset"]))
    else:
        runs_sorted = None
    return comm.bcast(runs_sorted, root=0)


def _load_ct_density(mesh, rho_min: float, rho_max: float, comm: MPI.Comm, logger) -> tuple[Any, Any, Any]:
    if not POP_STATS_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Population stats checkpoint not found: {POP_STATS_CHECKPOINT}. "
            "Run morpho_mapper.py to generate cohort stats first."
        )

    if comm.rank == 0:
        logger.info(f"Loading cohort mean intensity from {POP_STATS_CHECKPOINT}")

    V = fem.functionspace(mesh, ("Lagrange", 1))
    I_mean = load_checkpoint_function(POP_STATS_CHECKPOINT, "I_mean", V, time=0.0)
    I_lo = load_checkpoint_function(POP_STATS_CHECKPOINT, "I_mean_minus_2sigma", V, time=0.0)
    I_hi = load_checkpoint_function(POP_STATS_CHECKPOINT, "I_mean_plus_2sigma", V, time=0.0)

    vals = I_mean.x.array
    local_min = float(vals.min()) if vals.size else float("inf")
    local_max = float(vals.max()) if vals.size else float("-inf")
    i_min = comm.allreduce(local_min, op=MPI.MIN)
    i_max = comm.allreduce(local_max, op=MPI.MAX)

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


def _compute_css_transform(comm: MPI.Comm, logger) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, t) for world→CSS: x_css = R x + t (with t=-R fhc)."""
    if comm.rank == 0:
        import pyvista as pv

        femur_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
        le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)

        css = FemurCSS(femur_mesh, head_line, le_me_line, side="left")
        R = np.vstack([css.axes[a] for a in ("x", "y", "z")]).astype(np.float64)
        t = (-R @ css.fhc).astype(np.float64)

        logger.info("Computed Femur CSS transform (world→CSS) from landmarks.")
    else:
        R = None
        t = None
    R = comm.bcast(R, root=0)
    t = comm.bcast(t, root=0)
    return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _assemble_scalar(form, comm: MPI.Comm) -> float:
    from dolfinx import fem

    local = fem.assemble_scalar(form)
    return float(comm.allreduce(local, op=MPI.SUM))


def _mass_moments_z(rho, z_css, dx, comm: MPI.Comm) -> tuple[float, float, float]:
    """Return (mass, mean_z, std_z) using mass-weighted moments."""
    from dolfinx import fem

    m = _assemble_scalar(fem.form(rho * dx), comm)
    mz = _assemble_scalar(fem.form(rho * z_css * dx), comm)
    mz2 = _assemble_scalar(fem.form(rho * (z_css * z_css) * dx), comm)
    if m <= 0.0:
        return m, float("nan"), float("nan")
    zbar = mz / m
    ez2 = mz2 / m
    var = max(0.0, ez2 - zbar * zbar)
    return m, float(zbar), float(np.sqrt(var))


def _plot_analysis_summary(
    metrics: list[RunMetrics],
    mean_rho_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]],
    ct_mean_rho: float,
    ct_mean_rho_lo: float,
    ct_mean_rho_hi: float,
    out_dir: Path,
) -> None:
    apply_style()

    a = np.array([m.alpha_front_offset_deg for m in metrics], dtype=float)
    rmse = np.array([m.rmse_total for m in metrics], dtype=float)
    rmse_lo = np.array([m.rmse_total_lo for m in metrics], dtype=float)
    rmse_hi = np.array([m.rmse_total_hi for m in metrics], dtype=float)
    corr = np.array([m.corr_total for m in metrics], dtype=float)
    corr_lo = np.array([m.corr_total_lo for m in metrics], dtype=float)
    corr_hi = np.array([m.corr_total_hi for m in metrics], dtype=float)
    zdiff = np.array([m.zbar_diff for m in metrics], dtype=float)
    zdiff_lo = np.array([m.zbar_diff_lo for m in metrics], dtype=float)
    zdiff_hi = np.array([m.zbar_diff_hi for m in metrics], dtype=float)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10.0, 2.5), constrained_layout=True)

    # (a) CT agreement (RMSE)
    ax1.fill_between(a, rmse_lo, rmse_hi, color=TOL_COLORS["orange"], alpha=0.20, linewidth=0.0)
    ax1.plot(
        a,
        rmse,
        color=TOL_COLORS["orange"],
        marker="s",
        linewidth=PLOT_LINEWIDTH,
        markersize=PLOT_MARKERSIZE,
    )
    ax1.axvline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        alpha=REFERENCE_ALPHA,
        zorder=0,
    )
    ax1.set_xlabel(r"$\Delta\alpha_{\mathrm{front}}$ [deg]")
    ax1.set_ylabel(r"RMSE in $\rho$ [g/cm$^3$]")
    ax1.set_title(r"(a) CT agreement")

    # (b) Pattern agreement (correlation)
    ax2.fill_between(a, corr_lo, corr_hi, color=TOL_COLORS["blue"], alpha=0.20, linewidth=0.0)
    ax2.plot(
        a,
        corr,
        color=TOL_COLORS["blue"],
        marker="o",
        linewidth=PLOT_LINEWIDTH,
        markersize=PLOT_MARKERSIZE,
    )
    ax2.axvline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        alpha=REFERENCE_ALPHA,
        zorder=0,
    )
    ax2.set_xlabel(r"$\Delta\alpha_{\mathrm{front}}$ [deg]")
    ax2.set_ylabel("Pearson corr")
    ax2.set_title(r"(b) Pattern agreement")
    
    # (c) Medial–lateral redistribution (center-of-density shift)
    ax3.fill_between(a, zdiff_lo, zdiff_hi, color=TOL_COLORS["teal"], alpha=0.20, linewidth=0.0)
    ax3.plot(
        a,
        zdiff,
        color=TOL_COLORS["teal"],
        marker="^",
        linewidth=PLOT_LINEWIDTH,
        markersize=PLOT_MARKERSIZE,
    )
    ax3.axhline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        alpha=REFERENCE_ALPHA,
        zorder=0,
    )
    ax3.axvline(
        0.0,
        color=REFERENCE_COLOR,
        linestyle=REFERENCE_LINESTYLE,
        linewidth=REFERENCE_LINEWIDTH,
        alpha=REFERENCE_ALPHA,
        zorder=0,
    )
    ax3.set_xlabel(r"$\Delta\alpha_{\mathrm{front}}$ [deg]")
    ax3.set_ylabel(r"$\bar z_\rho - \bar z_{\mathrm{CT}}$ [mm]")
    ax3.set_title(r"(c) Medial shift")
    
    # (d) Global density evolution
    alphas = sorted(mean_rho_by_alpha.keys())
    colors = []
    if alphas:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize, TwoSlopeNorm

        vmin = float(min(alphas))
        vmax = float(max(alphas))
        if vmin < 0.0 < vmax:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            cmap = plt.cm.RdBu_r
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis
        sm = ScalarMappable(norm=norm, cmap=cmap)
        colors = [sm.to_rgba(float(a_val)) for a_val in alphas]

    for c, a_val in zip(colors, alphas):
        t, rho_bar = mean_rho_by_alpha[a_val]
        ax4.plot(t, rho_bar, color=c, linewidth=PLOT_LINEWIDTH)

    ax4.axhspan(ct_mean_rho_lo, ct_mean_rho_hi, color="black", alpha=0.12, label=r"CT mean $\pm 2\sigma$")
    ax4.axhline(ct_mean_rho, color="black", linestyle="--", linewidth=REFERENCE_LINEWIDTH, label="CT mean")
    ax4.set_xlabel("Time [day]")
    ax4.set_ylabel(r"Mean density $\bar\rho$ [g/cm$^3$]")
    ax4.set_title(r"(d) Global density evolution")

    if alphas:
        cbar = fig.colorbar(sm, ax=ax4, aspect=20, pad=0.05)
        cbar.set_label(r"$\Delta\alpha_{\mathrm{front}}$ [deg]")

    ax4.legend(loc="lower left")

    save_manuscript_figure(fig, "alpha_front_summary", dpi=PUBLICATION_DPI, close=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "alpha_front_summary.png", dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _write_sensitivity_field(
    V,
    alphas: np.ndarray,
    sum_rho: np.ndarray,
    sum_alpha_rho: np.ndarray,
    out_path: Path,
    comm: MPI.Comm,
) -> None:
    """Write d rho / d alpha (linear regression slope) to VTX for ParaView."""
    from dolfinx import fem, io

    n = float(alphas.size)
    if n < 2:
        return
    mean_a = float(alphas.mean())
    var_a = float((alphas * alphas).sum() - n * mean_a * mean_a)
    if var_a <= 0.0:
        return

    # cov(a, rho_j) = sum(a*rho) - mean(a) * sum(rho)
    slope_owned = (sum_alpha_rho - mean_a * sum_rho) / var_a

    drho = fem.Function(V, name="drho_dalpha_front")
    n_owned = _owned_dofs(drho)
    drho.x.array[:] = 0.0
    drho.x.array[:n_owned] = slope_owned
    drho.x.scatter_forward()

    if comm.rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    with io.VTXWriter(comm, str(out_path), [drho], engine="BP4") as vtx:
        vtx.write(0.0)


def main() -> None:
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="AlphaFrontAnalysis")

    runs = _load_sweep_runs(comm)
    if not runs:
        if comm.rank == 0:
            logger.error("No runs found in sweep summary.")
        return

    params = load_default_params(PARAMS_FILE)
    rho_min = float(params["density"].rho_min)
    rho_max = float(params["density"].rho_max)

    # Load mesh from first checkpoint
    first_ckpt = SWEEP_DIR / runs[0]["output_dir"] / "checkpoint.bp"
    if not first_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {first_ckpt}")

    mesh = load_checkpoint_mesh(first_ckpt, comm)
    from dolfinx import fem
    import ufl

    V = fem.functionspace(mesh, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=mesh)

    # CSS transform for medial axis metrics
    R, t = _compute_css_transform(comm, logger)
    x = ufl.SpatialCoordinate(mesh)
    z_css = float(R[2, 0]) * x[0] + float(R[2, 1]) * x[1] + float(R[2, 2]) * x[2] + float(t[2])

    # Cohort mean density (and mean±2σ bounds) on mesh
    ct, ct_lo, ct_hi = _load_ct_density(mesh, rho_min=rho_min, rho_max=rho_max, comm=comm, logger=logger)

    # Cohort global mean density (volume-weighted)
    vol = _assemble_scalar(fem.form(1.0 * dx), comm)
    ct_int = _assemble_scalar(fem.form(ct * dx), comm)
    ct_int_lo = _assemble_scalar(fem.form(ct_lo * dx), comm)
    ct_int_hi = _assemble_scalar(fem.form(ct_hi * dx), comm)
    ct_mean_rho = float(ct_int / vol) if vol > 0 else float("nan")
    ct_mean_rho_lo = float(ct_int_lo / vol) if vol > 0 else float("nan")
    ct_mean_rho_hi = float(ct_int_hi / vol) if vol > 0 else float("nan")

    # Cohort medial moments
    _, zbar_ct, zstd_ct = _mass_moments_z(ct, z_css, dx, comm)
    _, zbar_ct_lo, _ = _mass_moments_z(ct_lo, z_css, dx, comm)
    _, zbar_ct_hi, _ = _mass_moments_z(ct_hi, z_css, dx, comm)

    # Owned DOFs for pointwise CT metrics
    n_owned = _owned_dofs(ct)
    ct_vals_local = ct.x.array[:n_owned].copy()
    ct_vals_lo_local = ct_lo.x.array[:n_owned].copy()
    ct_vals_hi_local = ct_hi.x.array[:n_owned].copy()

    # Accumulators for sensitivity field (owned DOFs only)
    alphas: list[float] = []
    sum_rho = np.zeros_like(ct_vals_local)
    sum_alpha_rho = np.zeros_like(ct_vals_local)

    metrics: list[RunMetrics] = []
    mean_rho_time_by_alpha: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for run in runs:
        alpha = float(run["alpha_front_offset"])
        run_hash = str(run["output_dir"])
        ckpt = SWEEP_DIR / run_hash / "checkpoint.bp"
        if not ckpt.exists():
            if comm.rank == 0:
                logger.warning(f"Missing checkpoint for alpha={alpha:g}: {ckpt}")
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
        _, zbar_rho, zstd_rho = _mass_moments_z(rho, z_css, dx, comm)

        zdiff_lo = float(zbar_rho - zbar_ct_lo)
        zdiff_hi = float(zbar_rho - zbar_ct_hi)
        zdiff_band_lo = float(min(zdiff_lo, zdiff_hi))
        zdiff_band_hi = float(max(zdiff_lo, zdiff_hi))

        metrics.append(
            RunMetrics(
                alpha_front_offset_deg=alpha,
                run_hash=run_hash,
                rmse_total=rmse_total,
                mae_total=mae_total,
                corr_total=corr_total,
                rmse_total_lo=rmse_band_lo,
                rmse_total_hi=rmse_band_hi,
                corr_total_lo=corr_band_lo,
                corr_total_hi=corr_band_hi,
                zbar_rho=zbar_rho,
                zbar_ct=zbar_ct,
                zbar_diff=float(zbar_rho - zbar_ct),
                zbar_diff_lo=zdiff_band_lo,
                zbar_diff_hi=zdiff_band_hi,
                zstd_rho=zstd_rho,
                zstd_ct=zstd_ct,
                zstd_diff=float(zstd_rho - zstd_ct),
            )
        )

        alphas.append(alpha)
        sum_rho += rho_vals_local
        sum_alpha_rho += alpha * rho_vals_local

        # Time course (rank 0 only)
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
                        t_days = float(row["time_days"])
                        mass_g = float(row["total_mass_g"])
                        if vol <= 0.0:
                            continue
                        rho_mean = mass_g / (vol * 1e-3)  # g/cm^3
                        times.append(t_days)
                        rho_bar.append(rho_mean)
                if times:
                    mean_rho_time_by_alpha[alpha] = (np.array(times), np.array(rho_bar))

        if comm.rank == 0:
            logger.info(
                f"alpha={alpha:6.2f}°  RMSE={rmse_total:.4f}  corr={corr_total:.3f}  "
                f"zbar_diff={zbar_rho - zbar_ct:+.2f} mm"
            )

    metrics = sorted(metrics, key=lambda m: m.alpha_front_offset_deg)

    # Rank-0 outputs
    if comm.rank == 0 and metrics:
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        metrics_csv = SWEEP_DIR / "alpha_front_metrics.csv"
        metrics_json = SWEEP_DIR / "alpha_front_metrics.json"

        import csv

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].to_dict().keys()))
            writer.writeheader()
            for m in metrics:
                writer.writerow(m.to_dict())
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in metrics], f, indent=2)
        logger.info(f"Wrote metrics: {metrics_csv}")

        _plot_analysis_summary(
            metrics,
            mean_rho_time_by_alpha,
            ct_mean_rho=ct_mean_rho,
            ct_mean_rho_lo=ct_mean_rho_lo,
            ct_mean_rho_hi=ct_mean_rho_hi,
            out_dir=SWEEP_DIR,
        )

    # Sensitivity field (all ranks; uses owned-DOF sums)
    alphas_arr = np.array(alphas, dtype=float)
    if alphas_arr.size >= 2:
        out_path = SWEEP_DIR / "alpha_front_sensitivity.bp"
        _write_sensitivity_field(V, alphas_arr, sum_rho, sum_alpha_rho, out_path, comm)
        if comm.rank == 0:
            logger.info(f"Wrote sensitivity field: {out_path} (drho_dalpha_front)")

    comm.Barrier()


if __name__ == "__main__":
    main()
