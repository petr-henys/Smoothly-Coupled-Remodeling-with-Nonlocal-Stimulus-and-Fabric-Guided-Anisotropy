"""Density comparison analysis and plotting.

Loads simulation checkpoint and CT image, computes density histograms,
and saves spatial residuum.

Usage:
    mpirun -n 4 python analysis/density_comparison_plot.py
"""

import sys
import json
from pathlib import Path
import numpy as np
import ants
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import basix.ufl
from dolfinx import fem, mesh
from dolfinx.io import VTXWriter

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from morpho_mapper import idw
from simulation.checkpoint import load_checkpoint_mesh, load_checkpoint_function
from simulation.logger import get_logger
from simulation.params import load_default_params

# Configuration
OUTPUT_DIR = project_root / "results/density_comparison"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.bp"
CT_IMAGE_PATH = project_root / "anatomy/raw/proximal_femur/template_new3.nii.gz"
PARAMS_FILE = "stiff_params_femur.json"

# IDW interpolation parameters
IDW_THRESHOLD = 0.2
IDW_POWER = 1
IDW_K_NEIGHBORS = 16
CT_SMOOTHING_SIGMA = 1.5  # Gaussian smoothing sigma in mm (0 = disabled)


def load_from_checkpoint(
    checkpoint_path: Path,
    params_file: str = "stiff_params_femur.json",
) -> tuple[fem.Function, mesh.Mesh, float, float]:
    """Load model density from existing checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint.bp.
        params_file: Parameter file name (for total_time and density bounds).
    
    Returns:
        Tuple of (density function, mesh, rho_min, rho_max).
    """
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="DensityAnalysis")
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Run simulation first."
        )
    
    if comm.rank == 0:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load mesh from checkpoint
    mesh = load_checkpoint_mesh(checkpoint_path, comm)
    
    # Load config for final time and density bounds
    output_dir = checkpoint_path.parent
    config_path = output_dir / "config.json"
    
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        final_time = cfg.get("time", {}).get("total_time", 0.0)
        rho_min = cfg.get("density", {}).get("rho_min", 0.1)
        rho_max = cfg.get("density", {}).get("rho_max", 2.0)
    else:
        # Fallback: read from params file
        params = load_default_params(params_file)
        final_time = params["time"].total_time
        rho_min = params["density"].rho_min
        rho_max = params["density"].rho_max
    
    if comm.rank == 0:
        logger.info(f"Loading density at t={final_time} days")
    
    # Create function space for density (P1 scalar)
    cell_name = mesh.topology.cell_name()
    element = basix.ufl.element("Lagrange", cell_name, 1)
    Q = fem.functionspace(mesh, element)
    
    # Load density from checkpoint
    rho = load_checkpoint_function(checkpoint_path, "rho", Q, time=final_time)
    
    return rho, mesh, rho_min, rho_max


def load_ct_density(
    mesh: mesh.Mesh,
    rho_min: float,
    rho_max: float,
    image_path: str | Path,
    sigma: float = CT_SMOOTHING_SIGMA,
    verbose: bool = True,
) -> fem.Function:
    """Load and interpolate CT density onto mesh, normalized to [rho_min, rho_max]."""
    comm = mesh.comm
    logger = get_logger(comm, name="DensityAnalysis")
    
    if comm.rank == 0 and verbose:
        logger.info(f"Loading CT image: {image_path}")
        logger.info(f"Normalizing to rho_min={rho_min:.4f}, rho_max={rho_max:.4f}")
    
    # Load CT image
    image = ants.image_read(str(image_path))

    # Apply Gaussian smoothing if requested
    if sigma > 0:
        if comm.rank == 0 and verbose:
            logger.info(f"Smoothing CT image with sigma={sigma} mm")
        image = ants.smooth_image(image, sigma)
    
    # Interpolate onto mesh using IDW (CG1 for comparison with model)
    ct_func = idw(
        image, mesh,
        threshold=IDW_THRESHOLD,
        power=IDW_POWER,
        k_neighbors=IDW_K_NEIGHBORS,
        method='nodes',  # CG1 space
        verbose=verbose,
    )
    ct_func.name = "rho_ct"
    
    # Normalize CT values to [rho_min, rho_max] from config
    local_vals = ct_func.x.array.copy()
    
    # Global min/max for consistent normalization across MPI ranks
    local_min = local_vals.min() if local_vals.size > 0 else np.inf
    local_max = local_vals.max() if local_vals.size > 0 else -np.inf
    
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    
    if comm.rank == 0 and verbose:
        logger.info(f"CT raw range: [{global_min:.4f}, {global_max:.4f}]")
    
    # Linear scaling from [global_min, global_max] to [rho_min, rho_max]
    if global_max - global_min > 1e-10:
        normalized = (local_vals - global_min) / (global_max - global_min)
        ct_func.x.array[:] = rho_min + normalized * (rho_max - rho_min)
    else:
        # Constant CT image - use midpoint
        ct_func.x.array[:] = (rho_min + rho_max) / 2
    
    ct_func.x.scatter_forward()
    
    if comm.rank == 0 and verbose:
        # Verify actual range after normalization
        actual_min = comm.allreduce(ct_func.x.array.min(), op=MPI.MIN)
        actual_max = comm.allreduce(ct_func.x.array.max(), op=MPI.MAX)
        logger.info(f"CT normalized range: [{actual_min:.4f}, {actual_max:.4f}]")
    
    return ct_func


def compute_histogram_data(
    model_rho: fem.Function,
    ct_rho: fem.Function,
    n_bins: int = 50,
    rho_range: tuple[float, float] | None = None,
) -> dict:
    """Compute histogram data for model and CT densities."""
    comm = model_rho.function_space.mesh.comm
    
    # Get local values (owned DOFs only to avoid double-counting)
    dofmap = model_rho.function_space.dofmap
    n_owned = dofmap.index_map.size_local * dofmap.index_map_bs
    
    model_vals = model_rho.x.array[:n_owned].copy()
    ct_vals = ct_rho.x.array[:n_owned].copy()
    
    # Determine histogram range
    if rho_range is None:
        local_min = min(model_vals.min(), ct_vals.min()) if len(model_vals) > 0 else np.inf
        local_max = max(model_vals.max(), ct_vals.max()) if len(model_vals) > 0 else -np.inf
        
        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)
        rho_range = (global_min, global_max)
    
    # Compute local histograms
    bin_edges = np.linspace(rho_range[0], rho_range[1], n_bins + 1)
    
    model_hist, _ = np.histogram(model_vals, bins=bin_edges)
    ct_hist, _ = np.histogram(ct_vals, bins=bin_edges)
    
    # Sum histograms across all ranks
    global_model_hist = np.zeros_like(model_hist)
    global_ct_hist = np.zeros_like(ct_hist)
    
    comm.Allreduce(model_hist, global_model_hist, op=MPI.SUM)
    comm.Allreduce(ct_hist, global_ct_hist, op=MPI.SUM)
    
    # Compute statistics
    local_model_sum = model_vals.sum()
    local_model_sq = (model_vals ** 2).sum()
    local_ct_sum = ct_vals.sum()
    local_ct_sq = (ct_vals ** 2).sum()
    local_n = len(model_vals)
    
    global_model_sum = comm.allreduce(local_model_sum, op=MPI.SUM)
    global_model_sq = comm.allreduce(local_model_sq, op=MPI.SUM)
    global_ct_sum = comm.allreduce(local_ct_sum, op=MPI.SUM)
    global_ct_sq = comm.allreduce(local_ct_sq, op=MPI.SUM)
    global_n = comm.allreduce(local_n, op=MPI.SUM)
    
    model_mean = global_model_sum / global_n if global_n > 0 else 0.0
    model_std = np.sqrt(global_model_sq / global_n - model_mean ** 2) if global_n > 0 else 0.0
    ct_mean = global_ct_sum / global_n if global_n > 0 else 0.0
    ct_std = np.sqrt(global_ct_sq / global_n - ct_mean ** 2) if global_n > 0 else 0.0
    
    # Residuum statistics
    residuum = model_vals - ct_vals
    local_res_sum = residuum.sum()
    local_res_sq = (residuum ** 2).sum()
    local_res_abs = np.abs(residuum).sum()
    
    global_res_sum = comm.allreduce(local_res_sum, op=MPI.SUM)
    global_res_sq = comm.allreduce(local_res_sq, op=MPI.SUM)
    global_res_abs = comm.allreduce(local_res_abs, op=MPI.SUM)
    
    res_mean = global_res_sum / global_n if global_n > 0 else 0.0
    res_rmse = np.sqrt(global_res_sq / global_n) if global_n > 0 else 0.0
    res_mae = global_res_abs / global_n if global_n > 0 else 0.0
    
    return {
        "bin_edges": bin_edges.tolist(),
        "bin_centers": ((bin_edges[:-1] + bin_edges[1:]) / 2).tolist(),
        "model_hist": global_model_hist.tolist(),
        "ct_hist": global_ct_hist.tolist(),
        "model_mean": float(model_mean),
        "model_std": float(model_std),
        "ct_mean": float(ct_mean),
        "ct_std": float(ct_std),
        "residuum_mean": float(res_mean),
        "residuum_rmse": float(res_rmse),
        "residuum_mae": float(res_mae),
        "n_dofs": int(global_n),
    }


def plot_histograms(
    hist_data: dict,
    output_dir: Path,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """Plot histogram comparison (rank 0 only)."""
    bin_centers = np.array(hist_data["bin_centers"])
    model_hist = np.array(hist_data["model_hist"])
    ct_hist = np.array(hist_data["ct_hist"])
    
    # Normalize to density (area = 1)
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    model_density = model_hist / (model_hist.sum() * bin_width) if model_hist.sum() > 0 else model_hist
    ct_density = ct_hist / (ct_hist.sum() * bin_width) if ct_hist.sum() > 0 else ct_hist
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Histograms
    ax1 = axes[0]
    ax1.fill_between(bin_centers, model_density, alpha=0.5, label="Model", color="C0", step="mid")
    ax1.fill_between(bin_centers, ct_density, alpha=0.5, label="CT (normalized)", color="C1", step="mid")
    ax1.set_xlabel(r"Density $\rho$ [g/cm³]")
    ax1.set_ylabel("Probability density")
    ax1.set_title("Density Distribution Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f"Model: μ={hist_data['model_mean']:.3f}, σ={hist_data['model_std']:.3f}\n"
        f"CT:    μ={hist_data['ct_mean']:.3f}, σ={hist_data['ct_std']:.3f}\n"
        f"RMSE:  {hist_data['residuum_rmse']:.4f}\n"
        f"MAE:   {hist_data['residuum_mae']:.4f}"
    )
    ax1.text(
        0.02, 0.98, stats_text,
        transform=ax1.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    
    # Right: Q-Q plot (quantile-quantile)
    ax2 = axes[1]
    
    # Compute quantiles from histograms
    model_cdf = np.cumsum(model_hist) / model_hist.sum() if model_hist.sum() > 0 else np.zeros_like(model_hist)
    ct_cdf = np.cumsum(ct_hist) / ct_hist.sum() if ct_hist.sum() > 0 else np.zeros_like(ct_hist)
    
    # Sample quantiles at percentiles
    percentiles = np.linspace(0.01, 0.99, 50)
    model_quantiles = np.interp(percentiles, model_cdf, bin_centers)
    ct_quantiles = np.interp(percentiles, ct_cdf, bin_centers)
    
    ax2.scatter(ct_quantiles, model_quantiles, alpha=0.7, s=20)
    
    # Diagonal reference line
    lims = [
        min(bin_centers.min(), bin_centers.min()),
        max(bin_centers.max(), bin_centers.max()),
    ]
    ax2.plot(lims, lims, "k--", alpha=0.5, label="Perfect agreement")
    
    ax2.set_xlabel(r"CT quantiles $\rho$ [g/cm³]")
    ax2.set_ylabel(r"Model quantiles $\rho$ [g/cm³]")
    ax2.set_title("Q-Q Plot")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig(output_dir / "density_histograms.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "density_histograms.pdf", bbox_inches="tight")
    plt.close(fig)


def compute_and_save_residuum(
    model_rho: fem.Function,
    ct_rho: fem.Function,
    output_dir: Path,
    comm: MPI.Comm,
) -> fem.Function:
    """Compute spatial residuum and save to VTX."""
    # Create residuum function in same space
    residuum = fem.Function(model_rho.function_space, name="residuum")
    residuum.x.array[:] = model_rho.x.array - ct_rho.x.array
    residuum.x.scatter_forward()
    
    # Also compute absolute error
    abs_error = fem.Function(model_rho.function_space, name="abs_error")
    abs_error.x.array[:] = np.abs(residuum.x.array)
    abs_error.x.scatter_forward()
    
    # Relative error (avoid division by zero)
    rel_error = fem.Function(model_rho.function_space, name="rel_error")
    ct_safe = np.maximum(np.abs(ct_rho.x.array), 1e-10)
    rel_error.x.array[:] = np.abs(residuum.x.array) / ct_safe
    rel_error.x.scatter_forward()
    
    # Save to VTX
    vtx_path = output_dir / "residuum.bp"
    with VTXWriter(comm, str(vtx_path), [residuum, abs_error, rel_error, ct_rho], engine="bp4") as writer:
        writer.write(0.0)
    
    return residuum


def main() -> None:
    """Run density comparison analysis."""
    comm = MPI.COMM_WORLD
    logger = get_logger(comm, name="DensityAnalysis")
    
    if comm.rank == 0:
        logger.info("=" * 60)
        logger.info("DENSITY COMPARISON ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
        logger.info(f"CT image:   {CT_IMAGE_PATH}")
        logger.info(f"Output:     {OUTPUT_DIR}")
    
    # Step 1: Load model density from checkpoint
    model_rho, mesh, rho_min, rho_max = load_from_checkpoint(CHECKPOINT_PATH, PARAMS_FILE)
    
    # Step 2: Load CT density
    ct_rho = load_ct_density(mesh, rho_min, rho_max, CT_IMAGE_PATH)
    
    # Step 3: Compute histogram data
    if comm.rank == 0:
        logger.info("Computing histogram statistics...")
    
    hist_data = compute_histogram_data(model_rho, ct_rho, n_bins=50)
    
    # Step 4: Plot histograms (rank 0 only)
    if comm.rank == 0:
        logger.info("Plotting histograms...")
        plot_histograms(hist_data, OUTPUT_DIR)
        
        # Save statistics to JSON
        stats_path = OUTPUT_DIR / "density_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(hist_data, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")
    
    # Ensure plotting is done before VTX write
    comm.Barrier()
    
    # Step 5: Compute and save spatial residuum
    if comm.rank == 0:
        logger.info("Computing and saving spatial residuum...")
    
    compute_and_save_residuum(model_rho, ct_rho, OUTPUT_DIR, comm)
    
    if comm.rank == 0:
        logger.info(f"Residuum saved to {OUTPUT_DIR}/residuum.bp")
        logger.info("=" * 60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
