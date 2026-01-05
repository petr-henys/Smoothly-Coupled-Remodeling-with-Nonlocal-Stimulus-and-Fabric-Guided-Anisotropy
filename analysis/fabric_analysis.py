"""Fabric parameter sweep analysis: compute anisotropy metrics and generate visualizations.

Loads checkpoints from fabric sweep results, computes fabric-related metrics
(anisotropy index, eigenvalue ratios), and visualizes parameter sensitivity.

Requires adios4dolfinx: pip install adios4dolfinx

Usage:
    mpirun -n 1 python analysis/fabric_analysis.py
    
Inputs:
    results/fabric_sweep/
    ├── sweep_summary.csv
    ├── sweep_summary.json
    ├── <hash1>/
    │   ├── config.json
    │   ├── checkpoint.bp/   <- contains L, sigma, rho, S, psi
    │   └── steps.csv
    ...

Outputs:
    results/fabric_sweep/
    ├── fabric_metrics.csv
    ├── fabric_diagnostic.png
    └── (manuscript/images/fabric_diagnostic.png)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from mpi4py import MPI


# =============================================================================
# Imports from plot_utils (CMAME style)
# =============================================================================

from analysis.plot_utils import (
    apply_style,
    setup_axis_style,
    save_figure,
    save_manuscript_figure,
    PUBLICATION_DPI,
    PLOT_LINEWIDTH,
    PLOT_MARKERSIZE,
    FIELD_COLORS,
    FIGSIZE_FULL_WIDTH,
)


# =============================================================================
# Fabric Parameter Styling (consistent with FIELD_COLORS)
# =============================================================================

FABRIC_PARAM_COLORS = {
    "fabric_tau": "#CC78BC",     # Purple (fabric)
    "fabric_gammaF": "#029E73",  # Green
}
FABRIC_PARAM_LABELS = {
    "fabric_tau": r"$\tau_A$ (time constant)",
    "fabric_gammaF": r"$\gamma_F$ (exponent)",
}
FABRIC_PARAM_MARKERS = {
    "fabric_tau": "o",
    "fabric_gammaF": "^",
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FabricMetrics:
    """Fabric analysis metrics computed from checkpoint data.
    
    Metrics characterizing fabric tensor L:
    - anisotropy_index: Measures deviation from isotropy (0=isotropic, 1=max anisotropic)
    - eigenvalue_ratio: max(m_i) / min(m_i) where m_i = exp(l_i)
    - fabric_magnitude: ||L||_F (Frobenius norm)
    - qbar_anisotropy: Anisotropy of Qbar (directional signal strength)
    """
    # Run identification
    run_hash: str
    output_dir: str
    
    # Fabric parameters
    fabric_tau: float
    fabric_gammaF: float
    
    # Anisotropy metrics (volume-averaged)
    anisotropy_index_mean: float    # Mean over domain
    anisotropy_index_std: float     # Std dev
    anisotropy_index_max: float     # Max value
    
    # Eigenvalue ratio metrics
    eigenvalue_ratio_mean: float    # Mean m_max / m_min
    eigenvalue_ratio_std: float
    eigenvalue_ratio_max: float
    
    # Fabric magnitude
    fabric_magnitude_mean: float
    fabric_magnitude_std: float
    
    # Qbar anisotropy (directional signal strength)
    qbar_anisotropy_mean: float     # If ~0, eigenvectors are unstable
    qbar_anisotropy_std: float
    
    # Field statistics
    rho_mean: float
    rho_std: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "run_hash": self.run_hash,
            "output_dir": self.output_dir,
            "fabric_tau": self.fabric_tau,
            "fabric_gammaF": self.fabric_gammaF,
            "anisotropy_index_mean": self.anisotropy_index_mean,
            "anisotropy_index_std": self.anisotropy_index_std,
            "anisotropy_index_max": self.anisotropy_index_max,
            "eigenvalue_ratio_mean": self.eigenvalue_ratio_mean,
            "eigenvalue_ratio_std": self.eigenvalue_ratio_std,
            "eigenvalue_ratio_max": self.eigenvalue_ratio_max,
            "fabric_magnitude_mean": self.fabric_magnitude_mean,
            "fabric_magnitude_std": self.fabric_magnitude_std,
            "qbar_anisotropy_mean": self.qbar_anisotropy_mean,
            "qbar_anisotropy_std": self.qbar_anisotropy_std,
            "rho_mean": self.rho_mean,
            "rho_std": self.rho_std,
        }


# =============================================================================
# Sweep Records Loading
# =============================================================================

def load_fabric_sweep_records(
    base_dir: Path,
    comm,
) -> list[dict[str, Any]]:
    """Load fabric sweep CSV records (MPI-aware).
    
    Args:
        base_dir: Path to sweep output directory.
        comm: MPI communicator.
    
    Returns:
        List of sweep records sorted by fabric parameters.
    """
    csv_file = base_dir / "sweep_summary.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Sweep summary not found: {csv_file}")
    
    if comm.rank == 0:
        df_sweep = pd.read_csv(csv_file)
        # Sort by fabric parameters
        sort_cols = []
        for col in ["fabric.fabric_tau", "fabric.fabric_gammaF"]:
            if col in df_sweep.columns:
                sort_cols.append(col)
        if sort_cols:
            df_sorted = df_sweep.sort_values(sort_cols)
        else:
            df_sorted = df_sweep
        records = df_sorted.to_dict("records")
    else:
        records = None
    
    records = comm.bcast(records, root=0)
    return records


# =============================================================================
# Checkpoint Loading
# =============================================================================

def create_tensor_space(domain, gdim: int = 3):
    """Create P1 tensor function space for fabric L."""
    import basix.ufl
    from dolfinx import fem
    
    cell_name = domain.topology.cell_name()
    element = basix.ufl.element("Lagrange", cell_name, 1, shape=(gdim, gdim))
    return fem.functionspace(domain, element)


def create_dg0_tensor_space(domain, gdim: int = 3):
    """Create DG0 tensor function space for Qbar (cellwise)."""
    import basix.ufl
    from dolfinx import fem
    
    cell_name = domain.topology.cell_name()
    element = basix.ufl.element("DG", cell_name, 0, shape=(gdim, gdim))
    return fem.functionspace(domain, element)


def create_scalar_space(domain):
    """Create P1 scalar function space."""
    import basix.ufl
    from dolfinx import fem
    
    cell_name = domain.topology.cell_name()
    element = basix.ufl.element("Lagrange", cell_name, 1)
    return fem.functionspace(domain, element)


def load_fields_from_checkpoint(
    run_dir: Path,
    comm,
    final_time: float | None = None,
) -> tuple:
    """Load L, Qbar, rho from checkpoint.
    
    Args:
        run_dir: Path to simulation output directory.
        comm: MPI communicator.
        final_time: Time to load. If None, reads from config.json.
    
    Returns:
        Tuple of (L, Qbar, rho, domain).
    """
    from dolfinx import fem
    
    checkpoint_path = run_dir / "checkpoint.bp"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Read final_time from config if not provided
    if final_time is None:
        config_path = run_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            final_time = cfg.get("time", {}).get("total_time", 0.0)
    
    from analysis.analysis_utils import load_checkpoint_mesh, load_checkpoint_function
    
    domain = load_checkpoint_mesh(checkpoint_path, comm)
    gdim = domain.geometry.dim
    
    # Create function spaces:
    # - L is P1 tensor (nodal values)
    # - Qbar is DG0 tensor (cellwise values from GaitDriver)
    # - rho is P1 scalar
    V_L = create_tensor_space(domain, gdim)
    V_Qbar = create_dg0_tensor_space(domain, gdim)  # DG0, not P1!
    V_rho = create_scalar_space(domain)
    
    # Load functions
    L = load_checkpoint_function(checkpoint_path, "L", V_L, time=final_time)
    Qbar = load_checkpoint_function(checkpoint_path, "Qbar", V_Qbar, time=final_time)
    rho = load_checkpoint_function(checkpoint_path, "rho", V_rho, time=final_time)
    
    return L, Qbar, rho, domain


# =============================================================================
# Fabric Metrics Computation
# =============================================================================

def compute_anisotropy_index(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute anisotropy index from fabric eigenvalues.
    
    Anisotropy index A ∈ [0, 1]:
        A = sqrt(3/2) * ||dev(m)|| / ||m||
    
    where m = exp(l) are the fabric eigenvalues and dev(m) = m - mean(m).
    A = 0 for isotropic, A → 1 for maximally anisotropic.
    
    Args:
        eigenvalues: Array of shape (n_points, 3) with l1, l2, l3.
    
    Returns:
        Anisotropy index for each point.
    """
    # Convert log-fabric to fabric eigenvalues
    m = np.exp(eigenvalues)  # (n, 3)
    
    # Mean eigenvalue
    m_mean = np.mean(m, axis=1, keepdims=True)
    
    # Deviatoric part
    m_dev = m - m_mean
    
    # Frobenius norms
    norm_dev = np.sqrt(np.sum(m_dev**2, axis=1))
    norm_m = np.sqrt(np.sum(m**2, axis=1))
    
    # Anisotropy index with safety for zero norm
    tiny = 1e-12
    A = np.sqrt(3.0 / 2.0) * norm_dev / np.maximum(norm_m, tiny)
    
    return np.clip(A, 0.0, 1.0)


def compute_eigenvalue_ratio(eigenvalues: np.ndarray) -> np.ndarray:
    """Compute eigenvalue ratio m_max / m_min.
    
    Args:
        eigenvalues: Array of shape (n_points, 3) with l1, l2, l3.
    
    Returns:
        Ratio max(m) / min(m) for each point.
    """
    m = np.exp(eigenvalues)
    m_max = np.max(m, axis=1)
    m_min = np.min(m, axis=1)
    
    tiny = 1e-12
    ratio = m_max / np.maximum(m_min, tiny)
    
    return ratio


def compute_fabric_metrics(
    L, Qbar, rho,
    record: dict[str, Any],
    comm,
) -> FabricMetrics:
    """Compute fabric metrics from checkpoint fields.
    
    Args:
        L: Log-fabric tensor function (P1 - nodal).
        Qbar: Cycle-averaged stress outer product (DG0 - cellwise).
        rho: Density function (P1).
        record: Sweep record with parameter values.
        comm: MPI communicator.
    
    Returns:
        FabricMetrics dataclass.
    """
    from dolfinx import fem
    from mpi4py import MPI
    
    # Scatter forward for ghost consistency
    L.x.scatter_forward()
    Qbar.x.scatter_forward()
    rho.x.scatter_forward()
    
    # Get owned DOF count - for tensor fields, total owned values = size_local * bs
    # where bs = block_size (9 for 3x3 tensor)
    def get_tensor_array(f):
        """Extract owned tensor values and reshape to (n, 3, 3)."""
        n_local = f.function_space.dofmap.index_map.size_local
        bs = f.function_space.dofmap.index_map_bs
        n_owned = n_local * bs
        arr = f.x.array[:n_owned]
        # Reshape: n_owned = n_nodes_or_cells * 9 for 3x3 tensor
        n_tensors = n_owned // 9
        if n_tensors == 0:
            return np.zeros((0, 3, 3))
        return arr[:n_tensors * 9].reshape(n_tensors, 3, 3)
    
    # Get L array (P1 - nodal)
    L_array = get_tensor_array(L)
    
    # Get Qbar array (DG0 - cellwise)
    Qbar_array = get_tensor_array(Qbar)
    
    n_local_rho = rho.function_space.dofmap.index_map.size_local
    rho_array = rho.x.array[:n_local_rho]
    
    # Symmetrize L tensor
    L_sym = 0.5 * (L_array + L_array.transpose(0, 2, 1))
    
    # Eigendecomposition of L
    L_eigenvalues, L_eigenvectors = np.linalg.eigh(L_sym)
    
    # Compute local metrics from L (nodal)
    anisotropy = compute_anisotropy_index(L_eigenvalues)
    ev_ratio = compute_eigenvalue_ratio(L_eigenvalues)
    
    # Fabric magnitude (Frobenius norm)
    fabric_mag = np.sqrt(np.sum(L_sym**2, axis=(1, 2)))
    
    # Compute Qbar anisotropy - measures directional signal strength
    Qbar_sym = 0.5 * (Qbar_array + Qbar_array.transpose(0, 2, 1))
    Qbar_eigenvalues, _ = np.linalg.eigh(Qbar_sym)
    qbar_anisotropy = compute_anisotropy_index(np.log(np.maximum(Qbar_eigenvalues, 1e-30)))
    
    # Local statistics
    def local_stats(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0
        return np.sum(arr), np.sum(arr**2), np.min(arr), np.max(arr), len(arr)
    
    # Aggregate via MPI
    aniso_sum, aniso_sq_sum, aniso_min, aniso_max, aniso_n = local_stats(anisotropy)
    ev_sum, ev_sq_sum, ev_min, ev_max, ev_n = local_stats(ev_ratio)
    mag_sum, mag_sq_sum, _, _, mag_n = local_stats(fabric_mag)
    qbar_aniso_sum, qbar_aniso_sq_sum, _, _, qbar_aniso_n = local_stats(qbar_anisotropy)
    rho_sum, rho_sq_sum, _, _, rho_n = local_stats(rho_array)
    
    # Global reduction
    global_aniso_sum = comm.allreduce(aniso_sum, op=MPI.SUM)
    global_aniso_sq_sum = comm.allreduce(aniso_sq_sum, op=MPI.SUM)
    global_aniso_max = comm.allreduce(aniso_max, op=MPI.MAX)
    global_aniso_n = comm.allreduce(aniso_n, op=MPI.SUM)
    
    global_ev_sum = comm.allreduce(ev_sum, op=MPI.SUM)
    global_ev_sq_sum = comm.allreduce(ev_sq_sum, op=MPI.SUM)
    global_ev_max = comm.allreduce(ev_max, op=MPI.MAX)
    global_ev_n = comm.allreduce(ev_n, op=MPI.SUM)
    
    global_mag_sum = comm.allreduce(mag_sum, op=MPI.SUM)
    global_mag_sq_sum = comm.allreduce(mag_sq_sum, op=MPI.SUM)
    global_mag_n = comm.allreduce(mag_n, op=MPI.SUM)
    
    global_rho_sum = comm.allreduce(rho_sum, op=MPI.SUM)
    global_rho_sq_sum = comm.allreduce(rho_sq_sum, op=MPI.SUM)
    global_rho_n = comm.allreduce(rho_n, op=MPI.SUM)
    
    global_qbar_aniso_sum = comm.allreduce(qbar_aniso_sum, op=MPI.SUM)
    global_qbar_aniso_sq_sum = comm.allreduce(qbar_aniso_sq_sum, op=MPI.SUM)
    global_qbar_aniso_n = comm.allreduce(qbar_aniso_n, op=MPI.SUM)
    
    # Compute means and stds
    def mean_std(s, sq_s, n):
        if n == 0:
            return 0.0, 0.0
        mean = s / n
        var = max(0.0, sq_s / n - mean**2)
        return mean, np.sqrt(var)
    
    aniso_mean, aniso_std = mean_std(global_aniso_sum, global_aniso_sq_sum, global_aniso_n)
    ev_mean, ev_std = mean_std(global_ev_sum, global_ev_sq_sum, global_ev_n)
    mag_mean, mag_std = mean_std(global_mag_sum, global_mag_sq_sum, global_mag_n)
    qbar_aniso_mean, qbar_aniso_std = mean_std(global_qbar_aniso_sum, global_qbar_aniso_sq_sum, global_qbar_aniso_n)
    rho_mean, rho_std = mean_std(global_rho_sum, global_rho_sq_sum, global_rho_n)
    
    return FabricMetrics(
        run_hash=record.get("run_hash", ""),
        output_dir=record.get("output_dir", ""),
        fabric_tau=float(record.get("fabric.fabric_tau", 0.0)),
        fabric_gammaF=float(record.get("fabric.fabric_gammaF", 0.0)),
        anisotropy_index_mean=aniso_mean,
        anisotropy_index_std=aniso_std,
        anisotropy_index_max=global_aniso_max,
        eigenvalue_ratio_mean=ev_mean,
        eigenvalue_ratio_std=ev_std,
        eigenvalue_ratio_max=global_ev_max,
        fabric_magnitude_mean=mag_mean,
        fabric_magnitude_std=mag_std,
        qbar_anisotropy_mean=qbar_aniso_mean,
        qbar_anisotropy_std=qbar_aniso_std,
        rho_mean=rho_mean,
        rho_std=rho_std,
    )


# =============================================================================
# Analysis Pipeline
# =============================================================================

def analyze_sweep(
    sweep_dir: Path,
    comm,
    verbose: bool = True,
) -> pd.DataFrame:
    """Analyze all runs in a fabric sweep.
    
    Args:
        sweep_dir: Path to sweep output directory.
        comm: MPI communicator.
        verbose: Print progress.
    
    Returns:
        DataFrame with fabric metrics for all runs.
    """
    records = load_fabric_sweep_records(sweep_dir, comm)
    if not records:
        raise ValueError(f"No sweep records found in {sweep_dir}")
    
    if verbose and comm.rank == 0:
        print(f"Found {len(records)} sweep runs")
    
    metrics_list = []
    
    for idx, record in enumerate(records, start=1):
        output_dir = record["output_dir"]
        run_dir = sweep_dir / output_dir
        
        if verbose and comm.rank == 0:
            tau = record.get("fabric.fabric_tau", "?")
            gammaF = record.get("fabric.fabric_gammaF", "?")
            print(f"  [{idx}/{len(records)}] tau={tau}, gammaF={gammaF}")
        
        try:
            L, Qbar, rho, _ = load_fields_from_checkpoint(run_dir, comm)
            metrics = compute_fabric_metrics(L, Qbar, rho, record, comm)
            metrics_list.append(metrics.to_dict())
            
        except FileNotFoundError as e:
            if comm.rank == 0:
                print(f"    Warning: {e}")
            continue
        except Exception as e:
            if comm.rank == 0:
                print(f"    Error: {e}")
            continue
    
    return pd.DataFrame(metrics_list)


# =============================================================================
# Visualization
# =============================================================================

def _summarize(values: np.ndarray) -> tuple[float, float, float]:
    """Return (median, q25, q75) for a 1D array."""
    if values.size == 0:
        return (np.nan, np.nan, np.nan)
    q25, med, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return float(med), float(q25), float(q75)


def create_diagnostic_figure(
    df: pd.DataFrame,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Create diagnostic figure for fabric sweep (single-panel).
    
    Layout:
      Anisotropy Index vs parameters - how anisotropic is the adapted fabric?
    
    We intentionally report a single scalar anisotropy measure here because common
    alternatives (e.g. eigenvalue ratios) are monotone-equivalent under the
    trace-free log-fabric normalization used in the model.
    
    Args:
        df: DataFrame with fabric metrics.
        output_dir: Directory to save plots.
        metadata: Sweep metadata.
    """
    apply_style()
    
    # Get unique parameter values
    tau_vals = sorted(df["fabric_tau"].unique())
    gammaF_vals = sorted(df["fabric_gammaF"].unique())
    
    # Baselines
    baseline_tau = float(metadata.get("baseline_fabric_tau", tau_vals[len(tau_vals) // 2]))
    baseline_gammaF = float(metadata.get("baseline_fabric_gammaF", gammaF_vals[len(gammaF_vals) // 2]))
    
    # Create figure (single panel)
    width = FIGSIZE_FULL_WIDTH[0] * 0.55
    fig, ax = plt.subplots(1, 1, figsize=(width, 3.0))
    
    # Panel: Anisotropy Index
    _plot_parameter_sensitivity(
        ax, df,
        metric_col="anisotropy_index_mean",
        param_values={
            "fabric_tau": (tau_vals, baseline_tau),
            "fabric_gammaF": (gammaF_vals, baseline_gammaF),
        },
    )
    setup_axis_style(
        ax,
        xlabel="Parameter value / baseline",
        ylabel="Anisotropy index $A$",
        title="Fabric anisotropy",
        grid=True,
    )
    ax.set_ylim(0, None)
    
    # Add legend below the plots
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=FABRIC_PARAM_COLORS["fabric_tau"], marker="o",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=FABRIC_PARAM_LABELS["fabric_tau"]),
        Line2D([0], [0], color=FABRIC_PARAM_COLORS["fabric_gammaF"], marker="^",
               linestyle="-", linewidth=PLOT_LINEWIDTH, markersize=5,
               label=FABRIC_PARAM_LABELS["fabric_gammaF"]),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    
    # Save figures
    save_manuscript_figure(fig, "fabric_diagnostic", dpi=PUBLICATION_DPI, close=False)
    save_figure(fig, output_dir / "fabric_diagnostic.png", dpi=PUBLICATION_DPI, close=True)


def _plot_parameter_sensitivity(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric_col: str,
    param_values: dict[str, tuple[list, float]],
) -> None:
    """Plot metric sensitivity to each fabric parameter.
    
    For each parameter, varies it while keeping others at baseline.
    Shows median line (IQR removed - it showed variability due to other
    parameters, not statistical uncertainty).
    
    Args:
        ax: Matplotlib axis.
        df: DataFrame with metrics.
        metric_col: Column name for the metric to plot.
        param_values: Dict mapping param names to (values, baseline).
    """
    for param_name, (values, baseline) in param_values.items():
        color = FABRIC_PARAM_COLORS[param_name]
        marker = FABRIC_PARAM_MARKERS[param_name]
        
        # Normalize values by baseline
        x_normalized = np.array(values) / baseline
        
        # Compute median metric for each parameter value (marginalizing over others)
        y_med = []
        
        for val in values:
            mask = df[param_name] == val
            metric_vals = df.loc[mask, metric_col].values
            med, _, _ = _summarize(metric_vals)
            y_med.append(med)
        
        y_med = np.array(y_med)
        
        # Plot median line with markers
        ax.plot(
            x_normalized, y_med,
            color=color, marker=marker, linestyle="-",
            linewidth=PLOT_LINEWIDTH, markersize=PLOT_MARKERSIZE + 2,
        )
    
    # Add vertical line at baseline (x=1)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Run fabric sweep analysis."""
    sweep_dir = Path("results/fabric_sweep")

    print("=" * 70)
    print("FABRIC PARAMETER SWEEP ANALYSIS")
    print("=" * 70)
    print(f"Sweep directory: {sweep_dir}")

    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        print("Run the sweep first: mpirun -n 4 python run_fabric_sweep.py")
        return
    
    # Load metadata
    metadata_file = sweep_dir / "sweep_summary.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            sweep_json = json.load(f)
            metadata = sweep_json.get("metadata", {})
    else:
        metadata = {}

    # Prefer precomputed metrics (plot-only mode does not require MPI/DOLFINx).
    metrics_file = sweep_dir / "fabric_metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        comm_rank = 0
        print(f"\nLoaded precomputed metrics: {metrics_file}")
    else:
        # Analyze sweep - compute metrics from checkpoints (requires MPI + DOLFINx stack).
        try:
            from mpi4py import MPI
        except ModuleNotFoundError as e:
            print("Error: mpi4py is required to compute metrics from checkpoints.")
            print("Either install mpi4py and run via mpirun, or provide precomputed fabric_metrics.csv.")
            print(f"Details: {e}")
            return

        comm = MPI.COMM_WORLD
        comm_rank = comm.rank

        if comm_rank == 0:
            print("\nComputing fabric metrics from checkpoints...")
        df = analyze_sweep(sweep_dir, comm, verbose=(comm_rank == 0))

        if df.empty:
            if comm_rank == 0:
                print("No valid runs found!")
            return

        if comm_rank == 0:
            df.to_csv(metrics_file, index=False)
            print(f"\nSaved metrics to {metrics_file}")

    # Create diagnostic figure (rank 0 only)
    if comm_rank == 0:
        print("\nGenerating diagnostic figure...")
        create_diagnostic_figure(df, sweep_dir, metadata)
        print("\nAnalysis complete!")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Figure: {sweep_dir / 'fabric_diagnostic.png'}")


if __name__ == "__main__":
    main()
