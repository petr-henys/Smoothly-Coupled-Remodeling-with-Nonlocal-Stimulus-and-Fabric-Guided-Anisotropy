"""Shared plotting utilities for analysis scripts.

Provides common styling, helper functions, and figure management
to avoid code duplication across plotting scripts.
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================================
# Shared styling configuration
# ============================================================================

# Field styling (for convergence plots)
FIELD_NAMES = ["u", "rho", "S", "A"]
FIELD_LABELS = {
    "u": r"Displacement $u$",
    "rho": r"Density $\rho$",
    "S": r"Stimulus $S$",
    "A": r"Orientation $A$",
}
FIELD_COLORS = {
    "u": "#1f77b4",      # Blue
    "rho": "#ff7f0e",    # Orange
    "S": "#2ca02c",      # Green
    "A": "#d62728",      # Red
}
FIELD_MARKERS = {
    "u": "o",
    "rho": "s",
    "S": "^",
    "A": "D",
}

# Timestep styling (for performance/anderson plots)
DT_COLORS = {
    6.25: "#1f77b4",
    12.5: "#ff7f0e",
    25.0: "#2ca02c",
    50.0: "#d62728",
    100.0: "#9467bd",
}
DT_MARKERS = {
    6.25: "o",
    12.5: "s",
    25.0: "^",
    50.0: "D",
    100.0: "v",
}

# Reference line styling
REFERENCE_COLOR = "gray"
REFERENCE_ALPHA = 0.5
REFERENCE_LINESTYLE = "--"
REFERENCE_LINEWIDTH = 1.5

# Default figure settings
DEFAULT_DPI = 300
DEFAULT_FIGURE_FORMAT = "png"


# ============================================================================
# Helper functions
# ============================================================================

def setup_axis_style(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str,
    loglog: bool = False,
    grid: bool = True,
) -> None:
    """Apply consistent axis styling.
    
    Args:
        ax: Matplotlib axis
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Subplot title
        loglog: Use log-log scale
        grid: Show grid
    """
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    if grid:
        ax.grid(True, which="both", alpha=0.3, linestyle=":")


def remove_all_legends(axes) -> None:
    """Remove legends from all subplots in axes array."""
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat:
        legend = ax.legend()
        if legend:
            legend.set_visible(False)


def create_unified_legend(
    fig: plt.Figure,
    handles,
    labels,
    ncol: Optional[int] = None,
    bbox_y: float = -0.02,
) -> None:
    """Create unified legend below subplots.
    
    Args:
        fig: Matplotlib figure
        handles: Legend handles
        labels: Legend labels
        ncol: Number of columns (auto if None)
        bbox_y: Y-position for legend (0 = bottom of figure)
    """
    if ncol is None:
        ncol = min(len(handles), 5)
    
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=ncol,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, bbox_y),
    )


def save_figure(
    fig: plt.Figure,
    output_file: Path,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> None:
    """Save figure with consistent settings.
    
    Args:
        fig: Matplotlib figure
        output_file: Output path
        dpi: Resolution
        close: Close figure after saving
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    
    if close:
        plt.close(fig)


def estimate_convergence_order(x: np.ndarray, y: np.ndarray, n_pts: int = 3) -> float:
    """Estimate convergence order from log-log data via linear regression.
    
    Fits log(y) = log(C) + p*log(x) and returns slope p.
    
    Args:
        x: Independent variable (h or dt)
        y: Dependent variable (error)
        n_pts: Number of points to use (from end, for robustness)
        
    Returns:
        Estimated convergence order (slope)
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan
    
    # Use last n_pts (or all if less)
    n = min(n_pts, len(x))
    x_fit = x[-n:]
    y_fit = y[-n:]
    
    # Log-log linear regression
    log_x = np.log10(x_fit)
    log_y = np.log10(y_fit)
    poly = np.polyfit(log_x, log_y, 1)
    return poly[0]  # Slope = convergence order


def add_reference_line(
    ax: plt.Axes,
    x_range: tuple[float, float],
    order: float,
    ref_scale: float,
    label: str,
    linestyle: str = REFERENCE_LINESTYLE,
) -> None:
    """Add power-law reference line to log-log plot.
    
    Args:
        ax: Matplotlib axis
        x_range: (x_min, x_max) for reference line
        order: Power-law exponent
        ref_scale: Scaling constant for vertical positioning
        label: Legend label
        linestyle: Line style
    """
    x_ref = np.array(x_range)
    C = ref_scale / (x_ref[1] ** order)
    y_ref = C * x_ref ** order
    
    ax.loglog(
        x_ref, y_ref,
        color=REFERENCE_COLOR,
        linestyle=linestyle,
        linewidth=REFERENCE_LINEWIDTH,
        alpha=REFERENCE_ALPHA,
        label=label,
    )


def print_banner(title: str, width: int = 80) -> None:
    """Print formatted banner for console output."""
    print("=" * width)
    print(title)
    print("=" * width)


def format_dt_label(dt: float) -> str:
    """Format dt value for legend labels."""
    if dt == int(dt):
        return f"dt={int(dt)} days"
    else:
        return f"dt={dt:.2f} days"
