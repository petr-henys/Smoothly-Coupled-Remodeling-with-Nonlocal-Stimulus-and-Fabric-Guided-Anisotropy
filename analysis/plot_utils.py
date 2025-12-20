"""Shared Matplotlib styling and helpers for analysis plots."""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# Allow running analysis scripts from this directory.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# --- CMAME-style Matplotlib settings ---

# Use absolute font sizes (points) for consistent printed size.
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    # Base font size: matches manuscript body text (10pt)
    'font.size': 8,           # Default for all text
    'axes.labelsize': 8,      # X/Y axis labels
    'axes.titlesize': 9,      # Subplot titles (slightly larger)
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
    'legend.fontsize': 7,     # Legend text
    'figure.titlesize': 10,   # Main figure title (if used)
    'text.usetex': False,     # Set True if LaTeX available
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Figure size presets for CMAME (A4 paper).
FIGSIZE_SINGLE_COLUMN = (3.5, 2.8)      # Single column width
FIGSIZE_DOUBLE_COLUMN = (8.0, 6.0)      # Double column width (larger for readability)
FIGSIZE_FULL_WIDTH = (8.0, 6.0)         # Full A4 width
FIGSIZE_TALL = (8.0, 10.0)              # Tall format (3x2 subplots)

# DPI presets.
PUBLICATION_DPI = 600
DEFAULT_DPI = 300


# --- Shared styling configuration ---

# Field styling (convergence plots).
FIELD_NAMES = ["u", "rho", "S", "A"]
FIELD_LABELS = {
    "u": r"$\mathbf{u}$ (displacement)",
    "rho": r"$\rho$ (density)",
    "S": r"$S$ (stimulus)",
    "A": r"$\mathbf{A}$ (orientation)",
}
# Color-blind friendly palette.
FIELD_COLORS = {
    "u": "#0173B2",      # Blue (mechanics)
    "rho": "#DE8F05",    # Orange (density)
    "S": "#029E73",      # Green (stimulus)
    "A": "#CC78BC",      # Purple (orientation/direction)
}
FIELD_MARKERS = {
    "u": "o",
    "rho": "s",
    "S": "^",
    "A": "D",
}

# Subsolver styling (aligned with field colors).
SUBSOLVER_NAMES = ["mech", "stim", "dens", "dir"]
SUBSOLVER_LABELS = {
    "mech": r"$\mathbf{u}$ (mechanics)",
    "stim": r"$S$ (stimulus)", 
    "dens": r"$\rho$ (density)",
    "dir": r"$\mathbf{A}$ (direction)",
}
SUBSOLVER_COLORS = {
    "mech": FIELD_COLORS["u"],      # Blue
    "stim": FIELD_COLORS["S"],      # Green
    "dens": FIELD_COLORS["rho"],    # Orange
    "dir": FIELD_COLORS["A"],       # Purple
}
SUBSOLVER_MARKERS = {
    "mech": FIELD_MARKERS["u"],
    "stim": FIELD_MARKERS["S"],
    "dens": FIELD_MARKERS["rho"],
    "dir": FIELD_MARKERS["A"],
}

# Timestep styling (performance/Anderson plots): grayscale for print.
DT_COLORS = {
    6.25: "#000000",   # Black
    12.5: "#404040",   # Dark gray
    25.0: "#707070",   # Medium gray
    50.0: "#A0A0A0",   # Light gray
    100.0: "#D0D0D0",  # Very light gray
}
DT_MARKERS = {
    6.25: "v",   # Triangle down
    12.5: "<",   # Triangle left
    25.0: "o",   # Circle
    50.0: "s",   # Square
    100.0: "^",  # Triangle up
}
DT_LINESTYLES = {
    6.25: "-",
    12.5: "--",
    25.0: "-.",
    50.0: ":",
    100.0: "-",
}

# Reference line styling.
REFERENCE_COLOR = "#808080"
REFERENCE_ALPHA = 0.6
REFERENCE_LINESTYLE = "--"
REFERENCE_LINEWIDTH = 1.0

# Plot styling constants.
PLOT_LINEWIDTH = 1.2
PLOT_MARKERSIZE = 3
PLOT_ALPHA_OVERLAY = 0.5  # For overlapping curves

# Legend styling constants.
LEGEND_FONTSIZE = 7  # Matches rcParams['legend.fontsize']
LEGEND_FRAMEALPHA = 0.95
LEGEND_EDGECOLOR = "black"
LEGEND_FANCYBOX = False

DEFAULT_FIGURE_FORMAT = "png"


# --- Helper functions ---

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
        ax: Matplotlib axis.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Subplot title.
        loglog: If True, use log-log scale.
        grid: If True, draw a background grid.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    
    if grid:
        ax.grid(True, which="both", alpha=0.25, linestyle=":", linewidth=0.5, color="gray")
    
    # Ensure tick labels are visible
    ax.tick_params(axis='both', which='major', direction='in')
    ax.tick_params(axis='both', which='minor', direction='in', length=2)


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
    loc: str = "upper left",
    bbox_to_anchor: Optional[tuple] = None,
) -> None:
    """Create a single legend for a figure.

    Args:
        fig: Matplotlib figure.
        handles: Legend handles.
        labels: Legend labels.
        ncol: Number of columns (auto if None).
        loc: Legend location.
        bbox_to_anchor: Optional custom anchor position.
    """
    if ncol is None:
        ncol = min(len(handles), 5)
    
    if bbox_to_anchor is None:
        if loc == "upper left":
            bbox_to_anchor = (0.05, 0.98)
        elif loc == "lower center":
            bbox_to_anchor = (0.5, -0.02)
        else:
            bbox_to_anchor = None
    
    fig.legend(
        handles, labels,
        loc=loc,
        ncol=ncol,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=bbox_to_anchor,
        framealpha=LEGEND_FRAMEALPHA,
        edgecolor=LEGEND_EDGECOLOR,
        fancybox=LEGEND_FANCYBOX,
    )


def add_subplot_legend(ax: plt.Axes, loc: str = "upper left") -> None:
    """Add a legend to a subplot with consistent styling.

    Args:
        ax: Matplotlib axis.
        loc: Legend location.
    """
    ax.legend(
        loc=loc,
        fontsize=LEGEND_FONTSIZE,
        framealpha=LEGEND_FRAMEALPHA,
        frameon=True,
        edgecolor=LEGEND_EDGECOLOR,
        fancybox=LEGEND_FANCYBOX,
    )


def save_figure(
    fig: plt.Figure,
    output_file: Path,
    dpi: int = PUBLICATION_DPI,
    close: bool = True,
) -> None:
    """Save a figure using project defaults (PNG).

    Args:
        fig: Matplotlib figure.
        output_file: Output path.
        dpi: Raster resolution.
        close: If True, close the figure after saving.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    
    if close:
        plt.close(fig)


def estimate_convergence_order(x: np.ndarray, y: np.ndarray, n_pts: int = 3, from_start: bool = False) -> float:
    """Estimate convergence order from log-log data via linear regression.

    Args:
        x: Independent variable (h or dt).
        y: Dependent variable (error).
        n_pts: Number of points to use.
        from_start: If True, use first n points (smallest x, finest resolution).
            If False, use last n points (largest x).

    Returns:
        Estimated convergence order (slope).
    """
    if len(x) < 2 or len(y) < 2:
        return np.nan
    
    # Use n_pts from appropriate end
    n = min(n_pts, len(x))
    if from_start:
        # Use first n points (smallest x = finest resolution)
        x_fit = x[:n]
        y_fit = y[:n]
    else:
        # Use last n points (largest x)
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
    """Add a power-law reference line to a log-log plot.

    Args:
        ax: Matplotlib axis.
        x_range: `(x_min, x_max)` for the reference line.
        order: Power-law exponent.
        ref_scale: Scaling constant for vertical positioning.
        label: Legend label.
        linestyle: Line style.
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
    """Print a simple console banner."""
    print("=" * width)
    print(title)
    print("=" * width)


def format_dt_label(dt: float) -> str:
    """Format `dt` for legend labels."""
    if dt == int(dt):
        return f"dt={int(dt)} days"
    else:
        return f"dt={dt:.2f} days"
