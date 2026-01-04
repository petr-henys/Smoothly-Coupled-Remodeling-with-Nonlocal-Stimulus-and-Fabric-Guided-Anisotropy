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

# Master style dictionary for consistent plots across all analysis scripts.
# This is the SINGLE source of truth for plot styling.
MASTER_STYLE = {
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,           # Default for all text
    'axes.labelsize': 8,      # X/Y axis labels
    'axes.titlesize': 9,      # Subplot titles (slightly larger)
    'axes.titleweight': 'bold', # Bold titles
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
    'legend.fontsize': 7,     # Legend text
    'figure.titlesize': 10,   # Main figure title (if used)
    'text.usetex': False,     # Set True if LaTeX available
    'mathtext.fontset': 'dejavuserif',
    # Line widths
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
    'lines.markersize': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
    # Grid settings
    'axes.grid': True,
    'grid.alpha': 0.2,
    # Spine visibility (modern look)
    'axes.spines.top': False,
    'axes.spines.right': False,
    # Background colors
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    # Legend
    'legend.frameon': False,
    # DPI and save settings
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
}

# Apply master style on import
plt.rcParams.update(MASTER_STYLE)


def apply_style() -> None:
    """Apply the unified CMAME-style Matplotlib settings.
    
    Call this at the start of any plotting script to ensure consistent styling.
    The style is also applied on import, but calling this explicitly is recommended
    for clarity and to reset any modifications made during script execution.
    """
    plt.rcParams.update(MASTER_STYLE)


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
FIELD_NAMES = ["psi", "rho", "S", "L"]
FIELD_LABELS = {
    "psi": r"$\psi$ (SED)",
    "rho": r"$\rho$ (density)",
    "S": r"$S$ (stimulus)",
    "L": r"$\mathbf{L}$ (log-fabric)",
}
# Color-blind friendly palette.
FIELD_COLORS = {
    "psi": "#0173B2",    # Blue (mechanics/SED)
    "rho": "#DE8F05",    # Orange (density)
    "S": "#029E73",      # Green (stimulus)
    "L": "#CC78BC",      # Purple (fabric)
}
FIELD_MARKERS = {
    "psi": "o",
    "rho": "s",
    "S": "^",
    "L": "D",
}

# Subsolver styling (aligned with field colors).
SUBSOLVER_NAMES = ["mech", "stim", "dens", "fab"]
SUBSOLVER_LABELS = {
    "mech": r"$\psi$ (mechanics)",
    "stim": r"$S$ (stimulus)", 
    "dens": r"$\rho$ (density)",
    "fab": r"$\mathbf{L}$ (fabric)",
}
SUBSOLVER_COLORS = {
    "mech": FIELD_COLORS["psi"],    # Blue
    "stim": FIELD_COLORS["S"],      # Green
    "dens": FIELD_COLORS["rho"],    # Orange
    "fab": FIELD_COLORS["L"],       # Purple
}
SUBSOLVER_MARKERS = {
    "mech": FIELD_MARKERS["psi"],
    "stim": FIELD_MARKERS["S"],
    "dens": FIELD_MARKERS["rho"],
    "fab": FIELD_MARKERS["L"],
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
    ax.set_title(title)
    
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


# =============================================================================
# Manuscript Figure Utilities (constitutive/analysis plots)
# =============================================================================

# Output directory for manuscript figures.
MANUSCRIPT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "manuscript" / "images"

# Tol colorblind-safe palette for constitutive plots.
COLORS = {
    'blue': '#0077BB',
    'cyan': '#33BBEE',
    'teal': '#009988',
    'orange': '#EE7733',
    'red': '#CC3311',
    'magenta': '#EE3377',
    'grey': '#BBBBBB',
    'black': '#000000',
}


def smooth_max(x: np.ndarray, xmin: float, eps: float = 1e-6) -> np.ndarray:
    """C¹ approximation of max(x, xmin): smooth_max(x, xmin) ≥ xmin always."""
    dx = x - xmin
    return xmin + 0.5 * (dx + np.sqrt(dx * dx + eps * eps))


def smoothstep01(t: np.ndarray) -> np.ndarray:
    """Cubic smoothstep: 0 for t≤0, 1 for t≥1, t²(3-2t) in between."""
    t_clamped = np.clip(t, 0.0, 1.0)
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)


def create_figure(nrows: int = 2, ncols: int = 3, 
                  figsize: tuple = None) -> tuple:
    """Create figure with consistent styling for manuscript.
    
    Returns:
        (fig, axes) tuple.
    """
    if figsize is None:
        figsize = (12, 7) if ncols >= 3 else (8, 6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_manuscript_figure(fig: plt.Figure, filename: str, 
                           dpi: int = 300, close: bool = True) -> Path:
    """Save figure to manuscript/images directory.
    
    Args:
        fig: Matplotlib figure.
        filename: Output filename (with or without extension).
        dpi: Resolution.
        close: Close figure after saving.
        
    Returns:
        Path to saved file.
    """
    MANUSCRIPT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    output_path = MANUSCRIPT_IMAGE_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    
    if close:
        plt.close(fig)
    
    print(f"Generated {output_path}")
    return output_path
