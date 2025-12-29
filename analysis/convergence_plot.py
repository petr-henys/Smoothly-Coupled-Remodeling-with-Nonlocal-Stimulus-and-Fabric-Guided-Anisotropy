"""Extended convergence plot: errors + performance.

Reads `analysis/convergence_analysis/convergence_data.xlsx` produced by
`analysis/convergence_errors.py`.

Figure layout (2x2):
- (a) Spatial L2 errors vs h for multiple fields
- (b) Temporal L2 errors vs dt for multiple fields  
- (c) Spatial performance vs h: per-subsolver wall time + peak memory
- (d) Temporal performance vs dt: per-subsolver wall time + peak memory

This keeps the same plotting style utilities as other manuscript plots.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.plot_utils import (
    FIELD_NAMES as _ALL_FIELD_NAMES, FIELD_LABELS, FIELD_COLORS, FIELD_MARKERS, COLORS,
    SUBSOLVER_COLORS, SUBSOLVER_MARKERS, SUBSOLVER_LABELS,
    REFERENCE_COLOR, REFERENCE_ALPHA, REFERENCE_LINESTYLE, REFERENCE_LINEWIDTH,
    FIGSIZE_DOUBLE_COLUMN, PUBLICATION_DPI,
    PLOT_LINEWIDTH, PLOT_MARKERSIZE,
    estimate_convergence_order, add_reference_line, setup_axis_style,
    save_manuscript_figure, print_banner, apply_style,
)

# Exclude psi from convergence plots
FIELD_NAMES = [f for f in _ALL_FIELD_NAMES if f != "psi"]


def _try_read_sheet(xlsx: Path, sheet: str) -> pd.DataFrame | None:
    try:
        return pd.read_excel(xlsx, sheet_name=sheet)
    except Exception:
        return None


def load_spatial_data(xlsx_file: Path, dt_value: float) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for field in FIELD_NAMES:
        sheet_name = f"spatial_{field}_dt{dt_value}"
        df = _try_read_sheet(xlsx_file, sheet_name)
        if df is not None and not df.empty:
            data[field] = df
    return data


def load_temporal_data(xlsx_file: Path, N_value: int) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for field in FIELD_NAMES:
        sheet_name = f"temporal_{field}_N{N_value}"
        df = _try_read_sheet(xlsx_file, sheet_name)
        if df is not None and not df.empty:
            data[field] = df
    return data


def load_spatial_perf(xlsx_file: Path, dt_value: float) -> pd.DataFrame | None:
    return _try_read_sheet(xlsx_file, f"spatial_perf_dt{dt_value}")


def load_temporal_perf(xlsx_file: Path, N_value: int) -> pd.DataFrame | None:
    return _try_read_sheet(xlsx_file, f"temporal_perf_N{N_value}")


def plot_spatial_errors(ax: plt.Axes, spatial_data: dict[str, pd.DataFrame], error_type: str, dt_value: float) -> None:
    if error_type == "L2_error":
        ylabel = r"$L^2$ error"
        title_prefix = r"(a) Spatial $L^2$"
    else:
        ylabel = r"$H^1$ seminorm error"
        title_prefix = r"Spatial $H^1$"

    for field, df in spatial_data.items():
        h = df["h"].to_numpy(dtype=float)
        err = df[error_type].to_numpy(dtype=float)
        if h.size == 0:
            continue
        order = estimate_convergence_order(h, err)
        label = FIELD_LABELS.get(field, field)
        line, = ax.loglog(
            h, err,
            marker=FIELD_MARKERS.get(field, "o"),
            color=FIELD_COLORS.get(field, "C0"),
            label=label,
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
        )
        # Add order annotation near the line (at rightmost point)
        ax.annotate(
            f"p={order:.2f}",
            xy=(h[-1], err[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=6,
            color=line.get_color(),
            va="center",
        )

    if spatial_data:
        first_df = next(iter(spatial_data.values()))
        h_vals = first_df["h"].to_numpy(dtype=float)
        if h_vals.size:
            h_range = (float(np.min(h_vals)), float(np.max(h_vals)))
            mid_errors = [float(df[error_type].iloc[len(df) // 2]) for df in spatial_data.values() if not df.empty]
            if mid_errors:
                ref_scale = float(np.median(mid_errors))
                add_reference_line(ax, h_range, 1.0, ref_scale, r"$O(h)$")
                add_reference_line(ax, h_range, 2.0, ref_scale, r"$O(h^2)$", linestyle=":")

    setup_axis_style(
        ax,
        r"Mesh size $h$ [mm]",
        ylabel,
        rf"{title_prefix} ($\Delta t = {dt_value}$ days)",
        loglog=True,
    )


def plot_temporal_errors(ax: plt.Axes, temporal_data: dict[str, pd.DataFrame], error_type: str, N_value: int) -> None:
    if error_type == "L2_error":
        ylabel = r"$L^2$ error"
        title_prefix = r"(b) Temporal $L^2$"
    else:
        ylabel = r"$H^1$ seminorm error"
        title_prefix = r"Temporal $H^1$"

    for field, df in temporal_data.items():
        dt = df["dt_days"].to_numpy(dtype=float)
        err = df[error_type].to_numpy(dtype=float)
        if dt.size == 0:
            continue
        order = estimate_convergence_order(dt, err, from_start=True)
        label = FIELD_LABELS.get(field, field)
        line, = ax.loglog(
            dt, err,
            marker=FIELD_MARKERS.get(field, "o"),
            color=FIELD_COLORS.get(field, "C0"),
            label=label,
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
        )
        # Add order annotation near the line (at rightmost point)
        ax.annotate(
            f"p={order:.2f}",
            xy=(dt[-1], err[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=6,
            color=line.get_color(),
            va="center",
        )

    if temporal_data:
        first_df = next(iter(temporal_data.values()))
        dt_vals = first_df["dt_days"].to_numpy(dtype=float)
        if dt_vals.size:
            dt_range = (float(np.min(dt_vals)), float(np.max(dt_vals)))
            mid_errors = [float(df[error_type].iloc[len(df) // 2]) for df in temporal_data.values() if not df.empty]
            if mid_errors:
                ref_scale = float(np.median(mid_errors))
                add_reference_line(ax, dt_range, 1.0, ref_scale, r"$O(\Delta t)$")

    setup_axis_style(
        ax,
        r"Timestep $\Delta t$ [days]",
        ylabel,
        rf"{title_prefix} ($N = {N_value}$)",
        loglog=True,
    )


def plot_performance(ax: plt.Axes, df: pd.DataFrame, x_col: str, title: str) -> None:
    # Per-subsolver wall time (primary y)
    x = df[x_col].to_numpy(dtype=float)

    lines = [
        ("mech_time", "mech"),
        ("fab_time", "fab"),
        ("stim_time", "stim"),
        ("dens_time", "dens"),
    ]

    for col, subsolver in lines:
        if col not in df:
            continue
        y = df[col].to_numpy(dtype=float)
        ax.loglog(
            x, y,
            color=SUBSOLVER_COLORS.get(subsolver, "C0"),
            marker=SUBSOLVER_MARKERS.get(subsolver, "o"),
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
        )

    # KSP iterations (secondary y)
    ax2 = ax.twinx()
    iters = [
        ("mech_iters", "mech"),
        ("fab_iters", "fab"),
        ("stim_iters", "stim"),
        ("dens_iters", "dens"),
    ]
    for col, subsolver in iters:
        if col not in df:
            continue
        y = df[col].to_numpy(dtype=float)
        ax2.semilogx(
            x,
            y,
            linestyle=":",
            color=SUBSOLVER_COLORS.get(subsolver, "C0"),
            linewidth=PLOT_LINEWIDTH,
        )

    ax2.set_ylabel("KSP iterations")
    ax2.tick_params(axis="y")

    # Don't add legend here - will be added as unified legend below figure

    if x_col == "h":
        xlabel = r"Mesh size $h$ [mm]"
    else:
        xlabel = r"Timestep $\Delta t$ [days]"

    setup_axis_style(ax, xlabel, "Wall time [s]", title, loglog=True)
    
    # Store ax2 reference for legend collection
    ax._perf_ax2 = ax2


def create_figure(xlsx: Path, dt_spatial: float, N_temporal: int, out: Path) -> None:
    spatial_data = load_spatial_data(xlsx, dt_spatial)
    temporal_data = load_temporal_data(xlsx, N_temporal)

    df_sp_perf = load_spatial_perf(xlsx, dt_spatial)
    df_tm_perf = load_temporal_perf(xlsx, N_temporal)

    # Layout: 3 rows (2 plot rows + 1 legend row) using GridSpec
    fig = plt.figure(figsize=(FIGSIZE_DOUBLE_COLUMN[0], FIGSIZE_DOUBLE_COLUMN[1] * 1.18))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.18], hspace=0.35, wspace=0.3)

    # Row 0: L2 errors (spatial, temporal)
    ax_err_sp = fig.add_subplot(gs[0, 0])
    ax_err_tm = fig.add_subplot(gs[0, 1])
    plot_spatial_errors(ax_err_sp, spatial_data, "L2_error", dt_spatial)
    plot_temporal_errors(ax_err_tm, temporal_data, "L2_error", N_temporal)

    # Row 1: Performance (spatial, temporal)
    ax_perf_sp = fig.add_subplot(gs[1, 0])
    ax_perf_tm = fig.add_subplot(gs[1, 1])
    
    if df_sp_perf is not None and not df_sp_perf.empty:
        plot_performance(ax_perf_sp, df_sp_perf, "h", r"(c) Spatial performance")
    else:
        setup_axis_style(ax_perf_sp, "", "", r"(c) Spatial performance", loglog=False)
        ax_perf_sp.text(0.5, 0.5, "No performance data", ha="center", va="center", transform=ax_perf_sp.transAxes)

    if df_tm_perf is not None and not df_tm_perf.empty:
        plot_performance(ax_perf_tm, df_tm_perf, "dt_days", r"(d) Temporal performance")
    else:
        setup_axis_style(ax_perf_tm, "", "", r"(d) Temporal performance", loglog=False)
        ax_perf_tm.text(0.5, 0.5, "No performance data", ha="center", va="center", transform=ax_perf_tm.transAxes)

    # Row 2: Unified legend for all plots
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.axis('off')
    
    # Build unified legend: field symbols + line style indicators
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []
    
    # Add field/subsolver entries (shared across all plots)
    all_fields = list(spatial_data.keys()) if spatial_data else []
    # Add psi if present in performance data (not in error data)
    if df_sp_perf is not None and "mech_time" in df_sp_perf.columns:
        if "psi" not in all_fields:
            all_fields = ["psi"] + all_fields
    
    for field in all_fields:
        legend_handles.append(Line2D(
            [0], [0],
            marker=FIELD_MARKERS.get(field, "o"),
            color=FIELD_COLORS.get(field, "C0"),
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
        ))
        legend_labels.append(FIELD_LABELS.get(field, field))
    
    # Add line style indicators (solid = wall time / error, dashed = KSP iters)
    legend_handles.append(Line2D([0], [0], color="gray", linewidth=PLOT_LINEWIDTH, linestyle="-"))
    legend_labels.append("error / wall time")
    legend_handles.append(Line2D([0], [0], color="gray", linewidth=PLOT_LINEWIDTH, linestyle=":"))
    legend_labels.append("KSP iterations")
    
    # Add reference line indicators
    legend_handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH, 
                                  linestyle=REFERENCE_LINESTYLE, alpha=REFERENCE_ALPHA))
    legend_labels.append(r"$O(h)$, $O(\Delta t)$")
    legend_handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linewidth=REFERENCE_LINEWIDTH, 
                                  linestyle=":", alpha=REFERENCE_ALPHA))
    legend_labels.append(r"$O(h^2)$")
    
    ax_leg.legend(legend_handles, legend_labels, loc="center", ncol=4, fontsize=7, frameon=False)

    save_manuscript_figure(fig, out.name, dpi=PUBLICATION_DPI)


if __name__ == "__main__":
    apply_style()
    
    xlsx_file = Path("analysis/convergence_analysis/convergence_data.xlsx")
    output_file = Path("manuscript/images/convergence_full_plot.png")

    if not xlsx_file.exists():
        print(f"ERROR: {xlsx_file} not found!")
        print("Run: mpirun -n 2 python analysis/convergence_errors.py")
        sys.exit(1)

    print_banner("CONVERGENCE (ERRORS + PERFORMANCE)")

    xl = pd.ExcelFile(xlsx_file)
    sheet_names = xl.sheet_names

    dt_values = sorted({float(s.split("_dt")[1]) for s in sheet_names if s.startswith("spatial_perf_dt")})
    N_values = sorted({int(s.split("_N")[1]) for s in sheet_names if s.startswith("temporal_perf_N")})

    if not dt_values:
        # Fall back to old naming (spatial_u_dt...)
        dt_values = sorted({float(s.split("_dt")[1]) for s in sheet_names if s.startswith("spatial_") and "_dt" in s})
    if not N_values:
        N_values = sorted({int(s.split("_N")[1]) for s in sheet_names if s.startswith("temporal_") and "_N" in s})

    dt_spatial = dt_values[len(dt_values) // 2] if dt_values else 5.0
    N_temporal = N_values[-1] if N_values else 24  # Use highest resolution for temporal

    print(f"Using dt={dt_spatial} for spatial panels")
    print(f"Using N={N_temporal} for temporal panels")

    create_figure(xlsx_file, dt_spatial, N_temporal, output_file)
    print(f"Saved: {output_file}")
