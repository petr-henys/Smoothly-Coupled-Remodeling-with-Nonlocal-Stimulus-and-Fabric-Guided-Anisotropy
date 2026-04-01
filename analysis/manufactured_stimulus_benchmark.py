"""Manufactured-solution benchmark for the scalar stimulus transport operator.

This script validates the linear diffusion-decay operator used in the stimulus
update with a smooth exact solution on the unit cube. The nonlinear mechanostat
drive is replaced by a manufactured source term so the FE operator can be
checked in isolation. It produces:

- `results/reviewer_benchmarks/manufactured_stimulus/metrics.csv`
- `results/reviewer_benchmarks/manufactured_stimulus/summary.json`
- `manuscript/images/manufactured_stimulus.png`

The benchmark is intentionally lightweight so it can be rerun during revision.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import basix.ufl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

conda_bin = project_root / ".conda" / "bin"
if conda_bin.exists():
    os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("CC", str(conda_bin / "gcc"))
    os.environ.setdefault("CXX", str(conda_bin / "g++"))

from analysis.plot_utils import (
    COLORS,
    LEGEND_FONTSIZE,
    PUBLICATION_DPI,
    add_reference_line,
    apply_style,
    estimate_convergence_order,
    save_figure,
    save_manuscript_figure,
    setup_axis_style,
)
from simulation.utils import compute_mean_element_length


RESULTS_DIR = project_root / "results" / "reviewer_benchmarks" / "manufactured_stimulus"

# Problem parameters chosen to keep the solve well conditioned while remaining
# representative of the diffusion-decay stimulus operator.
TAU = 2.0
DIFFUSIVITY = 0.15
OMEGA = 0.7
FINAL_TIME = 1.0

# Lightweight grids suitable for a quick reviewer-facing benchmark.
SPATIAL_LEVELS = [4, 6, 8, 10]
TEMPORAL_STEPS = [1.0, 0.5, 0.25, 0.125]
SPATIAL_DT = 0.05
TEMPORAL_MESH_N = 24
FIGSIZE_COMPACT_TWO_PANEL = (5.5, 2.45)


def exact_solution(domain: mesh.Mesh, time_value: float) -> ufl.core.expr.Expr:
    x = ufl.SpatialCoordinate(domain)
    phi = ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]) * ufl.cos(ufl.pi * x[2])
    return phi * math.exp(-OMEGA * time_value)


def continuous_source(domain: mesh.Mesh, time_value: float) -> ufl.core.expr.Expr:
    u_exact = exact_solution(domain, time_value)
    dudt = -OMEGA * u_exact
    return TAU * dudt - TAU * DIFFUSIVITY * ufl.div(ufl.grad(u_exact)) + u_exact


def discrete_be_source(
    domain: mesh.Mesh, time_old: float, time_new: float, dt_value: float
) -> ufl.core.expr.Expr:
    u_old = exact_solution(domain, time_old)
    u_new = exact_solution(domain, time_new)
    return TAU * (u_new - u_old) / dt_value - TAU * DIFFUSIVITY * ufl.div(ufl.grad(u_new)) + u_new


def create_space(domain: mesh.Mesh) -> fem.FunctionSpace:
    element = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    return fem.functionspace(domain, element)


def interpolate_exact(space: fem.FunctionSpace, time_value: float) -> fem.Function:
    func = fem.Function(space, name=f"u_exact_t{time_value:g}")

    def _expr(x: np.ndarray) -> np.ndarray:
        return np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]) * np.cos(np.pi * x[2]) * math.exp(
            -OMEGA * time_value
        )

    func.interpolate(_expr)
    func.x.scatter_forward()
    return func


def solve_stimulus_step(
    space: fem.FunctionSpace,
    u_old: fem.Function,
    rhs_expr: ufl.core.expr.Expr,
    dt_value: float,
) -> fem.Function:
    trial = ufl.TrialFunction(space)
    test = ufl.TestFunction(space)
    dx = ufl.Measure("dx", domain=space.mesh, metadata={"quadrature_degree": 6})

    a_form = (
        (TAU / dt_value + 1.0) * trial * test
        + (TAU * DIFFUSIVITY) * ufl.dot(ufl.grad(trial), ufl.grad(test))
    ) * dx
    l_form = ((TAU / dt_value) * u_old + rhs_expr) * test * dx

    uh = fem.Function(space, name="S")
    problem = LinearProblem(
        a_form,
        l_form,
        u=uh,
        petsc_options_prefix="manufactured_stimulus_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    problem.solve()
    uh.x.scatter_forward()
    return uh


def compute_error_norms(
    uh: fem.Function, exact_expr: ufl.core.expr.Expr
) -> tuple[float, float]:
    dx = ufl.Measure("dx", domain=uh.function_space.mesh, metadata={"quadrature_degree": 8})
    diff = uh - exact_expr
    l2_sq = fem.assemble_scalar(fem.form(ufl.inner(diff, diff) * dx))
    h1_sq = fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(diff), ufl.grad(diff)) * dx))
    return math.sqrt(l2_sq), math.sqrt(h1_sq)


def spatial_convergence() -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for n in SPATIAL_LEVELS:
        domain = mesh.create_unit_cube(MPI.COMM_SELF, n, n, n)
        space = create_space(domain)
        u0 = interpolate_exact(space, 0.0)
        rhs_expr = discrete_be_source(domain, 0.0, SPATIAL_DT, SPATIAL_DT)
        uh = solve_stimulus_step(space, u0, rhs_expr, SPATIAL_DT)
        u_exact = exact_solution(domain, SPATIAL_DT)
        l2_error, h1_error = compute_error_norms(uh, u_exact)
        records.append(
            {
                "study": "spatial",
                "n": float(n),
                "h": float(compute_mean_element_length(domain)),
                "dt": SPATIAL_DT,
                "L2_error": l2_error,
                "H1_error": h1_error,
            }
        )
    return records


def temporal_convergence() -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    domain = mesh.create_unit_cube(MPI.COMM_SELF, TEMPORAL_MESH_N, TEMPORAL_MESH_N, TEMPORAL_MESH_N)
    space = create_space(domain)
    h_value = float(compute_mean_element_length(domain))

    for dt_value in TEMPORAL_STEPS:
        u_prev = interpolate_exact(space, 0.0)
        current_time = 0.0
        while current_time < FINAL_TIME - 1e-14:
            dt_step = min(dt_value, FINAL_TIME - current_time)
            t_new = current_time + dt_step
            rhs_expr = continuous_source(domain, t_new)
            u_prev = solve_stimulus_step(space, u_prev, rhs_expr, dt_step)
            current_time = t_new

        u_exact = exact_solution(domain, FINAL_TIME)
        l2_error, h1_error = compute_error_norms(u_prev, u_exact)
        records.append(
            {
                "study": "temporal",
                "n": float(TEMPORAL_MESH_N),
                "h": h_value,
                "dt": float(dt_value),
                "L2_error": l2_error,
                "H1_error": h1_error,
            }
        )
    return records


def summarize_records(records: list[dict[str, float]]) -> dict[str, float]:
    spatial = [r for r in records if r["study"] == "spatial"]
    temporal = [r for r in records if r["study"] == "temporal"]

    h_values = np.array([r["h"] for r in spatial], dtype=float)
    spatial_l2 = np.array([r["L2_error"] for r in spatial], dtype=float)
    spatial_h1 = np.array([r["H1_error"] for r in spatial], dtype=float)

    dt_values = np.array([r["dt"] for r in temporal], dtype=float)
    temporal_l2 = np.array([r["L2_error"] for r in temporal], dtype=float)
    return {
        "spatial_L2_order": float(abs(estimate_convergence_order(h_values, spatial_l2))),
        "spatial_H1_order": float(abs(estimate_convergence_order(h_values, spatial_h1))),
        "temporal_L2_order": float(abs(estimate_convergence_order(dt_values, temporal_l2, from_start=True))),
        "spatial_L2_finest": float(spatial_l2[-1]),
        "spatial_H1_finest": float(spatial_h1[-1]),
        "temporal_L2_finest": float(temporal_l2[-1]),
    }


def write_metrics(records: list[dict[str, float]], summary: dict[str, float]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["study", "n", "h", "dt", "L2_error", "H1_error"])
        writer.writeheader()
        writer.writerows(records)

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def create_figure(records: list[dict[str, float]], summary: dict[str, float]) -> None:
    apply_style()

    spatial = [r for r in records if r["study"] == "spatial"]
    temporal = [r for r in records if r["study"] == "temporal"]

    h_values = np.array([r["h"] for r in spatial], dtype=float)
    spatial_l2 = np.array([r["L2_error"] for r in spatial], dtype=float)
    spatial_h1 = np.array([r["H1_error"] for r in spatial], dtype=float)

    dt_values = np.array([r["dt"] for r in temporal], dtype=float)
    temporal_l2 = np.array([r["L2_error"] for r in temporal], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_COMPACT_TWO_PANEL)

    axes[0].loglog(
        h_values,
        spatial_l2,
        marker="o",
        color=COLORS["blue"],
        linewidth=1.4,
        label=rf"spatial $L^2$ (p={summary['spatial_L2_order']:.2f})",
    )
    axes[0].loglog(
        h_values,
        spatial_h1,
        marker="s",
        color=COLORS["orange"],
        linewidth=1.4,
        label=rf"spatial $H^1$ (p={summary['spatial_H1_order']:.2f})",
    )
    ref_scale = float(np.median(spatial_l2))
    add_reference_line(axes[0], (float(np.min(h_values)), float(np.max(h_values))), 2.0, ref_scale, r"$O(h^2)$")
    add_reference_line(
        axes[0],
        (float(np.min(h_values)), float(np.max(h_values))),
        1.0,
        ref_scale,
        r"$O(h)$",
        linestyle=":",
    )
    setup_axis_style(
        axes[0],
        r"Mesh size $h$",
        "Error",
        "(a) Spatial convergence",
        loglog=True,
    )

    axes[1].loglog(
        dt_values,
        temporal_l2,
        marker="o",
        color=COLORS["blue"],
        linewidth=1.4,
        label=rf"temporal $L^2$ (p={summary['temporal_L2_order']:.2f})",
    )
    ref_scale = float(np.median(temporal_l2))
    add_reference_line(
        axes[1],
        (float(np.min(dt_values)), float(np.max(dt_values))),
        1.0,
        ref_scale,
        r"$O(\Delta t)$",
    )
    setup_axis_style(
        axes[1],
        r"Timestep $\Delta t$",
        "Error",
        "(b) Temporal convergence",
        loglog=True,
    )
    tick_values = np.sort(dt_values)
    axes[1].xaxis.set_major_locator(mticker.FixedLocator(tick_values))
    axes[1].xaxis.set_major_formatter(mticker.FixedFormatter(["0.125", "0.25", "0.5", "1"]))
    axes[1].xaxis.set_minor_locator(mticker.NullLocator())
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)

    fig.subplots_adjust(bottom=0.3, wspace=0.32)
    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    save_figure(fig, RESULTS_DIR / "manufactured_stimulus.png", dpi=PUBLICATION_DPI, close=False)
    save_manuscript_figure(fig, "manufactured_stimulus", dpi=PUBLICATION_DPI, close=True)


def main() -> None:
    records = spatial_convergence() + temporal_convergence()
    summary = summarize_records(records)
    write_metrics(records, summary)
    create_figure(records, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
