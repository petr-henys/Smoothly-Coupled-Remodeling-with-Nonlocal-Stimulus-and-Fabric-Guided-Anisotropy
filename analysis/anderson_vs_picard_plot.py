"""Anderson vs Picard comparison plot with explicit *failure* markers.

This script is meant to make the conclusion unambiguous:

  - For each (dt, method) we compute whether each timestep converged
    (final residual <= tolerance).
  - We highlight timesteps where Picard hits `max_subiters` without
    reaching tolerance.

It reads the sweep output produced by `run_anderson_sweep_strict.py`.

Usage:
    python anderson_vs_picard_plot_strict.py

Optional:
    python anderson_vs_picard_plot_strict.py results/anderson_sweep_strict
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocessor import SweepLoader


def _get_cfg_value(cfg: dict, *paths: str, default=None):
    """Try multiple key paths against either nested or flat config dicts."""
    # 1) Nested paths like ("solver", "coupling_tol")
    for path in paths:
        cur = cfg
        ok = True
        for k in path.split("."):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return cur

    # 2) Flat keys like "solver.coupling_tol"
    for path in paths:
        if path in cfg:
            return cfg[path]

    return default


def _step_convergence(subiters: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Return per-step final residual, iterations and convergence flag."""
    if subiters.empty:
        return pd.DataFrame(columns=["step", "final_res", "iters", "converged"])

    g = subiters.sort_values(["step", "iter"]).groupby("step", as_index=False)
    last = g.tail(1)[["step", "iter", "proj_res"]].rename(
        columns={"iter": "iters", "proj_res": "final_res"}
    )
    last["converged"] = last["final_res"].astype(float) <= float(tol)
    return last


def main() -> None:
    sweep_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/anderson_sweep_strict")

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if comm.rank != 0:
        return

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {sweep_dir}")

    sweep = SweepLoader(str(sweep_dir), comm)
    summary = sweep.get_summary()
    if summary.empty:
        raise RuntimeError("No runs found in sweep summary.")

    # We expect at least these columns from the strict sweep
    for col in ("dt_days", "accel_type", "output_dir"):
        if col not in summary.columns:
            raise ValueError(f"Missing '{col}' in sweep_summary.csv. Columns: {list(summary.columns)}")

    # Build run objects grouped by dt + accel
    runs: dict[tuple[float, str], dict] = {}
    for _, row in summary.iterrows():
        key = (float(row["dt_days"]), str(row["accel_type"]))
        if key in runs:
            # Multiple runs per bucket: keep the first (strict sweep shouldn't do this).
            continue
        loader = sweep.get_loader(str(row["output_dir"]))
        cfg = loader.get_config()
        tol = float(_get_cfg_value(cfg, "solver.coupling_tol", "coupling_tol", default=1e-6))
        max_subiters = int(_get_cfg_value(cfg, "solver.max_subiters", "max_subiters", default=0))
        subiters = loader.get_subiterations_metrics()
        steps = loader.get_steps_metrics()

        per_step = _step_convergence(subiters, tol)
        # Mark hard failures: reached max_subiters but still not converged
        if max_subiters > 0 and not per_step.empty:
            per_step["hit_max"] = (per_step["iters"].astype(int) >= max_subiters) & (~per_step["converged"])
        else:
            per_step["hit_max"] = False

        runs[key] = {
            "tol": tol,
            "max_subiters": max_subiters,
            "subiters": subiters,
            "steps": steps,
            "per_step": per_step,
        }

    dt_values = sorted({k[0] for k in runs.keys()})
    if not dt_values:
        raise RuntimeError("No dt values found.")

    # Console summary (this is the quickest sanity check)
    print("\n=== ANDERSON vs PICARD: convergence summary ===")
    for dt in dt_values:
        for accel in ("picard", "anderson"):
            key = (dt, accel)
            if key not in runs:
                print(f"dt={dt:>6.1f}  {accel:<8}: MISSING")
                continue
            per = runs[key]["per_step"]
            if per.empty:
                print(f"dt={dt:>6.1f}  {accel:<8}: no data")
                continue
            n = len(per)
            conv = int(per["converged"].sum())
            hit_max = int(per["hit_max"].sum())
            med_it = float(np.median(per["iters"].values))
            p90_it = float(np.percentile(per["iters"].values, 90))
            worst_res = float(np.max(per["final_res"].values))
            tol = runs[key]["tol"]
            ms = runs[key]["max_subiters"]
            print(
                f"dt={dt:>6.1f}  {accel:<8}: conv {conv:>3}/{n:<3}  "
                f"hit_max {hit_max:>3}  it_med {med_it:>6.1f}  it_p90 {p90_it:>6.1f}  "
                f"worst_res {worst_res:.2e}  (tol={tol:.1e}, max_subiters={ms})"
            )

    # ----------------- plotting -----------------
    n_dt = len(dt_values)
    fig, axes = plt.subplots(2, n_dt, figsize=(min(3.6 * n_dt, 12.0), 6.0), squeeze=False)

    for j, dt in enumerate(dt_values):
        ax1 = axes[0, j]
        ax2 = axes[1, j]

        for accel in ("picard", "anderson"):
            key = (dt, accel)
            if key not in runs:
                continue
            sub = runs[key]["subiters"]
            per = runs[key]["per_step"]
            steps = runs[key]["steps"]

            # Convergence curves: overlay all timesteps
            for step in sorted(sub["step"].unique()):
                sd = sub[sub["step"] == step].sort_values("iter")
                ax1.plot(sd["iter"].values, sd["proj_res"].values, alpha=0.25, linestyle="-" if accel == "picard" else "--")

            # Subiteration counts, with failure markers
            ax2.plot(
                steps["step"].values,
                steps["num_subiters"].values,
                marker="o" if accel == "picard" else "s",
                linestyle="-" if accel == "picard" else "--",
                linewidth=1.5,
                label=accel,
            )

            # Overlay hard failures as 'x'
            if "hit_max" in per.columns and per["hit_max"].any():
                failed_steps = per.loc[per["hit_max"], "step"].values
                # Map to y values using steps dataframe
                y = steps.set_index("step").loc[failed_steps, "num_subiters"].values
                ax2.scatter(failed_steps, y, marker="x", s=40)

        ax1.set_title(f"Δt = {dt:.0f} days")
        ax1.set_yscale("log")
        ax1.set_xlabel("Subiteration")
        if j == 0:
            ax1.set_ylabel("Residual (proj_res)")
        ax1.grid(True, which="both", alpha=0.2)
        ax1.set_ylim(1e-10, 1e2)

        ax2.set_xlabel("Timestep")
        if j == 0:
            ax2.set_ylabel("Subiterations")
        ax2.grid(True, alpha=0.2)

    axes[1, 0].legend(loc="upper right")

    out = sweep_dir / "anderson_vs_picard_strict.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
