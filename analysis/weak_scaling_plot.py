"""Plot weak-scaling results produced by `run_weak_scaling.py`.

Reads:
  results/weak_scaling/weak_scaling.csv

Writes:
  manuscript/images/weak_scaling.png
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from analysis.plot_utils import (
    FIGSIZE_SINGLE_COLUMN,
    PUBLICATION_DPI,
    COLORS,
    apply_style,
    save_manuscript_figure,
    setup_axis_style,
)


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing weak-scaling CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Weak-scaling CSV is empty: {csv_path}")
    return df


def main() -> None:
    apply_style()

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "results" / "weak_scaling" / "weak_scaling.csv"
    df = _load_csv(csv_path)

    required = {"problem", "ranks", "t_solve_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Aggregate repeats: median time per rank and problem.
    df["ranks"] = df["ranks"].astype(int)
    grouped = (
        df.groupby(["problem", "ranks"], as_index=False)
        .agg(t_solve_s=("t_solve_s", "median"))
        .sort_values(["problem", "ranks"])
    )

    problems = list(grouped["problem"].unique())
    ncols = 2 if len(problems) > 1 else 1
    fig, axes = plt.subplots(1, ncols, figsize=FIGSIZE_SINGLE_COLUMN if ncols == 1 else (7.5, 2.8))
    axes = np.atleast_1d(axes)

    for ax, problem in zip(axes, problems, strict=False):
        sub = grouped[grouped["problem"] == problem].sort_values("ranks")
        x = sub["ranks"].to_numpy(dtype=int)
        t = sub["t_solve_s"].to_numpy(dtype=float)

        if x.size == 0:
            continue

        # Normalize by the smallest rank count in the series (weak-scaling efficiency).
        t0 = float(t[0])
        eff = t0 / t

        ax2 = ax.twinx()
        ax.plot(x, t, marker="o", color=COLORS["blue"], label="Solve time")
        ax2.plot(x, eff, marker="s", color=COLORS["orange"], linestyle="--", label="Efficiency")

        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(FuncFormatter(lambda val, _: f"{int(val)}"))

        setup_axis_style(
            ax,
            xlabel="MPI ranks",
            ylabel="Wall time per solve [s]",
            title=f"Weak scaling ({problem})",
            loglog=False,
        )
        ax2.set_ylabel("Weak-scaling efficiency")
        ax2.set_ylim(0.0, 1.05)

        # Unified legend (combine both axes).
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")

    fig.tight_layout()
    save_manuscript_figure(fig, "weak_scaling", dpi=PUBLICATION_DPI)


if __name__ == "__main__":
    main()
