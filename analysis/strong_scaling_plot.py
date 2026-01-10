"""Plot scaling results produced by `run_weak_scaling.py`.

Reads:
  results/scaling_stiff/scaling.csv

Writes:
  manuscript/images/strong_scaling.png
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
        raise FileNotFoundError(f"Missing scaling CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Scaling CSV is empty: {csv_path}")
    return df


def main() -> None:
    apply_style()

    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "results" / "scaling_stiff" / "scaling.csv"
    df = _load_csv(csv_path)

    # Use 't_avg_step_s'
    required = {"ranks", "t_avg_step_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    # Aggregate repeats: median time per rank
    df["ranks"] = df["ranks"].astype(int)
    grouped = (
        df.groupby("ranks", as_index=False)
        .agg(t_solve_s=("t_avg_step_s", "median"))
        .sort_values("ranks")
    )

    ranks = grouped["ranks"].to_numpy(dtype=int)
    times = grouped["t_solve_s"].to_numpy(dtype=float)

    if ranks.size == 0:
        print("No data found.")
        return

    # Base values (rank 1)
    t1 = times[0]
    r1 = ranks[0] # Should be 1, but handle generic base
    
    # Ideal time for strong scaling: T(N) = T(1) * (1/N)
    ideal_times = t1 * (float(r1) / ranks)
    
    # Efficiency: E(N) = T(1) / (N * T(N))
    efficiency = (t1 / (ranks / float(r1))) / times

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE_COLUMN)

    # Left axis: Time
    ax.loglog(ranks, times, marker="o", color=COLORS["blue"], label="Measured time", linewidth=1.5)
    ax.loglog(ranks, ideal_times, linestyle="--", color="gray", label="Ideal strong scaling", linewidth=1.0)
    
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda val, _: f"{int(val)}"))
    ax.minorticks_off()

    setup_axis_style(
        ax,
        xlabel="MPI ranks",
        ylabel="Wall time per step [s]",
        title="Strong Scaling (Stiff Femur)",
        loglog=True,
    )

    # Right axis: Efficiency
    ax2 = ax.twinx()
    ax2.plot(ranks, efficiency, marker="s", color=COLORS["orange"], linestyle="-.", label="Efficiency")
    ax2.set_ylabel("Parallel Efficiency")
    ax2.set_ylim(0.0, 1.2)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.5)

    # Unified legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    save_manuscript_figure(fig, "strong_scaling", dpi=PUBLICATION_DPI)
    print("Generated manuscript/images/strong_scaling.png")


if __name__ == "__main__":
    main()
