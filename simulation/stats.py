"""Structured statistics for solver blocks.

Defines the SweepStats dataclass that all CouplingBlock.sweep() methods must return.
Mandatory stats: label, ksp_iters, ksp_reason, solve_time.
Physics-specific stats go in the `extra` dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class SweepStats:
    """Statistics returned by every CouplingBlock.sweep() call.

    Mandatory fields capture common solver metrics. Physics-specific
    data (e.g., anisotropy measures, field bounds) goes in `extra`.

    Attributes:
        label: Short block identifier (e.g., "mech", "fab", "stim", "dens").
        ksp_iters: Total KSP iterations for this sweep.
        ksp_reason: PETSc KSP convergence reason (>=0 means converged).
        solve_time: Wall-clock time for the solve [seconds].
        extra: Dict of physics-specific stats (e.g., {"a_min": 0.5, "a_max": 2.1}).
    """

    label: str
    ksp_iters: int
    ksp_reason: int
    solve_time: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        """True if KSP converged (reason >= 0)."""
        return self.ksp_reason >= 0

    def format_short(self, width: int = 4) -> str:
        """Compact block summary: 'mech  81it  0.57s'."""
        return f"{self.label:<{width}}  {self.ksp_iters:>3}it  {self.solve_time:.3f}s"

    def format_extra(self) -> str:
        """Format physics-specific extras as ranges: 'a=[0.56, 1.82] p2=[0.01, 0.09]'."""
        if not self.extra:
            return ""
        # Group min/max pairs into ranges
        pairs: dict[str, list[float]] = {}
        standalone: list[str] = []
        for k, v in self.extra.items():
            # Check for _min/_max pattern
            if k.endswith("_min"):
                base = k[:-4]
                pairs.setdefault(base, [None, None])[0] = v
            elif k.endswith("_max"):
                base = k[:-4]
                pairs.setdefault(base, [None, None])[1] = v
            else:
                if isinstance(v, float):
                    standalone.append(f"{k}={v:.2g}")
                else:
                    standalone.append(f"{k}={v}")
        
        parts = []
        for base, (vmin, vmax) in pairs.items():
            if vmin is not None and vmax is not None:
                parts.append(f"{base}=[{vmin:.2g}, {vmax:.2g}]")
            elif vmin is not None:
                parts.append(f"{base}_min={vmin:.2g}")
            elif vmax is not None:
                parts.append(f"{base}_max={vmax:.2g}")
        parts.extend(standalone)
        return "  ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Full dict for JSON/metrics storage."""
        return {
            "label": self.label,
            "ksp_iters": self.ksp_iters,
            "ksp_reason": self.ksp_reason,
            "solve_time": self.solve_time,
            **self.extra,
        }


@dataclass
class StepSummary:
    """Aggregated statistics for one complete timestep (all Picard iterations)."""

    picard_iters: int
    total_ksp_iters: Dict[str, int]
    total_solve_time: Dict[str, float]
    max_cond: float
    max_hist: int
    rejections: int
    restarts: int
    mem_peak_mb: float

    @classmethod
    def from_iteration_records(cls, records: List[Dict[str, Any]]) -> "StepSummary":
        """Aggregate from per-iteration metric records."""
        if not records:
            return cls(
                picard_iters=0,
                total_ksp_iters={},
                total_solve_time={},
                max_cond=0.0,
                max_hist=0,
                rejections=0,
                restarts=0,
                mem_peak_mb=0.0,
            )

        ksp_iters: Dict[str, int] = {}
        solve_time: Dict[str, float] = {}

        for rec in records:
            for stats in rec.get("block_stats", []):
                lbl = stats.label
                ksp_iters[lbl] = ksp_iters.get(lbl, 0) + stats.ksp_iters
                solve_time[lbl] = solve_time.get(lbl, 0.0) + stats.solve_time

        return cls(
            picard_iters=len(records),
            total_ksp_iters=ksp_iters,
            total_solve_time=solve_time,
            max_cond=max((r.get("condH", 0.0) for r in records), default=0.0),
            max_hist=max((r.get("aa_hist", 0) for r in records), default=0),
            rejections=sum(1 for r in records if not r.get("aa_accepted", True)),
            restarts=sum(1 for r in records if r.get("aa_restart")),
            mem_peak_mb=max((r.get("mem_mb", 0.0) for r in records), default=0.0),
        )

    def format_summary(self, step_index: int = 0, sim_time: float = 0.0) -> str:
        """Format as ASCII table for INFO-level timestep summary log.

        Args:
            step_index: Current timestep index (1-based).
            sim_time: Current simulation time [days].

        Example output:
        ┌───────────────────────────────────────────────────────────────┐
        │ Step 5 (t=0.50 days): 17 Picard, 13.2s total                  │
        ├────────┬─────────┬─────────┬─────────┬───────────────────────┤
        │ Block  │ KSP its │  Time   │    %    │         Rate          │
        ├────────┼─────────┼─────────┼─────────┼───────────────────────┤
        │  mech  │    1394 │   9.42s │   87.3% │            148 it/s   │
        │  fab   │      85 │   1.21s │   11.2% │             70 it/s   │
        │  stim  │      89 │   0.05s │    0.5% │           1780 it/s   │
        │  dens  │      56 │   0.05s │    0.5% │           1120 it/s   │
        ├────────┴─────────┴─────────┴─────────┴───────────────────────┤
        │ Anderson: m_max=6  cond_max=1.8e+04  rej=0  rst=0  mem=428MB │
        └───────────────────────────────────────────────────────────────┘
        """
        if not self.total_ksp_iters:
            return f"\nStep {step_index} (t={sim_time:.2f}): 0 Picard iters (no blocks)"

        # Compute totals
        total_time = sum(self.total_solve_time.values())
        labels = sorted(self.total_ksp_iters.keys())

        # Column widths (content only, excluding │ separators)
        c1, c2, c3, c4, c5 = 8, 9, 9, 9, 23  # Block, KSP, Time, %, Rate
        w = c1 + c2 + c3 + c4 + c5 + 6  # +6 for │ separators

        # Build table (start with newline so [FixedPoint] prefix doesn't break alignment)
        lines = [""]  # empty first line after [FixedPoint] prefix
        lines.append(f"┌{'─' * (w - 2)}┐")
        header = f"Step {step_index} (t={sim_time:.2f} days): {self.picard_iters} Picard, {total_time:.1f}s total"
        lines.append(f"│ {header:<{w - 4}} │")
        lines.append(f"├{'─' * c1}┬{'─' * c2}┬{'─' * c3}┬{'─' * c4}┬{'─' * c5}┤")
        lines.append(f"│{'Block':^{c1}}│{'KSP its':^{c2}}│{'Time':^{c3}}│{'%':^{c4}}│{'Rate':^{c5}}│")
        lines.append(f"├{'─' * c1}┼{'─' * c2}┼{'─' * c3}┼{'─' * c4}┼{'─' * c5}┤")

        for lbl in labels:
            iters = self.total_ksp_iters[lbl]
            t = self.total_solve_time[lbl]
            pct = 100.0 * t / total_time if total_time > 0 else 0.0
            rate = iters / t if t > 0 else 0.0
            rate_str = f"{rate:.0f} it/s"
            lines.append(
                f"│{lbl:^{c1}}│{iters:>{c2 - 1}} │{t:>{c3 - 2}.2f}s │{pct:>{c4 - 2}.1f}% │{rate_str:>{c5 - 1}} │"
            )

        lines.append(f"├{'─' * c1}┴{'─' * c2}┴{'─' * c3}┴{'─' * c4}┴{'─' * c5}┤")
        aa_line = (
            f"Anderson: m_max={self.max_hist}  cond_max={self.max_cond:.1e}  "
            f"rej={self.rejections}  rst={self.restarts}  mem={self.mem_peak_mb:.0f}MB"
        )
        lines.append(f"│ {aa_line:<{w - 4}} │")
        lines.append(f"└{'─' * (w - 2)}┘")

        return "\n".join(lines)
