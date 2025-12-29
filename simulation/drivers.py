"""GaitDriver: multi-load mechanics with cycle-weighted power-mean SED."""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.solvers import MechanicsSolver
from simulation.logger import get_logger
from simulation.stats import SweepStats
from simulation.utils import assign, get_owned_size, symm

if TYPE_CHECKING:
    from femur.loader import LoadingCase, Loader


class StimulusCalculator:
    """Computes SED (ψ) and fabric tensor (Q) from mechanics solution."""

    def __init__(self, mech: MechanicsSolver):
        """Initialize with mechanics solver to access state fields."""
        self.mech = mech
        self.mesh = mech.u.function_space.mesh
        self.gdim = self.mesh.geometry.dim

        # Function spaces for output
        self.V_psi = fem.functionspace(self.mesh, ("DG", 0))
        self.V_Q = fem.functionspace(self.mesh, ("DG", 0, (self.gdim, self.gdim)))

        # Pre-compile expressions
        self._sed_expr = self._build_sed_expression()
        self._Q_expr = self._build_Q_expression()

    def compute_sed(self, target: fem.Function) -> None:
        """Compute SED (psi) and interpolate into target function."""
        target.interpolate(self._sed_expr)

    def compute_Q(self, target: fem.Function) -> None:
        """Compute fabric tensor (Q) and interpolate into target function."""
        target.interpolate(self._Q_expr)

    def _build_sed_expression(self) -> fem.Expression:
        """Build UFL expression for psi = 0.5 * sigma : epsilon (clamped >= 0)."""
        u = self.mech.u
        rho = self.mech.rho
        L = self.mech.L
        sig = self.mech.sigma(u, rho, L)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)

    def _build_Q_expression(self) -> fem.Expression:
        """Build UFL expression for Q_case = sym(dev(sigma)) (DG0 tensor 3×3)."""
        u = self.mech.u
        rho = self.mech.rho
        L = self.mech.L

        sig = self.mech.sigma(u, rho, L)
        I = ufl.Identity(self.gdim)
        sig_dev = sig - (ufl.tr(sig)/3.0)*I
        Q = ufl.dot(sig_dev, ufl.transpose(sig_dev))
        Q_sym = symm(Q)

        return fem.Expression(Q_sym, self.V_Q.element.interpolation_points)


class GaitDriver:
    """Multi-load mechanics: solves each case, accumulates cycle-weighted SED and Q̄."""

    def __init__(
        self,
        mech: MechanicsSolver,
        config: Config,
        loader: "Loader",
        loading_cases: List["LoadingCase"],
    ):
        """Bind mechanics solver, loader, and list of loading cases."""
        self.mech = mech
        self.cfg = config
        self.loader = loader
        self.loading_cases = loading_cases

        mesh = self.mech.u.function_space.mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.logger = get_logger(self.comm, name="Driver", log_file=self.cfg.log_file)

        # Physics calculator
        self.calculator = StimulusCalculator(self.mech)

        # Accumulated SED (DG0 - cellwise values).
        self.V_psi = self.calculator.V_psi
        self.psi = fem.Function(self.V_psi, name="psi")
        
        # Temporary per-case SED (DG0).
        self._psi_temp = fem.Function(self.V_psi, name="psi_temp")

        # Weighted average of Q_case = sigma*sigma^T (DG0 tensor 3×3).
        self.V_Q = self.calculator.V_Q
        self.Qbar = fem.Function(self.V_Q, name="Qbar")
        self._Q_temp = fem.Function(self.V_Q, name="Q_temp")

        # Precompute all loading cases (expensive interpolation done once)
        self.loader.precompute_loading_cases(self.loading_cases)

        self.logger.debug(f"GaitDriver initialized with {len(loading_cases)} loading case(s), loads precomputed")

    def setup(self) -> None:
        """Initialize mechanics solver."""
        self.mech.setup()

    def assemble_lhs(self) -> None:
        """Reassemble stiffness matrix K(rho)."""
        self.mech.assemble_lhs()

    def destroy(self) -> None:
        """Release solver resources."""
        self.mech.destroy()

    def _update_snapshots(self) -> Tuple[int, int, float, Dict[str, Any]]:
        """Solve all enabled loading cases and update averaged `psi` and `Qbar`.

        Returns:
            Tuple of (n_enabled_cases, avg_ksp_iters, total_solve_time, extra_stats_from_last_solve).
        """
        # Zero out accumulated fields (owned DOFs only; single scatter at the end)
        assign(self.psi, 0.0, scatter=False)
        assign(self.Qbar, 0.0, scatter=False)
        n_owned_psi = get_owned_size(self.psi)
        n_owned_Q = get_owned_size(self.Qbar)
        p = float(self.cfg.stimulus.stimulus_power_p)
        sum_cycles = 0.0
        tiny = 1e-30

        total_iters = 0
        total_time = 0.0
        last_extra: Dict[str, Any] = {}
        n_enabled_cases = 0

        for case in self.loading_cases:
            day_cycles = float(case.day_cycles)
            if day_cycles <= 0.0:
                # Disabled case (day_cycles == 0) → skip solve.
                continue
            n_enabled_cases += 1
            sum_cycles += day_cycles

            # Load cached traction (cheap, no geometry work)
            self.loader.set_loading_case(case.name)

            # Reassemble RHS (traction is referenced in the form) in solve
            stats = self.mech.solve()
            total_iters += stats.ksp_iters
            total_time += stats.solve_time
            last_extra = stats.extra

            # Compute SED + Q_case for this case
            self.calculator.compute_sed(self._psi_temp)
            self.calculator.compute_Q(self._Q_temp)

            # Accumulate stimulus as a cycle-weighted power mean:
            #   psi = ( sum_i day_cycles_i * psi_i^p / sum_i day_cycles_i )^(1/p)
            #   p = 1 -> cycle-weighted mean; p > 1 -> peak-biased
            contrib_p = self._psi_temp.x.array[:n_owned_psi] ** p
            self.psi.x.array[:n_owned_psi] += day_cycles * contrib_p

            # Accumulate Qbar as a plain weighted average (no power-mean).
            self.Qbar.x.array[:n_owned_Q] += day_cycles * self._Q_temp.x.array[:n_owned_Q]

        # Finalize: psi = ( sum(day_cycles * psi^p) / sum(day_cycles) )^(1/p)
        if sum_cycles <= tiny:
            raise ValueError("Total day_cycles is zero; cannot compute cycle-weighted stimulus.")
        owned = self.psi.x.array[:n_owned_psi]
        owned[:] = (owned / sum_cycles) ** (1.0 / p)

        # Finalize Qbar average and scatter.
        self.Qbar.x.array[:n_owned_Q] *= 1.0 / max(sum_cycles, tiny)

        self.psi.x.scatter_forward()
        self.Qbar.x.scatter_forward()

        avg_ksp_iters = int(round(total_iters / n_enabled_cases)) if n_enabled_cases > 0 else 0

        extra = dict(last_extra)
        extra["n_load_cases"] = n_enabled_cases
        extra["ksp_iters_total"] = total_iters
        extra["ksp_iters_avg"] = float(total_iters) / float(n_enabled_cases) if n_enabled_cases > 0 else 0.0

        return n_enabled_cases, avg_ksp_iters, total_time, extra

    def stimulus_field(self) -> fem.Function:
        """Return the averaged `psi` function (DG0 field)."""
        return self.psi

    def Qbar_field(self) -> fem.Function:
        """Return averaged `Qbar` (DG0 tensor field)."""
        return self.Qbar

    # -------------------------------------------------------------------------
    # CouplingBlock protocol - GaitDriver produces derived quantities, no state
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        # The driver does not own coupled state variables (rho/S). It produces psi/Qbar.
        return ()

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        """Return psi and Qbar for VTX output."""
        return (self.psi, self.Qbar)

    def post_step_update(self) -> None:
        """No post-step processing needed for GaitDriver."""
        pass

    def sweep(self) -> SweepStats:
        """One Gauss-Seidel sweep for the mechanics block.

        - Reassemble stiffness K(rho)
        - Solve each enabled loading case
        - Update averaged psi (DG0)
        """
        self.assemble_lhs()
        _n_cases, avg_iters, total_time, extra = self._update_snapshots()
        return SweepStats(
            label="mech",
            ksp_iters=avg_iters,
            ksp_reason=int(self.mech.last_reason),
            solve_time=total_time,
            extra=extra,
        )
