"""GaitDriver: mechanics + averaged SED stimulus over multiple loading cases."""

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING

from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, symm
from simulation.logger import get_logger
from simulation.utils import assign, get_owned_size

if TYPE_CHECKING:
    from simulation.loader import LoadingCase, Loader


class GaitDriver:
    """Mechanics driver that computes accumulated SED stimulus.

    For each loading case: set cached traction, solve mechanics, compute
    element-wise psi = 0.5 * sigma : epsilon (DG0), and accumulate
    weighted by day_cycles (number of loading cycles per day).
    """

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

        # Accumulated SED (DG0 - cellwise values).
        self.V_psi = fem.functionspace(mesh, ("DG", 0))
        self.psi = fem.Function(self.V_psi, name="psi")
        
        # Temporary per-case SED (DG0).
        self._psi_temp = fem.Function(self.V_psi, name="psi_temp")

        # Weighted average of Q_case = sigma*sigma^T (DG0 tensor 3×3).
        gdim = mesh.geometry.dim
        self.V_Q = fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))
        self.Qbar = fem.Function(self.V_Q, name="Qbar")
        self._Q_temp = fem.Function(self.V_Q, name="Q_temp")

        # Pre-compile SED expression (reused every update)
        self._sed_expr = self._build_sed_expression()
        self._Q_expr = self._build_Q_expression()

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

    def update_snapshots(self) -> Dict:
        """Solve all enabled loading cases and update averaged `psi` and `Qbar` (collective)."""
        # Zero out accumulated fields (owned DOFs only; single scatter at the end)
        assign(self.psi, 0.0, scatter=False)
        assign(self.Qbar, 0.0, scatter=False)
        n_owned_psi = get_owned_size(self.psi)
        n_owned_Q = get_owned_size(self.Qbar)
        p = float(self.cfg.stimulus_power_p)
        sum_cycles = 0.0
        tiny = 1e-30

        for case in self.loading_cases:
            day_cycles = float(case.day_cycles)
            if day_cycles <= 0.0:
                # Disabled case (day_cycles == 0) → skip solve.
                continue
            sum_cycles += day_cycles
            
            # Load cached traction (cheap, no geometry work)
            self.loader.set_loading_case(case.name)
            
            # Reassemble RHS (traction is referenced in the form)
            self.mech.assemble_rhs()
            self.mech.solve()
            
            # Compute SED + Q_case for this case
            self._psi_temp.interpolate(self._sed_expr)
            self._Q_temp.interpolate(self._Q_expr)

            # Accumulate stimulus weighted by day_cycles:
            #   p = 1 -> sum of day_cycles * psi
            #   p > 1 → peak-biased accumulation
            contrib_p = self._psi_temp.x.array[:n_owned_psi] ** p
            self.psi.x.array[:n_owned_psi] += day_cycles * contrib_p

            # Accumulate Qbar as a plain weighted average (no power-mean).
            self.Qbar.x.array[:n_owned_Q] += day_cycles * self._Q_temp.x.array[:n_owned_Q]

        # Finalize: take (.)^(1/p) after accumulation of sum(day_cycles * psi^p)
        owned = self.psi.x.array[:n_owned_psi]
        owned[:] = owned ** (1.0 / p)

        # Finalize Qbar average and scatter.
        self.Qbar.x.array[:n_owned_Q] *= 1.0 / max(sum_cycles, tiny)

        self.psi.x.scatter_forward()
        self.Qbar.x.scatter_forward()
        return {}

    def stimulus_field(self) -> fem.Function:
        """Return the averaged `psi` function (DG0 field)."""
        return self.psi

    def Qbar_field(self) -> fem.Function:
        """Return averaged `Qbar` (DG0 tensor field)."""
        return self.Qbar


    @property
    def state_fields(self):
        # The driver does not own coupled state variables (rho/S). It produces psi.
        return ()

    def sweep(self) -> Dict:
        """One Gauss–Seidel sweep for the mechanics block.

        - Reassemble stiffness K(rho)
        - Solve each enabled loading case
        - Update averaged psi (DG0)
        """
        self.assemble_lhs()
        self.update_snapshots()
        return {"label": "mech", "reason": int(self.mech.last_reason)}

    def _build_sed_expression(self) -> fem.Expression:
        """Build UFL expression for psi = 0.5 * sigma : epsilon (clamped >= 0)."""
        u = self.mech.u
        rho = self.mech.rho
        
        L = getattr(self.mech, "L", None)
        sig = self.mech.sigma(u, rho, L)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)

    def _build_Q_expression(self) -> fem.Expression:
        """Build UFL expression for Q_case = sym(sigma*sigma^T) (DG0 tensor 3×3)."""
        u = self.mech.u
        rho = self.mech.rho
        L = getattr(self.mech, "L", None)

        sig = self.mech.sigma(u, rho, L)
        Q = ufl.dot(sig, ufl.transpose(sig))
        Q_sym = symm(Q)
        return fem.Expression(Q_sym, self.V_Q.element.interpolation_points)
