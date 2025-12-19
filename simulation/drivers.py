"""GaitDriver: mechanics + averaged SED stimulus over multiple loading cases."""

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING, Optional

from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.logger import get_logger
from simulation.utils import assign, field_stats, get_owned_size

if TYPE_CHECKING:
    from simulation.loader import LoadingCase, Loader
    from simulation.storage import UnifiedStorage


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
        storage: Optional["UnifiedStorage"] = None,
    ):
        """Bind mechanics solver, loader, and list of loading cases."""
        self.mech = mech
        self.cfg = config
        self.loader = loader
        self.loading_cases = loading_cases
        self.storage = storage

        mesh = self.mech.u.function_space.mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.logger = get_logger(self.comm, name="Driver", log_file=self.cfg.log_file)

        # Stimulus field (DG0 - element-wise constant)
        self.V_psi = fem.functionspace(mesh, ("DG", 0))
        self.psi = fem.Function(self.V_psi, name="Stimulus_SED")
        
        # Temporary per-case SED (DG0)
        self._psi_temp = fem.Function(self.V_psi, name="Stimulus_SED_temp")
        
        # Storage for per-case psi fields
        self.case_psi_functions: Dict[str, fem.Function] = {}
        if self.storage:
            all_psi_funcs = []
            for case in self.loading_cases:
                func = fem.Function(self.V_psi, name=f"psi_{case.name}")
                self.case_psi_functions[case.name] = func
                all_psi_funcs.append(func)
            
            # Also save the total accumulated stimulus
            all_psi_funcs.append(self.psi)
            
            if all_psi_funcs:
                self.storage.fields.register("psi_cases", all_psi_funcs, filename="psi_cases.bp")

        # Pre-compile SED expression (reused every update)
        self._sed_expr = self._build_sed_expression()

        # Precompute all loading cases (expensive interpolation done once)
        self.loader.precompute_loading_cases(self.loading_cases)

        self.logger.debug(f"GaitDriver initialized with {len(loading_cases)} loading case(s), loads precomputed")

    def setup(self) -> None:
        """Initialize mechanics solver."""
        self.mech.setup()

    def destroy(self) -> None:
        """Release solver resources."""
        self.mech.destroy()

    def update_stiffness(self) -> None:
        """Reassemble stiffness matrix K(rho)."""
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """Solve all enabled phases and update averaged `psi` (collective).

        Returns `phase_iters`, `phase_times`, and `total_time` (MPI max).
        """
        start = MPI.Wtime()
        
        phase_iters = []
        phase_times = []

        # Zero out accumulated psi (owned DOFs only; single scatter at the end)
        assign(self.psi, 0.0, scatter=False)
        n_owned_psi = get_owned_size(self.psi)
        p = float(self.cfg.stimulus_power_p)

        for case in self.loading_cases:
            day_cycles = float(case.day_cycles)
            if day_cycles <= 0.0:
                # Disabled case (day_cycles == 0) → skip solve.
                continue

            case_start = MPI.Wtime()
            
            # Load cached traction (cheap, no geometry work)
            self.loader.set_loading_case(case.name)
            
            # Reassemble RHS (traction is referenced in the form)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            
            # Compute SED for this case
            self._psi_temp.interpolate(self._sed_expr)
            
            # Store per-case psi if storage is enabled
            if self.storage and case.name in self.case_psi_functions:
                self.case_psi_functions[case.name].x.array[:] = self._psi_temp.x.array[:]

            # Accumulate stimulus weighted by day_cycles:
            #   p = 1 -> sum of day_cycles * psi
            #   p > 1 → peak-biased accumulation
            contrib_p = self._psi_temp.x.array[:n_owned_psi] ** p
            self.psi.x.array[:n_owned_psi] += day_cycles * contrib_p

            
            case_elapsed = self.comm.allreduce(MPI.Wtime() - case_start, op=MPI.MAX)
            phase_iters.append(int(its))
            phase_times.append(float(case_elapsed))


        # Finalize: take (.)^(1/p) after accumulation of sum(day_cycles * psi^p)
        owned = self.psi.x.array[:n_owned_psi]
        owned[:] = owned ** (1.0 / p)

        self.psi.x.scatter_forward()

        elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)

        return {
            "phase_iters": phase_iters,
            "phase_times": phase_times,
            "total_time": float(elapsed),
        }

    def stimulus_field(self) -> fem.Function:
        """Return the averaged psi function (DG0 field)."""
        return self.psi

    def save_case_snapshots(self, t: float) -> None:
        """Write per-case psi fields to storage."""
        if not self.storage:
            return
        if self.case_psi_functions:
            self.storage.fields.write("psi_cases", t)

    def get_stimulus_stats(self) -> Dict[str, float]:
        """
        Return global min/max/mean of psi across all MPI ranks.
        
        Returns:
            dict with psi_avg, psi_min, psi_max
        """
        psi_min, psi_max, psi_avg = field_stats(self.psi, self.comm)
        return {
            "psi_avg": psi_avg,
            "psi_min": psi_min,
            "psi_max": psi_max,
        }

    def _build_sed_expression(self) -> fem.Expression:
        """Build UFL expression for psi = 0.5 * sigma : epsilon (clamped >= 0)."""
        u = self.mech.u
        rho = self.mech.rho
        
        sig = self.mech.sigma(u, rho)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)
