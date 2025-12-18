"""GaitDriver: mechanics + averaged SED stimulus over multiple loading cases."""

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING

from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.logger import get_logger
from simulation.utils import assign, field_stats, get_owned_size

if TYPE_CHECKING:
    from simulation.loader import LoadingCase, Loader


class GaitDriver:
    """Mechanics driver that computes averaged SED stimulus.

    For each enabled loading case: set cached traction, solve mechanics, compute
    element-wise $\Psi=\tfrac12\sigma:\varepsilon$ (DG0), and accumulate with
    normalized weights ($\sum w=1$ each call).
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

        # Stimulus field (DG0 - element-wise constant)
        self.V_psi = fem.functionspace(mesh, ("DG", 0))
        self.psi = fem.Function(self.V_psi, name="Stimulus_SED")
        
        # Temporary per-case SED (DG0)
        self._psi_temp = fem.Function(self.V_psi, name="Stimulus_SED_temp")
        
        # Pre-compile SED expression (reused every update)
        self._sed_expr = self._build_sed_expression()

        # Normalized weights are recomputed each `update_snapshots()`.
        self._weights_norm: Dict[str, float] = {}
        self._recompute_normalized_weights()

        # Precompute all loading cases (expensive interpolation done once)
        self.loader.precompute_loading_cases(self.loading_cases)

        self.logger.debug(f"GaitDriver initialized with {len(loading_cases)} loading case(s), loads precomputed")

    def _recompute_normalized_weights(self) -> None:
        """Recompute per-case normalized weights.

        Conventions:
        - weight >= 0
        - at least one positive weight
        - Σ w_norm = 1 over cases with weight > 0
        """
        weights: Dict[str, float] = {}
        total = 0.0
        for case in self.loading_cases:
            w = float(case.weight)
            if w < 0.0:
                raise ValueError(f"Loading case '{case.name}' has negative weight {w}")
            if w > 0.0:
                weights[case.name] = w
                total += w

        if total <= 0.0:
            raise ValueError("At least one loading case must have positive weight")

        self._weights_norm = {name: (w / total) for name, w in weights.items()}

    def setup(self) -> None:
        """Initialize mechanics solver."""
        self.mech.setup()

    def destroy(self) -> None:
        """Release solver resources."""
        self.mech.destroy()

    def update_stiffness(self) -> None:
        """Reassemble stiffness matrix K(ρ)."""
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """Solve all enabled phases and update averaged `psi` (collective).

        Returns `phase_iters`, `phase_times`, and `total_time` (MPI max).
        """
        start = MPI.Wtime()
        
        phase_iters = []
        phase_times = []
        
        # Recompute normalization each call (supports quick weight sweeps).
        self._recompute_normalized_weights()

        # Zero out averaged psi (owned DOFs only; single scatter at the end)
        assign(self.psi, 0.0, scatter=False)
        n_owned_psi = get_owned_size(self.psi)
        p = float(self.cfg.stimulus_power_p)

        
        for case in self.loading_cases:
            w_norm = self._weights_norm.get(case.name, 0.0)
            if w_norm <= 0.0:
                # Disabled case (weight == 0) → skip solve and timing.
                continue

            case_start = MPI.Wtime()
            
            # Load cached traction (cheap, no geometry work)
            self.loader.set_loading_case(case.name)
            
            # Reassemble RHS (traction is referenced in the form)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            
            # Compute SED for this case
            self._psi_temp.interpolate(self._sed_expr)
            
            # Accumulate multi-case stimulus via power-mean:
            #   p = 1 → arithmetic mean (original behaviour)
            #   p > 1 → increasingly peak-biased
            contrib_p = self._psi_temp.x.array[:n_owned_psi] ** p
            self.psi.x.array[:n_owned_psi] += w_norm * contrib_p

            
            case_elapsed = self.comm.allreduce(MPI.Wtime() - case_start, op=MPI.MAX)
            phase_iters.append(int(its))
            phase_times.append(float(case_elapsed))


        # Finalize power-mean: take (.)^(1/p) after accumulation of Σ w Ψ^p

        owned = self.psi.x.array[:n_owned_psi]
        owned[:] = owned ** (1.0 / p)

        # No further normalization needed (Σ w_norm = 1)
        self.psi.x.scatter_forward()

        elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)

        return {
            "phase_iters": phase_iters,
            "phase_times": phase_times,
            "total_time": float(elapsed),
        }

    def stimulus_field(self) -> fem.Function:
        """Return the averaged Ψ function (DG0 field)."""
        return self.psi

    def get_stimulus_stats(self) -> Dict[str, float]:
        """
        Return global min/max/mean of Ψ across all MPI ranks.
        
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
        """Build UFL expression for $\Psi=\tfrac12\sigma:\varepsilon$ (clamped ≥ 0)."""
        u = self.mech.u
        rho = self.mech.rho
        
        sig = self.mech.sigma(u, rho)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)
