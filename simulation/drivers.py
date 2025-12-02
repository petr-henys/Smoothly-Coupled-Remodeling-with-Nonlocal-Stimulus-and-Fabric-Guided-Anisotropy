"""GaitDriver: mechanics + SED stimulus computation."""

from typing import Dict
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.logger import get_logger
from simulation.utils import field_stats


class GaitDriver:
    """
    Solves mechanics and computes Strain Energy Density: Ψ = ½σ:ε.
    """

    def __init__(
        self,
        mech: MechanicsSolver,
        config: Config,
    ):
        """Initialize with MechanicsSolver and config."""
        self.mech = mech
        self.cfg = config

        mesh = self.mech.u.function_space.mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.logger = get_logger(self.comm, name="Driver", log_file=self.cfg.log_file)

        # Stimulus field (Strain Energy Density - SED)
        # Using DG0 (element-wise constant) is standard for SED
        self.V_psi = fem.functionspace(mesh, ("DG", 0))
        self.psi = fem.Function(self.V_psi, name="Stimulus_SED")
        
        # Pre-compile the SED expression
        self._sed_expr = self._build_sed_expression()

        self.logger.debug("GaitDriver initialized.")

    def setup(self) -> None:
        """Initialize solver."""
        self.mech.setup()

    def destroy(self) -> None:
        """Release solver resources."""
        self.mech.destroy()

    def update_stiffness(self) -> None:
        """Reassemble K(ρ)."""
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """Solve mechanics and compute SED."""
        start = MPI.Wtime()

        # Solve mechanics equilibrium
        self.mech.assemble_rhs()
        its, _ = self.mech.solve()

        elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)

        # Calculate SED (DG0 space - no ghost sharing needed, but scatter
        # ensures consistency for any downstream consumers)
        self.psi.interpolate(self._sed_expr)
        # Note: DG0 has cell-local DOFs. Scatter updates ghost cells which
        # are needed if psi is used in forms assembled over ghost cells.
        self.psi.x.scatter_forward()

        return {
            "phase_iters": [int(its)],
            "phase_times": [float(elapsed)],
            "total_time": float(elapsed),
        }

    def stimulus_field(self) -> fem.Function:
        """Return Ψ function."""
        return self.psi

    def get_stimulus_stats(self) -> Dict[str, float]:
        """Return global min/max/mean of Ψ."""
        psi_min, psi_max, psi_avg = field_stats(self.psi, self.comm)
        return {
            "psi_avg": psi_avg,
            "psi_min": psi_min,
            "psi_max": psi_max,
        }

    def _build_sed_expression(self) -> fem.Expression:
        """Build UFL expression: Ψ = ½σ:ε."""
        u = self.mech.u
        rho = self.mech.rho
        
        sig = self.mech.sigma(u, rho)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)
