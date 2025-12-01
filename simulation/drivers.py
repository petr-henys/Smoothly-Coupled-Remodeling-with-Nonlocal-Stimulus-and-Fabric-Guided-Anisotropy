"""Gait driver: solves mechanics and computes stimulus (SED)."""

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
    Solves mechanics and computes Strain Energy Density (SED) stimulus.
    
    Workflow:
        1. update_stiffness() - reassemble K(ρ) matrix
        2. update_snapshots() - solve Ku=f, compute Ψ
    
    Stimulus definition:
        Ψ = ½ σ(u,ρ) : ε(u)  [MPa]
        
    Note: Since σ = C(ρ):ε and C(ρ) ~ E(ρ), the stimulus Ψ depends on ρ.
    This creates a positive feedback: higher ρ → higher E → higher σ → higher Ψ.
    Consider using normalized stimulus Ψ/E(ρ) for stability if issues arise.
    """

    def __init__(
        self,
        mech: MechanicsSolver,
        config: Config,
    ):
        """
        Initialize driver with mechanics solver.
        
        Args:
            mech: Pre-configured MechanicsSolver with Neumann BCs already set.
            config: Simulation configuration.
        """
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
        """Initialize mechanics solver."""
        self.mech.setup()

    def destroy(self) -> None:
        """Release solver resources."""
        self.mech.destroy()

    def update_stiffness(self) -> None:
        """Reassemble mechanics stiffness matrix."""
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """Solve mechanics and compute Strain Energy Density."""
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
        """Return the computed stimulus function (Psi)."""
        return self.psi

    def get_stimulus_stats(self) -> Dict[str, float]:
        """Compute MPI-reduced stimulus statistics."""
        psi_min, psi_max, psi_avg = field_stats(self.psi, self.comm)
        return {
            "psi_avg": psi_avg,
            "psi_min": psi_min,
            "psi_max": psi_max,
        }

    def _build_sed_expression(self) -> fem.Expression:
        """
        Build UFL expression for Strain Energy Density (SED).
        
        Formula: Ψ = ½ σ:ε = ½ (C(ρ):ε):ε
        
        Note: This expression is evaluated at interpolation time, using
        current values of u and ρ. The result is stored in DG0 space
        (element-wise constant), which is appropriate for P1 displacements.
        
        Stability consideration:
            Since E(ρ) = E₀(ρ/ρ_ref)^n appears in σ, Ψ scales with E(ρ).
            For material-independent stimulus, consider Ψ_norm = Ψ / E(ρ).
        """
        u = self.mech.u
        rho = self.mech.rho
        
        sig = self.mech.sigma(u, rho)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)
