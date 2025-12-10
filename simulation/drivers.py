"""
GaitDriver: mechanics + averaged SED stimulus computation over multiple loading cases.
"""

from __future__ import annotations
from typing import Dict, List, TYPE_CHECKING

from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.logger import get_logger
from simulation.utils import field_stats

if TYPE_CHECKING:
    from simulation.loader import LoadingCase, Loader


class GaitDriver:
    """
    Solves mechanics and computes weighted-averaged Strain Energy Density Ψ = ½σ:ε.
    
    Workflow for each loading case:
    1. Apply loads to traction field (via Loader, MPI-parallel)
    2. Reassemble RHS and solve mechanics (MPI-parallel)
    3. Compute element-wise SED (DG0)
    4. Accumulate weighted SED
    
    Final stimulus: Ψ = Σ(wᵢ·Ψᵢ) / Σ(wᵢ)
    
    MPI: All operations are MPI-safe. Loader handles coordinate gather/scatter.
    """

    def __init__(
        self,
        mech: MechanicsSolver,
        config: Config,
        loader: "Loader",
        loading_cases: List["LoadingCase"],
    ):
        """
        Initialize GaitDriver.
        
        Args:
            mech: MechanicsSolver (already configured with Neumann BC pointing to loader.traction)
            config: Simulation configuration
            loader: Loader instance for applying loads
            loading_cases: List of LoadingCase objects to average over
        """
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
        
        # Temporary SED for each loading case
        self._psi_temp = fem.Function(self.V_psi, name="Stimulus_SED_temp")
        
        # Pre-compile SED expression
        self._sed_expr = self._build_sed_expression()
        
        # Total weight for normalization
        self._total_weight = sum(case.weight for case in self.loading_cases)
        if self._total_weight <= 0:
            raise ValueError("Total weight of loading cases must be positive")
        
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
        """Reassemble stiffness matrix K(ρ)."""
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """
        Solve mechanics for each loading case and compute weighted-averaged SED.
        
        MPI-parallel: All ranks participate in loader.apply_loading_case() and mech.solve().
        
        Returns:
            dict with keys:
            - phase_iters: list of KSP iterations per case
            - phase_times: list of wall-clock times per case
            - total_time: total elapsed time (max across ranks)
        """
        start = MPI.Wtime()
        
        phase_iters = []
        phase_times = []
        
        # Zero out averaged psi
        self.psi.x.array[:] = 0.0
        
        for case in self.loading_cases:
            case_start = MPI.Wtime()
            
            # Set cached loading case - copies precomputed traction arrays
            self.loader.set_loading_case(case.name)
            
            # Reassemble RHS (traction field is already referenced in form)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            
            # Compute SED for this case
            self._psi_temp.interpolate(self._sed_expr)
            self._psi_temp.x.scatter_forward()
            
            # Accumulate weighted SED
            self.psi.x.array[:] += case.weight * self._psi_temp.x.array
            
            case_elapsed = self.comm.allreduce(MPI.Wtime() - case_start, op=MPI.MAX)
            phase_iters.append(int(its))
            phase_times.append(float(case_elapsed))

        # Normalize by total weight
        self.psi.x.array[:] /= self._total_weight
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
        """
        Build UFL expression for Strain Energy Density: Ψ = ½σ:ε.
        
        Uses max_value(..., 0) to clamp any numerical noise to non-negative.
        """
        u = self.mech.u
        rho = self.mech.rho
        
        sig = self.mech.sigma(u, rho)
        eps = self.mech.eps(u)
        
        Psi = 0.5 * ufl.inner(sig, eps)
        Psi_safe = ufl.max_value(Psi, 0.0)

        return fem.Expression(Psi_safe, self.V_psi.element.interpolation_points)
