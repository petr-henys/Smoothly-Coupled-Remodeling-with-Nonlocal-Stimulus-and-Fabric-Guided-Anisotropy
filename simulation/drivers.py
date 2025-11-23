"""Remodeling drivers: translate mechanics to stimulus ψ(u) and structure M(u)."""

from __future__ import annotations

from typing import Protocol, Dict, Tuple, List, Optional, TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config
    from simulation.subsolvers import MechanicsSolver
    from simulation.femur_gait import FemurRemodellerGait


class RemodelingDriver(Protocol):
    """Protocol for drivers that provide mechanical fields to remodeling PDEs."""

    def stimulus_expr(self) -> ufl.core.expr.Expr: ...
    def structure_expr(self) -> ufl.core.expr.Expr: ...
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> Optional[Dict]: ...
    def setup(self) -> None: ...
    def destroy(self) -> None: ...
    def update_stiffness(self) -> None: ...
    def get_stimulus_stats(self) -> Dict[str, float]: ...


from simulation.utils import matrix_ln

class GaitDriver:
    """Gait-averaged Carter-Beaupré stimulus and strain-aligned target tensor.
    
    ψ = N_cyc * ⟨(σ/ψ_ref)^m⟩, M = ⟨ε_dev^T ε_dev⟩, L_target = log(M).
    """

    def __init__(self, mech: MechanicsSolver, gait_loader: FemurRemodellerGait, config: Config):
        self.mech = mech
        self.gait = gait_loader
        self.cfg = config
        self.comm = self.mech.u.function_space.mesh.comm
        self.logger = get_logger(self.comm, verbose=(self.cfg.verbose is True), name="Driver")

        # Cache config parameters
        self.psi_ref = float(config.psi_ref)
        self.exponent = float(config.n_power)

        # Quadrature setup
        quad = list(self.gait.get_quadrature())
        if not quad:
            raise ValueError("Gait quadrature must provide at least one sample.")

        self.phases = [float(p) for p, _ in quad]
        self.weights = [float(w) for _, w in quad]

        # Snapshots for displacement field at each phase
        V = self.mech.u.function_space
        self.u_snap = [fem.Function(V, name=f"u_snap_{i}") for i in range(len(self.phases))]

        # Tractions to update
        self._tractions = [self.gait.t_hip, self.gait.t_glmed, self.gait.t_glmax]
        #self._tractions = [self.gait.t_hip]
        # Precompute load vectors for all phases to avoid re-interpolation
        self.loads = self._precompute_loads()

        # UFL Expressions
        self.psi_expr: Optional[ufl.core.expr.Expr] = None
        self.M_expr: Optional[ufl.core.expr.Expr] = None
        self.L_target_expr: Optional[ufl.core.expr.Expr] = None
        self._build_expressions()
        
        # Auxiliary function space for statistics
        self.V_stats = fem.functionspace(self.mech.u.function_space.mesh, ("DG", 0))
        self.psi_stats = fem.Function(self.V_stats)

        self._last_stats: Optional[Dict] = None

    def setup(self) -> None:
        """Initialize underlying mechanics solver."""
        self.mech.setup()

    def destroy(self) -> None:
        """Clean up underlying mechanics solver."""
        self.mech.destroy()

    def update_stiffness(self) -> None:
        """Reassemble mechanics stiffness matrix (LHS)."""
        self.mech.assemble_lhs()

    def invalidate(self) -> None:
        """Rebuild expressions if psi_ref or exponent change in Config."""
        dirty = False
        if abs(self.psi_ref - float(self.cfg.psi_ref)) > 1e-9:
            self.psi_ref = float(self.cfg.psi_ref)
            dirty = True

        if abs(self.exponent - float(self.cfg.n_power)) > 1e-9:
            self.exponent = float(self.cfg.n_power)
            dirty = True

        if dirty:
            self._build_expressions()

    def update_snapshots(self) -> Dict:
        """Solve mechanics at each gait phase and refresh displacement snapshots."""
        times: List[float] = []
        iters: List[float] = []

        # Reset stats
        total_weight = 0.0

        for idx, (phase, weight) in enumerate(zip(self.phases, self.weights)):
            start = MPI.Wtime()
            
            # 1. Apply loads for this phase
            self._apply_load(idx)
            
            # 2. Solve mechanics
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            
            elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
            times.append(float(elapsed))
            iters.append(float(its))

            # 3. Store displacement snapshot
            # We copy the vector from the solver's u to our snapshot u_snap[idx]
            # This automatically updates the UFL expressions that depend on u_snap[idx]
            self.u_snap[idx].x.array[:] = self.mech.u.x.array
            
            total_weight += weight

        stats = {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": float(sum(times)),
            "median_time": float(np.median(times)) if times else 0.0,
            "median_iters": float(np.median(iters)) if iters else 0.0,
        }
        return stats

    def get_stimulus_stats(self) -> Dict[str, float]:
        """Compute statistics of the daily stimulus field (min, max, mean, median)."""
        # Compute domain-average of the daily stress for reporting
        # This uses the updated u_snap fields via psi_expr
        psi_int = self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.psi_expr * self.cfg.dx)), op=MPI.SUM
        )
        vol = self.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * self.cfg.dx)), op=MPI.SUM
        )
        psi_avg = psi_int / vol if vol > 0 else 0.0

        # Compute min, max, median
        psi_expr_compiled = fem.Expression(self.psi_expr, self.V_stats.element.interpolation_points)
        self.psi_stats.interpolate(psi_expr_compiled)
        local_vals = self.psi_stats.x.array

        local_min = np.min(local_vals) if local_vals.size > 0 else float('inf')
        local_max = np.max(local_vals) if local_vals.size > 0 else float('-inf')

        psi_min = self.comm.allreduce(local_min, op=MPI.MIN)
        psi_max = self.comm.allreduce(local_max, op=MPI.MAX)

        # Median (approximate via gather to rank 0)
        all_vals = self.comm.gather(local_vals, root=0)
        psi_median = 0.0
        if self.comm.rank == 0:
            full_data = np.concatenate(all_vals)
            if full_data.size > 0:
                psi_median = float(np.median(full_data))
        psi_median = self.comm.bcast(psi_median, root=0)

        return {
            "psi_avg": psi_avg,
            "psi_min": psi_min,
            "psi_max": psi_max,
            "psi_median": psi_median,
        }

    def stimulus_expr(self) -> ufl.core.expr.Expr:
        return self.psi_expr

    def structure_expr(self) -> ufl.core.expr.Expr:
        return self.M_expr
    
    def log_structure_expr(self) -> ufl.core.expr.Expr:
        return self.L_target_expr

    def _precompute_loads(self) -> List[Tuple[np.ndarray, ...]]:
        """Pre-calculate traction vector arrays for all phases."""
        loads = []
        for phase in self.phases:
            self.gait.update_loads(phase)
            # Copy the arrays so they are stored independently
            phase_loads = tuple(t.x.array.copy() for t in self._tractions)
            loads.append(phase_loads)
        return loads

    def _apply_load(self, idx: int) -> None:
        """Apply precomputed loads for phase index `idx` to the traction functions."""
        for traction, data in zip(self._tractions, self.loads[idx]):
            traction.x.array[:] = data

    def _build_expressions(self) -> None:
        """Construct UFL expressions for daily stimulus and structure tensor."""
        if self.exponent <= 0.0:
            raise ValueError(f"n_power must be positive, got {self.exponent}")

        N_cyc = float(self.cfg.gait_cycles_per_day)
        if N_cyc <= 0.0:
            raise ValueError(f"gait_cycles_per_day must be positive, got {N_cyc}")

        psi_p_terms = []
        structure_terms = []
        total_weight = 0.0
        
        # We use the snapshots u_snap to build the expression.
        # When u_snap values change, these expressions evaluate to new values.
        for u_i, weight in zip(self.u_snap, self.weights):
            # Stress and Strain from snapshot u_i
            # Note: rho and A_dir are shared (current state of remodeling)
            sig_i = self.mech.sigma(u_i, self.mech.rho, self.mech.A_dir)
            e_i = self.mech.get_strain_tensor(u_i)
            e_dev_i = ufl.dev(e_i)
            structure_i = ufl.dot(ufl.transpose(e_dev_i), e_dev_i)

            # Carter–Beaupré / Jacobs-style stimulus:
            # use an effective stress derived from the strain energy density,
            # without dividing by density (no specific-SED scaling).
            # U = 0.5 * σ : ε  (strain energy density)
            U_i = 0.5 * ufl.inner(sig_i, e_i)
            U_safe = ufl.max_value(U_i, 0.0)
            # Effective scalar stress measure σ_eff = sqrt(2 U)
            sigma_eff = ufl.sqrt(2.0 * U_safe + self.cfg.smooth_eps)

            term = (sigma_eff / self.psi_ref) ** self.exponent

            # Accumulate weighted terms
            psi_p_terms.append(weight * term)

            # M_term = w * (ε_devᵀ ε_dev)
            structure_terms.append(weight * structure_i)

            total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("Gait quadrature weights must sum to a positive value.")

        # Average over cycle
        J_cycle = sum(psi_p_terms) / total_weight
        M_cycle = sum(structure_terms) / total_weight

        # Final expressions
        self.psi_expr = N_cyc * J_cycle
        self.M_expr = M_cycle
        
        # Log-Euclidean Target
        # Ensure M is SPD and unit trace before taking log
        # M_cycle is already PSD (sum of outer products), so we use unittrace_psd
        
        d = self.mech.u.function_space.mesh.geometry.dim
        from simulation.utils import unittrace_psd
        M_hat = unittrace_psd(M_cycle, d, eps=self.cfg.smooth_eps)
        self.L_target_expr = matrix_ln(M_hat)


