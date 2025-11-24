"""
Remodeling drivers: translate mechanics to stimulus ψ(u) and structure M(u).

This module implements the 'Daily Stress Stimulus' theory as defined by 
Beaupré, Orr, and Carter (1990) [cite: 1] and Jacobs et al. (1995) [cite: 4].
"""

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
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> Optional[Dict]: ...
    def setup(self) -> None: ...
    def destroy(self) -> None: ...
    def update_stiffness(self) -> None: ...
    def get_stimulus_stats(self) -> Dict[str, float]: ...


class GaitDriver:
    """
    Gait-averaged Carter-Beaupré stimulus.
    
    Implements the Daily Stress Stimulus (ψ)[cite: 1]:
        ψ = ( Σ n_i * σ_eff_i^m )^(1/m)
    
    Where σ_eff is the effective stress at the tissue level[cite: 4]:
        σ_eff = (ρ_cortical / ρ)^k * sqrt(2 * E(ρ) * U)
    
    And U is the Strain Energy Density (SED).
    
    Attributes:
        psi_expr (ufl.Expr): The symbolic expression for the daily stimulus ψ.
    """

    def __init__(self, mech: MechanicsSolver, gait_loader: FemurRemodellerGait, config: Config):
        self.mech = mech
        self.gait = gait_loader
        self.cfg = config
        self.comm = self.mech.u.function_space.mesh.comm
        self.logger = get_logger(self.comm, verbose=(self.cfg.verbose is True), name="Driver")

        # Cache config parameters for stimulus definition
        # m: Weighting exponent (usually 4.0 for energy/stress) [cite: 1]
        self.m_exp = float(config.n_power) 
        
        # N: Number of daily cycles [cite: 1]
        self.n_cycles = float(config.gait_cycles_per_day)
        
        # k: Tissue stress concentration exponent. 
        # Jacobs et al. (1995) use power 2 for (rho_c/rho)^2 applied to energy, 
        # which implies k=1 for stress (sqrt of energy) or k=2 depending on formulation.
        # Here we assume applied to stress directly.
        self.k_stimulus = 1.0 

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

        # Precompute load vectors for all phases to avoid re-interpolation
        self.loads = self._precompute_loads()

        # UFL Expressions
        self.psi_expr: Optional[ufl.core.expr.Expr] = None
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
        """Rebuild expressions if configuration parameters change."""
        dirty = False
        if abs(self.m_exp - float(self.cfg.n_power)) > 1e-9:
            self.m_exp = float(self.cfg.n_power)
            dirty = True
        
        if abs(self.n_cycles - float(self.cfg.gait_cycles_per_day)) > 1e-9:
            self.n_cycles = float(self.cfg.gait_cycles_per_day)
            dirty = True

        if dirty:
            self._build_expressions()

    def update_snapshots(self) -> Dict:
        """Solve mechanics at each gait phase and refresh displacement snapshots."""
        times: List[float] = []
        iters: List[float] = []

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
        # Interpolate the complex UFL expression into a DG0 field to query values
        psi_expr_compiled = fem.Expression(self.psi_expr, self.V_stats.element.interpolation_points)
        self.psi_stats.interpolate(psi_expr_compiled)
        
        # Domain integral average
        psi_int = self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.psi_stats * self.cfg.dx)), op=MPI.SUM
        )
        vol = self.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * self.cfg.dx)), op=MPI.SUM
        )
        psi_avg = psi_int / vol if vol > 0 else 0.0

        # Min/Max/Median
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
        """
        Construct UFL expressions for daily stimulus.
        
        Strictly follows Beaupré et al. (1990) :
        psi = ( Sum( n_i * sigma_eff_i^m ) ) ^ (1/m)
        
        Where sigma_eff_i is the effective stress at tissue level.
        1. Continuum Effective Stress: sigma_cont = sqrt(2 * E * U)
        2. Tissue Effective Stress: sigma_tissue = (rho_max / rho)^k * sigma_cont
        """
        if self.m_exp <= 0.0:
            raise ValueError(f"n_power (m) must be positive, got {self.m_exp}")

        if self.n_cycles <= 0.0:
            raise ValueError(f"gait_cycles_per_day must be positive, got {self.n_cycles}")

        # Reconstruct Young's Modulus E(rho) field from config parameters
        # E is needed to convert SED (U) to Effective Stress dimensionally [MPa]
        # E = E_max * (rho/rho_max)^p_stiffness
        rho = self.mech.rho
        # Note: Config values are stored, we use E0 as E_max reference
        rho_max = self.cfg.rho_max
        E_max = self.cfg.E0 
        # We need to use the same variable exponent logic as in MechanicsSolver if we want consistency.
        # But here we just need a scalar E for the stress conversion.
        # MechanicsSolver uses E = E0 * rho^k_var.
        # Let's replicate that logic or simplify.
        # The original code used p_exponent = self.cfg.k_stiff.
        # But MechanicsSolver now uses variable k.
        # Let's use k_stiff from config as a representative exponent for stimulus calculation,
        # or replicate the variable exponent logic.
        # Replicating variable exponent logic is safer for consistency.
        
        rho_safe = ufl.max_value(rho, self.cfg.smooth_eps)
        
        # Variable exponent logic
        def smoothstep(x, edge0, edge1):
            t = ufl.max_value(0.0, ufl.min_value(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        w = smoothstep(rho_safe, self.cfg.rho_trab_max, self.cfg.rho_cort_min)
        k_var = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w
        
        E_field = E_max * (rho_safe**k_var)

        psi_summation = 0.0
        total_weight = 0.0
        
        # We use the snapshots u_snap to build the expression.
        for u_i, weight in zip(self.u_snap, self.weights):
            # 1. Kinematics & Stress
            sig_i = self.mech.sigma(u_i, rho)
            e_i = self.mech.get_strain_tensor(u_i)
            
            # 2. Strain Energy Density (U = 1/2 * sigma : epsilon)
            U_i = 0.5 * ufl.inner(sig_i, e_i)
            U_safe = ufl.max_value(U_i, 0.0)
            
            # 3. Continuum Effective Stress: sigma_bar = sqrt(2 * E * U)
            # This converts energy density back to a scalar stress equivalent 
            # consistent with continuum elasticity[cite: 1, 17].
            sigma_continuum = ufl.sqrt(2.0 * E_field * U_safe + self.cfg.smooth_eps)
            
            # 4. Tissue Effective Stress: sigma_t = (rho_max / rho)^k * sigma_continuum
            # Accounts for porous stress concentration.
            tissue_scaling = (rho_max / rho_safe)**self.k_stimulus
            sigma_tissue = tissue_scaling * sigma_continuum
            
            # 5. Accumulate Stimulus Dose: n_i * sigma^m
            # n_i = weight * total_daily_cycles
            n_i = weight * self.n_cycles
            psi_summation += n_i * (sigma_tissue ** self.m_exp)

            total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("Gait quadrature weights must sum to a positive value.")

        # Final Stimulus Expression: psi = (Sum)^ (1/m)
        # This is the daily stress stimulus described by Beaupré [cite: 1]
        self.psi_expr = ufl.max_value(psi_summation, 0.0) ** (1.0 / self.m_exp)
