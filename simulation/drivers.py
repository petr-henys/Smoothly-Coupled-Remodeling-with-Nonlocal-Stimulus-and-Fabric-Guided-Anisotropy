"""Remodeling drivers for stimulus and direction solvers.

This module provides driver objects that translate mechanics results (displacements u) into:
- a scalar stimulus driver ψ(u) [-] (dimensionless daily load dose)
- a structure tensor M(u) capturing preferred loading directions

Used as inputs for:
- StimulusSolver (source term from ψ)
- DirectionSolver (evolution of fabric tensor from M)
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


def von_mises_stress(sig: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
    """Von Mises equivalent stress σ_vm from Cauchy stress tensor."""
    s = ufl.dev(sig)
    # Add small epsilon for numerical stability of sqrt(0)
    return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-16)


class RemodelingDriver(Protocol):
    """Protocol for drivers that provide mechanical fields to remodeling PDEs."""

    def stimulus_expr(self) -> ufl.core.expr.Expr: ...
    def structure_expr(self) -> ufl.core.expr.Expr: ...
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> Optional[Dict]: ...
    def setup(self) -> None: ...
    def destroy(self) -> None: ...
    def update_stiffness(self) -> None: ...


class GaitDriver:
    """Gait-averaged Carter–Beaupré daily stress stimulus + structure tensor.

    ψ_day(x) = N_cyc * ⟨(σ_eff(x)/ψ_ref)^m⟩_cycle   [-]
    M(x) = ⟨ε_devᵀ ε_dev⟩_cycle
    """

    def __init__(self, mech: MechanicsSolver, gait_loader: FemurRemodellerGait, config: Config):
        self.mech = mech
        self.gait = gait_loader
        self.cfg = config
        self.comm = self.mech.u.function_space.mesh.comm
        self.logger = get_logger(self.comm, verbose=self.cfg.verbose, name="Driver")

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
        
        # Precompute load vectors for all phases to avoid re-interpolation
        self.loads = self._precompute_loads()

        # UFL Expressions
        self.psi_expr: Optional[ufl.core.expr.Expr] = None
        self.M_expr: Optional[ufl.core.expr.Expr] = None
        self._build_expressions()
        
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
            self.u_snap[idx].x.scatter_forward()
            
            total_weight += weight

        # Compute domain-average of the daily stress for reporting
        # This uses the updated u_snap fields via psi_expr
        psi_int = self.comm.allreduce(
            fem.assemble_scalar(fem.form(self.psi_expr * self.cfg.dx)), op=MPI.SUM
        )
        vol = self.comm.allreduce(
            fem.assemble_scalar(fem.form(1.0 * self.cfg.dx)), op=MPI.SUM
        )
        psi_avg = psi_int / vol if vol > 0 else 0.0

        stats = {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": float(sum(times)),
            "median_time": float(np.median(times)) if times else 0.0,
            "median_iters": float(np.median(iters)) if iters else 0.0,
            "psi_avg": psi_avg,
        }
        self._last_stats = stats
        return stats

    def stimulus_expr(self) -> ufl.core.expr.Expr:
        return self.psi_expr

    def structure_expr(self) -> ufl.core.expr.Expr:
        return self.M_expr

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
            traction.x.scatter_forward()

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
            sigma_vm_i = von_mises_stress(sig_i)

            e_i = self.mech.get_strain_tensor(u_i)
            e_dev_i = ufl.dev(e_i)
            structure_i = ufl.dot(ufl.transpose(e_dev_i), e_dev_i)

            # Accumulate weighted terms
            # ψ_term = w * (σ_vm / σ_ref)^m
            psi_p_terms.append(weight * (sigma_vm_i / self.psi_ref) ** self.exponent)
            
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

