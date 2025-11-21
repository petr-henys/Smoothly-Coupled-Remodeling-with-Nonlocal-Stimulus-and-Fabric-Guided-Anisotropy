"""Remodeling drivers for stimulus and direction solvers.

This module provides driver objects that translate mechanics results (displacements u) into:
- a scalar stimulus driver ψ(u) [-] (dimensionless daily load dose)
- a structure tensor M(u) capturing preferred loading directions

Used as inputs for:
- StimulusSolver (source term from ψ)
- DirectionSolver (evolution of fabric tensor from M)
"""

from __future__ import annotations

from typing import Protocol, Dict, Tuple, List, Optional

import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl
import basix

from simulation.config import Config
from simulation.projector import L2Projector
from simulation.logger import get_logger, Level


def von_mises_stress(sig: ufl.core.expr.Expr, gdim: int) -> ufl.core.expr.Expr:
    """Von Mises equivalent stress σ_vm from Cauchy stress tensor."""
    s = ufl.dev(sig)
    return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-16)


class RemodelingDriver(Protocol):
    """Protocol for drivers that provide mechanical fields to remodeling PDEs."""

    def stimulus_expr(self) -> ufl.core.expr.Expr: ...
    def structure_expr(self) -> ufl.core.expr.Expr: ...
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> Optional[Dict]: ...


class InstantDriver:
    """Instantaneous effective stress and structure tensor from current mechanics state."""

    def __init__(self, mech):
        self.mech = mech

    def invalidate(self) -> None:
        pass

    def update_snapshots(self) -> None:
        pass

    def stimulus_expr(self) -> ufl.core.expr.Expr:
        """Effective stress driver σ_vm(u) [MPa]."""
        sig = self.mech.sigma(self.mech.u, self.mech.rho, self.mech.A_dir)
        return von_mises_stress(sig, self.mech.gdim)

    def structure_expr(self) -> ufl.core.expr.Expr:
        """Deviatoric structure tensor M = ε_devᵀ ε_dev."""
        e = self.mech.get_strain_tensor()
        e_dev = ufl.dev(e)
        return ufl.dot(ufl.transpose(e_dev), e_dev)


class GaitDriver:
    """Gait-averaged Carter–Beaupré daily stress stimulus + structure tensor.

    ψ_day(x) = N_cyc * ⟨(σ_eff(x)/ψ_ref)^m⟩_cycle   [-]
    M(x) = ⟨ε_devᵀ ε_dev⟩_cycle
    """

    def __init__(self, mech, gait_loader, config: Config):
        self.mech = mech
        self.gait = gait_loader
        self.cfg = config
        self.psi_ref = float(config.psi_ref)
        self.exponent = float(config.n_power)
        self.comm = self.mech.u.function_space.mesh.comm

        quad = list(self.gait.get_quadrature())
        if not quad:
            raise ValueError("Gait quadrature must provide at least one sample.")

        self.phases = [float(p) for p, _ in quad]
        self.weights = [float(w) for _, w in quad]

        V = self.mech.u.function_space
        self.u_snap = [fem.Function(V, name=f"u_snap_{i}") for i in range(len(self.phases))]

        # Setup L2 projector for VM stress monitoring
        mesh = V.mesh
        E_dg0 = basix.ufl.element("DG", mesh.basix_cell(), 0)
        V_stress = fem.functionspace(mesh, E_dg0)
        
        self.projector = L2Projector(V_stress)
        self.vm_stress = fem.Function(V_stress, name="vm_stress")
        self.vm_stress_avg = fem.Function(V_stress, name="vm_stress_avg")

        self._tractions = (self.gait.t_hip, self.gait.t_glmed, self.gait.t_glmax)
        self.loads = self._precompute_loads()

        self.psi_expr: ufl.core.expr.Expr
        self.M_expr: ufl.core.expr.Expr
        self._build_expressions()
        self._last_stats: Optional[Dict] = None
        self.logger = get_logger(self.comm, verbose=self.cfg.verbose, name="Driver")

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

        self.vm_stress_avg.x.array[:] = 0.0
        total_weight = 0.0

        for idx in range(len(self.phases)):
            start = MPI.Wtime()
            self._apply_load(idx)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)

            times.append(float(elapsed))
            iters.append(float(its))

            # Project VM stress
            #sig = self.mech.sigma(self.mech.u, self.mech.rho, self.mech.A_dir)
            #vm = von_mises_stress(sig, self.mech.gdim)
            #self.projector.project(vm, result=self.vm_stress)
            
            # Accumulate average
            w = self.weights[idx]
            #self.vm_stress_avg.x.array[:] += w * self.vm_stress.x.array[:]
            #self.logger.info(f"Phase {idx}: minmax VM: {self.vm_stress.x.array.min():.5e} .. {self.vm_stress.x.array.max():.5e}")
            total_weight += w

            self.u_snap[idx].x.array[:] = self.mech.u.x.array
            self.u_snap[idx].x.scatter_forward()

        #if total_weight > 0:
        #    self.vm_stress_avg.x.array[:] /= total_weight
        #self.vm_stress_avg.x.scatter_forward()
        
        #self.logger.info(f"Phase {idx}: minmax avg VM: {self.vm_stress_avg.x.array.min():.3f} .. {self.vm_stress_avg.x.array.max():.3f}")
        self.mech.u.x.scatter_forward()

        # Domain-average of the daily stress
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

    def _precompute_loads(self) -> List[Tuple]:
        loads = []
        for phase in self.phases:
            self.gait.update_loads(phase)
            loads.append(tuple(t.x.array.copy() for t in self._tractions))
        return loads

    def _apply_load(self, idx: int) -> None:
        for traction, data in zip(self._tractions, self.loads[idx]):
            traction.x.array[:] = data
            traction.x.scatter_forward()

    def _build_expressions(self) -> None:
        if self.exponent <= 0.0:
            raise ValueError(f"n_power must be positive, got {self.exponent}")

        N_cyc = float(self.cfg.gait_cycles_per_day)
        if N_cyc <= 0.0:
            raise ValueError(f"gait_cycles_per_day must be positive, got {N_cyc}")

        psi_p_terms = []
        structure_terms = []
        total_weight = 0.0
        gdim = self.mech.gdim

        for u_i, weight in zip(self.u_snap, self.weights):
            sig_i = self.mech.sigma(u_i, self.mech.rho, self.mech.A_dir)
            sigma_vm_i = von_mises_stress(sig_i, gdim)

            e_i = self.mech.get_strain_tensor(u_i)
            e_dev_i = ufl.dev(e_i)
            structure_i = ufl.dot(ufl.transpose(e_dev_i), e_dev_i)

            psi_p_terms.append(weight * (sigma_vm_i / self.psi_ref) ** self.exponent)
            structure_terms.append(weight * structure_i)
            total_weight += weight

        if total_weight <= 0.0:
            raise ValueError("Gait quadrature weights must sum to a positive value.")

        J_cycle = sum(psi_p_terms) / total_weight
        self.psi_expr = N_cyc * J_cycle
        self.M_expr = sum(structure_terms) / total_weight

