"""Linear elasticity solver with density/fabric-dependent anisotropic stiffness."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
import ufl

from simulation.solvers.base import BaseLinearSolver
from simulation.stats import SweepStats
from simulation.utils import (
    build_nullspace,
    smooth_max,
    smoothstep01,
    clamp,
    symm,
    eigenvalues_sym3,
    projectors_sylvester,
)

if TYPE_CHECKING:
    from simulation.config import Config


class MechanicsSolver(BaseLinearSolver):
    """Linear elasticity with density/fabric-dependent anisotropic stiffness.

    Solves the static equilibrium problem:
        -div(σ(u)) = 0     in Ω
        u = 0               on Γ_D
        σ·n = t             on Γ_N

    The stress σ depends on:
    - Local density ρ (power-law E(ρ))
    - Log-fabric tensor L (anisotropic stiffness via eigenvalue decomposition)
    """

    _label = "mech"

    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        config: "Config",
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
        L: fem.Function | None = None,
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho
        self.L = L
        self._nullspace = None

    def _compile_forms(self):
        a_ufl = ufl.inner(self.sigma(self.trial, self.rho, self.L), self.eps(self.test)) * self.dx
        self.a_form = fem.form(a_ufl)

        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_ufl = ufl.inner(zero_vec, self.test) * self.ds
        n = ufl.FacetNormal(self.mesh)

        for t, tag in self.neumann_bcs:
            if len(t.ufl_shape) == 0:
                L_ufl = L_ufl + ufl.inner(-t * n, self.test) * self.ds(tag)
            else:
                L_ufl = L_ufl + ufl.inner(t, self.test) * self.ds(tag)

        self.L_form = fem.form(L_ufl)

    def assemble_lhs(self) -> None:
        self.rho.x.scatter_forward()
        if isinstance(self.L, fem.Function):
            self.L.x.scatter_forward()
        super().assemble_lhs()

    def _setup_ksp(self):
        self._nullspace = build_nullspace(self.function_space)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(self._nullspace)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)

    @staticmethod
    def eps(u):
        return ufl.sym(ufl.grad(u))

    def _E_iso(self, rho):
        rho_eff = smooth_max(rho, self.cfg.density.rho_min, self.smooth_eps)
        rho_rel = rho_eff / self.cfg.density.rho_ref

        denom = float(self.cfg.material.rho_cort_min - self.cfg.material.rho_trab_max)
        t = (rho_eff - self.cfg.material.rho_trab_max) / denom
        w = smoothstep01(t)
        k = self.cfg.material.n_trab * (1.0 - w) + self.cfg.material.n_cort * w

        return self.cfg.material.E0 * (rho_rel ** k)

    def sigma(self, u, rho, L: fem.Function | None = None):
        eps = self.eps(u)

        E_iso = self._E_iso(rho)
        nu0 = float(self.cfg.material.nu0)

        # Isotropic baseline
        mu_iso = E_iso / (2.0 * (1.0 + nu0))
        lmbda_iso = E_iso * nu0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
        sigma_iso = 2.0 * mu_iso * eps + lmbda_iso * ufl.tr(eps) * ufl.Identity(self.gdim)

        if L is None:
            return sigma_iso

        if self.gdim != 3:
            raise ValueError("Anisotropic MechanicsSolver currently requires gdim==3.")

        I3 = ufl.Identity(3)

        # Fabric spectral decomposition
        Ls = symm(L)
        l1, l2, l3 = eigenvalues_sym3(Ls)
        P1, P2, P3 = projectors_sylvester(Ls, l1, l2, l3)

        mean_l = (l1 + l2 + l3) / 3.0

        # Clamp BEFORE exp() to prevent overflow
        m_min = float(self.cfg.fabric.fabric_m_min)
        m_max = float(self.cfg.fabric.fabric_m_max)
        a_cap = max(m_max, 1.0 / m_min)
        dmax = math.log(a_cap)

        d1 = clamp(l1 - mean_l, -dmax, dmax)
        d2 = clamp(l2 - mean_l, -dmax, dmax)
        d3 = clamp(l3 - mean_l, -dmax, dmax)

        a1_hat = ufl.exp(d1)
        a2_hat = ufl.exp(d2)
        a3_hat = ufl.exp(d3)

        pE = float(self.cfg.material.stiff_pE)
        pG = float(self.cfg.material.stiff_pG)

        E1 = E_iso * (a1_hat ** pE)
        E2 = E_iso * (a2_hat ** pE)
        E3 = E_iso * (a3_hat ** pE)

        G_iso = E_iso / (2.0 * (1.0 + nu0))
        G12 = G_iso * ((a1_hat * a2_hat) ** (0.5 * pG))
        G23 = G_iso * ((a2_hat * a3_hat) ** (0.5 * pG))
        G31 = G_iso * ((a3_hat * a1_hat) ** (0.5 * pG))

        # Normal part via closed-form inverse
        nu = nu0
        a = 1.0 / (1.0 + nu)
        b = nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        e1 = ufl.inner(P1, eps)
        e2 = ufl.inner(P2, eps)
        e3 = ufl.inner(P3, eps)

        sqrtE1 = ufl.sqrt(E1)
        sqrtE2 = ufl.sqrt(E2)
        sqrtE3 = ufl.sqrt(E3)

        sum_term = sqrtE1 * e1 + sqrtE2 * e2 + sqrtE3 * e3
        s1 = a * E1 * e1 + b * sqrtE1 * sum_term
        s2 = a * E2 * e2 + b * sqrtE2 * sum_term
        s3 = a * E3 * e3 + b * sqrtE3 * sum_term

        sigma_normal = s1 * P1 + s2 * P2 + s3 * P3

        def _P_eps_P(A, B):
            return ufl.dot(A, ufl.dot(eps, B))

        sigma_shear = (
            2.0 * G12 * (_P_eps_P(P1, P2) + _P_eps_P(P2, P1))
            + 2.0 * G23 * (_P_eps_P(P2, P3) + _P_eps_P(P3, P2))
            + 2.0 * G31 * (_P_eps_P(P3, P1) + _P_eps_P(P1, P3))
        )

        sigma_aniso = sigma_normal + sigma_shear

        # C¹ blend: isotropic ↔ anisotropic
        q = ufl.tr(Ls) / 3.0
        B = Ls - q * I3
        p2 = ufl.tr(ufl.dot(B, B)) / 6.0
        scale2 = ufl.max_value(q * q + p2, 1.0)

        r = p2 / scale2

        tol_iso = 1e-8
        r0 = 0.1 * tol_iso
        r1 = 10.0 * tol_iso

        t = (r - r0) / (r1 - r0)
        w = smoothstep01(t)

        return (1.0 - w) * sigma_iso + w * sigma_aniso

    def assemble_rhs(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.dirichlet_bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.dirichlet_bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def _compute_extra_stats(self) -> Dict[str, Any]:
        if self.L is None:
            return {}

        bs = self.L.function_space.dofmap.index_map_bs
        n_local = self.L.function_space.dofmap.index_map.size_local
        n_owned = n_local * bs
        arr = self.L.x.array[:n_owned]

        if arr.size == 0:
            a_min_loc, a_max_loc = 1e30, -1e30
            p2_min_loc, p2_max_loc = 1e30, -1e30
        else:
            L_flat = arr.reshape(-1, 3, 3)
            L_sym = 0.5 * (L_flat + L_flat.transpose(0, 2, 1))
            w = np.linalg.eigvalsh(L_sym)

            mean_l = np.mean(w, axis=1, keepdims=True)
            a_hat = np.exp(w - mean_l)

            a_min_loc = np.min(a_hat)
            a_max_loc = np.max(a_hat)

            dev = w - mean_l
            p2 = np.sum(dev**2, axis=1) / 6.0

            p2_min_loc = np.min(p2)
            p2_max_loc = np.max(p2)

        comm = self.comm

        glob_min = np.array([a_min_loc, p2_min_loc], dtype=float)
        glob_max = np.array([a_max_loc, p2_max_loc], dtype=float)

        comm.Allreduce(MPI.IN_PLACE, glob_min, op=MPI.MIN)
        comm.Allreduce(MPI.IN_PLACE, glob_max, op=MPI.MAX)

        if glob_min[0] > glob_max[0]:
            return {}

        return {
            "a_min": float(glob_min[0]),
            "a_max": float(glob_max[0]),
            "p2_min": float(glob_min[1]),
            "p2_max": float(glob_max[1]),
        }

    def solve(self) -> SweepStats:
        self.assemble_rhs()
        return self._solve()

    # -------------------------------------------------------------------------
    # CouplingBlock protocol
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        return ()

    def sweep(self) -> SweepStats:
        self.assemble_rhs()
        return self._solve()
