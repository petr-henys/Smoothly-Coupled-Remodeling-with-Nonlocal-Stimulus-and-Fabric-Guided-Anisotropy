"""Log-fabric evolution solver with diffusion and activity-gated relaxation."""

from __future__ import annotations

import time
from typing import Tuple, TYPE_CHECKING

import basix.ufl
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import functionspace, Function
from dolfinx.fem.petsc import assemble_vector
import ufl

from simulation.solvers.base import BaseLinearSolver
from simulation.stats import SweepStats
from simulation.utils import symm, eigenvalues_sym3, projectors_sylvester, clamp

if TYPE_CHECKING:
    from simulation.config import Config


class FabricSolver(BaseLinearSolver):
    """Solves log-fabric evolution toward stress-aligned target with activity gating.

    Equation:
        dL/dt + (act/τ)·(L - L_target) - div(D·grad(L)) = 0

    - L_target: Traceless tensor derived from cycle-averaged stress Q̄.
    - act: Activity gate function (0 for isotropic stress, 1 for anisotropic).
    """

    _label = "fab"

    def __init__(
        self,
        L: fem.Function,
        L_old: fem.Function,
        Qbar: fem.Function,
        config: "Config",
    ):
        super().__init__(config, L, [], [])
        self.L = self.state
        self.L_old = L_old
        self.Qbar = Qbar

        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))
        self.tau_c = fem.Constant(self.mesh, float(self.cfg.fabric.fabric_tau))
        self.D_c = fem.Constant(self.mesh, float(self.cfg.fabric.fabric_D))

        self._use_diffusion = float(self.D_c.value) > 0.0

        # Scalar space for diagnostics derived from L (anisotropy magnitude, eigenvalue spread, etc.).
        P1 = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1)
        self._V_scalar = functionspace(self.mesh, P1)

        # Vector function space for eigenvector output
        P1_vec = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1, shape=(self.gdim,))
        self._V_vec = functionspace(self.mesh, P1_vec)

        # Principal direction output fields
        self.n1 = Function(self._V_vec, name="n1")
        self.n2 = Function(self._V_vec, name="n2")
        self.n3 = Function(self._V_vec, name="n3")

        # Scalar diagnostics for visualization/post-processing.
        # - A_fabric: scale-free anisotropy index A in [0, 1] (0 isotropic).
        # - m_ratio: fabric eigenvalue spread m_max/m_min (m_i = exp(l_i)).
        # - L_mag: Frobenius norm ||L||_F (trace-free, so dev(L)=L).
        self.A_fabric = Function(self._V_scalar, name="A_fabric")
        self.m_ratio = Function(self._V_scalar, name="m_ratio")
        self.L_mag = Function(self._V_scalar, name="L_mag")

    def _activity_factor(self):
        """Smooth activity factor [0, 1] based on directional information in Qbar."""
        if self.gdim != 3:
            raise ValueError("FabricSolver currently requires gdim==3.")

        epsQ = float(self.cfg.fabric.fabric_epsQ)
        I = ufl.Identity(3)
        Q = symm(self.Qbar) + epsQ * I

        trQ = ufl.tr(Q)
        Q_iso = (trQ / 3.0) * I
        Q_dev = Q - Q_iso

        r = ufl.sqrt(ufl.inner(Q_dev, Q_dev) + self.smooth_eps * self.smooth_eps) / (trQ + epsQ)

        aniso_eps = float(self.cfg.fabric.fabric_aniso_eps)
        act = r / (r + aniso_eps)
        return act

    def _compile_forms(self):
        dt = self.dt_c
        tau = self.tau_c
        D = self.D_c

        L_trial = ufl.TrialFunction(self.function_space)
        T = ufl.TestFunction(self.function_space)

        alpha = 1.0 / dt
        act = self._activity_factor()
        beta = act / tau

        a_ufl = (alpha + beta) * ufl.inner(L_trial, T) * self.dx
        if self._use_diffusion:
            a_ufl += D * ufl.inner(ufl.grad(L_trial), ufl.grad(T)) * self.dx
        self.a_form = fem.form(a_ufl)

        L_target = self._L_target_from_Qbar()
        rhs_ufl = (alpha * ufl.inner(self.L_old, T) + beta * ufl.inner(L_target, T)) * self.dx
        self.L_form = fem.form(rhs_ufl)

    def _L_target_from_Qbar(self):
        gdim = self.mesh.geometry.dim
        if gdim != 3:
            raise ValueError("FabricSolver currently requires gdim==3.")

        epsQ = float(self.cfg.fabric.fabric_epsQ)
        gammaF = float(self.cfg.fabric.fabric_gammaF)
        m_min = float(self.cfg.fabric.fabric_m_min)
        m_max = float(self.cfg.fabric.fabric_m_max)

        I = ufl.Identity(3)
        Q = symm(self.Qbar) + epsQ * I

        lam1, lam2, lam3 = eigenvalues_sym3(Q)
        P1, P2, P3 = projectors_sylvester(Q, lam1, lam2, lam3)

        prod = ufl.max_value(lam1 * lam2 * lam3, 1e-30)
        geo = ufl.exp((1.0 / 3.0) * ufl.ln(prod))

        a1 = lam1 / geo
        a2 = lam2 / geo
        a3 = lam3 / geo

        m1 = clamp(a1**gammaF, m_min, m_max, float(self.cfg.numerics.smooth_eps))
        m2 = clamp(a2**gammaF, m_min, m_max, float(self.cfg.numerics.smooth_eps))
        m3 = clamp(a3**gammaF, m_min, m_max, float(self.cfg.numerics.smooth_eps))

        # Normalize in log-space so tr(L_target)=0 exactly.
        # This ensures det(exp(L_target)) = 1 (volume preserving fabric).
        lnm1 = ufl.ln(m1)
        lnm2 = ufl.ln(m2)
        lnm3 = ufl.ln(m3)
        lnm_bar = (lnm1 + lnm2 + lnm3) / 3.0

        L_target = (lnm1 - lnm_bar) * P1 + (lnm2 - lnm_bar) * P2 + (lnm3 - lnm_bar) * P3
        return symm(L_target)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="fabric", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        self.dt_c.value = float(self.cfg.dt)
        self.Qbar.x.scatter_forward()
        self.L_old.x.scatter_forward()
        super().assemble_lhs()

    def assemble_rhs(self):
        self.L_old.x.scatter_forward()
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def _symmetrize_L(self) -> None:
        bs = int(self.function_space.dofmap.index_map_bs)
        n_owned = int(self.function_space.dofmap.index_map.size_local * bs)
        if n_owned <= 0:
            return
        gdim = self.mesh.geometry.dim
        if bs != gdim * gdim:
            raise ValueError(
                f"Unexpected tensor block size for L: bs={bs}, expected gdim*gdim={gdim*gdim}."
            )
        A = self.L.x.array[:n_owned].reshape(-1, gdim, gdim)
        A[:] = 0.5 * (A + np.swapaxes(A, 1, 2))
        self.L.x.scatter_forward()

    # -------------------------------------------------------------------------
    # CouplingBlock protocol
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return (self.L,)

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return (self.L_old,)

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        # Export L itself (for scale) alongside the principal directions.
        # The direction fields alone are ambiguous near isotropy and lack eigenvalue magnitudes.
        return (self.L, self.A_fabric, self.m_ratio, self.L_mag, self.n1, self.n2, self.n3)

    def post_step_update(self) -> None:
        self._update_diagnostics()

    def _update_diagnostics(self) -> None:
        """Update principal directions and scalar diagnostics derived from the current L."""
        bs = int(self.function_space.dofmap.index_map_bs)
        n_owned = int(self.function_space.dofmap.index_map.size_local * bs)
        gdim = self.gdim

        if n_owned <= 0:
            return

        L_arr = self.L.x.array[:n_owned].reshape(-1, gdim, gdim)
        L_sym = 0.5 * (L_arr + np.swapaxes(L_arr, 1, 2))

        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

        n_owned_v = int(self._V_vec.dofmap.index_map.size_local * self._V_vec.dofmap.index_map_bs)
        n_owned_s = int(self._V_scalar.dofmap.index_map.size_local * self._V_scalar.dofmap.index_map_bs)

        # np.linalg.eigh returns eigenpairs sorted in ascending eigenvalue order.
        # Use n1 as the dominant direction (largest eigenvalue), then n2, n3.
        self.n1.x.array[:n_owned_v] = eigenvectors[:, :, 2].flatten()  # max
        self.n2.x.array[:n_owned_v] = eigenvectors[:, :, 1].flatten()  # mid
        self.n3.x.array[:n_owned_v] = eigenvectors[:, :, 0].flatten()  # min

        self.n1.x.scatter_forward()
        self.n2.x.scatter_forward()
        self.n3.x.scatter_forward()

        # Scalar diagnostics (defined on the same P1 nodes as L).
        # Fabric eigenvalues are m_i = exp(l_i) for log-fabric eigenvalues l_i.
        m = np.exp(eigenvalues)
        m_mean = np.mean(m, axis=1, keepdims=True)
        m_dev = m - m_mean
        norm_dev = np.sqrt(np.sum(m_dev * m_dev, axis=1))
        norm_m = np.sqrt(np.sum(m * m, axis=1))
        tiny = 1e-12

        A = np.sqrt(3.0 / 2.0) * norm_dev / np.maximum(norm_m, tiny)
        A = np.clip(A, 0.0, 1.0)

        m_max = np.max(m, axis=1)
        m_min = np.min(m, axis=1)
        ratio = m_max / np.maximum(m_min, tiny)

        mag = np.sqrt(np.sum(L_sym * L_sym, axis=(1, 2)))

        self.A_fabric.x.array[:n_owned_s] = A
        self.m_ratio.x.array[:n_owned_s] = ratio
        self.L_mag.x.array[:n_owned_s] = mag
        self.A_fabric.x.scatter_forward()
        self.m_ratio.x.scatter_forward()
        self.L_mag.x.scatter_forward()

    def solve(self) -> SweepStats:
        t0 = time.perf_counter()
        self.assemble_lhs()
        self.assemble_rhs()
        t1 = time.perf_counter()
        stats = self._solve(assemble_time=t1 - t0)
        self._symmetrize_L()
        return stats

    def sweep(self) -> SweepStats:
        return self.solve()
