"""Linear subsolvers for mechanics, stimulus, and density."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import basix.ufl
import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem import functionspace, Function
from dolfinx.fem.petsc import (
    assemble_matrix,
    assemble_vector,
    apply_lifting,
    set_bc,
    create_matrix,
    create_vector,
)
import ufl

from simulation.utils import (
    build_nullspace,
    smooth_max,
    smoothstep01,
    clamp,
    symm,
    eigenvalues_sym3,
    projectors_sylvester,
)
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config


class _BaseLinearSolver:
    """Base class for PETSc KSP linear solvers."""

    def __init__(
        self,
        cfg: "Config",
        state_function: fem.Function,
        dirichlet_bcs: List[fem.DirichletBC],
        neumann_bcs: List[Tuple[fem.Function, int]],
    ):
        self.state = state_function
        self.function_space = state_function.function_space
        self.mesh = self.function_space.mesh
        self.comm = self.mesh.comm
        self.rank = self.comm.rank
        self.gdim = self.mesh.geometry.dim

        self.cfg = cfg
        self.dx = self.cfg.dx
        self.ds = self.cfg.ds
        self.logger = get_logger(self.comm, name=self.__class__.__name__, log_file=self.cfg.log_file)
        self.smooth_eps: float = self.cfg.numerics.smooth_eps

        self.trial = ufl.TrialFunction(self.function_space)
        self.test = ufl.TestFunction(self.function_space)

        self.last_reason: int = 0

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: PETSc.KSP = None
        self.A: PETSc.Mat = None
        self.b: PETSc.Vec = None

        self.a_form: fem.Form = None
        self.L_form: fem.Form = None

    def destroy(self):
        if self.ksp is not None:
            self.ksp.destroy()
            self.ksp = None
        if self.A is not None:
            self.A.destroy()
            self.A = None
        if self.b is not None:
            self.b.destroy()
            self.b = None

    # -------------------------------------------------------------------------
    # CouplingBlock protocol - default implementations
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        """Fields participating in fixed-point coupling. Override in subclass."""
        return ()

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        """Old-step counterparts for state_fields. Override in subclass."""
        return ()

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        """Fields to write to VTX storage. Override in subclass."""
        return ()

    def post_step_update(self) -> None:
        """Hook called after each accepted timestep. Override in subclass."""
        pass

    def setup(self):
        self._compile_forms()
        self.A = create_matrix(self.a_form)
        self.b = create_vector(self.function_space)
        self.assemble_lhs()
        self._setup_ksp()

    def assemble_lhs(self) -> None:
        self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()
        if self.ksp is not None:
            self.ksp.setOperators(self.A)

    def assemble_rhs(self) -> None:
        raise NotImplementedError

    def _solve(self) -> int:
        self.ksp.solve(self.b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()

        reason = int(self.ksp.getConvergedReason())
        self.last_reason = reason
        return reason

    def _maybe_warn(self, reason: int, label: str):
        if reason < 0:
            self.logger.warning(f"{label} solver failed to converge (reason: {reason})")

    def create_ksp(self, prefix: str, ksp_options: dict[str, object]) -> PETSc.KSP:
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOptionsPrefix(prefix + "_")

        opts = PETSc.Options()
        for k, v in ksp_options.items():
            if v is not None:
                opts[f"{prefix}_{k}"] = v

        opts[f"{prefix}_ksp_rtol"] = self.cfg.solver.ksp_rtol
        opts[f"{prefix}_ksp_atol"] = self.cfg.solver.ksp_atol
        opts[f"{prefix}_ksp_max_it"] = self.cfg.solver.ksp_max_it

        self.ksp.setInitialGuessNonzero(True)
        if self.A is not None:
            self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()
        self.ksp.setUp()
        return self.ksp


class MechanicsSolver(_BaseLinearSolver):
    """Linear elasticity with density-dependent stiffness."""

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
        mu_iso = E_iso / (2.0 * (1.0 + nu0))
        lmbda_iso = E_iso * nu0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
        sigma_iso = 2.0 * mu_iso * eps + lmbda_iso * ufl.tr(eps) * ufl.Identity(self.gdim)

        if L is None:
            return sigma_iso

        if self.gdim != 3:
            raise ValueError("Anisotropic MechanicsSolver currently requires gdim==3.")

        Ls = symm(L)
        l1, l2, l3 = eigenvalues_sym3(Ls)
        P1, P2, P3 = projectors_sylvester(Ls, l1, l2, l3)

        mean_l = (l1 + l2 + l3) / 3.0
        a1_hat = ufl.exp(l1 - mean_l)
        a2_hat = ufl.exp(l2 - mean_l)
        a3_hat = ufl.exp(l3 - mean_l)

        pE = float(self.cfg.material.stiff_pE)
        pG = float(self.cfg.material.stiff_pG)

        E1 = E_iso * (a1_hat**pE)
        E2 = E_iso * (a2_hat**pE)
        E3 = E_iso * (a3_hat**pE)

        G_iso = E_iso / (2.0 * (1.0 + nu0))
        G12 = G_iso * ((a1_hat * a2_hat) ** (0.5 * pG))
        G23 = G_iso * ((a2_hat * a3_hat) ** (0.5 * pG))
        G31 = G_iso * ((a3_hat * a1_hat) ** (0.5 * pG))

        # Orthotropic Poisson ratios: use nu0 for all directions.
        # With reciprocity, nu_ij/E_j = nu_ji/E_i.
        nu12 = nu0
        nu23 = nu0
        nu31 = nu0

        nu21 = nu12 * E2 / E1
        nu32 = nu23 * E3 / E2
        nu13 = nu31 * E1 / E3

        Sn = ufl.as_matrix(
            [
                [1.0 / E1, -nu21 / E2, -nu31 / E3],
                [-nu12 / E1, 1.0 / E2, -nu32 / E3],
                [-nu13 / E1, -nu23 / E2, 1.0 / E3],
            ]
        )
        Cn = ufl.inv(Sn)

        e1 = ufl.inner(P1, eps)
        e2 = ufl.inner(P2, eps)
        e3 = ufl.inner(P3, eps)
        e = ufl.as_vector([e1, e2, e3])
        s = ufl.dot(Cn, e)

        sigma_normal = s[0] * P1 + s[1] * P2 + s[2] * P3

        def _P_eps_P(A, B):
            return ufl.dot(A, ufl.dot(eps, B))

        sigma_shear = (
            2.0 * G12 * (_P_eps_P(P1, P2) + _P_eps_P(P2, P1))
            + 2.0 * G23 * (_P_eps_P(P2, P3) + _P_eps_P(P3, P2))
            + 2.0 * G31 * (_P_eps_P(P3, P1) + _P_eps_P(P1, P3))
        )

        sigma_aniso = sigma_normal + sigma_shear

        # Robust isotropic fallback when L is nearly spherical (avoid 0/0 in projectors).
        q = ufl.tr(Ls) / 3.0
        B = Ls - q * ufl.Identity(3)
        p2 = ufl.tr(ufl.dot(B, B)) / 6.0
        iso_L = ufl.lt(p2, 1e-14)
        return ufl.as_tensor([[ufl.conditional(iso_L, sigma_iso[i, j], sigma_aniso[i, j]) for j in range(3)] for i in range(3)])

    def assemble_rhs(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.dirichlet_bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.dirichlet_bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return int(reason)

    # -------------------------------------------------------------------------
    # CouplingBlock protocol - MechanicsSolver produces no coupled state
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return ()

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        # Displacement u is managed by GaitDriver which owns the output
        return ()

    def sweep(self):
        reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return {"label": "mech", "reason": int(reason)}


class FabricSolver(_BaseLinearSolver):
    """Log-fabric evolution: implicit Euler reaction–diffusion towards L_target(Qbar)."""

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

        self.logger = get_logger(self.comm, name="Fabric", log_file=self.cfg.log_file)

        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))
        self.cA_c = fem.Constant(self.mesh, float(self.cfg.fabric.fabric_cA))
        self.tau_c = fem.Constant(self.mesh, float(self.cfg.fabric.fabric_tau))
        self.D_c = fem.Constant(self.mesh, float(self.cfg.fabric.fabric_D))

        self._use_diffusion = float(self.D_c.value) > 0.0

        # Create vector function space for eigenvector output
        P1_vec = basix.ufl.element("Lagrange", self.mesh.basix_cell(), 1, shape=(self.gdim,))
        self._V_vec = functionspace(self.mesh, P1_vec)

        # Principal direction output fields (eigenvectors of L)
        # n1 -> largest eigenvalue (principal fabric direction)
        # n3 -> smallest eigenvalue
        self.n1 = Function(self._V_vec, name="n1")
        self.n2 = Function(self._V_vec, name="n2")
        self.n3 = Function(self._V_vec, name="n3")

    def _compile_forms(self):
        dt = self.dt_c
        cA = self.cA_c
        tau = self.tau_c
        D = self.D_c

        L_trial = ufl.TrialFunction(self.function_space)
        T = ufl.TestFunction(self.function_space)

        alpha = cA / dt
        beta = cA / tau

        a_ufl = (alpha + beta) * ufl.inner(L_trial, T) * self.dx
        if self._use_diffusion:
            a_ufl += (cA * D) * ufl.inner(ufl.grad(L_trial), ufl.grad(T)) * self.dx
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

        m1 = clamp(a1**gammaF, m_min, m_max)
        m2 = clamp(a2**gammaF, m_min, m_max)
        m3 = clamp(a3**gammaF, m_min, m_max)

        # Normalize by trace (ensures tr(M) = 3, i.e., tr(L_target) = 0).
        s = (m1 + m2 + m3) / 3.0

        m1n = m1 / s
        m2n = m2 / s
        m3n = m3 / s

        L_target = ufl.ln(m1n) * P1 + ufl.ln(m2n) * P2 + ufl.ln(m3n) * P3
        return symm(L_target)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="fabric", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        self.dt_c.value = float(self.cfg.dt)
        # Qbar (from driver) is referenced in L_target; L_old in the RHS.
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
                f"Unexpected tensor block size for L: bs={bs}, expected gdim*gdim={gdim*gdim}. "
                "This symmetrization assumes contiguous (gdim,gdim) blocks per DOF."
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
        """Return principal direction fields for visualization."""
        return (self.n1, self.n2, self.n3)

    def post_step_update(self) -> None:
        """Extract eigenvectors from L tensor after each accepted step."""
        self._update_eigenvectors()

    def _update_eigenvectors(self) -> None:
        """Extract eigenvectors from L tensor and store in n1, n2, n3 fields.

        n1 corresponds to largest eigenvalue (principal fabric direction),
        n3 to smallest eigenvalue.
        """
        bs = int(self.function_space.dofmap.index_map_bs)
        n_owned = int(self.function_space.dofmap.index_map.size_local * bs)
        gdim = self.gdim

        if n_owned <= 0:
            return

        # Reshape L to (n_nodes, 3, 3) tensor array
        L_arr = self.L.x.array[:n_owned].reshape(-1, gdim, gdim)

        # Symmetrize for eigendecomposition
        L_sym = 0.5 * (L_arr + np.swapaxes(L_arr, 1, 2))

        # Compute eigenvalues and eigenvectors for each node
        # np.linalg.eigh returns eigenvalues in ascending order
        _, eigenvectors = np.linalg.eigh(L_sym)

        # eigenvectors[:, :, i] is the eigenvector for eigenvalues[:, i]
        # eigenvalues are sorted ascending, so:
        # - [:, 2] -> largest eigenvalue -> n1
        # - [:, 1] -> middle eigenvalue -> n2
        # - [:, 0] -> smallest eigenvalue -> n3
        n_owned_v = int(self._V_vec.dofmap.index_map.size_local * self._V_vec.dofmap.index_map_bs)

        self.n1.x.array[:n_owned_v] = eigenvectors[:, :, 2].flatten()
        self.n2.x.array[:n_owned_v] = eigenvectors[:, :, 1].flatten()
        self.n3.x.array[:n_owned_v] = eigenvectors[:, :, 0].flatten()

        self.n1.x.scatter_forward()
        self.n2.x.scatter_forward()
        self.n3.x.scatter_forward()

    def solve(self) -> int:
        if float(self.dt_c.value) != float(self.cfg.dt):
            self.assemble_lhs()
        self.assemble_rhs()
        reason = self._solve()
        self._symmetrize_L()
        self._maybe_warn(reason, "Fabric")
        return int(reason)

    def sweep(self):
        reason = self.solve()
        return {"label": "fab", "reason": int(reason)}


class StimulusSolver(_BaseLinearSolver):
    """Update stimulus field `S` via diffusion/decay with a saturating drive.

    Model (dimensionless `S`):
      dS/dt = D_S ΔS - (1/tau_S) S + (1/tau_S) S_max tanh(delta/kappa),
      with the local quasi-static limit S = S_max tanh(delta/kappa) for tau_S=0.
    where delta is computed from the specific energy `m = psi/rho_safe` relative to `m_ref`.
    """

    def __init__(
        self,
        S: fem.Function,
        S_old: fem.Function,
        psi_field,
        rho: fem.Function,
        config: "Config",
    ):
        super().__init__(config, S, [], [])
        self.S = self.state
        self.S_old = S_old
        self.psi = psi_field
        self.rho = rho

        self.logger = get_logger(self.comm, name="Stimulus", log_file=self.cfg.log_file)

        # Time step is adaptive, so keep dt as a Constant (updates reassemble the LHS).
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

        # Stimulus parameters (Constants for UFL forms)
        self.tau_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_tau))
        self.D_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_D))
        self.S_max_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_S_max))
        self.kappa_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_kappa))
        self.delta0_c = fem.Constant(self.mesh, float(self.cfg.stimulus.stimulus_delta0))

        # Compile-time flag: avoid forming grad-grad if D_S is identically zero (helps DG spaces).
        self._use_diffusion = float(self.cfg.stimulus.stimulus_D) > 0.0

    def _compile_forms(self):
        dt = self.dt_c
        tau = self.tau_c

        S_trial = ufl.TrialFunction(self.function_space)
        v = ufl.TestFunction(self.function_space)

        # (tau/dt) * S^{n+1} + S^{n+1}  on the LHS, plus diffusion if enabled.
        alpha = tau / dt
        a_ufl = (alpha + 1.0) * S_trial * v * self.dx
        if self._use_diffusion:
            a_ufl += (tau * self.D_c) * ufl.dot(ufl.grad(S_trial), ufl.grad(v)) * self.dx
        self.a_form = fem.form(a_ufl)

        # Production term (explicit in time): lazy-zone + saturation
        #   delta = (m - m_ref) / m_ref
        #   |delta| <= delta0 -> drive = 0
        #   otherwise drive = S_max * tanh((|delta|-delta0)/kappa) * sign(delta)
        eps = float(self.cfg.numerics.smooth_eps)
        rho_safe = smooth_max(self.rho, self.cfg.density.rho_min, eps)
        m = self.psi / rho_safe
        m_ref = float(self.cfg.stimulus.psi_ref) / float(self.cfg.density.rho_ref)
        delta = (m - m_ref) / m_ref
        delta_abs = ufl.sqrt(delta * delta + eps * eps)
        delta_excess = smooth_max(delta_abs - self.delta0_c, 0.0, eps)
        sgn = delta / delta_abs
        drive = self.S_max_c * ufl.tanh(delta_excess / self.kappa_c) * sgn

        L_ufl = (alpha * self.S_old + drive) * v * self.dx
        self.L_form = fem.form(L_ufl)

    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        # dt may change between steps; update Constant and reassemble LHS
        self.dt_c.value = float(self.cfg.dt)
        super().assemble_lhs()

    def assemble_rhs(self):
        self.S_old.x.scatter_forward()
        if hasattr(self.psi, "x"):
            self.psi.x.scatter_forward()
        self.rho.x.scatter_forward()

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    # -------------------------------------------------------------------------
    # CouplingBlock protocol
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return (self.S,)

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return (self.S_old,)

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        return (self.S,)

    def solve(self) -> int:
        # Reassemble LHS when dt changes (adaptive stepping).
        if float(self.dt_c.value) != float(self.cfg.dt):
            self.assemble_lhs()
        self.assemble_rhs()
        reason = self._solve()
        self._maybe_warn(reason, "Stimulus")
        return int(reason)

    def sweep(self):
        reason = self.solve()
        return {"label": "stim", "reason": int(reason)}


class DensitySolver(_BaseLinearSolver):
    """Density evolution (implicit Euler diffusion + bounded formation/resorption kinetics).

    Continuous model implemented (up to smoothing of clamps):

        dρ/dt = Dρ Δρ
               + k_form S_+ (1 - ρ/ρ_max)
               - k_res  S_- (1 - ρ/ρ_min),

    optionally scaled by a surface availability A(ρ_old) when cfg.surface_use=True.
    """

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        stimulus: fem.Function,
        config: "Config",
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.S = stimulus
        self.dt_c = fem.Constant(self.mesh, float(self.cfg.dt))

    def _compile_forms(self):
        dt = self.dt_c

        # Smooth helpers (avoid hard clamps / if-branches in UFL)
        eps = float(self.cfg.numerics.smooth_eps)

        def smooth_min(a, b, eps_):
            # smooth_min(a,b) = a + b - smooth_max(a,b)
            return a + b - smooth_max(a, b, eps_)

        # Positive / negative stimulus parts (non-negative)
        S_pos = smooth_max(self.S, 0.0, eps)
        S_neg = smooth_max(-self.S, 0.0, eps)

        # Surface availability A(rho_old) from apparent density via a vascular-porosity proxy.
        if bool(self.cfg.density.surface_use):
            rho_t = float(self.cfg.density.rho_tissue)
            f_raw = 1.0 - (self.rho_old / rho_t)
            f = smooth_min(smooth_max(f_raw, 0.0, eps), 1.0, eps)

            # Martin polynomial for specific surface S_V(f) [1/mm]
            S_v = (
                32.3 * f
                - 93.9 * f**2
                + 134.0 * f**3
                - 101.0 * f**4
                + 28.8 * f**5
            )
            S_v = smooth_max(S_v, 0.0, eps)

            A_min = float(self.cfg.density.surface_A_min)
            S0 = float(self.cfg.density.surface_S0)
            x = S_v / S0
            x = smooth_min(x, 1.0, eps)  # linear in S_v, capped
            A_surf = A_min + (1.0 - A_min) * x
        else:
            A_surf = 1.0

        # Soft-bounded kinetics: formation vanishes as rho -> rho_max; resorption vanishes as rho -> rho_min.
        rho_min = float(self.cfg.density.rho_min)
        rho_max = float(self.cfg.density.rho_max)

        k_form = float(self.cfg.density.k_rho_form) * A_surf
        k_res = float(self.cfg.density.k_rho_resorb) * A_surf

        # Linear reaction coefficient on rho^{n+1}
        reaction = (k_form * S_pos / rho_max) + (k_res * S_neg / rho_min)

        a_ufl = (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.density.D_rho * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction * self.trial * self.test * self.dx
        )
        self.a_form = fem.form(a_ufl)

        # RHS constants (note: the reaction term ensures the bounds are attracting).
        L_ufl = ((self.rho_old / dt) + (k_form * S_pos) + (k_res * S_neg)) * self.test * self.dx
        self.L_form = fem.form(L_ufl)
    def _setup_ksp(self):
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.solver.ksp_type, "pc_type": self.cfg.solver.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)

    def assemble_lhs(self) -> None:
        self.dt_c.value = float(self.cfg.dt)
        # Coefficient fields appear in the bilinear form -> must be ghost-synchronized before matrix assembly.
        self.S.x.scatter_forward()
        self.rho_old.x.scatter_forward()
        super().assemble_lhs()

    def assemble_rhs(self):
        self.S.x.scatter_forward()
        self.rho_old.x.scatter_forward()

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self.L_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    # -------------------------------------------------------------------------
    # CouplingBlock protocol
    # -------------------------------------------------------------------------

    @property
    def state_fields(self) -> Tuple[fem.Function, ...]:
        return (self.rho,)

    @property
    def state_fields_old(self) -> Tuple[fem.Function, ...]:
        return (self.rho_old,)

    @property
    def output_fields(self) -> Tuple[fem.Function, ...]:
        return (self.rho,)

    def solve(self) -> int:
        # LHS depends on dt, stimulus, and rho_old (via surface availability) -> reassemble each call
        self.assemble_lhs()
        self.assemble_rhs()
        reason = self._solve()
        self._maybe_warn(reason, "Density")
        return int(reason)

    def sweep(self):
        reason = self.solve()
        return {"label": "dens", "reason": int(reason)}
