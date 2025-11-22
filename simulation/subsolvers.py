"""PDE subsolvers: mechanics (u), stimulus (S), density (ρ), direction (A)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
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
    build_nullspace, unittrace_psd, spectral_decomposition_3x3, matrix_exp,
    smooth_abs, smooth_plus, smooth_max, smooth_heaviside
)
from simulation.logger import get_logger

if TYPE_CHECKING:
    from simulation.config import Config


def get_distal_damping_mask(mesh, z_min: float, height: float = 15.0, transition: float = 5.0):
    """Spatial mask dampening distal boundary: 0 below z_min+height, smooth transition, 1 above."""
    x = ufl.SpatialCoordinate(mesh)
    z = x[2]
    z_start = z_min + height
    
    if transition <= 1e-14:
        # Step function if transition is effectively zero
        return ufl.conditional(ufl.ge(z, z_start), 1.0, 0.0)
    
    t = (z - z_start) / transition
    return ufl.max_value(0.0, ufl.min_value(1.0, t))


# --- Base linear solver class ---

class _BaseLinearSolver:
    """Base linear solver with assembly, KSP solve, and iteration tracking."""

    def __init__(
        self,
        cfg: Config,
        state_function: fem.Function,
        dirichlet_bcs: list[fem.DirichletBC],
        neumann_bcs: list[tuple[fem.Function, int]],
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
        self.logger = get_logger(self.comm, verbose=self.cfg.verbose, name=self.__class__.__name__)
        self.smooth_eps: float = self.cfg.smooth_eps

        self.trial = ufl.TrialFunction(self.function_space)
        self.test = ufl.TestFunction(self.function_space)

        self.total_iters = 0
        self.ksp_steps = 0
        self.last_reason: int | None = None
        self.last_iters: int | None = None

        self.dirichlet_bcs = dirichlet_bcs
        self.neumann_bcs = neumann_bcs

        self.ksp: PETSc.KSP | None = None
        self.A: PETSc.Mat | None = None
        self.b: PETSc.Vec | None = create_vector(self.function_space)
        self.a_form: ufl.Form | None = None

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

    def _reset_stats(self):
        self.last_reason = None
        self.last_iters = None

    def build_lhs_form(self):
        """Return UFL bilinear form for the current LHS."""
        raise NotImplementedError(f"{self.__class__.__name__}.build_lhs_form() not implemented")

    def assemble_lhs(self):
        """(Re)build and assemble LHS matrix from subclass-provided form."""
        self.a_form = fem.form(self.build_lhs_form())
        if self.A is None:
            self.A = create_matrix(self.a_form)
        else:
            self.A.zeroEntries()
        assemble_matrix(self.A, self.a_form, bcs=self.dirichlet_bcs)
        self.A.assemble()

        if self.ksp is not None:
            self.ksp.setOperators(self.A)
            self.ksp.setUp()
        self._reset_stats()

    def _solve(self) -> tuple[int, int]:
        self.ksp.solve(self.b, self.state.x.petsc_vec)
        self.state.x.scatter_forward()
        its = self.ksp.getIterationNumber()
        reason = self.ksp.getConvergedReason()
        self.total_iters += its
        self.ksp_steps += 1
        self.last_iters = its
        self.last_reason = reason
        return its, reason

    @property
    def ksp_its(self) -> int:
        return int(self.total_iters)

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
        opts[f"{prefix}_ksp_rtol"] = self.cfg.ksp_rtol
        opts[f"{prefix}_ksp_atol"] = self.cfg.ksp_atol
        opts[f"{prefix}_ksp_max_it"] = self.cfg.ksp_max_it

        self.ksp.setInitialGuessNonzero(True)
        if self.A is not None:
            self.ksp.setOperators(self.A)

        self.ksp.setFromOptions()
        self.ksp.setUp()
        return self.ksp


# --- Subsolver implementations ---

class MechanicsSolver(_BaseLinearSolver):
    """Anisotropic elastic equilibrium solver (Zysset-Curnier model)."""

    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        A_dir: fem.Function,
        config: Config,
        dirichlet_bcs: list[fem.DirichletBC],
        neumann_bcs: list[tuple[fem.Function, int]],
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho
        self.A_dir = A_dir

        # Neumann RHS template
        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_form = ufl.inner(zero_vec, self.test) * self.ds
        for t, tag in self.neumann_bcs:
            L_form = L_form + ufl.inner(t, self.test) * self.ds(tag)
        self.L_ufl = L_form
        self.L_form = fem.form(L_form)

    def build_lhs_form(self):
        return ufl.inner(self.sigma(self.trial, self.rho, self.A_dir), self.eps(self.test)) * self.dx

    def eps(self, u):
        """Strain tensor: sym(grad(u))."""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho, L_dir):
        """Cauchy stress [MPa] via Zysset-Curnier model with log-fabric L_dir."""
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)
        
        # 1. Spectral Decomposition of L
        l1, l2, l3 = spectral_decomposition_3x3(L_dir)
        
        # 2. Eigenvalues of A = exp(L)
        m1 = ufl.exp(l1)
        m2 = ufl.exp(l2)
        m3 = ufl.exp(l3)
        
        # 3. Zysset-Curnier Stiffness Construction
        # Parameters
        # k = self.cfg.k_stiff  <-- Replaced by variable exponent
        p = self.cfg.p_stiff
        E0 = self.cfg.E0_z
        G0 = self.cfg.G0_z
        nu0 = self.cfg.nu0_z
        
        # Variable exponent k(rho) interpolating between n_trab and n_cort
        # We use a smooth transition based on rho
        # k(rho) = n_trab * (1 - w) + n_cort * w
        # w(rho) = smoothstep(rho, rho_trab_max, rho_cort_min)
        
        def smoothstep(x, edge0, edge1):
            # Clamp x to [edge0, edge1] and map to [0, 1]
            t = ufl.max_value(0.0, ufl.min_value(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        w = smoothstep(rho_eff, self.cfg.rho_trab_max, self.cfg.rho_cort_min)
        k_var = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w
        
        # Eigenvalues of stiffness
        # E_i = E0 * rho^k * m_i^p
        E1 = E0 * (rho_eff**k_var) * (m1**p)
        E2 = E0 * (rho_eff**k_var) * (m2**p)
        E3 = E0 * (rho_eff**k_var) * (m3**p)
        
        # Shear moduli (approximate geometric mean)
        # G_ij = G0 * rho^k * (mi*mj)^(p/2)
        G12 = G0 * (rho_eff**k_var) * ((m1*m2)**(p/2.0))
        G23 = G0 * (rho_eff**k_var) * ((m2*m3)**(p/2.0))
        G31 = G0 * (rho_eff**k_var) * ((m3*m1)**(p/2.0))
        
        # Poisson ratios (assumed constant or weak dependence)
        # nu_ij = nu0 * (mj/mi)^(p/2) ? 
        # Simplified: nu_ij = nu0
        nu12 = nu21 = nu23 = nu32 = nu31 = nu13 = nu0
        
        # Construct Stiffness Tensor C in principal frame of A
        # We need eigenprojectors M_i = v_i (x) v_i
        # Use Sylvester's formula on L to get M_i (same eigenvectors as A)
        I = ufl.Identity(3)
        eps = 1e-5
        def safe_denom(d):
            return ufl.conditional(ufl.lt(abs(d), eps), eps, d)
            
        d12 = safe_denom(l1 - l2)
        d13 = safe_denom(l1 - l3)
        d23 = safe_denom(l2 - l3)
        
        M1 = (L_dir - l2*I) * (L_dir - l3*I) / (d12 * d13)
        M2 = (L_dir - l1*I) * (L_dir - l3*I) / (-d12 * d23)
        M3 = (L_dir - l1*I) * (L_dir - l2*I) / (-d13 * -d23)
        
        # Strain
        eps_ten = self.eps(u)
        
        # Stress calculation: sigma = C : eps
        
        eps11 = ufl.inner(eps_ten, M1)
        eps22 = ufl.inner(eps_ten, M2)
        eps33 = ufl.inner(eps_ten, M3)
        
        factor = 1.0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
        lam_11 = E1 * (1.0 - nu0) * factor
        lam_22 = E2 * (1.0 - nu0) * factor
        lam_33 = E3 * (1.0 - nu0) * factor
        
        lam_12 = ufl.sqrt(E1 * E2) * nu0 * factor
        lam_13 = ufl.sqrt(E1 * E3) * nu0 * factor
        lam_23 = ufl.sqrt(E2 * E3) * nu0 * factor
        
        # Normal stress parts
        # sigma_N = sum_i (sum_j lam_ij eps_jj) M_i
        sig_N1 = (lam_11 * eps11 + lam_12 * eps22 + lam_13 * eps33) * M1
        sig_N2 = (lam_12 * eps11 + lam_22 * eps22 + lam_23 * eps33) * M2
        sig_N3 = (lam_13 * eps11 + lam_23 * eps22 + lam_33 * eps33) * M3
        
        # Shear stress parts
        # sigma_S = sum_{i<j} 2 G_ij (Mi eps Mj)_sym
        
        # We can compute T_ij = (Mi eps Mj + Mj eps Mi) directly
        T12 = M1 * eps_ten * M2 + M2 * eps_ten * M1
        T23 = M2 * eps_ten * M3 + M3 * eps_ten * M2
        T31 = M3 * eps_ten * M1 + M1 * eps_ten * M3
        
        sig_S12 = 2.0 * G12 * T12 # G12 is shear modulus, T12 contains shear strain
        sig_S23 = 2.0 * G23 * T23
        sig_S31 = 2.0 * G31 * T31
        
        return sig_N1 + sig_N2 + sig_N3 + sig_S12 + sig_S23 + sig_S31

    def get_strain_tensor(self, u=None):
        return self.eps(self.u if u is None else u)

    def setup(self):
        self.assemble_lhs()

        ns = build_nullspace(self.function_space)
        self.A.setBlockSize(self.gdim)
        self.A.setNearNullSpace(ns)
        self.A.setOption(PETSc.Mat.Option.SPD, True)

        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="mechanics", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self):
        with self.b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(self.b, self.L_form)
        apply_lifting(self.b, [self.a_form], bcs=[self.dirichlet_bcs])
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        set_bc(self.b, self.dirichlet_bcs)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Mechanics")
        return its, reason

    def energy_balance(self):
        """Internal/external work and relative error."""
        # Internal work: a(u,u)
        a_uu_local = fem.assemble_scalar(fem.form(ufl.inner(self.sigma(self.u, self.rho, self.A_dir), self.eps(self.u)) * self.dx))
        W_int = self.comm.allreduce(a_uu_local, op=MPI.SUM)
        
        # External work: l(u)
        # We need l(u). L_form is defined as inner(t, v) * ds.
        # We can replace v with u.
        L_u_form = fem.form(ufl.replace(self.L_ufl, {self.test: self.u}))
        l_u_local = fem.assemble_scalar(L_u_form)
        W_ext = self.comm.allreduce(l_u_local, op=MPI.SUM)
        
        denom = max(abs(W_int), abs(W_ext), 1e-300)
        rel_error = abs(W_int - W_ext) / denom
        return W_int, W_ext, rel_error

    def average_strain_energy(self):
        """Domain-averaged strain energy density."""
        psi = 0.5 * ufl.inner(self.sigma(self.u, self.rho, self.A_dir), self.eps(self.u))
        E_local = fem.assemble_scalar(fem.form(psi * self.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        
        E_total = self.comm.allreduce(E_local, op=MPI.SUM)
        vol_total = self.comm.allreduce(vol_local, op=MPI.SUM)
        
        return E_total / max(vol_total, 1e-300)


class StimulusSolver(_BaseLinearSolver):
    """Reaction-diffusion stimulus solver with mechanical driver."""

    def __init__(
        self,
        S: fem.Function,
        S_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, S, [], [])
        self.S = self.state
        self.S_old = S_old
        self._rhs_form = None
        self.z_min = None

    def _compute_z_min(self):
        if self.z_min is None:
            z_coords = self.mesh.geometry.x[:, 2]
            local_min = z_coords.min() if z_coords.size > 0 else 1e30
            self.z_min = self.comm.allreduce(local_min, op=MPI.MIN)
        return self.z_min

    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if self.cfg.tauS <= 0:
            raise ValueError(f"tauS must be positive, got {self.cfg.tauS}")
            
        return (
            (self.cfg.cS / dt + self.cfg.cS / self.cfg.tauS) * self.trial * self.test * self.dx
            + self.cfg.kappaS * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="stimulus", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self, psi_expr):
        with self.b.localForm() as b_local:
            b_local.set(0.0)
        self.S_old.x.scatter_forward()
        dt = self.cfg.dt

        # Apply distal damping to psi to avoid artifacts at u=0 boundary
        z_min = self._compute_z_min()
        mask = get_distal_damping_mask(self.mesh, z_min, height=self.cfg.distal_damping_height, 
                                       transition=self.cfg.distal_damping_transition)
        psi_effective = psi_expr * mask

        rhs = (self.cfg.cS / dt) * self.S_old + self.cfg.rS_gain * (psi_effective - 1.0)
        self._rhs_form = fem.form(rhs * self.test * self.dx)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Stimulus")
        return its, reason

    def power_balance_residual(self, psi_expr):
        """Compute power balance residual: Storage + Decay - Source."""
        dt = self.cfg.dt
        S, S_old = self.S, self.S_old
        
        # Terms
        storage = (self.cfg.cS / dt) * (S - S_old)
        decay = (self.cfg.cS / self.cfg.tauS) * S
        
        # Source
        z_min = self._compute_z_min()
        mask = get_distal_damping_mask(self.mesh, z_min, height=self.cfg.distal_damping_height, 
                                       transition=self.cfg.distal_damping_transition)
        psi_eff = psi_expr * mask
        source = self.cfg.rS_gain * (psi_eff - 1.0)
        
        integrand = storage + decay - source
        res_local = fem.assemble_scalar(fem.form(integrand * self.dx))
        res_abs = self.comm.allreduce(res_local, op=MPI.SUM)
        
        # Relative to source magnitude
        src_mag_local = fem.assemble_scalar(fem.form(abs(source) * self.dx))
        src_mag = self.comm.allreduce(src_mag_local, op=MPI.SUM)
        
        return res_abs, abs(res_abs) / max(src_mag, 1e-300)


class DensitySolver(_BaseLinearSolver):
    """Density evolution with anisotropic diffusion and mechanostat remodeling."""

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        L_dir: fem.Function,
        S: fem.Function,
        config: Config,
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.L_dir = L_dir
        self.S = S
        self.L_form_template = None
        self.a_form = fem.form(self.build_lhs_form())

    
    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        d = self.mesh.geometry.dim
        I = ufl.Identity(d)

        # Recover Fabric A = exp(L)
        A = matrix_exp(self.L_dir)
        
        # Normalize A to unit trace for diffusion tensor construction
        Ahat = unittrace_psd(A, d, eps=self.smooth_eps)
        
        Bten = self.cfg.beta_perp * I + (self.cfg.beta_par - self.cfg.beta_perp) * Ahat

        # Linear driver: rate = k_rho * (S_plus*(rho_max - rho) + S_minus*(rho_min - rho))
        # LHS contribution: + k_rho * (S_plus + S_minus) * rho
        S_plus = smooth_plus(self.S, self.smooth_eps)
        S_minus = smooth_plus(-self.S, self.smooth_eps)
        
        reaction_coeff = self.cfg.k_rho * (S_plus + S_minus)

        return (
            (self.trial / dt) * self.test * self.dx
            + ufl.inner(Bten * ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction_coeff * self.trial * self.test * self.dx
        )

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self):
        with self.b.localForm() as b_local:
            b_local.set(0.0)

        dt = self.cfg.dt
        
        # Linear driver source term: k_rho * (S_plus*rho_max + S_minus*rho_min)
        S_plus = smooth_plus(self.S, self.smooth_eps)
        S_minus = smooth_plus(-self.S, self.smooth_eps)
        
        source_term = self.cfg.k_rho * (S_plus * self.cfg.rho_max + S_minus * self.cfg.rho_min)

        rhs_expr = (self.rho_old / dt) + source_term
        self.L_form_template = fem.form(rhs_expr * self.test * self.dx)

        assemble_vector(self.b, self.L_form_template)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Density")
        return its, reason

    def mass_balance_residual(self):
        """Compute mass balance residual."""
        dt = self.cfg.dt
        rho, rho_old = self.rho, self.rho_old
        
        S_plus = smooth_plus(self.S, self.smooth_eps)
        S_minus = smooth_plus(-self.S, self.smooth_eps)
        
        # rate = k_rho * (S_plus*(rho_max - rho) + S_minus*(rho_min - rho))
        rate = self.cfg.k_rho * (S_plus * (self.cfg.rho_max - rho) + S_minus * (self.cfg.rho_min - rho))
        
        lhs = (rho - rho_old) / dt
        rhs = rate
        
        res_local = fem.assemble_scalar(fem.form((lhs - rhs) * self.dx))
        res_abs = self.comm.allreduce(res_local, op=MPI.SUM)
        
        rhs_mag_local = fem.assemble_scalar(fem.form(abs(rhs) * self.dx))
        rhs_mag = self.comm.allreduce(rhs_mag_local, op=MPI.SUM)
        
        return res_abs, abs(res_abs) / max(rhs_mag, 1e-300)


class DirectionSolver(_BaseLinearSolver):
    """Log-fabric tensor solver: reaction-diffusion toward strain-aligned target.
    
    Evolves L = log(A) with cA dL/dt - div(D grad L) + r L = r L_M.
    """

    def __init__(
        self,
        L_dir: fem.Function,
        L_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, L_dir, [], [])
        self.L_dir = self.state
        self.L_old = L_old
        self._rhs_form = None

    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        tauA = self.cfg.tauA
        if tauA <= 0:
            raise ValueError(f"tauA must be positive, got {tauA}")
        ell2 = self.cfg.ell ** 2
        
        # Diffusion of L
        return (
            (self.cfg.cA / dt + self.cfg.cA / tauA) * ufl.inner(self.trial, self.test) * self.dx
            + self.cfg.cA * (ell2 / tauA) * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
        )

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="direction", ksp_options=ksp_options)
        self._reset_stats()

    def assemble_rhs(self, L_target_expr):
        dt = self.cfg.dt
        tauA = self.cfg.tauA
        
        rhs_ten = (self.cfg.cA / dt) * self.L_old + (self.cfg.cA / tauA) * L_target_expr
        self._rhs_form = fem.form(ufl.inner(rhs_ten, self.test) * self.dx)

        with self.b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self.b, self._rhs_form)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Direction")
        return its, reason

    def trace_balance_residual(self, L_target_expr):
        """Compute domain-averaged trace of L and L_target."""
        trL = ufl.tr(self.L_dir)
        trLM = ufl.tr(L_target_expr)
        
        vol = self.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * self.dx)), op=MPI.SUM)
        trL_avg = self.comm.allreduce(fem.assemble_scalar(fem.form(trL * self.dx)), op=MPI.SUM) / vol
        trLM_avg = self.comm.allreduce(fem.assemble_scalar(fem.form(trLM * self.dx)), op=MPI.SUM) / vol
        
        # Residual of L - L_target
        res_sq = self.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(self.L_dir - L_target_expr, self.L_dir - L_target_expr) * self.dx)), op=MPI.SUM)
        import numpy as np
        return trL_avg, trLM_avg, np.sqrt(res_sq)
