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
    build_nullspace,
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
        self.logger = get_logger(self.comm, verbose=(self.cfg.verbose is True), name=self.__class__.__name__)
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
    """Isotropic elastic equilibrium solver."""

    def __init__(
        self,
        u: fem.Function,
        rho: fem.Function,
        config: Config,
        dirichlet_bcs: list[fem.DirichletBC],
        neumann_bcs: list[tuple[fem.Function, int]],
    ):
        super().__init__(config, u, dirichlet_bcs, neumann_bcs)
        self.u = self.state
        self.rho = rho

        # Neumann RHS template
        zero_vec = fem.Constant(self.mesh, (0.0,) * self.gdim)
        L_form = ufl.inner(zero_vec, self.test) * self.ds
        for t, tag in self.neumann_bcs:
            L_form = L_form + ufl.inner(t, self.test) * self.ds(tag)
        self.L_ufl = L_form
        self.L_form = fem.form(L_form)

    def build_lhs_form(self):
        return ufl.inner(self.sigma(self.trial, self.rho), self.eps(self.test)) * self.dx

    def eps(self, u):
        """Strain tensor: sym(grad(u))."""
        return ufl.sym(ufl.grad(u))

    def sigma(self, u, rho):
        """Cauchy stress [MPa] via Isotropic model."""
        rho_eff = smooth_max(rho, self.cfg.rho_min, self.smooth_eps)
        
        # Parameters
        E0 = self.cfg.E0
        nu0 = self.cfg.nu0
        
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
        
        # Young's modulus
        E = E0 * (rho_eff**k_var)
        
        # Lame parameters
        mu = E / (2.0 * (1.0 + nu0))
        lmbda = E * nu0 / ((1.0 + nu0) * (1.0 - 2.0 * nu0))
        
        # Stress
        return 2.0 * mu * self.eps(u) + lmbda * ufl.tr(self.eps(u)) * ufl.Identity(self.gdim)

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
        a_uu_local = fem.assemble_scalar(fem.form(ufl.inner(self.sigma(self.u, self.rho), self.eps(self.u)) * self.dx))
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
        psi = 0.5 * ufl.inner(self.sigma(self.u, self.rho), self.eps(self.u))
        E_local = fem.assemble_scalar(fem.form(psi * self.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * self.dx))
        
        E_total = self.comm.allreduce(E_local, op=MPI.SUM)
        vol_total = self.comm.allreduce(vol_local, op=MPI.SUM)
        
        return E_total / max(vol_total, 1e-300)


class DensitySolver(_BaseLinearSolver):
    """Density evolution with isotropic diffusion and local remodeling."""

    def __init__(
        self,
        rho: fem.Function,
        rho_old: fem.Function,
        config: Config,
    ):
        super().__init__(config, rho, [], [])
        self.rho = self.state
        self.rho_old = rho_old
        self.L_form_template = None
        self.z_min = None
        
        # Internal function to store the driving force (S = psi/psi_ref - 1)
        # This allows us to use it in both LHS (implicit reaction) and RHS (source)
        self.S_driving = fem.Function(self.function_space, name="S_driving")
        
        # Initialize LHS form
        self.a_form = fem.form(self.build_lhs_form())

    def _compute_z_min(self):
        if self.z_min is None:
            z_coords = self.mesh.geometry.x[:, 2]
            local_min = z_coords.min() if z_coords.size > 0 else 1e30
            self.z_min = self.comm.allreduce(local_min, op=MPI.MIN)
        return self.z_min
    
    def update_driving_force(self, psi_expr):
        """
        Update the internal driving force field S = (psi / psi_ref) - 1.
        Applies distal damping to psi before calculation.
        """
        z_min = self._compute_z_min()
        mask = get_distal_damping_mask(self.mesh, z_min, height=self.cfg.distal_damping_height, 
                                       transition=self.cfg.distal_damping_transition)
        psi_effective = psi_expr * mask
        
        driving_force_expr = (psi_effective / self.cfg.psi_ref) - 1.0
        
        # Interpolate expression into S_driving function
        expr = fem.Expression(driving_force_expr, self.function_space.element.interpolation_points)
        self.S_driving.interpolate(expr)
        self.S_driving.x.scatter_forward()

    def build_lhs_form(self):
        dt = self.cfg.dt
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        
        # Use the stored driving force function
        # Note: S_driving is updated before assemble_lhs is called in the loop
        S_driving = self.S_driving

        S_plus = smooth_plus(S_driving, self.smooth_eps)
        S_minus = smooth_plus(-S_driving, self.smooth_eps)
        
        # Reaction coefficient for implicit treatment: k_rho * (S_plus + S_minus) * rho
        reaction_coeff = self.cfg.k_rho * (S_plus + S_minus)

        return (
            (self.trial / dt) * self.test * self.dx
            + self.cfg.beta * ufl.inner(ufl.grad(self.trial), ufl.grad(self.test)) * self.dx
            + reaction_coeff * self.trial * self.test * self.dx
        )

    def assemble_rhs(self):
        with self.b.localForm() as b_local:
            b_local.set(0.0)

        dt = self.cfg.dt
        S_driving = self.S_driving

        S_plus = smooth_plus(S_driving, self.smooth_eps)
        S_minus = smooth_plus(-S_driving, self.smooth_eps)

        # Source term: k_rho * (S_plus * rho_max + S_minus * rho_min)
        source_term = self.cfg.k_rho * (S_plus * self.cfg.rho_max + S_minus * self.cfg.rho_min)

        rhs_expr = (self.rho_old / dt) + source_term
        self.L_form_template = fem.form(rhs_expr * self.test * self.dx)

        assemble_vector(self.b, self.L_form_template)
        self.b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        self.b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

    def setup(self):
        self.assemble_lhs()
        self.A.setOption(PETSc.Mat.Option.SPD, True)
        ksp_options = {"ksp_type": self.cfg.ksp_type, "pc_type": self.cfg.pc_type}
        self.create_ksp(prefix="density", ksp_options=ksp_options)
        self._reset_stats()

    def solve(self):
        its, reason = self._solve()
        self._maybe_warn(reason, "Density")
        return its, reason

    def mass_balance_residual(self):
        """Compute mass balance residual."""
        dt = self.cfg.dt
        rho, rho_old = self.rho, self.rho_old
        S_driving = self.S_driving

        S_plus = smooth_plus(S_driving, self.smooth_eps)
        S_minus = smooth_plus(-S_driving, self.smooth_eps)
        
        # rate = k_rho * (S_plus*(rho_max - rho) + S_minus*(rho_min - rho))
        rate = self.cfg.k_rho * (S_plus * (self.cfg.rho_max - rho) + S_minus * (self.cfg.rho_min - rho))
        
        lhs = (rho - rho_old) / dt
        rhs = rate
        
        res_local = fem.assemble_scalar(fem.form((lhs - rhs) * self.dx))
        res_abs = self.comm.allreduce(res_local, op=MPI.SUM)
        
        rhs_mag_local = fem.assemble_scalar(fem.form(abs(rhs) * self.dx))
        rhs_mag = self.comm.allreduce(rhs_mag_local, op=MPI.SUM)
        
        return res_abs, abs(res_abs) / max(rhs_mag, 1e-300)
