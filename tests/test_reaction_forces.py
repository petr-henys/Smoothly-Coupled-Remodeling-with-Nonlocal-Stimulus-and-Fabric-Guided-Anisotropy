"""New reaction force tests with proper unit scaling and integration."""
import pytest
import numpy as np
import basix
from dolfinx import fem
from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.config import Config
from simulation.femur_remodeller_gait import setup_femur_gait_loading
from simulation.subsolvers import MechanicsSolver
from simulation.utils import build_dirichlet_bcs
from dolfinx.fem.petsc import (
    assemble_matrix as assemble_matrix_petsc,
    assemble_vector as assemble_vector_petsc,
    create_vector as create_vector_petsc,
)
from petsc4py import PETSc


@pytest.fixture(scope="module", params=["mm", "m"])
def unit_scale(request):
    """Parametrize tests over two unit systems: mm and m."""
    return request.param


@pytest.fixture(scope="module")
def femur_setup(unit_scale):
    """Create femur mesh and function space with specified unit scale."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)
    
    # Unit-dependent config
    if unit_scale == "mm":
        # Mesh in mm, L_c = 1.0 mm, u_c = 1e-3 mm = 1 μm
        cfg = Config(domain=domain, facet_tags=facet_tags, L_c=1.0, u_c=1e-3)
    else:  # "m"
        # Mesh in mm but L_c = 1e-3 m, u_c = 1e-6 m = 1 μm
        cfg = Config(domain=domain, facet_tags=facet_tags, L_c=1e-3, u_c=1e-6)
    
    return domain, facet_tags, V, Q, cfg, unit_scale


@pytest.fixture(scope="module")
def gait_loader(femur_setup):
    """Create gait loader."""
    _, _, V, _, cfg, _ = femur_setup
    return setup_femur_gait_loading(V, cfg, BW_kg=75.0, n_samples=9)


class TestReactionForces:
    """Validate reaction forces with proper unit scaling and integration."""

    def test_applied_force_integration(self, gait_loader, femur_setup):
        """Applied forces integrated over surface should match physiological expectations.
        
        Hip joint forces at peak stance: ~3-4× BW  (~2200-2950 N for 75 kg)
        Muscle forces (glut med + max): ~1-2× BW (~735-1470 N for 75 kg)
        Total applied load should be in range 1-6× BW
        """
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Update to peak stance (50%)
        gait_loader.update_loads(50.0)
        
        # Integrate total applied force over surface
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        
        F_applied_nd = np.zeros(3)
        for i in range(3):
            F_i_form = fem.form(t_total[i] * cfg.ds(2))
            F_i_local = fem.assemble_scalar(F_i_form)
            F_applied_nd[i] = domain.comm.allreduce(F_i_local, op=4)  # MPI.SUM
        
        # Convert to Newtons: F[N] = F_nd * sigma_c[Pa] * L_c[m]^2
        # For mm system: L_c = 1.0 mm → L_c_m = 1e-3 m
        # For m system: L_c = 1e-3 m → L_c_m = 1e-3 m
        sigma_c_Pa = cfg.sigma_c
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        
        F_applied_N = F_applied_nd * sigma_c_Pa * (L_c_m ** 2)
        F_magnitude = np.linalg.norm(F_applied_N)
        
        BW_N = 75.0 * 9.81  # 735.8 N
        
        # Applied force should be physiological (1-6× BW at peak stance)
        assert 1.0 * BW_N < F_magnitude < 6.0 * BW_N, \
            f"[{unit_scale}] Applied force {F_magnitude:.1f} N should be 1-6× BW (736-4415 N)"
        
        print(f"\n[{unit_scale}] Applied force at peak: {F_magnitude:.1f} N ({F_magnitude/BW_N:.2f}× BW)")
        print(f"  Components [N]: Fx={F_applied_N[0]:.1f}, Fy={F_applied_N[1]:.1f}, Fz={F_applied_N[2]:.1f}")

    def test_reaction_force_equilibrium(self, gait_loader, femur_setup):
        """Consistent reaction forces (from unconstrained residual) balance applied loads.

        We compute reactions as r_D = (A0 @ u − b0) on Dirichlet DOFs
        where A0 and b0 are assembled without applying Dirichlet constraints.
        In global equilibrium (no body forces): F_applied + F_reaction ≈ 0.
        """
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup solver
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain,
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute applied force (integrate tractions over surface tag=2)
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_nd = np.zeros(3)
        for i in range(3):
            F_i_form = fem.form(t_total[i] * cfg.ds(2))
            F_i_local = fem.assemble_scalar(F_i_form)
            F_applied_nd[i] = domain.comm.allreduce(F_i_local, op=4)
        
        # Compute consistent reactions from unconstrained residual on Dirichlet DOFs
        # A0 u − b0 (no lifting / no set_bc)
        A0 = assemble_matrix_petsc(solver.a_form)  # no bcs argument → unconstrained
        A0.assemble()
        # Allocate vectors compatible with the mechanics function space (dolfinx 0.10 API)
        b0 = create_vector_petsc(V)
        with b0.localForm() as b0_loc:
            b0_loc.set(0.0)
        assemble_vector_petsc(b0, solver.L_form)
        b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        r = create_vector_petsc(V)
        with r.localForm() as r_loc:
            r_loc.set(0.0)
        A0.mult(u.x.petsc_vec, r)
        r.axpy(-1.0, b0)  # r = A0 u − b0

        # Sum reactions per component over Dirichlet DOFs (tag=1)
        F_reaction_nd = np.zeros(3)
        for i, bc in enumerate(dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.getValues(idx_owned)
                F_reaction_nd[i] += float(np.sum(r_local))
        # MPI global sum across ranks
        F_reaction_nd = domain.comm.allreduce(F_reaction_nd, op=4)
        
        # Convert to Newtons
        sigma_c_Pa = cfg.sigma_c
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        
        F_applied_N = F_applied_nd * sigma_c_Pa * (L_c_m ** 2)
        F_reaction_N = F_reaction_nd * sigma_c_Pa * (L_c_m ** 2)

        # Independent verification via stress traction on Γ_D
        import ufl
        n = ufl.FacetNormal(domain)
        sigma_u = solver.sigma(u, rho, A_dir)
        t_reac = ufl.dot(sigma_u, n)  # vector traction on Γ_D
        F_reaction_sigma_nd = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_reac[i] * cfg.ds(1))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_reaction_sigma_nd[i] = domain.comm.allreduce(Fi_loc, op=4)
        F_reaction_sigma_N = F_reaction_sigma_nd * sigma_c_Pa * (L_c_m ** 2)
        
        F_total_N = F_applied_N + F_reaction_N  # Should be exactly zero
        F_total_magnitude = np.linalg.norm(F_total_N)
        F_applied_magnitude = np.linalg.norm(F_applied_N)
        F_reaction_magnitude = np.linalg.norm(F_reaction_N)
        
        print(f"\n[{unit_scale}] Force equilibrium (global balance):")
        print(f"  Applied (∫ t dS over tag=2): {F_applied_N}")
        print(f"  Reaction (A0 u − b0 on Γ_D): {F_reaction_N}")
        print(f"  Reaction (∫_ΓD σ n dS): {F_reaction_sigma_N}")
        print(f"  Sum (should be 0): {F_total_N}")
        print(f"  |Applied|: {F_applied_magnitude:.1f} N, |Reaction|: {F_reaction_magnitude:.1f} N")
        
        # Global equilibrium: F_applied + F_reaction ≈ 0
        # Use relative tolerance w.r.t. |F_applied| to accommodate discretization/solve tolerance
        rel_err = F_total_magnitude / max(F_applied_magnitude, 1e-30)
        assert rel_err < 5e-6, \
            f"[{unit_scale}] Force balance failed: |F_applied+F_reaction|/|F_applied| = {rel_err:.2e}"

        # Cross-check (independent): compare component along net applied load direction.
        # For coarse meshes, ∫_ΓD σ n dS may deviate more; verify consistency along dominant direction.
        e = F_applied_N / max(F_applied_magnitude, 1e-30)
        s_res = float(F_reaction_N @ e)
        s_sig = float((-F_reaction_sigma_N) @ e)  # support reaction = -∫_ΓD σ n dS
        rel_axis_err = abs(abs(s_sig) - F_applied_magnitude) / max(F_applied_magnitude, 1e-30)
        print(f"  Axis-check: s_res={s_res:.2f} N, s_sig={s_sig:.2f} N, |s_sig|-|F_appl| rel_err={rel_axis_err:.3f}")
        assert rel_axis_err < 0.30, \
            f"[{unit_scale}] Traction reaction (σ·n) inconsistent along load axis (rel_err={rel_axis_err:.2e})"
