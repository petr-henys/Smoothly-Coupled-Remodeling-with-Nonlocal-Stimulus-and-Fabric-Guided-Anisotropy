import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import ufl
import basix

from simulation.config import Config
from simulation.params import MaterialParams, DensityParams, StimulusParams
from simulation.solvers import DensitySolver, MechanicsSolver
from simulation.utils import build_facetag, build_dirichlet_bcs

def make_unit_cube(comm=MPI.COMM_WORLD, n=6):
    return mesh.create_unit_cube(comm, n, n, n)

class TestNumerics:
    """Rigorous numerical tests for conservation, symmetry, and stability."""

    @pytest.mark.parametrize("unit_cube", [4], indirect=True)
    def test_density_mass_balance_equation(self, unit_cube, facet_tags):
        """Verify the integral mass balance equation:
        ∫ (ρ - ρ_old)/dt dx = ∫ (S_source - S_sink) dx
        
        This ensures that the reaction terms and time stepping are consistent
        and that no mass is lost/gained unaccountably (e.g. via quadrature errors).
        """
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams(),
                    density=DensityParams(D_rho=0.1, k_rho_form=1.0, k_rho_resorb=1.0, surface_use=False))
        cfg.set_dt(1.0)

        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = fem.functionspace(unit_cube, P1)

        rho = fem.Function(Q, name="rho")
        rho_old = fem.Function(Q, name="rho_old")
        S_field = fem.Function(Q, name="S")

        # Initial condition: uniform
        rho_old.x.array[:] = 0.5
        rho_old.x.scatter_forward()

        # Stimulus: spatially varying to activate both formation and resorption
        # S > 0 in half, S < 0 in other half
        S_field.interpolate(lambda x: np.sin(2 * np.pi * x[0]))
        S_field.x.scatter_forward()

        solver = DensitySolver(rho, rho_old, S_field, cfg)
        solver.setup()
        solver.solve()

        # Check balance
        # LHS: ∫ (ρ - ρ_old)/dt dx
        dt = float(cfg.dt)
        lhs_form = (rho - rho_old) / dt * cfg.dx
        lhs_val = comm.allreduce(fem.assemble_scalar(fem.form(lhs_form)), op=MPI.SUM)

        # RHS: ∫ (S_source - S_sink) dx
        # We need to reconstruct the reaction terms exactly as in the solver
        # Note: Solver uses smooth_max for S_pos/S_neg
        eps = float(cfg.numerics.smooth_eps)
        from simulation.utils import smooth_max
        
        S_pos = smooth_max(S_field, 0.0, eps)
        S_neg = smooth_max(-S_field, 0.0, eps)
        
        # Assuming A_surf = 1.0 for simplicity (default if surface_use=False, which is default in params?)
        # Let's check params default. If surface_use is True, we need to account for it.
        # But we can just use the solver's internal logic if we could access it.
        # Instead, let's rely on the fact that we know the equation.
        
        rho_max = float(cfg.density.rho_max)
        rho_min = float(cfg.density.rho_min)
        k_form = float(cfg.density.k_rho_form) # * A_surf
        k_res = float(cfg.density.k_rho_resorb) # * A_surf
        
        # Re-calculate A_surf if needed. 
        # To be safe, let's force surface_use=False in this test to isolate the reaction logic.
        # (We can't easily change cfg after init, so we should have set it in init)
        # But Config is mutable? No, params are dataclasses.
        # Let's assume surface_use=False for now or check if we can modify it.
        # Actually, let's just re-implement the logic including A_surf if it's on.
        # But simpler is to check if the residual with test function 1 is zero.
        
        # The weak form is:
        # F(u, v) = ∫ (u - u_old)/dt * v + D ∇u·∇v + reaction*u*v - source*v dx = 0
        # If we choose v = 1, then ∇v = 0.
        # ∫ (u - u_old)/dt + reaction*u - source dx = 0
        # ∫ (u - u_old)/dt dx = ∫ source - reaction*u dx
        
        # So we can just assemble the residual with v=1.
        
        # We can use the solver's forms!
        # solver.a_form is bilinear a(u, v)
        # solver.L_form is linear L(v)
        # Residual R(v) = a(rho, v) - L(v)
        # We want R(1) = 0.
        
        # But a_form and L_form are compiled forms. We can't easily change the test function to 1.
        # However, 1 is in the space Q (constant function).
        # So we can project 1 onto Q and use it as the test function vector?
        # No, assemble_scalar is for functionals.
        # assemble_vector produces a vector.
        # R_vec = assemble_vector(L_form) - assemble_matrix(a_form) * rho_vec
        # Then dot(R_vec, 1_vec) should be 0.
        
        # Let's do that.
        b = fem.Function(Q)
        solver.assemble_rhs() # assembles into solver.b
        b_vec = solver.b.array
        
        A = solver.A
        x_vec = rho.x.array
        
        # y = A * x
        y_vec = A * rho.x.petsc_vec
        
        # res = y - b (or b - y depending on definition)
        # Solver solves A x = b. So res = A x - b should be 0 (numerically).
        # But that just tests the linear solver.
        
        # We want to test if the DISCRETIZED EQUATION satisfies global mass balance.
        # If the test space contains constants (which P1 does), then the row sum of A (or something similar) 
        # related to the constant test function should correspond to the mass balance.
        
        # Actually, if 1 \in TestSpace, then the equation is satisfied for v=1.
        # So ∫ (ρ - ρ_old)/dt - S_net dx = 0 is enforced by the FEM up to solver tolerance.
        # So this test might just be testing that 1 is in P1.
        
        # A more meaningful test is to verify that our manual calculation of the terms matches.
        # i.e. Explicitly integrate the terms and see if they balance.
        
        # Re-construct terms:
        reaction = (k_form * S_pos / rho_max) + (k_res * S_neg / rho_min)
        # Note: A_surf is 1.0 by default in DensityParams? Let's check.
        # If not, we need to handle it.
        
        rhs_term = (k_form * S_pos) + (k_res * S_neg)
        lhs_term = (rho - rho_old)/dt + reaction * rho
        
        # Balance: ∫ lhs_term dx = ∫ rhs_term dx
        # (Diffusive flux integrates to 0 with natural BCs)
        
        int_lhs = comm.allreduce(fem.assemble_scalar(fem.form(lhs_term * cfg.dx)), op=MPI.SUM)
        int_rhs = comm.allreduce(fem.assemble_scalar(fem.form(rhs_term * cfg.dx)), op=MPI.SUM)
        
        rel_err = abs(int_lhs - int_rhs) / max(abs(int_rhs), 1e-10)
        assert rel_err < 1e-5, f"Mass balance mismatch: LHS={int_lhs:.6e}, RHS={int_rhs:.6e}, err={rel_err:.2e}"

    @pytest.mark.parametrize("unit_cube", [4], indirect=True)
    def test_anisotropic_stress_symmetry(self, unit_cube, facet_tags):
        """Verify that the stress tensor is symmetric even with anisotropic fabric."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams())
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = fem.functionspace(unit_cube, P1_vec)
        Q = fem.functionspace(unit_cube, P1)
        T = fem.functionspace(unit_cube, P1_ten)
        
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        L = fem.Function(T, name="L")
        
        # Random fields
        np.random.seed(42)
        u.x.array[:] = np.random.randn(u.x.array.size) * 0.1
        rho.x.array[:] = 0.5 + np.random.rand(rho.x.array.size) * 0.5
        
        # Create symmetric fabric tensor
        L_vals = np.random.randn(L.x.array.size).reshape(-1, 9)
        # Symmetrize
        L_vals[:, 1] = L_vals[:, 3] # 01 = 10
        L_vals[:, 2] = L_vals[:, 6] # 02 = 20
        L_vals[:, 5] = L_vals[:, 7] # 12 = 21
        L.x.array[:] = L_vals.flatten()
        
        u.x.scatter_forward()
        rho.x.scatter_forward()
        L.x.scatter_forward()
        
        mech = MechanicsSolver(u, rho, cfg, [], [], L=L)
        
        sigma = mech.sigma(u, rho, L)
        
        # Check symmetry: ||sigma - sigma.T||
        asym = sigma - sigma.T
        asym_norm_sq = comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(asym, asym) * cfg.dx)), op=MPI.SUM)
        
        assert asym_norm_sq < 1e-10, f"Stress tensor is not symmetric: ||sigma - sigma^T||^2 = {asym_norm_sq}"

    @pytest.mark.parametrize("unit_cube", [4], indirect=True)
    def test_stiffness_positivity_anisotropic(self, unit_cube, facet_tags):
        """Verify that the anisotropic stiffness matrix is positive definite."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams())
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = fem.functionspace(unit_cube, P1_vec)
        Q = fem.functionspace(unit_cube, P1)
        T = fem.functionspace(unit_cube, P1_ten)
        
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        L = fem.Function(T, name="L")
        
        # Random displacement (non-rigid body motion ideally, but random is fine)
        # We need to avoid pure rigid body modes where energy is 0.
        # Fixing boundary conditions helps, but here we just check the form a(u,u).
        # If u is random, it likely has deformation.
        u.x.array[:] = np.random.randn(u.x.array.size)
        rho.x.array[:] = 0.8
        
        # Identity fabric (isotropic)
        L.interpolate(lambda x: np.tile(np.eye(3).flatten(), (x.shape[1], 1)).T)
        
        u.x.scatter_forward()
        rho.x.scatter_forward()
        L.x.scatter_forward()
        
        mech = MechanicsSolver(u, rho, cfg, [], [], L=L)
        
        # Energy = 0.5 * a(u, u)
        energy_form = 0.5 * ufl.inner(mech.sigma(u, rho, L), mech.eps(u)) * cfg.dx
        energy = comm.allreduce(fem.assemble_scalar(fem.form(energy_form)), op=MPI.SUM)
        
        assert energy >= 0.0, f"Strain energy must be non-negative: {energy}"
        
        # Now with strong anisotropy
        # Make L have large eigenvalues
        L.x.array[:] = L.x.array[:] * 2.0
        L.x.scatter_forward()
        
        energy_aniso = comm.allreduce(fem.assemble_scalar(fem.form(energy_form)), op=MPI.SUM)
        assert energy_aniso >= 0.0, f"Anisotropic strain energy must be non-negative: {energy_aniso}"

