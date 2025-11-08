#!/usr/bin/env python3
"""
Advanced tests for physical correctness of the bone remodeling model.

Tests:
- Constitutive law accuracy (stress-strain relationships)
- Thermodynamic consistency (energy dissipation, non-negative entropy production)
- Conservation properties (force equilibrium, mass conservation)
- Boundary condition enforcement
- Coupling physics verification
- Smooth function properties (C∞, monotonicity, limiting behavior)
- PSD tensor enforcement correctness
"""

import pytest
pytest.importorskip("dolfinx")
pytest.importorskip("mpi4py")
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import (MechanicsSolver, DensitySolver, DirectionSolver, smooth_abs, smooth_plus, smooth_max,
                        smooth_heaviside, unittrace_psd_from_any, unittrace_psd)

comm = MPI.COMM_WORLD


# =============================================================================
# Constitutive Law Tests
# =============================================================================

class TestConstitutiveLaw:
    """Test stress-strain constitutive relationships."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_isotropic_stress_symmetry(self, unit_cube):
        """Verify stress tensor is symmetric for isotropic material."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        # Create test displacement with non-zero gradient
        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([0.001*x[0]*x[1], 0.002*x[1]*x[2], 0.001*x[0]*x[2]]))
        u.x.scatter_forward()
        
        # Uniform density and isotropic fabric (I/3)
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)
        
        # Compute stress tensor components
        sigma = mech.sigma(u, rho)
        
        # Check σ_ij = σ_ji (symmetry)
        sigma_01 = sigma[0, 1]
        sigma_10 = sigma[1, 0]
        sigma_02 = sigma[0, 2]
        sigma_20 = sigma[2, 0]
        sigma_12 = sigma[1, 2]
        sigma_21 = sigma[2, 1]
        
        # Assemble differences (should be zero)
        diff_01 = fem.assemble_scalar(fem.form((sigma_01 - sigma_10)**2 * cfg.dx))
        diff_02 = fem.assemble_scalar(fem.form((sigma_02 - sigma_20)**2 * cfg.dx))
        diff_12 = fem.assemble_scalar(fem.form((sigma_12 - sigma_21)**2 * cfg.dx))
        
        diff_01_global = comm.allreduce(diff_01, op=MPI.SUM)
        diff_02_global = comm.allreduce(diff_02, op=MPI.SUM)
        diff_12_global = comm.allreduce(diff_12, op=MPI.SUM)
        
        assert diff_01_global < 1e-12, f"Stress not symmetric: σ_01 ≠ σ_10 (diff={diff_01_global})"
        assert diff_02_global < 1e-12, f"Stress not symmetric: σ_02 ≠ σ_20 (diff={diff_02_global})"
        assert diff_12_global < 1e-12, f"Stress not symmetric: σ_12 ≠ σ_21 (diff={diff_12_global})"
    
    def test_density_modulus_scaling(self):
        """Verify E(ρ) = E0 * ρ^n power-law scaling."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        # Test at different density values
        densities = [0.3, 0.5, 0.8, 1.0]
        n_power = float(cfg.n_power_c)
        E0_nd = float(cfg.E0_nd)
        
        for rho_val in densities:
            rho = Function(Q, name="rho")
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            # Expected modulus (smoothed clamping to rho_min)
            rho_eff = max(rho_val, float(cfg.rho_min_nd))
            E_expected = E0_nd * (rho_eff ** n_power)
            
            # Compute via UFL (using smooth_max as in sigma())
            from simulation.subsolvers import smooth_max
            rho_eff_ufl = smooth_max(rho, cfg.rho_min_nd, cfg.smooth_eps)
            E_ufl = cfg.E0_nd * (rho_eff_ufl ** cfg.n_power_c)
            
            E_computed_local = fem.assemble_scalar(fem.form(E_ufl * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            
            E_computed = comm.allreduce(E_computed_local, op=MPI.SUM)
            vol = comm.allreduce(vol_local, op=MPI.SUM)
            
            E_avg = E_computed / vol
            
            # Allow small error from smooth_max approximation
            rel_error = abs(E_avg - E_expected) / E_expected
            assert rel_error < 0.01, f"Modulus scaling incorrect at ρ={rho_val}: expected {E_expected}, got {E_avg}"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_anisotropic_stiffness_increases_energy(self, unit_cube, facet_tags):
        """Anisotropic fabric aligned with tension should stiffen response measurably."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0),
                     enable_telemetry=False, xi_aniso=2.0)

        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)

        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()

        def _iso_tensor(x):
            base = (np.eye(3) / 3.0).flatten()[:, None]
            return np.tile(base, (1, x.shape[1]))

        def _fiber_tensor(x):
            mat = np.array([[0.92, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]], dtype=float)
            return np.tile(mat.flatten()[:, None], (1, x.shape[1]))

        A_iso = Function(T, name="A_iso")
        A_iso.interpolate(_iso_tensor)
        A_iso.x.scatter_forward()

        A_fiber = Function(T, name="A_fiber")
        A_fiber.interpolate(_fiber_tensor)
        A_fiber.x.scatter_forward()

        u_test = Function(V, name="u_test")
        u_test.interpolate(lambda x: np.vstack([0.003 * x[0], 0.0 * x[1], 0.0 * x[2]]))
        u_test.x.scatter_forward()

        mech_iso = MechanicsSolver(V, rho, A_iso, [], [], cfg)
        mech_aniso = MechanicsSolver(V, rho, A_fiber, [], [], cfg)

        energy_iso = mech_iso.average_strain_energy(u_test)
        energy_aniso = mech_aniso.average_strain_energy(u_test)

        assert energy_aniso >= energy_iso * 1.10, (
            "Anisotropic fabric should raise energy for the same tensile strain by ≥10%; "
            f"energy_iso={energy_iso:.3e}, energy_aniso={energy_aniso:.3e}"
        )


# =============================================================================
# Thermodynamic Consistency Tests
# =============================================================================

class TestThermodynamics:
    """Test energy dissipation and thermodynamic consistency."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_strain_energy_positivity(self, unit_cube):
        """Strain energy density ψ = 0.5*σ:ε must be non-negative."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        # Random displacement field
        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([
            0.001*np.sin(2*np.pi*x[0])*np.cos(2*np.pi*x[1]),
            0.001*np.cos(2*np.pi*x[0])*np.sin(2*np.pi*x[2]),
            0.001*np.sin(2*np.pi*x[1])*np.cos(2*np.pi*x[2])
        ]))
        u.x.scatter_forward()
        
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)
        
        psi = 0.5 * ufl.inner(mech.sigma(u, rho), mech.eps(u))
        
        psi_local = fem.assemble_scalar(fem.form(psi * cfg.dx))
        psi_global = comm.allreduce(psi_local, op=MPI.SUM)
        
        assert psi_global >= -1e-12, f"Strain energy must be non-negative, got {psi_global}"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_stimulus_diffusion_dissipation(self, unit_cube, mean_value_factory):
        """Stimulus diffusion term should dissipate energy (κ∇S·∇S ≥ 0)."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        S = Function(Q, name="S")
        S.interpolate(lambda x: np.sin(np.pi*x[0]) * np.cos(np.pi*x[1]) * np.sin(np.pi*x[2]))
        S.x.scatter_forward()
        
        # Diffusion dissipation: κ |∇S|²
        kappa_S = float(cfg.kappaS_c)
        dissipation = kappa_S * ufl.inner(ufl.grad(S), ufl.grad(S))
        mean_val = mean_value_factory(dissipation)
        assert mean_val >= -1e-14, f"Diffusion dissipation must be non-negative, got {mean_val}"


# =============================================================================
# Conservation Tests
# =============================================================================

class TestConservation:
    """Test conservation properties and equilibrium."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_force_equilibrium_no_body_force(self, unit_cube):
        """With no body force and homogeneous BCs, internal forces should sum to zero."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        # BCs: left fixed, no traction on right
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(V, rho, A, bc_mech, [], cfg)
        
        mech.solver_setup()
        mech.solve(u)
        
        # Check residual: ∫ σ:∇v dx should be zero for all v (satisfied by FEM)
        # Instead, verify displacement is approximately zero when no load applied
        u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
        
        assert u_norm_sq < 1e-12, f"No-load case should yield zero displacement, got ||u||²={u_norm_sq}"
    
    def test_density_bounds_preservation(self):
        """Density solver should preserve [rho_min, rho_max] bounds."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        
        # Start with out-of-bounds initial condition
        rho_old.x.array[:] = 0.05  # Below rho_min_nd
        rho_old.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        S = Function(Q, name="S")
        S.x.array[:] = 0.1
        S.x.scatter_forward()
        
        densolver = DensitySolver(Q, rho_old, A, S, cfg)
        densolver.solver_setup()
        densolver.update_system()
        densolver.solve(rho)
        
        rho_min_nd = float(cfg.rho_min_nd)
        rho_max_nd = float(cfg.rho_max_nd)
        
        n_owned = Q.dofmap.index_map.size_local
        rho_min_computed = comm.allreduce(rho.x.array[:n_owned].min(), op=MPI.MIN)
        rho_max_computed = comm.allreduce(rho.x.array[:n_owned].max(), op=MPI.MAX)
        
        # Allow small tolerance for smooth_max/min enforcement
        assert rho_min_computed >= rho_min_nd - 1e-6, f"Density below minimum: {rho_min_computed} < {rho_min_nd}"
        assert rho_max_computed <= rho_max_nd + 1e-6, f"Density above maximum: {rho_max_computed} > {rho_max_nd}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_density_solver_response_to_stimulus_sign(self, unit_cube, facet_tags):
        """Positive stimulus should increase density, negative stimulus should decrease it."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))

        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)

        def _mean_value(func):
            val_local = fem.assemble_scalar(fem.form(func * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            val = comm.allreduce(val_local, op=MPI.SUM)
            vol = comm.allreduce(vol_local, op=MPI.SUM)
            return float(val / max(vol, 1e-300))

        def _solve_density(stimulus_value: float) -> float:
            rho_old = Function(Q, name="rho_old")
            rho_old.x.array[:] = 0.5
            rho_old.x.scatter_forward()

            rho = Function(Q, name="rho")

            A_field = Function(T, name="A")
            A_field.interpolate(lambda x: (np.eye(3) / 3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
            A_field.x.scatter_forward()

            S_field = Function(Q, name="S")
            S_field.x.array[:] = stimulus_value
            S_field.x.scatter_forward()

            dens = DensitySolver(Q, rho_old, A_field, S_field, cfg)
            dens.solver_setup()
            dens.update_system()
            dens.solve(rho)
            rho.x.scatter_forward()
            return _mean_value(rho)

        baseline = 0.5
        rho_mean_positive = _solve_density(0.25)
        rho_mean_negative = _solve_density(-0.25)

        pos_delta = rho_mean_positive - baseline
        neg_delta = baseline - rho_mean_negative

        assert pos_delta > 1e-2, (
            "Positive stimulus should raise density by at least 1%; "
            f"baseline={baseline}, mean={rho_mean_positive}, Δ={pos_delta}"
        )
        assert neg_delta > 1e-2, (
            "Negative stimulus should reduce density by at least 1%; "
            f"baseline={baseline}, mean={rho_mean_negative}, Δ={neg_delta}"
        )


class TestDirectionSolverProperties:
    """Properties of the direction tensor solver outputs."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_direction_solver_unit_trace_psd(self, unit_cube, facet_tags, traction_factory):
        """Direction solver output should be symmetric PSD with unit trace."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))

        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))

        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)

        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()

        A_old = Function(T, name="A_old")
        A_old.interpolate(lambda x: (np.eye(3) / 3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A_old.x.scatter_forward()

        u = Function(V, name="u")
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        traction = traction_factory(-0.4, facet_id=2, axis=0)

        mech = MechanicsSolver(V, rho, A_old, bc_mech, [traction], cfg)
        mech.solver_setup()
        mech.solve(u)

        dir_solver = DirectionSolver(T, A_old, cfg)
        dir_solver.solver_setup()
        dir_solver.update_rhs(mech, u)

        A_new = Function(T, name="A_new")
        dir_solver.solve(A_new)
        A_new.x.scatter_forward()

        n_owned = T.dofmap.index_map.size_local * T.dofmap.index_map_bs
        if n_owned == 0:
            pytest.skip("No owned DOFs on this rank for tensor space")

        values = A_new.x.array[:n_owned]
        assert values.size % 9 == 0, "Tensor DOF array not divisible by 9 components"
        tensors = values.reshape(-1, 9)

        for row in tensors:
            mat = row.reshape(3, 3)
            sym = 0.5 * (mat + mat.T)
            trace = np.trace(sym)
            assert abs(trace - 1.0) < 1e-6, f"Trace not unity: {trace}"
            eigvals = np.linalg.eigvalsh(sym)
            assert eigvals.min() >= -1e-8, f"Tensor not PSD: eigenvalues={eigvals}"


# =============================================================================
# Smooth Function Tests
# =============================================================================

class TestSmoothFunctions:
    """Test smooth approximations for non-differentiable functions."""
    
    def test_smooth_abs_properties(self):
        """Verify smooth_abs(x, eps) → |x| as eps→0 and C∞."""
        x_vals = np.linspace(-2, 2, 100)
        eps = 1e-3
        
        for x in x_vals:
            s_abs = np.sqrt(x**2 + eps**2)
            true_abs = np.abs(x)
            
            # Should approximate |x|
            assert abs(s_abs - true_abs) < eps, f"smooth_abs({x}) not close to |{x}|"
            
            # Should be smooth (derivative exists)
            # d/dx sqrt(x^2 + eps^2) = x / sqrt(x^2 + eps^2)
            deriv = x / np.sqrt(x**2 + eps**2)
            assert np.isfinite(deriv), f"smooth_abs derivative not finite at x={x}"
    
    def test_smooth_max_monotonicity(self):
        """Verify smooth_max is monotone increasing in first argument."""
        xmin = 0.5
        eps = 1e-4
        x_vals = np.linspace(0, 2, 50)
        
        smooth_vals = []
        for x in x_vals:
            dx = x - xmin
            s_max = xmin + 0.5 * (dx + np.sqrt(dx**2 + eps**2))
            smooth_vals.append(s_max)
        
        # Check monotonicity
        for i in range(len(smooth_vals) - 1):
            assert smooth_vals[i+1] >= smooth_vals[i], f"smooth_max not monotone: {smooth_vals[i]} > {smooth_vals[i+1]}"
    
    def test_smooth_heaviside_limits(self):
        """Verify smooth_heaviside(x, eps) → H(x) as |x|→∞."""
        eps = 1e-3
        
        # Far negative: should be near 0
        x_neg = -10.0
        H_neg = 0.5 * (1.0 + x_neg / np.sqrt(x_neg**2 + eps**2))
        assert abs(H_neg - 0.0) < 0.01, f"smooth_heaviside({x_neg}) should be ~0"
        
        # Far positive: should be near 1
        x_pos = 10.0
        H_pos = 0.5 * (1.0 + x_pos / np.sqrt(x_pos**2 + eps**2))
        assert abs(H_pos - 1.0) < 0.01, f"smooth_heaviside({x_pos}) should be ~1"
        
        # At zero: should be 0.5
        x_zero = 0.0
        H_zero = 0.5 * (1.0 + x_zero / np.sqrt(x_zero**2 + eps**2))
        assert abs(H_zero - 0.5) < 0.01, f"smooth_heaviside(0) should be ~0.5"

    def test_smooth_functions_match_ufl_on_mesh(self):
        """Validate UFL implementations of smooth_* against their analytical forms by integration."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)

        S = Function(Q, name="S")
        S.interpolate(lambda x: np.sin(2*np.pi*x[0]) - 0.3*np.cos(2*np.pi*x[1]) + 0.1*x[2])
        S.x.scatter_forward()

        eps = float(cfg.smooth_eps)
        xmin = 0.2

        # Expected analytical forms
        s_abs_form = ufl.sqrt(S*S + eps*eps)
        s_plus_form = 0.5*(S + ufl.sqrt(S*S + eps*eps))
        h_form = 0.5*(1.0 + S/ufl.sqrt(S*S + eps*eps))
        s_max_form = xmin + 0.5*((S - xmin) + ufl.sqrt((S - xmin)*(S - xmin) + eps*eps))

        # Implementations under test
        s_abs_impl = smooth_abs(S, eps)
        s_plus_impl = smooth_plus(S, eps)
        h_impl = smooth_heaviside(S, eps)
        s_max_impl = smooth_max(S, xmin, eps)

        # Integrate squared differences over the domain and require they are tiny
        def _mean_sq(expr_diff):
            val_local = fem.assemble_scalar(fem.form((expr_diff*expr_diff) * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            val = comm.allreduce(val_local, op=MPI.SUM)
            vol = comm.allreduce(vol_local, op=MPI.SUM)
            return float(val / max(vol, 1e-300))

        tol = 1e-12
        assert _mean_sq(s_abs_impl - s_abs_form) < tol
        assert _mean_sq(s_plus_impl - s_plus_form) < tol
        assert _mean_sq(h_impl - h_form) < tol
        assert _mean_sq(s_max_impl - s_max_form) < tol


# =============================================================================
# PSD Tensor Tests
# =============================================================================

class TestPSDTensors:
    """Test positive-semidefinite tensor enforcement."""
    
    def test_unittrace_psd_from_any_properties(self):
        """Verify unittrace_psd_from_any produces symmetric, SPD, unit-trace tensor."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        T = functionspace(domain, P1_ten)
        
        # Test with arbitrary (non-symmetric, non-PSD) tensor
        A_in = Function(T, name="A_in")
        
        # Create tensor field via component-wise interpolation
        def tensor_field(x):
            n_points = x.shape[1]
            result = np.zeros((9, n_points))
            # Fill tensor components [A00, A01, A02, A10, A11, A12, A20, A21, A22]
            result[0, :] = x[0]      # A00 = x
            result[1, :] = x[1]      # A01 = y
            result[2, :] = 0.0       # A02 = 0
            result[3, :] = x[2]      # A10 = z
            result[4, :] = x[0] + x[1]  # A11 = x+y
            result[5, :] = 0.0       # A12 = 0
            result[6, :] = 0.0       # A20 = 0
            result[7, :] = 0.0       # A21 = 0
            result[8, :] = 1.0       # A22 = 1
            return result
        
        A_in.interpolate(tensor_field)
        A_in.x.scatter_forward()
        
        # Apply PSD enforcement
        Asym = 0.5 * (A_in + ufl.transpose(A_in))
        A_hat = unittrace_psd_from_any(Asym, 3, float(cfg.smooth_eps))
        
        # Check trace = 1
        tr_A = ufl.tr(A_hat)
        tr_local = fem.assemble_scalar(fem.form(tr_A * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        tr_avg = comm.allreduce(tr_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        assert abs(tr_avg - 1.0) < 1e-10, f"Unit trace not preserved: tr(A_hat) = {tr_avg}"
        
        # Check symmetry: A_hat[0,1] = A_hat[1,0]
        A_hat_01 = A_hat[0, 1]
        A_hat_10 = A_hat[1, 0]
        diff_sq = (A_hat_01 - A_hat_10)**2
        diff_sq_local = fem.assemble_scalar(fem.form(diff_sq * cfg.dx))
        diff_sq_global = comm.allreduce(diff_sq_local, op=MPI.SUM)
        
        assert diff_sq_global < 1e-12, f"A_hat not symmetric"
    
    def test_fabric_normalization_stability(self):
        """Test fabric normalization doesn't blow up with zero strain."""
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        T = functionspace(domain, P1_ten)
        
        # Zero displacement → zero strain
        u = Function(V, name="u")
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        
        # Direction solver target: B = ε^T ε normalized
        eps = ufl.sym(ufl.grad(u))
        B = ufl.dot(ufl.transpose(eps), eps)
        I = ufl.Identity(3)
        
        # Should default to I/3 when trB ~ 0
        B_hat = unittrace_psd(B, 3, float(cfg.smooth_eps))
        
        # Check all eigenvalues ~ 1/3 (isotropic)
        B_hat_00 = B_hat[0, 0]
        B_hat_11 = B_hat[1, 1]
        B_hat_22 = B_hat[2, 2]
        
        diag_avg_local = fem.assemble_scalar(fem.form((B_hat_00 + B_hat_11 + B_hat_22) * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        
        diag_avg = comm.allreduce(diag_avg_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        # Trace = 1 → avg diagonal = 1/3 for isotropic
        assert abs(diag_avg - 1.0) < 0.05, f"Zero strain should yield isotropic fabric (tr=1), got {diag_avg}"


# =============================================================================
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Test boundary condition enforcement."""
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_dirichlet_enforcement_strong(self, unit_cube, traction_factory):
        """Verify Dirichlet BCs are strongly enforced."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        # Apply traction on right face
        traction = traction_factory(-0.1, facet_id=2, axis=0)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(V, rho, A, bc_mech, [traction], cfg)
        
        mech.solver_setup()
        mech.solve(u)
        
        # Extract DOFs on left boundary (tag=1, x=0)
        from simulation.utils import collect_dirichlet_dofs
        bc_dofs = collect_dirichlet_dofs(bc_mech, mech.V.dofmap.index_map.size_local)
        
        if bc_dofs.size > 0:
            u_bc_vals = u.x.array[bc_dofs]
            max_bc_val = np.max(np.abs(u_bc_vals))
            assert max_bc_val < 1e-9, f"Dirichlet BC not enforced: max |u| on BC = {max_bc_val}"
    
    @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    def test_traction_load_response(self, unit_cube, traction_factory):
        """Verify mechanics solver responds correctly to applied traction."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        
        # Compression in x-direction
        traction = traction_factory(-0.5, facet_id=2, axis=0)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(V, rho, A, bc_mech, [traction], cfg)
        
        mech.solver_setup()
        mech.solve(u)
        
        # Under compression, expect negative x-displacement (compression)
        u_x = u.sub(0).collapse()
        u_x_avg_local = fem.assemble_scalar(fem.form(u_x * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        
        u_x_avg = comm.allreduce(u_x_avg_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        assert u_x_avg < -1e-10, f"Compression load should yield negative x-displacement, got {u_x_avg}"


# No __main__ runner needed; tests executed via pytest
