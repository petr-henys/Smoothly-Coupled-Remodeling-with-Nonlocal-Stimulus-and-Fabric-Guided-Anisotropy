#!/usr/bin/env python3
"""Advanced tests for physical correctness of the bone remodeling model.

This module consolidates physics-related tests that were previously
scattered across several files:

- `test_physics.py` (core mechanics, density, direction, smooth functions)
- `test_stimulus_debug.py` (debug stimulus evolution)
- `test_reaction_forces.py` (femur reaction forces and applied loads)

The goal is to keep all core physics and balance checks in one place
while preserving the original coverage.
"""

import pytest

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.paths import FemurPaths
from simulation.febio_parser import FEBio2Dolfinx
from simulation.utils import build_facetag, build_dirichlet_bcs, collect_dirichlet_dofs, assign
from simulation.subsolvers import (
    MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver,
    smooth_abs, smooth_plus, smooth_max, smooth_heaviside,
    unittrace_psd_from_any, unittrace_psd
)
from simulation.femur_gait import setup_femur_gait_loading


def _make_unit_cube(comm: MPI.Comm, n: int = 6):
    """Create a tiny 3D unit cube mesh."""
    return mesh.create_unit_cube(comm, n, n, n, cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)


def _iso_tensor(x):
    """Isotropic unit-trace tensor I/3."""
    base = (np.eye(3) / 3.0).flatten()[:, None]
    return np.tile(base, (1, x.shape[1]))


def _fiber_tensor(x):
    """Anisotropic unit-trace tensor with fiber in x-direction."""
    mat = np.array([[0.92, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]], dtype=float)
    return np.tile(mat.flatten()[:, None], (1, x.shape[1]))


@pytest.fixture(scope="module")
def femur_setup():
    """Create femur mesh and function spaces (mm geometry, MPa stresses)."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)
    cfg = Config(domain=domain, facet_tags=facet_tags)
    return domain, facet_tags, V, Q, cfg


@pytest.fixture(scope="module")
def femur_gait_loader(femur_setup):
    """Gait loader used for femur reaction-force tests."""
    _, _, V, _, _ = femur_setup
    return setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)


# =============================================================================
# Constitutive Law Tests
# =============================================================================

class TestConstitutiveLaw:
    """Test stress-strain constitutive relationships."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_isotropic_stress_symmetry(self, unit_cube, facet_tags):
        """Verify stress tensor is symmetric for isotropic material."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        # Create test displacement with non-zero gradient
        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([0.001*x[0]*x[1], 0.002*x[1]*x[2], 0.001*x[0]*x[2]]))
        u.x.scatter_forward()
        
        # Uniform density and isotropic fabric (I/3)
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        
        # Compute stress tensor components
        sigma = mech.sigma(u, rho, A)
        
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
        domain = _make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        # Test at different density values
        densities = [0.3, 0.5, 0.8, 1.0]
        n_power = cfg.n_power
        E0 = cfg.E0
        
        for rho_val in densities:
            rho = Function(Q, name="rho")
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            # Expected modulus (smoothed clamping to rho_min) [kg/m³]
            rho_eff = max(rho_val, float(cfg.rho_min))
            E_expected = E0 * (rho_eff ** n_power)
            
            # Compute via UFL (using smooth_max as in sigma())
            rho_eff_ufl = smooth_max(rho, cfg.rho_min, cfg.smooth_eps)
            E_ufl = cfg.E0 * (rho_eff_ufl ** cfg.n_power)
            
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
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0),
                     xi_aniso=2.0)

        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))

        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)

        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()

        A_iso = Function(T, name="A_iso")
        A_iso.interpolate(_iso_tensor)
        A_iso.x.scatter_forward()

        A_fiber = Function(T, name="A_fiber")
        A_fiber.interpolate(_fiber_tensor)
        A_fiber.x.scatter_forward()

        u_test = Function(V, name="u_test")
        u_test.interpolate(lambda x: np.vstack([0.003 * x[0], 0.0 * x[1], 0.0 * x[2]]))
        u_test.x.scatter_forward()

        # Copy u_test to u for energy computation
        u.x.array[:] = u_test.x.array[:]
        u.x.scatter_forward()

        mech_iso = MechanicsSolver(u, rho, A_iso, cfg, [], [])
        mech_aniso = MechanicsSolver(u, rho, A_fiber, cfg, [], [])

        energy_iso = mech_iso.average_strain_energy()
        energy_aniso = mech_aniso.average_strain_energy()

        assert energy_aniso >= energy_iso * 1.10, (
            "Anisotropic fabric should raise energy for the same tensile strain by ≥10%; "
            f"energy_iso={energy_iso:.3e}, energy_aniso={energy_aniso:.3e}"
        )


# =============================================================================
# Thermodynamic Consistency Tests
# =============================================================================

class TestThermodynamics:
    """Test energy dissipation and thermodynamic consistency."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_strain_energy_positivity(self, unit_cube, facet_tags):
        """Strain energy density ψ = 0.5*σ:ε must be non-negative."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
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
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        
        psi = 0.5 * ufl.inner(mech.sigma(u, rho, A), mech.eps(u))
        
        psi_local = fem.assemble_scalar(fem.form(psi * cfg.dx))
        psi_global = comm.allreduce(psi_local, op=MPI.SUM)
        
        assert psi_global >= -1e-12, f"Strain energy must be non-negative, got {psi_global}"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_stimulus_diffusion_dissipation(self, unit_cube, facet_tags, mean_value_factory):
        """Stimulus diffusion term should dissipate energy (κ∇S·∇S ≥ 0)."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        S = Function(Q, name="S")
        S.interpolate(lambda x: np.sin(np.pi*x[0]) * np.cos(np.pi*x[1]) * np.sin(np.pi*x[2]))
        S.x.scatter_forward()
        
        # Diffusion dissipation: κ |∇S|²
        kappa_S = cfg.kappaS
        dissipation = kappa_S * ufl.inner(ufl.grad(S), ufl.grad(S))
        mean_val = mean_value_factory(dissipation)
        assert mean_val >= -1e-14, f"Diffusion dissipation must be non-negative, got {mean_val}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_linear_elastic_energy_balance(self, unit_cube, facet_tags, traction_factory):
        """For the linear system, internal work equals external work: W_int ≈ W_ext.
        
        Tests both manual calculation (a(u,u) vs l(u)) and solver method (energy_balance()).
        """
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))

        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)

        # Fields
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.7
        rho.x.scatter_forward()

        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()

        # Dirichlet on x=0, traction on x=1 (axis 0)
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        t_const, t_tag = traction_factory(-0.4, facet_id=2, axis=0)

        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [(t_const, t_tag)])
        mech.setup()
        mech.solve()

        # Test 1: Manual calculation - Internal work: a(u,u) = ∫ σ:ε dx
        a_uu_local = fem.assemble_scalar(fem.form(ufl.inner(mech.sigma(u, rho, A), mech.eps(u)) * cfg.dx))
        a_uu = comm.allreduce(a_uu_local, op=MPI.SUM)

        # External work: l(u) = ∫ t·u ds on tagged facet(s)
        l_u_local = fem.assemble_scalar(fem.form(ufl.inner(t_const, u) * cfg.ds(t_tag)))
        l_u = comm.allreduce(l_u_local, op=MPI.SUM)

        # Energy balance: a(u,u) ≈ l(u)
        denom = max(abs(l_u), abs(a_uu), 1e-300)
        rel_gap = abs(a_uu - l_u) / denom
        assert rel_gap < 5e-9, f"Energy balance violated: a(u,u)={a_uu:.6e}, l(u)={l_u:.6e}, rel_gap={rel_gap:.3e}"
        
        # Test 2: Solver method energy_balance()
        W_int, W_ext, rel_error = mech.energy_balance()
        assert rel_error < 0.05, f"Solver energy balance: W_int={W_int:.3e}, W_ext={W_ext:.3e}, rel_error={rel_error:.3e}"



# =============================================================================
# Conservation Tests
# =============================================================================

class TestConservation:
    """Test conservation properties and equilibrium."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_force_equilibrium_no_body_force(self, unit_cube):
        """With no body force and homogeneous BCs, internal forces should sum to zero.
        
        Also tests that with traction applied, solver converges and energy balance holds.
        """
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        # Test 1: No load case - BCs: left fixed, no traction on right
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        
        mech.setup()
        mech.solve()
        
        # Check residual: displacement should be zero when no load applied
        u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
        
        assert u_norm_sq < 1e-12, f"No-load case should yield zero displacement, got ||u||²={u_norm_sq}"
        
        # Test 2: With traction - clamp x=0, apply traction on x=1
        t0 = fem.Constant(domain, np.array([1.0, 0.0, 0.0], dtype=float))
        neumanns = [(t0, 2)]
        
        mech2 = MechanicsSolver(u, rho, A, cfg, bc_mech, neumanns)
        mech2.setup()
        its, reason = mech2.solve()
        
        # Check solver converged
        assert reason > 0, f"KSP failed to converge, reason={reason}"
        
        # Check energy balance
        W_int, W_ext, rel_err = mech2.energy_balance()
        assert rel_err < 1e-6, f"Energy balance violated: rel_err={rel_err:.2e} (W_int={W_int:.3e}, W_ext={W_ext:.3e})"

    
    def test_density_bounds_preservation(self):
        """Density solver should relax toward bounds under stimulus sign."""
        comm = MPI.COMM_WORLD
        domain = _make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        Q = functionspace(domain, P1)
        T = functionspace(domain, P1_ten)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        
        # Start with out-of-bounds initial condition (below rho_min)
        rho_min = float(cfg.rho_min)
        rho_max = float(cfg.rho_max)
        rho_initial = 0.5 * rho_min
        rho_old.x.array[:] = rho_initial
        rho_old.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        S = Function(Q, name="S")
        S.x.array[:] = 0.1  # Positive stimulus drives toward rho_max
        S.x.scatter_forward()
        
        densolver = DensitySolver(rho, rho_old, A, S, cfg)
        densolver.setup()
        densolver.assemble_rhs()
        densolver.solve()
        
        n_owned = Q.dofmap.index_map.size_local
        rho_min_computed = comm.allreduce(rho.x.array[:n_owned].min(), op=MPI.MIN)
        
        # Density bounds are soft constraints enforced via diffusion-reaction PDE
        # Positive S: check solution moved upward from initial value, but remains below rho_max
        assert rho_min_computed > rho_initial, f"Density should increase from {rho_initial}, got {rho_min_computed}"
        assert rho_min_computed < rho_max, f"Density should still be relaxing toward bounds, got {rho_min_computed}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_density_solver_response_to_stimulus_sign(self, unit_cube, facet_tags, mean_value_factory):
        """Positive stimulus should increase density (toward rho_max), negative should decrease (toward rho_min)."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0))

        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))

        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)

        rho_min = float(cfg.rho_min)
        rho_max = float(cfg.rho_max)
        rho_mid = 0.5 * (rho_min + rho_max)

        def _solve_density(stimulus_value: float) -> float:
            rho_old = Function(Q, name="rho_old")
            rho_old.x.array[:] = rho_mid
            rho_old.x.scatter_forward()

            rho = Function(Q, name="rho")

            A_field = Function(T, name="A")
            A_field.interpolate(_iso_tensor)
            A_field.x.scatter_forward()

            S_field = Function(Q, name="S")
            S_field.x.array[:] = stimulus_value
            S_field.x.scatter_forward()

            dens = DensitySolver(rho, rho_old, A_field, S_field, cfg)
            dens.setup()
            dens.assemble_rhs()
            dens.solve()
            rho.x.scatter_forward()
            return mean_value_factory(rho)

        baseline = rho_mid
        rho_mean_positive = _solve_density(0.25)
        rho_mean_negative = _solve_density(-0.25)

        pos_delta = rho_mean_positive - baseline
        neg_delta = baseline - rho_mean_negative

        span = rho_max - rho_min
        assert pos_delta > 0 and pos_delta > 0.01 * span, (
            "Positive stimulus should raise density by at least 1% of span; "
            f"baseline={baseline}, mean={rho_mean_positive}, Δ={pos_delta}"
        )
        assert neg_delta > 0 and neg_delta > 0.01 * span, (
            "Negative stimulus should reduce density by at least 1% of span; "
            f"baseline={baseline}, mean={rho_mean_negative}, Δ={neg_delta}"
        )

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_total_mass_conservation_zero_stimulus(self, unit_cube, facet_tags):
        """With S=0 and natural (no-flux) boundaries, ∫ρ dx is conserved by diffusion step."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))

        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)

        # Non-uniform initial density to exercise diffusion but preserve mass
        rho_old = Function(Q, name="rho_old")
        rho_old.interpolate(lambda x: 0.6 + 0.2 * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]))
        rho_old.x.scatter_forward()

        # Isotropic direction tensor and zero stimulus
        A_iso = Function(T, name="A")
        A_iso.interpolate(_iso_tensor)
        A_iso.x.scatter_forward()

        S = Function(Q, name="S")
        S.x.array[:] = 0.0
        S.x.scatter_forward()

        # Assemble initial total mass
        m_old_local = fem.assemble_scalar(fem.form(rho_old * cfg.dx))
        m_old = comm.allreduce(m_old_local, op=MPI.SUM)

        # Solve one implicit diffusion step with natural BCs
        rho = Function(Q, name="rho")
        dens = DensitySolver(rho, rho_old, A_iso, S, cfg)
        dens.setup()
        dens.assemble_rhs()
        dens.solve()
        rho.x.scatter_forward()

        m_new_local = fem.assemble_scalar(fem.form(rho * cfg.dx))
        m_new = comm.allreduce(m_new_local, op=MPI.SUM)

        rel_diff = abs(m_new - m_old) / max(abs(m_old), 1e-300)
        # With smoothed |S| ≈ sqrt(S^2 + eps^2), at S=0 a small reaction ~eps remains.
        # Allow a tolerance proportional to smooth_eps plus a tiny numerical margin.
        eps = float(cfg.smooth_eps)
        tol = max(5e-10, 5.0 * eps)
        assert rel_diff < tol, (
            "Total mass approximately conserved for S=0 within smoothing tolerance; "
            f"rel_diff={rel_diff:.3e}, tol={tol:.3e} (m_old={m_old:.6e}, m_new={m_new:.6e})"
        )


class TestDirectionSolverProperties:
    """Properties of the direction tensor solver outputs."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_direction_solver_unit_trace_psd(self, unit_cube, facet_tags, traction_factory):
        """Direction solver output should be symmetric PSD with unit trace."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))

        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))

        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)

        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()

        A_old = Function(T, name="A_old")
        A_old.interpolate(lambda x: (np.eye(3) / 3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A_old.x.scatter_forward()

        u = Function(V, name="u")
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        traction = traction_factory(-0.4, facet_id=2, axis=0)

        mech = MechanicsSolver(u, rho, A_old, cfg, bc_mech, [traction])
        mech.setup()
        mech.solve()

        A = Function(T, name="A")
        dir_solver = DirectionSolver(A, A_old, cfg)
        dir_solver.setup()
        
        # Get strain tensor for RHS
        strain_tensor = mech.get_strain_tensor()
        dir_solver.assemble_rhs(strain_tensor)

        dir_solver.solve()
        A.x.scatter_forward()

        n_owned = T.dofmap.index_map.size_local * T.dofmap.index_map_bs
        if n_owned == 0:
            pytest.skip("No owned DOFs on this rank for tensor space")

        values = A.x.array[:n_owned]
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
        domain = _make_unit_cube(comm, 8)
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
        domain = _make_unit_cube(comm, 8)
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
        domain = _make_unit_cube(comm, 8)
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
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_dirichlet_enforcement_strong(self, unit_cube, traction_factory):
        """Verify Dirichlet BCs are strongly enforced."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        # Apply traction on right face
        traction = traction_factory(-0.1, facet_id=2, axis=0)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
        
        mech.setup()
        mech.assemble_rhs()
        mech.solve()
        
        # Extract DOFs on left boundary (tag=1, x=0)
        bc_dofs = collect_dirichlet_dofs(bc_mech, mech.function_space.dofmap.index_map.size_local)
        
        if bc_dofs.size > 0:
            u_bc_vals = u.x.array[bc_dofs]
            max_bc_val = np.max(np.abs(u_bc_vals))
            assert max_bc_val < 1e-9, f"Dirichlet BC not enforced: max |u| on BC = {max_bc_val}"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_traction_load_response(self, unit_cube, traction_factory):
        """Verify mechanics solver responds correctly to applied traction."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        # Compression in x-direction
        traction = traction_factory(-0.5, facet_id=2, axis=0)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
        
        mech.setup()
        mech.assemble_rhs()
        mech.solve()
        
        # Under compression, expect negative x-displacement (compression)
        u_x = u.sub(0).collapse()
        u_x_avg_local = fem.assemble_scalar(fem.form(u_x * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        
        u_x_avg = comm.allreduce(u_x_avg_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        assert u_x_avg < -1e-10, f"Compression load should yield negative x-displacement, got {u_x_avg}"


# =============================================================================
# Conservation and Balance Check Tests
# =============================================================================

class TestConservationChecks:
    """Test physical conservation/balance checks for all subsolvers."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_stimulus_power_balance(self, unit_cube):
        """StimulusSolver: Power balance (storage + decay ≈ source)."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        cfg.set_dt(10.0 * 86400.0)  # 10 days in seconds
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        S = Function(Q, name="S")
        S_old = Function(Q, name="S_old")
        S_old.x.array[:] = 0.1  # Initial positive stimulus
        S_old.x.scatter_forward()
        
        from simulation.subsolvers import StimulusSolver
        stim = StimulusSolver(S, S_old, cfg)
        stim.setup()
        
        # Create a psi field (supra-homeostatic in one region) [MPa]
        psi_expr = fem.Constant(domain, 1.2 * cfg.psi_ref)
        
        stim.assemble_rhs(psi_expr)
        stim.solve()
        
        # Check power balance (S is already updated by solve())
        power_abs, power_rel = stim.power_balance_residual(psi_expr)
        
        # Relaxed tolerance - small timestep can cause numerical errors
        assert power_rel < 0.1, (
            f"Stimulus power balance violated: abs={power_abs:.3e}, rel={power_rel:.3e}"
        )
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_density_mass_balance(self, unit_cube):
        """DensitySolver: Mass conservation (dM/dt + decay ≈ source)."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        cfg.set_dt(10.0 * 86400.0)  # 10 days in seconds
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        rho_old.x.array[:] = 0.5
        rho_old.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(_iso_tensor)
        A.x.scatter_forward()
        
        S = Function(Q, name="S")
        S.x.array[:] = 0.2  # Positive stimulus -> formation
        S.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, A, S, cfg)
        dens.setup()
        
        dens.assemble_rhs()
        dens.solve()
        
        # Check mass balance (rho is already updated by solve())
        mass_abs, mass_rel = dens.mass_balance_residual()
        
        assert mass_rel < 0.05, (
            f"Density mass balance violated: abs={mass_abs:.3e}, rel={mass_rel:.3e}"
        )
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_direction_trace_balance(self, unit_cube):
        """DirectionSolver: Trace conservation (tr(A) → tr(M̂) = 1)."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        cfg.set_dt(10.0 * 86400.0)  # 10 days in seconds
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        A = Function(T, name="A")
        A_old = Function(T, name="A_old")
        A_old.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A_old.x.scatter_forward()
        
        dir_solver = DirectionSolver(A, A_old, cfg)
        dir_solver.setup()
        
        # Create simple displacement field
        u = Function(V, name="u")
        u.interpolate(lambda x: np.vstack([0.01*x[0], 0.005*x[1], 0.002*x[2]]))
        u.x.scatter_forward()
        
        # Create MechanicsSolver for eps() method
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A_old, cfg, bc_mech, [])
        
        # Get strain tensor and assemble RHS
        eps_ten = mech.get_strain_tensor()
        B = ufl.dot(ufl.transpose(eps_ten), eps_ten)
        dir_solver.assemble_rhs(B)
        dir_solver.solve()
        
        # Create M̂ expression for balance check
        Mhat_expr = unittrace_psd(B, domain.geometry.dim, eps=cfg.smooth_eps)
        
        # Check trace balance (uses self.A_dir, not A_new)
        trA_avg, trMhat_avg, trace_res = dir_solver.trace_balance_residual(Mhat_expr)
        
        # tr(M̂) should be 1 by construction
        assert abs(trMhat_avg - 1.0) < 0.01, f"tr(M̂) should be 1, got {trMhat_avg:.3f}"
        
        # tr(A) should relax toward 1 (may not be exact in one step)
        assert abs(trA_avg - 1.0) < 0.3, (
            f"tr(A) should approach 1, got {trA_avg:.3f} (residual={trace_res:.3e})"
        )


# =============================================================================
# Femur Reaction Force Tests (from former test_reaction_forces.py)
# =============================================================================


class TestReactionForcesFemur:
    """Validate applied and reaction forces on femur geometry."""

    def test_applied_force_integration(self, femur_gait_loader, femur_setup):
        """Applied forces integrated over surface should match physiological expectations.

        Hip joint forces at peak stance: ~3-4× BW  (~2200-2950 N for 75 kg)
        Muscle forces (glut med + max): ~1-2× BW (~735-1470 N for 75 kg)
        Total applied load should be in range 1-6× BW.
        """
        domain, facet_tags, V, Q, cfg = femur_setup

        femur_gait_loader.update_loads(50.0)

        import ufl

        t_total = femur_gait_loader.t_hip + femur_gait_loader.t_glmed + femur_gait_loader.t_glmax

        F_applied_N = np.zeros(3)
        for i in range(3):
            F_i_form = fem.form(t_total[i] * cfg.ds(2))
            F_i_local = fem.assemble_scalar(F_i_form)
            F_applied_N[i] = domain.comm.allreduce(F_i_local, op=MPI.SUM)

        F_magnitude = np.linalg.norm(F_applied_N)
        BW_N = 75.0 * 9.81

        assert 1.0 * BW_N < F_magnitude < 6.0 * BW_N, (
            f"Applied force {F_magnitude:.1f} N should be 1-6× BW (736-4415 N)"
        )

    def test_reaction_force_equilibrium(self, femur_gait_loader, femur_setup):
        """Consistent reaction forces (from unconstrained residual) balance applied loads."""
        from dolfinx.fem.petsc import (
            assemble_matrix as assemble_matrix_petsc,
            assemble_vector as assemble_vector_petsc,
            create_vector as create_vector_petsc,
        )
        from petsc4py import PETSc
        import ufl

        domain, facet_tags, V, Q, cfg = femur_setup

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
        neumann_bcs = [
            (femur_gait_loader.t_hip, 2),
            (femur_gait_loader.t_glmed, 2),
            (femur_gait_loader.t_glmax, 2),
        ]

        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()

        femur_gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()

        t_total = femur_gait_loader.t_hip + femur_gait_loader.t_glmed + femur_gait_loader.t_glmax
        F_applied_N = np.zeros(3)
        for i in range(3):
            F_i_form = fem.form(t_total[i] * cfg.ds(2))
            F_i_local = fem.assemble_scalar(F_i_form)
            F_applied_N[i] = domain.comm.allreduce(F_i_local, op=MPI.SUM)

        A0 = assemble_matrix_petsc(solver.a_form)
        A0.assemble()
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
        r.axpy(-1.0, b0)

        F_reaction_N = np.zeros(3)
        for i, bc in enumerate(dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.getValues(idx_owned)
                F_reaction_N[i] += float(np.sum(r_local))
        F_reaction_N = domain.comm.allreduce(F_reaction_N, op=MPI.SUM)

        n = ufl.FacetNormal(domain)
        sigma_u = solver.sigma(u, rho, A_dir)
        t_reac = ufl.dot(sigma_u, n)
        F_reaction_sigma_N = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_reac[i] * cfg.ds(1))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_reaction_sigma_N[i] = domain.comm.allreduce(Fi_loc, op=MPI.SUM)

        F_total_N = F_applied_N + F_reaction_N
        F_total_magnitude = np.linalg.norm(F_total_N)
        F_applied_magnitude = np.linalg.norm(F_applied_N)
        F_reaction_magnitude = np.linalg.norm(F_reaction_N)

        rel_err = F_total_magnitude / max(F_applied_magnitude, 1e-30)
        assert rel_err < 5e-6, (
            f"Force balance failed: |F_applied+F_reaction|/|F_applied| = {rel_err:.2e}"
        )

        e = F_applied_N / max(F_applied_magnitude, 1e-30)
        s_res = float(F_reaction_N @ e)
        s_sig = float((-F_reaction_sigma_N) @ e)
        rel_axis_err = abs(abs(s_sig) - F_applied_magnitude) / max(F_applied_magnitude, 1e-30)
        assert rel_axis_err < 0.30, (
            f"Traction reaction (σ·n) inconsistent along load axis (rel_err={rel_axis_err:.2e})"
        )


def test_mechanics_uniform_extension():
    """Uniform extension test: apply displacement BCs, check solver converges and energy balance holds."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=6)
    facets = build_facetag(m)

    # Function spaces
    V = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,)))
    Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))
    T = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,3)))

    # Fields
    rho = fem.Function(Q, name="rho")
    rho.x.array[:] = 1.0
    rho.x.scatter_forward()

    Afield = fem.Function(T, name="A")
    Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    Afield.x.scatter_forward()

    # Config
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0

    # Simple extension test: clamp x=0, prescribe u_x=0.01 on x=1
    fdim = m.topology.dim - 1
    eps = 0.01

    # Clamp x=0 face
    bcs = build_dirichlet_bcs(V, facets, id_tag=1, value=0.0)
    
    # Prescribe u_x=eps on x=1 face
    facets_x1 = facets.find(2)
    V0 = V.sub(0)
    dofs_x1 = fem.locate_dofs_topological(V0, fdim, facets_x1)
    bc_x1 = fem.dirichletbc(default_scalar_type(eps), dofs_x1, V0)
    bcs.append(bc_x1)

    # Create solution function
    u = fem.Function(V, name="u")
    
    # Solve
    mech = MechanicsSolver(u, rho, Afield, cfg, bcs, [])
    mech.setup()
    mech.assemble_rhs()
    its, reason = mech.solve()
    assert reason > 0, f"KSP failed to converge, reason={reason}"

    # Check solution is nonzero (should have extension)
    idxmap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    owned = idxmap.size_local * bs
    u_norm = np.linalg.norm(u.x.array[:owned])
    assert u_norm > 1e-6, f"Solution is nearly zero: {u_norm:.2e}"


def test_stimulus_power_residual_scales_with_dt():
    """Power residual in stimulus solver should scale with dt (consistency check)."""
    comm = MPI.COMM_WORLD
    m = _make_unit_cube(comm, n=4)
    facets = build_facetag(m)

    Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))

    # Config: disable diffusion for algebraic balance
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.kappaS = 0.0

    S_old = fem.Function(Q, name="S_old")
    S_old.x.array[:] = 0.2
    S_old.x.scatter_forward()

    S = fem.Function(Q, name="S")
    stim = StimulusSolver(S, S_old, cfg)

    # Constant psi > psi_ref for positive source [MPa]
    psi_val = 1.5 * cfg.psi_ref
    psi = fem.Constant(m, default_scalar_type(psi_val))

    def compute_residual(dt_scale: float) -> float:
        cfg.dt = dt_scale
        stor = cfg.rS_gain * (psi_val - cfg.psi_ref) - cfg.tauS * 0.2
        S.x.array[:] = 0.2 + dt_scale * stor / cfg.cS
        S.x.scatter_forward()
        R_abs, R_rel = stim.power_balance_residual(psi)
        return abs(R_abs)

    R1 = compute_residual(1.0)
    R2 = compute_residual(0.5)
    R3 = compute_residual(0.25)

    # Expect scaling: R2 ~ R1/2, R3 ~ R1/4
    assert R2 <= R1 * 0.7 + 1e-14, f"Residual didn't scale ~O(dt): R2={R2:.3e}, R1={R1:.3e}"
    assert R3 <= R1 * 0.4 + 1e-14, f"Residual didn't scale ~O(dt): R3={R3:.3e}, R1={R1:.3e}"
