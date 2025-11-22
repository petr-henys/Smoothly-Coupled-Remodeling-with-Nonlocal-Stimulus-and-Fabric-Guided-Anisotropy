import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_dirichlet_bcs, build_facetag
from simulation.subsolvers import MechanicsSolver, smooth_max, DensitySolver

# =============================================================================
# Constitutive Law Tests
# =============================================================================

class TestConstitutiveLaw:
    """Test stress-strain constitutive relationships."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_isotropic_stress_symmetry(self, unit_cube, facet_tags, iso_tensor_factory):
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
        A.interpolate(iso_tensor_factory)
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
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_anisotropic_stiffness_increases_energy(self, unit_cube, facet_tags, iso_tensor_factory, fiber_tensor_factory):
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
        A_iso.interpolate(iso_tensor_factory)
        A_iso.x.scatter_forward()

        A_fiber = Function(T, name="A_fiber")
        A_fiber.interpolate(fiber_tensor_factory)
        A_fiber.x.scatter_forward()

        u_test = Function(V, name="u_test")
        u_test.interpolate(lambda x: np.vstack([0.003 * x[0], 0.0 * x[1], 0.0 * x[2]]))
        u_test.x.scatter_forward()

        # Copy u_test to u for energy computation
        u.x.array[:] = u_test.x.array[:]
        u.x.scatter_forward()

        mech_iso = MechanicsSolver(u, rho, A_iso, cfg, [], [])
        mech_aniso = MechanicsSolver(u, rho, A_fiber, cfg, [], [])

        # Helper to compute energy
        def get_energy(solver):
            # Energy = 0.5 * integral(sigma : eps)
            energy_form = 0.5 * ufl.inner(solver.sigma(u, rho, solver.A_dir), solver.eps(u)) * cfg.dx
            local_energy = fem.assemble_scalar(fem.form(energy_form))
            return comm.allreduce(local_energy, op=MPI.SUM)

        energy_iso = get_energy(mech_iso)
        energy_aniso = get_energy(mech_aniso)

        assert energy_aniso >= energy_iso * 1.10, (
            "Anisotropic fabric should raise energy for the same tensile strain by ≥10%; "
            f"energy_iso={energy_iso:.3e}, energy_aniso={energy_aniso:.3e}"
        )


# =============================================================================
# Advanced Constitutive Law Tests (Dual-Power Law & Mechanostat)
# =============================================================================

class TestAdvancedConstitutiveLaws:
    """Test new density-elasticity relationship and density evolution."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_dual_power_law_stiffness(self, unit_cube, facet_tags):
        """Verify E(ρ) follows different power laws in trabecular vs cortical regimes."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        # Set distinct exponents
        cfg.n_trab = 2.0
        cfg.n_cort = 1.0
        cfg.rho_trab_max = 0.4
        cfg.rho_cort_min = 0.8
        cfg.E0 = 1000.0
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        # Helper to compute effective modulus
        def compute_E_eff(rho_val):
            rho = Function(Q, name="rho")
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            # Use UFL expression from MechanicsSolver logic (simplified)
            rho_eff = smooth_max(rho, cfg.rho_min, cfg.smooth_eps)
            
            # Replicate smoothstep logic
            rho1 = float(cfg.rho_trab_max)
            rho2 = float(cfg.rho_cort_min)
            
            s_raw = (rho_eff - rho1) / (rho2 - rho1)
            s0 = smooth_max(s_raw, 0.0, cfg.smooth_eps)
            s1 = 1.0 - smooth_max(1.0 - s0, 0.0, cfg.smooth_eps)
            w = 3.0 * s1**2 - 2.0 * s1**3
            n_eff = (1.0 - w) * cfg.n_trab + w * cfg.n_cort
            
            E_expr = cfg.E0 * (rho_eff ** n_eff)
            
            E_local = fem.assemble_scalar(fem.form(E_expr * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            E_avg = comm.allreduce(E_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
            return E_avg

        # 1. Trabecular regime (rho < rho_trab_max)
        rho_trab = 0.2
        E_trab = compute_E_eff(rho_trab)
        E_expected_trab = cfg.E0 * (rho_trab ** cfg.n_trab)
        assert abs(E_trab - E_expected_trab) / E_expected_trab < 0.01,             f"Trabecular law failed: E({rho_trab})={E_trab:.2f}, expected {E_expected_trab:.2f}"

        # 2. Cortical regime (rho > rho_cort_min)
        rho_cort = 0.9
        E_cort = compute_E_eff(rho_cort)
        E_expected_cort = cfg.E0 * (rho_cort ** cfg.n_cort)
        assert abs(E_cort - E_expected_cort) / E_expected_cort < 0.01,             f"Cortical law failed: E({rho_cort})={E_cort:.2f}, expected {E_expected_cort:.2f}"

        # 3. Transition regime (rho_trab_max < rho < rho_cort_min)
        rho_trans = 0.6
        E_trans = compute_E_eff(rho_trans)
        # Should be between the two power laws
        E_low = cfg.E0 * (rho_trans ** cfg.n_trab)
        E_high = cfg.E0 * (rho_trans ** cfg.n_cort)
        # Since n_trab=2 > n_cort=1 and rho < 1, rho^2 < rho^1, so E_low < E_high
        assert E_low < E_trans < E_high,             f"Transition E({rho_trans})={E_trans:.2f} not between {E_low:.2f} and {E_high:.2f}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_linear_driver_rate(self, unit_cube, facet_tags, iso_tensor_factory):
        """Verify density evolution rate is proportional to stimulus."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        cfg.k_rho = 1.0
        cfg.dt = 0.1
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        rho_old.x.array[:] = 0.5
        rho_old.x.scatter_forward()
        
        A = Function(T, name="A")
        A.interpolate(iso_tensor_factory)
        A.x.scatter_forward()
        
        S = Function(Q, name="S")
        
        def get_rate(S_val):
            S.x.array[:] = S_val
            S.x.scatter_forward()
            
            dens = DensitySolver(rho, rho_old, A, S, cfg)
            dens.setup()
            dens.assemble_rhs()
            dens.solve()
            
            M_new = comm.allreduce(fem.assemble_scalar(fem.form(rho * cfg.dx)), op=MPI.SUM)
            M_old = comm.allreduce(fem.assemble_scalar(fem.form(rho_old * cfg.dx)), op=MPI.SUM)
            dM = M_new - M_old
            vol = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * cfg.dx)), op=MPI.SUM)
            
            return (dM / vol) / cfg.dt
        
        # Rate should be k_rho * S * (rho_max - rho) for S > 0
        # Rate should be k_rho * S * (rho - rho_min) for S < 0 (S is negative, so rate is negative)
        
        # Case 1: S = 0.1
        rate_pos = get_rate(0.1)
        expected_pos = cfg.k_rho * 0.1 * (cfg.rho_max - 0.5)
        assert abs(rate_pos - expected_pos) < 1e-3, f"Positive rate mismatch: got {rate_pos}, expected {expected_pos}"
        
        # Case 2: S = -0.1
        rate_neg = get_rate(-0.1)
        # Formula: k_rho * (S_plus*rho_max + S_minus*rho_min - (S_plus+S_minus)*rho)
        # S_plus=0, S_minus=0.1
        # Rate = k_rho * (0.1 * rho_min - 0.1 * rho) = k_rho * 0.1 * (rho_min - rho)
        expected_neg = cfg.k_rho * 0.1 * (cfg.rho_min - 0.5)
        assert abs(rate_neg - expected_neg) < 1e-3, f"Negative rate mismatch: got {rate_neg}, expected {expected_neg}"
