import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver, smooth_max, DensitySolver

# =============================================================================
# Constitutive Law Tests
# =============================================================================

class TestConstitutiveLaw:
    """Test stress-strain constitutive relationships."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_isotropic_stress_symmetry(self, unit_cube, facet_tags):
        """Verify stress tensor is symmetric for isotropic material."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags)
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        
        # Create test displacement with non-zero gradient
        u = Function(V, name="u")
        u.interpolate(lambda x: np.array([0.001*x[0]*x[1], 0.002*x[1]*x[2], 0.001*x[0]*x[2]]))
        u.x.scatter_forward()
        
        # Uniform density
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        
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


# =============================================================================
# Advanced Constitutive Law Tests (Dual-Power Law & Mechanostat)
# =============================================================================

class TestAdvancedConstitutiveLaws:
    """Test density-elasticity relationship and density evolution."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_power_law_stiffness(self, unit_cube, facet_tags):
        """Verify E(ρ) follows the power-law relationship E = E0 * (ρ/ρ_ref)^n."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags)
        
        # Set exponent
        cfg.n = 2.0
        cfg.E0 = 1000.0
        cfg.rho_ref = 1.0
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        # Helper to compute effective modulus
        def compute_E_eff(rho_val):
            rho = Function(Q, name="rho")
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            # Use UFL expression from MechanicsSolver logic
            rho_eff = smooth_max(rho, cfg.rho_min, cfg.smooth_eps)
            rho_rel = rho_eff / cfg.rho_ref
            E_expr = cfg.E0 * (rho_rel ** cfg.n)
            
            E_local = fem.assemble_scalar(fem.form(E_expr * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            E_avg = comm.allreduce(E_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
            return E_avg

        # Test at different densities
        test_densities = [0.5, 1.0, 1.5]
        for rho_val in test_densities:
            E_computed = compute_E_eff(rho_val)
            E_expected = cfg.E0 * ((rho_val / cfg.rho_ref) ** cfg.n)
            rel_err = abs(E_computed - E_expected) / E_expected
            assert rel_err < 0.01, \
                f"Power law failed at ρ={rho_val}: E={E_computed:.2f}, expected {E_expected:.2f}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_linear_driver_rate(self, unit_cube, facet_tags):
        """Verify density evolution rate follows specific energy stimulus."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags)
        
        cfg.k_rho = 1.0
        cfg.dt = 0.1
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        psi_field = Function(Q, name="psi")
        rho_val = cfg.rho_ref  # Use reference density for simpler math
        rho_old.x.array[:] = rho_val
        rho_old.x.scatter_forward()
        
        def get_rate(psi_val):
            psi_field.x.array[:] = psi_val
            psi_field.x.scatter_forward()
            
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            dens = DensitySolver(rho, rho_old, psi_field, cfg)
            dens.setup()
            dens.assemble_rhs()
            dens.solve()
            
            M_new = comm.allreduce(fem.assemble_scalar(fem.form(rho * cfg.dx)), op=MPI.SUM)
            M_old = comm.allreduce(fem.assemble_scalar(fem.form(rho_old * cfg.dx)), op=MPI.SUM)
            dM = M_new - M_old
            vol = comm.allreduce(fem.assemble_scalar(fem.form(1.0 * cfg.dx)), op=MPI.SUM)
            
            return (dM / vol) / cfg.dt
        
        # DensitySolver uses: ∂ρ/∂t = k_rho * (psi/rho - psi_ref/rho_ref) / (psi_ref/rho_ref)
        # With rho = rho_ref, this simplifies to: k_rho * (psi - psi_ref) / psi_ref
        
        # Case 1: psi = 2*psi_ref → S = 1.0
        rate_pos = get_rate(2.0 * cfg.psi_ref)
        expected_pos = cfg.k_rho * 1.0
        assert abs(rate_pos - expected_pos) < 0.05, f"Positive rate mismatch: got {rate_pos}, expected {expected_pos}"
        
        # Case 2: psi = 0.5*psi_ref → S = -0.5
        rate_neg = get_rate(0.5 * cfg.psi_ref)
        expected_neg = cfg.k_rho * (-0.5)
        assert abs(rate_neg - expected_neg) < 0.05, f"Negative rate mismatch: got {rate_neg}, expected {expected_neg}"

