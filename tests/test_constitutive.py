import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.params import MaterialParams, DensityParams
from simulation.solvers import DensitySolver, MechanicsSolver
from simulation.utils import build_dirichlet_bcs, smooth_max

# =============================================================================
# Constitutive Law Tests
# =============================================================================

class TestConstitutiveLaw:
    """Test stress-strain constitutive relationships."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_isotropic_stress_symmetry(self, unit_cube, facet_tags):
        """Verify stress tensor is symmetric for isotropic material."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
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
        """Verify E(ρ) follows the variable-exponent law from the manuscript."""
        comm = MPI.COMM_WORLD
        cfg = Config(
            domain=unit_cube, facet_tags=facet_tags,
            material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2, E0=1000.0),
            density=DensityParams(rho_ref=1.0)
        )
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        # Helper to compute effective modulus
        def compute_E_eff(rho_val):
            rho = Function(Q, name="rho")
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            # Use UFL expression from MechanicsSolver logic
            rho_eff = smooth_max(rho, cfg.density.rho_min, cfg.numerics.smooth_eps)
            rho_rel = rho_eff / cfg.density.rho_ref

            t = (rho_eff - cfg.material.rho_trab_max) / (cfg.material.rho_cort_min - cfg.material.rho_trab_max)
            w = ufl.conditional(ufl.le(t, 0.0), 0.0, ufl.conditional(ufl.ge(t, 1.0), 1.0, t))
            w = w * w * (3.0 - 2.0 * w)
            k = cfg.material.n_trab * (1.0 - w) + cfg.material.n_cort * w
            E_expr = cfg.material.E0 * (rho_rel ** k)
            
            E_local = fem.assemble_scalar(fem.form(E_expr * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            E_avg = comm.allreduce(E_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
            return E_avg

        # Test at different densities
        test_densities = [0.5, 1.0, 1.5]
        for rho_val in test_densities:
            E_computed = compute_E_eff(rho_val)

            # Python-side expected value (smooth_max ~ max for rho_val >> rho_min)
            t = (rho_val - cfg.material.rho_trab_max) / (cfg.material.rho_cort_min - cfg.material.rho_trab_max)
            t = max(0.0, min(1.0, t))
            w = t * t * (3.0 - 2.0 * t)
            k = cfg.material.n_trab * (1.0 - w) + cfg.material.n_cort * w
            E_expected = cfg.material.E0 * ((rho_val / cfg.density.rho_ref) ** k)
            rel_err = abs(E_computed - E_expected) / E_expected
            assert rel_err < 0.01, \
                f"Power law failed at ρ={rho_val}: E={E_computed:.2f}, expected {E_expected:.2f}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_linear_driver_rate(self, unit_cube, facet_tags):
        """Verify density evolution rate follows specific energy stimulus."""
        comm = MPI.COMM_WORLD
        # Note: k_rho is now in density params, but we set dt on cfg
        k_rho = 1.0  # Local test value
        
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2),
                    density=DensityParams(k_rho_form=k_rho, k_rho_resorb=k_rho, surface_use=False))

        cfg.set_dt(0.1)
    
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        psi_field = Function(Q, name="psi")
        rho_val = cfg.density.rho_ref  # Use reference density for simpler math
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
        
        # DensitySolver expects dimensionless stimulus S, not SED psi.
        
        def expected_rate_implicit(S_val, rho_val, k, limit):
            # Rate = (A - B*rho) / (1 + B*dt) for implicit Euler
            # Rate_explicit = k * |S| * (1 - rho/limit) = k*|S| - (k*|S|/limit)*rho
            # So A = k*|S|, B = k*|S|/limit
            S_abs = abs(S_val)
            A = k * S_abs
            B = k * S_abs / limit
            numerator = A - B * rho_val
            denominator = 1.0 + B * cfg.dt
            return numerator / denominator

        # Case 1: S = 1.0
        print(f"DEBUG: k_rho_form={cfg.density.k_rho_form}, k_rho_resorb={cfg.density.k_rho_resorb}")
        rate_pos = get_rate(1.0)
        # With S=1.0, rho=1.0, rho_max=2.0, k=1.0:
        expected_pos = expected_rate_implicit(1.0, rho_val, k_rho, cfg.density.rho_max)
        assert abs(rate_pos - expected_pos) < 0.05, f"Positive rate mismatch: got {rate_pos}, expected {expected_pos}"
        
        # Case 2: S = -0.5
        rate_neg = get_rate(-0.5)
        # With S=-0.5, rho=1.0, rho_min=0.1, k=1.0:
        expected_neg = expected_rate_implicit(-0.5, rho_val, k_rho, cfg.density.rho_min)
        assert abs(rate_neg - expected_neg) < 0.05, f"Negative rate mismatch: got {rate_neg}, expected {expected_neg}"


