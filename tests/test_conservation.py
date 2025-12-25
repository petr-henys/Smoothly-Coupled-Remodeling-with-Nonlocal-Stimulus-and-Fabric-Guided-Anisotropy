import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.params import MaterialParams, DensityParams, NumericsParams, StimulusParams
from simulation.utils import build_dirichlet_bcs, build_facetag
from simulation.solvers import MechanicsSolver, DensitySolver
from dolfinx import mesh

def make_unit_cube(comm=MPI.COMM_WORLD, n=6):
    """Create a unit cube mesh."""
    return mesh.create_unit_cube(comm, n, n, n)

# =============================================================================
# Thermodynamic Consistency Tests
# =============================================================================

class TestThermodynamics:
    """Test energy dissipation and thermodynamic consistency."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_strain_energy_positivity(self, unit_cube, facet_tags):
        """Strain energy density ψ = 0.5*σ:ε must be non-negative."""
        comm = MPI.COMM_WORLD
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        
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
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        
        psi = 0.5 * ufl.inner(mech.sigma(u, rho), mech.eps(u))
        
        psi_local = fem.assemble_scalar(fem.form(psi * cfg.dx))
        psi_global = comm.allreduce(psi_local, op=MPI.SUM)
        
        assert psi_global >= -1e-12, f"Strain energy must be non-negative, got {psi_global}"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_linear_elastic_energy_balance(self, unit_cube, facet_tags, traction_factory):
        """For the linear system, internal work equals external work: W_int ≈ W_ext.
        
        Tests manual calculation (a(u,u) vs l(u)).
        """
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))

        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)

        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)

        # Fields
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.7
        rho.x.scatter_forward()

        # Dirichlet on x=0, traction on x=1 (axis 0)
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        t_const, t_tag = traction_factory(-0.4, facet_id=2, axis=0)

        mech = MechanicsSolver(u, rho, cfg, bc_mech, [(t_const, t_tag)])
        mech.setup()
        mech.solve()

        # Internal work: a(u,u) = ∫ σ:ε dx
        a_uu_local = fem.assemble_scalar(fem.form(ufl.inner(mech.sigma(u, rho), mech.eps(u)) * cfg.dx))
        a_uu = comm.allreduce(a_uu_local, op=MPI.SUM)

        # External work: l(u) = ∫ t·u ds on tagged facet(s)
        l_u_local = fem.assemble_scalar(fem.form(ufl.inner(t_const, u) * cfg.ds(t_tag)))
        l_u = comm.allreduce(l_u_local, op=MPI.SUM)

        # Energy balance: a(u,u) ≈ l(u)
        denom = max(abs(l_u), abs(a_uu), 1e-300)
        rel_gap = abs(a_uu - l_u) / denom
        assert rel_gap < 1e-5, f"Energy balance violated: a(u,u)={a_uu:.6e}, l(u)={l_u:.6e}, rel_gap={rel_gap:.3e}"


# =============================================================================
# Conservation Tests
# =============================================================================

class TestConservation:
    """Test conservation properties and equilibrium."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_force_equilibrium_no_body_force(self, unit_cube):
        """With no body force and homogeneous BCs, internal forces should sum to zero.
        
        Also tests that with traction applied, solver converges.
        """
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        
        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        # Test 1: No load case - BCs: left fixed, no traction on right
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        
        mech.setup()
        mech.solve()
        
        # Check residual: displacement should be zero when no load applied
        u_norm_sq_local = fem.assemble_scalar(fem.form(ufl.inner(u, u) * cfg.dx))
        u_norm_sq = comm.allreduce(u_norm_sq_local, op=MPI.SUM)
        
        assert u_norm_sq < 1e-12, f"No-load case should yield zero displacement, got ||u||²={u_norm_sq}"
        
        # Test 2: With traction - clamp x=0, apply traction on x=1
        t0 = fem.Constant(domain, np.array([1.0, 0.0, 0.0], dtype=float))
        neumanns = [(t0, 2)]

        mech2 = MechanicsSolver(u, rho, cfg, bc_mech, neumanns)
        mech2.setup()
        stats = mech2.solve()

        # Check solver converged
        assert stats.converged, f"KSP failed to converge, reason={stats.ksp_reason}"
        
        # Check energy balance via manual calculation
        W_int_local = fem.assemble_scalar(fem.form(ufl.inner(mech2.sigma(u, rho), mech2.eps(u)) * cfg.dx))
        W_int = comm.allreduce(W_int_local, op=MPI.SUM)
        W_ext_local = fem.assemble_scalar(fem.form(ufl.inner(t0, u) * cfg.ds(2)))
        W_ext = comm.allreduce(W_ext_local, op=MPI.SUM)
        
        denom = max(abs(W_int), abs(W_ext), 1e-300)
        rel_err = abs(W_int - W_ext) / denom
        assert rel_err < 1e-6, f"Energy balance violated: rel_err={rel_err:.2e} (W_int={W_int:.3e}, W_ext={W_ext:.3e})"

    
    def test_density_bounds_preservation(self):
        """Density solver should relax toward bounds under stimulus sign."""
        comm = MPI.COMM_WORLD
        domain = make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        psi_field = Function(Q, name="psi")
        
        # Start with out-of-bounds initial condition (below rho_min)
        rho_min = float(cfg.density.rho_min)
        rho_max = float(cfg.density.rho_max)
        rho_initial = 0.5 * rho_min
        rho_old.x.array[:] = rho_initial
        rho_old.x.scatter_forward()
        
        # Set positive stimulus (psi > psi_ref) -> drives toward rho_max
        psi_val = 1.5 * cfg.stimulus.psi_ref
        psi_field.x.array[:] = psi_val
        psi_field.x.scatter_forward()
        
        densolver = DensitySolver(rho, rho_old, psi_field, cfg)
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
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
    
        # Note: k_rho is in density params but we use defaults here
        cfg.set_dt(10.0)
        # Disable distal damping for unit cube test
        # Note: distal_damping_height is not a standard param, skip if not available
    
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)

        rho_min = float(cfg.density.rho_min)
        rho_max = float(cfg.density.rho_max)
        rho_mid = 0.5 * (rho_min + rho_max)

        def _solve_density(stimulus_value: float) -> float:
            rho_old = Function(Q, name="rho_old")
            rho_old.x.array[:] = rho_mid
            rho_old.x.scatter_forward()

            rho = Function(Q, name="rho")
            psi_field = Function(Q, name="psi")
            
            # S = psi - psi_ref (dimensional).  So psi = psi_ref + S
            psi_val = cfg.stimulus.psi_ref + stimulus_value
            psi_field.x.array[:] = psi_val
            psi_field.x.scatter_forward()

            dens = DensitySolver(rho, rho_old, psi_field, cfg)
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

        # Just check that density moves in correct direction
        # The magnitude depends on many parameters (k_rho, dt, etc.)
        assert pos_delta > 0, (
            "Positive stimulus should raise density; "
            f"baseline={baseline}, mean={rho_mean_positive}, Δ={pos_delta}"
        )
        assert neg_delta > 0, (
            "Negative stimulus should reduce density; "
            f"baseline={baseline}, mean={rho_mean_negative}, Δ={neg_delta}"
        )

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_total_mass_conservation_zero_stimulus(self, unit_cube, facet_tags):
        """With S=0 and natural (no-flux) boundaries, ∫ρ dx is conserved by diffusion step."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        # Disable distal damping (skip if not available in new API)
    
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)

        # Non-uniform initial density to exercise diffusion but preserve mass
        rho_old = Function(Q, name="rho_old")
        rho_old.interpolate(lambda x: 0.6 + 0.2 * np.sin(2*np.pi*x[0]) * np.cos(2*np.pi*x[1]))
        rho_old.x.scatter_forward()

        # Solve one implicit diffusion step with natural BCs
        rho = Function(Q, name="rho")
        # Initialize rho to rho_old (solver uses rho in stimulus calculation)
        rho.x.array[:] = rho_old.x.array[:]
        rho.x.scatter_forward()
        
        # For zero stimulus S=0, we pass a zero function.
        # (DensitySolver expects dimensionless stimulus S, not SED psi)
        S_field = Function(Q, name="S")
        S_field.x.array[:] = 0.0
        S_field.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, S_field, cfg)
        dens.setup()
        dens.assemble_rhs()
        dens.solve()
        rho.x.scatter_forward()

        # Assemble initial total mass
        m_old_local = fem.assemble_scalar(fem.form(rho_old * cfg.dx))
        m_old = comm.allreduce(m_old_local, op=MPI.SUM)

        m_new_local = fem.assemble_scalar(fem.form(rho * cfg.dx))
        m_new = comm.allreduce(m_new_local, op=MPI.SUM)

        rel_diff = abs(m_new - m_old) / max(abs(m_old), 1e-300)
        # With smoothed |S| ≈ sqrt(S^2 + eps^2), at S=0 a small reaction ~eps remains.
        # Allow a tolerance proportional to smooth_eps plus a tiny numerical margin.
        eps = float(cfg.numerics.smooth_eps)
        tol = max(5e-10, 5.0 * eps)
        assert rel_diff < tol, (
            "Total mass approximately conserved for S=0 within smoothing tolerance; "
            f"rel_diff={rel_diff:.3e}, tol={tol:.3e} (m_old={m_old:.6e}, m_new={m_new:.6e})"
        )




# =============================================================================
# Conservation and Balance Check Tests
# =============================================================================

class TestConservationChecks:
    """Test physical conservation/balance checks for all subsolvers."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_density_evolves_correctly(self, unit_cube):
        """DensitySolver: Density should increase with positive stimulus."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags,
                    material=MaterialParams(n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2))
        cfg.set_dt(10.0)  # 10 days
        
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        Q = functionspace(unit_cube, P1)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        psi_field = Function(Q, name="psi")
        rho_old.x.array[:] = 0.5
        rho_old.x.scatter_forward()
        
        # Strong positive stimulus -> formation (S = 1.0 -> psi = 2 * psi_ref)
        psi_field.x.array[:] = 2.0 * cfg.stimulus.psi_ref
        psi_field.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, psi_field, cfg)
        dens.setup()
        dens.assemble_rhs()
        dens.solve()
        rho.x.scatter_forward()
        
        # Check density increased with positive stimulus
        n_owned = Q.dofmap.index_map.size_local
        rho_mean = comm.allreduce(rho.x.array[:n_owned].sum(), op=MPI.SUM) / comm.allreduce(n_owned, op=MPI.SUM)
        rho_old_mean = 0.5
        
        assert rho_mean > rho_old_mean, (
            f"Density should increase with positive stimulus: rho_mean={rho_mean:.4f}, rho_old={rho_old_mean}"
        )
