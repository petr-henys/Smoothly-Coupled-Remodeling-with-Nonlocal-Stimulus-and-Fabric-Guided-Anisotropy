import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_dirichlet_bcs, build_facetag
from simulation.subsolvers import MechanicsSolver, DensitySolver, DirectionSolver, unittrace_psd
from tests.physics_utils import iso_tensor, make_unit_cube

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
        A.interpolate(iso_tensor)
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
        A.interpolate(iso_tensor)
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
        A.interpolate(iso_tensor)
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
        domain = make_unit_cube(comm, 8)
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
        A.interpolate(iso_tensor)
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
        
        # Increase remodeling rate for this test to ensure measurable change in one step
        cfg.lambda_rho = 0.2

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
            A_field.interpolate(iso_tensor)
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
        A_iso.interpolate(iso_tensor)
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
        # Disable distal damping for unit cube tests
        cfg.distal_damping_height = 0.0
        cfg.distal_damping_transition = 0.0
        
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
        A.interpolate(iso_tensor)
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

    def test_stimulus_power_residual_scales_with_dt(self):
        """Power residual in stimulus solver should scale with dt (consistency check)."""
        comm = MPI.COMM_WORLD
        m = make_unit_cube(comm, n=4)
        facets = build_facetag(m)

        Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))

        # Config: disable diffusion for algebraic balance
        cfg = Config(domain=m, facet_tags=facets, verbose=False)
        cfg.kappaS = 0.0
        # Disable distal damping for unit cube tests
        cfg.distal_damping_height = 0.0
        cfg.distal_damping_transition = 0.0

        S_old = fem.Function(Q, name="S_old")
        S_old.x.array[:] = 0.2
        S_old.x.scatter_forward()

        S = fem.Function(Q, name="S")
        from simulation.subsolvers import StimulusSolver
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
