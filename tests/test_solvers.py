#!/usr/bin/env python3
"""Tests for solver matrix assembly, SPD properties, and numerical stability."""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function
import basix.ufl

from simulation.config import Config
from simulation.solvers import MechanicsSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver
from simulation.utils import build_dirichlet_bcs, build_facetag

from simulation.stats import SweepStats


class MockDriver:
    """Mock driver for testing solvers without full gait integration."""
    def __init__(self, mech):
        self.mech = mech
        self.psi_expr = fem.Constant(mech.u.function_space.mesh, 0.0)

    def stimulus_expr(self):
        return self.psi_expr

    def invalidate(self):
        pass

    def sweep(self) -> SweepStats:
        return SweepStats(label="mock", ksp_iters=0, ksp_reason=0, solve_time=0.0)

    def setup(self):
        self.mech.setup()

    def destroy(self):
        pass

    def update_stiffness(self):
        self.mech.assemble_lhs()


_RNG = np.random.default_rng(1234)

# =============================================================================
# Solver Statistics Tests
# =============================================================================

class TestSolverStatistics:
    """Test solver statistics tracking."""

    def test_ksp_reason_tracking(self, cfg, spaces, fields, bc_mech):
        """Verify KSP convergence reason is exposed after a solve via SweepStats."""
        from dolfinx import fem
        import numpy as np
        from simulation.stats import SweepStats
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, _, A, _, _, _ = fields

        # Create traction directly
        t_load = fem.Function(V, name="traction")
        vec = np.zeros(3, dtype=float)
        vec[0] = -0.1
        t_load.interpolate(lambda x: np.tile(vec.reshape(3, 1), (1, x.shape[1])))
        t_load.x.scatter_forward()

        mech = MechanicsSolver(u, rho, cfg, bc_mech, [(t_load, 2)])

        mech.setup()

        stats = mech.solve()
        assert isinstance(stats, SweepStats), "solve() should return SweepStats"
        assert stats.converged, "Solver should have converged"
        assert stats.ksp_reason == mech.last_reason, "last_reason not set correctly"
        assert stats.ksp_iters > 0, "Should have positive iteration count"
        assert stats.solve_time > 0, "Should have positive solve time"


# =============================================================================
# Matrix Assembly Tests
# =============================================================================

class TestMatrixAssembly:
    """Test matrix assembly correctness."""
    
    def test_mechanics_stiffness_spd(self, cfg, spaces, fields, bc_mech):
        """Mechanics stiffness matrix should be symmetric positive definite (modulo BCs).
        
        Tests: (1) No NaN/Inf entries, (2) K ≈ K^T symmetry, (3) z^T K z ≥ 0 for random z.
        """
        V, Q, T = spaces.V, spaces.Q, spaces.T
        _, rho, _, A, _, _, _ = fields
        u = fem.Function(V, name="u")
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        
        mech.setup()
        
        from petsc4py import PETSc
        mech.A.assemble()
        
        # Test 1: No NaN/Inf
        norm = mech.A.norm(PETSc.NormType.FROBENIUS)
        assert np.isfinite(norm), f"Stiffness matrix has non-finite entries"
        assert norm > 0, f"Stiffness matrix is zero"
        
        # Test 2: Symmetry K ≈ K^T
        KT = mech.A.transpose()
        diff = mech.A.copy()
        diff.axpy(-1.0, KT, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        diff.assemble()
        nK = mech.A.norm()
        nDiff = diff.norm()
        assert nDiff / max(1e-16, nK) < 1e-10, f"Matrix not symmetric: ||K-K^T||/||K|| = {nDiff/nK:.2e}"
        
        # Test 3: Positive definiteness z^T K z ≥ 0
        z = mech.A.createVecLeft()
        z.setRandom()
        y = mech.A.createVecLeft()
        mech.A.mult(z, y)
        energy = z.dot(y)
        assert energy >= -1e-10, f"Matrix not PSD: z^T K z = {energy:.2e}"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    @pytest.mark.parametrize("solver_type", ["mechanics", "density"])
    def test_solver_operator_positive_definite(self, unit_cube, cfg, spaces, bc_mech, solver_type):
        """Check x^T A x > 0 (or >= 0) for solver operators with random test vectors.
        
        Mechanics: Positive-definite with Dirichlet DOFs zeroed
        Density: Positive semi-definite (allow zero eigenvalues)
        """
        V, Q, T = spaces.V, spaces.Q, spaces.T
        
        if solver_type == "mechanics":
            # Mechanics solver: positive-definite for non-Dirichlet DOFs
            u = Function(V, name="u")
            rho = Function(Q, name="rho"); rho.x.array[:] = 0.6; rho.x.scatter_forward()
            solver = MechanicsSolver(u, rho, cfg, bc_mech, [])
            solver.setup()
            
            # Random vector with Dirichlet DOFs zeroed
            z = Function(V, name="z")
            n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            z.x.array[:n_owned] = _RNG.standard_normal(n_owned)
            from simulation.utils import collect_dirichlet_dofs
            fixed = collect_dirichlet_dofs(bc_mech, n_owned)
            if fixed.size:
                z.x.array[fixed] = 0.0
            z.x.scatter_forward()
            
            y = z.x.petsc_vec.duplicate()
            solver.A.mult(z.x.petsc_vec, y)
            energy = z.x.petsc_vec.dot(y)
            assert energy > 0.0, f"Mechanics stiffness not positive-definite: {energy}"
            
        elif solver_type == "density":
            # Density solver: positive semi-definite
            rho = Function(Q, name="rho")
            rho_old = Function(Q, name="rho_old"); rho_old.x.array[:] = 0.5; rho_old.x.scatter_forward()
            psi_field = Function(Q, name="psi"); psi_field.x.array[:] = cfg.stimulus.psi_ref_trab; psi_field.x.scatter_forward()
            solver = DensitySolver(rho, rho_old, psi_field, cfg)
            solver.setup()  # setup() already calls assemble_lhs()
            
            # Random vector
            z = Function(Q, name="z")
            n_owned = Q.dofmap.index_map.size_local
            z.x.array[:n_owned] = _RNG.standard_normal(n_owned)
            z.x.scatter_forward()
            
            y = z.x.petsc_vec.duplicate()
            solver.A.mult(z.x.petsc_vec, y)
            energy = z.x.petsc_vec.dot(y)
            assert energy >= 0.0, f"Density operator not positive semi-definite: {energy}"

    def test_fabric_stiffness_spd(self, cfg, spaces, fields):
        """Fabric stiffness matrix should be symmetric positive definite."""
        V, Q, T = spaces.V, spaces.Q, spaces.T
        _, rho, _, A, A_old, _, _ = fields
        
        # Fabric solver uses T (tensor space) for L
        L = A
        L_old = A_old
        Qbar = fem.Function(T, name="Qbar")
        
        from simulation.solvers import FabricSolver
        solver = FabricSolver(L, L_old, Qbar, cfg)
        solver.setup()
        
        from petsc4py import PETSc
        solver.A.assemble()
        
        # Test 1: No NaN/Inf
        norm = solver.A.norm(PETSc.NormType.FROBENIUS)
        assert np.isfinite(norm), f"Fabric matrix has non-finite entries"
        
        # Test 2: Symmetry K ≈ K^T
        KT = solver.A.transpose()
        diff = solver.A.copy()
        diff.axpy(-1.0, KT, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        diff.assemble()
        nK = solver.A.norm()
        nDiff = diff.norm()
        assert nDiff / max(1e-16, nK) < 1e-10, f"Fabric matrix not symmetric: {nDiff/nK:.2e}"
        
        # Test 3: Positive definiteness
        z = solver.A.createVecLeft()
        z.setRandom()
        y = solver.A.createVecLeft()
        solver.A.mult(z, y)
        energy = z.dot(y)
        assert energy > -1e-10, f"Fabric matrix not PSD: {energy:.2e}"

# =============================================================================
# Stimulus Solver Tests
# =============================================================================

from simulation.solvers import StimulusSolver, FabricSolver

@pytest.fixture
def stimulus_setup(cfg, spaces, fields):
    """Setup for StimulusSolver tests."""
    V, Q, T = spaces.V, spaces.Q, spaces.T
    u, rho, rho_old, A, A_old, S, S_old = fields
    
    # Set some initial values
    rho.x.array[:] = 1.0
    rho.x.scatter_forward()
    
    # Create a psi field (SED)
    psi = fem.Function(Q, name="psi")
    psi.x.array[:] = 0.0
    psi.x.scatter_forward()
    
    solver = StimulusSolver(S, S_old, psi, rho, cfg)
    return solver, S, S_old, psi, rho

@pytest.fixture
def fabric_setup(cfg, spaces, fields):
    """Setup for FabricSolver tests."""
    V, Q, T = spaces.V, spaces.Q, spaces.T
    u, rho, rho_old, A, A_old, S, S_old = fields
    
    # Fabric tensor L (deviatoric part of fabric)
    # We use A from fields as L here for convenience, assuming T is tensor space
    L = A
    L_old = A_old
    
    # Initialize L to zero (isotropic)
    L.x.array[:] = 0.0
    L.x.scatter_forward()
    L_old.x.array[:] = 0.0
    L_old.x.scatter_forward()
    
    # Qbar is the stimulus tensor (e.g. normalized stress/strain)
    Qbar = fem.Function(T, name="Qbar")
    Qbar.x.array[:] = 0.0
    Qbar.x.scatter_forward()
    
    solver = FabricSolver(L, L_old, Qbar, cfg)
    return solver, L, L_old, Qbar

class TestStimulusSolver:
    """Tests for StimulusSolver."""

    def test_initialization(self, stimulus_setup):
        """Test that solver initializes correctly."""
        solver, S, S_old, psi, rho = stimulus_setup
        assert solver.S == S
        assert solver.S_old == S_old
        assert solver.psi == psi
        assert solver.rho == rho

    def test_solve_zero_stimulus(self, stimulus_setup):
        """With zero psi, S should decay towards 0."""
        solver, S, S_old, psi, rho = stimulus_setup
        
        # Set initial S to something positive
        S_old.x.array[:] = 1.0
        S_old.x.scatter_forward()
        S.x.array[:] = 1.0
        S.x.scatter_forward()
        
        # Psi is 0.0 by default
        
        solver.setup()
        solver.solve()
        
        # S should decrease (decay)
        s_max = MPI.COMM_WORLD.allreduce(S.x.array.max(), op=MPI.MAX)
        assert s_max < 1.0, f"S did not decay: {s_max}"

    def test_solve_high_stimulus(self, stimulus_setup, cfg):
        """With high psi, S should grow towards S_max."""
        solver, S, S_old, psi, rho = stimulus_setup
        
        # Set initial S to 0
        S_old.x.array[:] = 0.0
        S_old.x.scatter_forward()
        S.x.array[:] = 0.0
        S.x.scatter_forward()
        
        # Set psi high enough to trigger formation
        # m = psi/rho. m_ref = psi_ref.
        # We want m > m_ref * (1 + delta0)
        rho_val = 1.0
        rho.x.array[:] = rho_val
        rho.x.scatter_forward()
        
        m_ref = cfg.stimulus.psi_ref_trab
        target_m = m_ref * 2.0 # Well above threshold
        psi.x.array[:] = target_m * rho_val
        psi.x.scatter_forward()
        
        solver.setup()
        solver.solve()
        
        # S should increase
        s_max = MPI.COMM_WORLD.allreduce(S.x.array.max(), op=MPI.MAX)
        assert s_max > 0.0, f"S did not grow: {s_max}"


class TestFabricSolver:
    """Tests for FabricSolver."""

    def test_initialization(self, fabric_setup):
        """Test that solver initializes correctly."""
        solver, L, L_old, Qbar = fabric_setup
        assert solver.L == L
        assert solver.L_old == L_old
        assert solver.Qbar == Qbar

    def test_solve_isotropic_target(self, fabric_setup):
        """If Qbar is isotropic, L should remain/become isotropic (zero)."""
        solver, L, L_old, Qbar = fabric_setup
        
        # Qbar = 0 -> Isotropic target
        Qbar.x.array[:] = 0.0
        Qbar.x.scatter_forward()
        
        # Start with some anisotropy
        L_old.x.array[:] = 0.1
        L_old.x.scatter_forward()
        L.x.array[:] = 0.1
        L.x.scatter_forward()
        
        solver.setup()
        solver.solve()
        
        # L should decay towards 0
        l_norm = MPI.COMM_WORLD.allreduce(np.linalg.norm(L.x.array), op=MPI.SUM)
        l_old_norm = MPI.COMM_WORLD.allreduce(np.linalg.norm(L_old.x.array), op=MPI.SUM)
        
        assert l_norm < l_old_norm, "L did not decay towards isotropic target"


# =============================================================================
# Physical Sanity Check Tests
# =============================================================================

class TestPhysicalSanityChecks:
    """Critical tests for physical validity: no inf, nan, negative where forbidden."""
    
    def test_mechanics_no_nan_inf_in_displacement(self, cfg, spaces, fields, bc_mech):
        """Mechanics solution should never contain NaN or Inf."""
        from dolfinx import fem
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, _, A, _, _, _ = fields
        
        # Set realistic density
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()
        
        # Create traction load
        t_load = fem.Function(V, name="traction")
        t_load.interpolate(lambda x: np.array([[-0.1], [0.0], [0.0]]) * np.ones((1, x.shape[1])))
        t_load.x.scatter_forward()
        
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [(t_load, 2)])
        mech.setup()
        mech.solve()
        
        # Check no NaN/Inf
        assert np.all(np.isfinite(u.x.array)), "Displacement contains NaN or Inf"
    
    def test_mechanics_no_nan_inf_with_extreme_density(self, cfg, spaces, fields, bc_mech):
        """Mechanics should handle extreme (but valid) density values."""
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, _, A, _, _, _ = fields
        
        # Test with minimum density (stiffness can be very low)
        rho_min = float(cfg.density.rho_min)
        rho.x.array[:] = rho_min * 1.01  # Just above minimum
        rho.x.scatter_forward()
        
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        mech.setup()
        mech.solve()
        
        assert np.all(np.isfinite(u.x.array)), "Displacement NaN/Inf at low density"
        
        # Test with maximum density
        rho_max = float(cfg.density.rho_max)
        rho.x.array[:] = rho_max * 0.99
        rho.x.scatter_forward()
        
        mech.assemble_lhs()
        mech.solve()
        
        assert np.all(np.isfinite(u.x.array)), "Displacement NaN/Inf at high density"
    
    def test_density_stays_positive(self, cfg, spaces, fields):
        """Density should never become zero or negative after solve."""
        V, Q, T = spaces.V, spaces.Q, spaces.T
        _, rho, rho_old, A, _, _, _ = fields
        
        # Start with low density
        rho_old.x.array[:] = float(cfg.density.rho_min) * 1.1
        rho_old.x.scatter_forward()
        
        # Strong negative stimulus (drives toward rho_min)
        psi_field = fem.Function(Q, name="psi")
        psi_field.x.array[:] = 0.0  # Very low SED -> resorption
        psi_field.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, psi_field, cfg)
        dens.setup()
        dens.assemble_rhs()
        dens.solve()
        
        n_owned = Q.dofmap.index_map.size_local
        rho_min_val = MPI.COMM_WORLD.allreduce(rho.x.array[:n_owned].min(), op=MPI.MIN)
        
        assert rho_min_val > 0, f"Density became non-positive: {rho_min_val}"
        assert np.all(np.isfinite(rho.x.array)), "Density contains NaN/Inf"
    
    def test_density_no_nan_inf_extreme_stimulus(self, cfg, spaces, fields):
        """Density solver should handle extreme stimulus without NaN/Inf."""
        V, Q, T = spaces.V, spaces.Q, spaces.T
        _, rho, rho_old, A, _, _, _ = fields
        
        rho_old.x.array[:] = 0.8
        rho_old.x.scatter_forward()
        
        # Very high stimulus
        psi_field = fem.Function(Q, name="psi")
        psi_field.x.array[:] = cfg.stimulus.psi_ref_trab * 100.0  # Extreme
        psi_field.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, psi_field, cfg)
        dens.setup()
        dens.assemble_rhs()
        dens.solve()
        
        assert np.all(np.isfinite(rho.x.array)), "Density NaN/Inf with extreme stimulus"
        
        # Check stays within bounds (soft bounds, allow some overshoot)
        n_owned = Q.dofmap.index_map.size_local
        rho_max_val = MPI.COMM_WORLD.allreduce(rho.x.array[:n_owned].max(), op=MPI.MAX)
        assert rho_max_val < 5.0, f"Density unreasonably large: {rho_max_val}"
    
    def test_stiffness_positive_for_all_valid_densities(self, cfg, spaces, fields, bc_mech):
        """Young's modulus E(ρ) should be positive for all ρ > ρ_min."""
        import ufl
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, _, A, _, _, _ = fields
        
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        
        # Test at various densities
        test_densities = [
            float(cfg.density.rho_min) * 1.01,
            0.3,
            0.6,
            1.0,
            float(cfg.density.rho_max) * 0.99,
        ]
        
        for rho_val in test_densities:
            rho.x.array[:] = rho_val
            rho.x.scatter_forward()
            
            # Compute E(rho) using solver's internal formula
            E_expr = mech._E_iso(rho)
            E_local = fem.assemble_scalar(fem.form(E_expr * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            E_avg = MPI.COMM_WORLD.allreduce(E_local, op=MPI.SUM) / MPI.COMM_WORLD.allreduce(vol_local, op=MPI.SUM)
            
            assert E_avg > 0, f"E(ρ={rho_val}) = {E_avg} is not positive"
            assert np.isfinite(E_avg), f"E(ρ={rho_val}) = {E_avg} is NaN/Inf"
    
    def test_strain_energy_density_non_negative(self, cfg, spaces, fields, bc_mech):
        """Strain energy density ψ = 0.5*σ:ε must be non-negative everywhere."""
        import ufl
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, _, A, _, _, _ = fields
        
        # Random displacement
        u.interpolate(lambda x: 0.01 * np.sin(np.pi * x[0]) * np.array([[1], [0.5], [0.2]]) * np.ones((1, x.shape[1])))
        u.x.scatter_forward()
        
        rho.x.array[:] = 0.7
        rho.x.scatter_forward()
        
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        
        # Compute SED pointwise minimum
        psi = 0.5 * ufl.inner(mech.sigma(u, rho), mech.eps(u))
        
        # Project to DG0 to get element-wise values
        DG0 = fem.functionspace(cfg.domain, ("DG", 0))
        psi_dg = fem.Function(DG0)
        psi_expr = fem.Expression(psi, DG0.element.interpolation_points)
        psi_dg.interpolate(psi_expr)
        
        psi_min = MPI.COMM_WORLD.allreduce(psi_dg.x.array.min(), op=MPI.MIN)
        
        assert psi_min >= -1e-12, f"Strain energy density negative: {psi_min}"


class TestNumericalStability:
    """Tests for numerical stability under edge cases."""
    
    def test_mechanics_convergence_with_varied_mesh_size(self, cfg, bc_mech):
        """Mechanics should converge for different mesh resolutions."""
        from dolfinx import mesh as dmesh
        comm = MPI.COMM_WORLD
        
        for n in [4, 8]:  # Skip very fine mesh in CI
            domain = dmesh.create_unit_cube(comm, n, n, n)
            
            from simulation.utils import build_facetag
            facet_tags = build_facetag(domain)
            cfg_local = Config(domain=domain, facet_tags=facet_tags)
            
            P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
            P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
            V = fem.functionspace(domain, P1_vec)
            Q = fem.functionspace(domain, P1)
            
            u = fem.Function(V, name="u")
            rho = fem.Function(Q, name="rho")
            rho.x.array[:] = 0.5
            rho.x.scatter_forward()
            
            bc = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
            t_load = fem.Constant(domain, np.array([1.0, 0.0, 0.0], dtype=float))
            
            mech = MechanicsSolver(u, rho, cfg_local, bc, [(t_load, 2)])
            mech.setup()
            stats = mech.solve()
            
            assert stats.converged, f"Mechanics failed to converge for n={n}"
            assert np.all(np.isfinite(u.x.array)), f"NaN/Inf for n={n}"
    
    def test_density_solver_mass_change_bounded(self, cfg, spaces, fields):
        """Mass change per step should be bounded (no blow-up)."""
        V, Q, T = spaces.V, spaces.Q, spaces.T
        _, rho, rho_old, A, _, _, _ = fields
        
        rho_old.x.array[:] = 0.5
        rho_old.x.scatter_forward()
        
        psi_field = fem.Function(Q, name="psi")
        psi_field.x.array[:] = cfg.stimulus.psi_ref_trab * 2.0
        psi_field.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, psi_field, cfg)
        dens.setup()
        dens.assemble_rhs()
        dens.solve()
        
        # Compute mass change
        mass_old_local = fem.assemble_scalar(fem.form(rho_old * cfg.dx))
        mass_new_local = fem.assemble_scalar(fem.form(rho * cfg.dx))
        mass_old = MPI.COMM_WORLD.allreduce(mass_old_local, op=MPI.SUM)
        mass_new = MPI.COMM_WORLD.allreduce(mass_new_local, op=MPI.SUM)
        
        rel_change = abs(mass_new - mass_old) / max(abs(mass_old), 1e-10)
        
        # Mass change per time step should be reasonable (< 100%)
        assert rel_change < 1.0, f"Mass change {rel_change:.1%} too large"


class TestFabricMath:
    """Tests for Fabric math operations (Exp/Ln behavior)."""

    def test_fabric_exp_ln_mapping(self, fabric_setup):
        """Test that L_target calculation from Qbar produces valid L."""
        solver, L, L_old, Qbar = fabric_setup
        
        # Set Qbar to a known state
        # Qbar = diag(2, 1, 0.5)
        def Q_init(x):
            vals = np.zeros((9, x.shape[1]))
            vals[0, :] = 2.0
            vals[4, :] = 1.0
            vals[8, :] = 0.5
            return vals
        Qbar.interpolate(Q_init)
        Qbar.x.scatter_forward()
        
        from simulation.utils import eigenvalues_sym3, projectors_sylvester, clamp, symm
        import ufl
        
        # Recreate the logic from _L_target_from_Qbar
        epsQ = float(solver.cfg.fabric.fabric_epsQ)
        gammaF = float(solver.cfg.fabric.fabric_gammaF)
        m_min = float(solver.cfg.fabric.fabric_m_min)
        m_max = float(solver.cfg.fabric.fabric_m_max)

        I = ufl.Identity(3)
        Q = symm(Qbar) + epsQ * I

        lam1, lam2, lam3 = eigenvalues_sym3(Q)
        P1, P2, P3 = projectors_sylvester(Q, lam1, lam2, lam3)

        prod = ufl.max_value(lam1 * lam2 * lam3, 1e-30)
        geo = ufl.exp((1.0 / 3.0) * ufl.ln(prod))

        a1 = lam1 / geo
        a2 = lam2 / geo
        a3 = lam3 / geo

        m1 = clamp(a1**gammaF, m_min, m_max)
        m2 = clamp(a2**gammaF, m_min, m_max)
        m3 = clamp(a3**gammaF, m_min, m_max)

        s = (m1 + m2 + m3) / 3.0

        m1n = m1 / s
        m2n = m2 / s
        m3n = m3 / s

        L_target_expr = ufl.ln(m1n) * P1 + ufl.ln(m2n) * P2 + ufl.ln(m3n) * P3
        
        # Evaluate L_target
        # In DOLFINx 0.8+, interpolation_points is a property returning an array
        try:
            points = L.function_space.element.interpolation_points
        except TypeError:
            points = L.function_space.element.interpolation_points()
            
        expr = fem.Expression(L_target_expr, points)
        
        # Project to function to check properties
        L_res = fem.Function(L.function_space)
        L_res.interpolate(expr)
        
        # Check trace(exp(L)) should be 3
        # tr(exp(L)) = exp(lam1) + exp(lam2) + exp(lam3)
        # Since we can't easily compute exp(L) of a tensor field in UFL without spectral decomp (which we have),
        # let's just check the property on the eigenvalues we computed: m1n + m2n + m3n = 3.
        
        # But we want to check the L_res field we computed.
        # Let's compute tr(exp(L_res))
        # We can use the same spectral decomposition on L_res
        
        lam1_L, lam2_L, lam3_L = eigenvalues_sym3(L_res)
        tr_exp_L = ufl.exp(lam1_L) + ufl.exp(lam2_L) + ufl.exp(lam3_L)
        
        tr_exp_int = fem.assemble_scalar(fem.form(tr_exp_L * ufl.dx))
        vol = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=solver.mesh)))
        avg_tr_exp = MPI.COMM_WORLD.allreduce(tr_exp_int, op=MPI.SUM) / MPI.COMM_WORLD.allreduce(vol, op=MPI.SUM)
        
        assert np.abs(avg_tr_exp - 3.0) < 1e-5, f"Trace of exp(L) should be 3, got {avg_tr_exp}"
        
        # Check symmetry error ||L - L.T||
        L_sym_err = fem.assemble_scalar(fem.form(ufl.inner(L_res - L_res.T, L_res - L_res.T) * ufl.dx))
        assert L_sym_err < 1e-10, "L is not symmetric"
