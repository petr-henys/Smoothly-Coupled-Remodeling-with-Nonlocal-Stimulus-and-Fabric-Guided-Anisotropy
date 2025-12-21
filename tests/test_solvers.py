#!/usr/bin/env python3
"""
Tests for solver internals, matrix properties, and numerical utilities.

Tests:
- Matrix assembly correctness (SPD properties)
- Solver statistics tracking
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, DensitySolver
from simulation.fixedsolver import FixedPointSolver

class MockDriver:
    """Mock driver for testing solvers without full gait integration."""
    def __init__(self, mech):
        self.mech = mech
        self.psi_expr = fem.Constant(mech.u.function_space.mesh, 0.0)

    def stimulus_expr(self):
        return self.psi_expr

    def invalidate(self):
        pass

    def update_snapshots(self):
        return {}

    def setup(self):
        self.mech.setup()

    def destroy(self):
        pass

    def update_stiffness(self):
        self.mech.assemble_lhs()


np.random.seed(1234)

# =============================================================================
# Solver Statistics Tests
# =============================================================================

class TestSolverStatistics:
    """Test solver statistics tracking."""
    
    def test_ksp_reason_tracking(self, cfg, spaces, fields, bc_mech):
        """Verify KSP convergence reason is exposed after a solve."""
        from dolfinx import fem
        import numpy as np
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

        reason = mech.solve()
        assert isinstance(reason, int)
        assert mech.last_reason == reason, "last_reason not set"


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
            z.x.array[:n_owned] = np.random.randn(n_owned)
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
            psi_field = Function(Q, name="psi"); psi_field.x.array[:] = cfg.stimulus.psi_ref; psi_field.x.scatter_forward()
            solver = DensitySolver(rho, rho_old, psi_field, cfg)
            solver.setup()  # setup() already calls assemble_lhs()
            
            # Random vector
            z = Function(Q, name="z")
            n_owned = Q.dofmap.index_map.size_local
            z.x.array[:n_owned] = np.random.randn(n_owned)
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
        
        from simulation.subsolvers import FabricSolver
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

from simulation.subsolvers import StimulusSolver, FabricSolver

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
        # m = psi/rho. m_ref = psi_ref/rho_ref.
        # We want m > m_ref * (1 + delta0)
        rho_val = 1.0
        rho.x.array[:] = rho_val
        rho.x.scatter_forward()
        
        m_ref = cfg.stimulus.psi_ref / cfg.density.rho_ref
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

    def test_eigenvectors_update(self, fabric_setup):
        """Test that eigenvectors are updated after step."""
        solver, L, L_old, Qbar = fabric_setup
        
        # Set L to a known anisotropic state (diagonal)
        # L = diag(1, 0, -1) roughly
        def L_init(x):
            vals = np.zeros((9, x.shape[1]))
            vals[0, :] = 1.0  # Lxx
            vals[4, :] = 0.0  # Lyy
            vals[8, :] = -1.0 # Lzz
            return vals
        
        L.interpolate(L_init)
        L.x.scatter_forward()
        
        solver.post_step_update()
        
        # Check n1 (principal direction)
        # Should be (1, 0, 0) for Lxx=1 (largest eigenvalue)
        n1_arr = solver.n1.x.array.reshape(-1, 3)
        if n1_arr.size > 0:
            # Check first node
            assert np.abs(np.abs(n1_arr[0, 0]) - 1.0) < 1e-5
            assert np.abs(n1_arr[0, 1]) < 1e-5
            assert np.abs(n1_arr[0, 2]) < 1e-5


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
