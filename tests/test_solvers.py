#!/usr/bin/env python3
"""
Tests for solver internals, matrix properties, and numerical utilities.

Tests:
- DOF ordering correctness in fixed-point solver
- Matrix assembly correctness (SPD properties)
- Solver statistics tracking
- Projected residual norm computation
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
# DOF Ordering Tests
# =============================================================================

class TestDOFOrdering:
    """Test critical DOF ordering in fixed-point solver (rho)."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_state_slices_match_field_sizes(self, unit_cube, cfg, spaces, fields, bc_mech):
        """Verify _build_state_slices produces correct offsets."""
        comm = MPI.COMM_WORLD
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, rho_old, A, A_old, S, S_old = fields
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        dens = DensitySolver(rho, rho_old, cfg)
        driver = MockDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, dens, rho, rho_old)
        
        # Check slice sizes
        s_rho, = fps.state_slices
        
        assert (s_rho.stop - s_rho.start) == fps.n_rho, "rho slice size mismatch"
        
        # Check contiguity: slices should be consecutive starting at zero
        assert s_rho.start == 0, "rho slice doesn't start at 0"
        assert s_rho.stop == fps.state_size, "rho slice doesn't end at state_size"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_flatten_restore_roundtrip(self, unit_cube, cfg, spaces, fields, bc_mech):
        """Flatten then restore should recover original state."""
        comm = MPI.COMM_WORLD
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, rho_old, A, A_old, S, S_old = fields
        # Set distinct values
        u.interpolate(lambda x: np.array([x[0], x[1], x[2]]))
        u.x.scatter_forward()
        rho.x.array[:] = 0.6; rho.x.scatter_forward()
        
        # Store originals
        u_orig = u.x.array.copy(); rho_orig = rho.x.array.copy()
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        dens = DensitySolver(rho, rho_old, cfg)
        driver = MockDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, dens, rho, rho_old)
        flat = fps._flatten_state(copy=True)
        # Modify fields and restore
        u.x.array[:] = 999.0; rho.x.array[:] = 888.0
        fps._restore_state(flat)
        assert np.allclose(rho.x.array, rho_orig), "rho not restored correctly"
        assert np.allclose(u.x.array, 999.0), "mechanics state should remain untouched"


# =============================================================================
# Solver Statistics Tests
# =============================================================================

class TestSolverStatistics:
    """Test solver statistics tracking."""
    
    def test_ksp_iteration_counting(self, cfg, spaces, fields, bc_mech):
        """Verify KSP iteration counts are tracked correctly."""
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
        
        # Reset stats before solve
        mech._reset_stats()
        assert mech.total_iters == 0, "Stats not reset"
        assert mech.ksp_steps == 0, "Stats not reset"
        
        # Solve
        its, reason = mech.solve()
        
        # Check stats updated
        assert mech.total_iters == its, f"total_iters ({mech.total_iters}) ≠ returned iters ({its})"
        assert mech.ksp_steps == 1, f"ksp_steps should be 1 after one solve"
        assert mech.last_iters == its, f"last_iters not set"
        assert mech.last_reason == reason, f"last_reason not set"


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
            solver = DensitySolver(rho, rho_old, cfg)
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


class TestProjectedResidual:
    """Tests for projected residual norm used in FixedPointSolver."""

    def test_proj_residual_matches_weighted_norm(self, cfg, spaces, fields, bc_mech):
        comm = MPI.COMM_WORLD
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, rho_old, A, A_old, S, S_old = fields
        rho.x.array[:] = 0.5; rho.x.scatter_forward()
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [])
        dens = DensitySolver(rho, rho_old, cfg)
        driver = MockDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, dens, rho, rho_old)

        x_old = fps._flatten_state(copy=True)
        x_raw = x_old.copy()
        rng = np.random.default_rng(42)
        x_test = x_raw.copy()
        for s in fps.state_slices:
            x_test[s] += rng.standard_normal(size=x_test[s].shape)

        weights = (0.5,)
        res = fps._weighted_dist(x_test, x_raw, weights)

        expected_sq = 0.0
        for s, w in zip(fps.state_slices, weights):
            diff = x_test[s] - x_raw[s]
            expected_sq += w * float(np.dot(diff, diff))
        
        # Reduce expected_sq across all ranks to match _proj_residual_norm's global reduction
        expected_sq = comm.allreduce(expected_sq, op=MPI.SUM)
        expected = expected_sq ** 0.5

        assert np.isclose(res, expected)


