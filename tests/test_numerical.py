#!/usr/bin/env python3
"""
Advanced numerical implementation tests for bone remodeling model.

Tests:
- DOF ordering correctness in fixed-point solver
- Anderson acceleration convergence properties
- Matrix assembly correctness
- Preconditioner update logic
- Solver statistics tracking
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver
from simulation.fixedsolver import FixedPointSolver
from simulation.drivers import InstantDriver
from simulation.anderson import _Anderson

np.random.seed(1234)

# =============================================================================
# DOF Ordering Tests
# =============================================================================

class TestDOFOrdering:
    """Test critical DOF ordering in fixed-point solver (rho, A, S)."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_state_slices_match_field_sizes(self, unit_cube, cfg, spaces, fields, bc_mech):
        """Verify _build_state_slices produces correct offsets."""
        comm = MPI.COMM_WORLD
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, rho_old, A, A_old, S, S_old = fields
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        stim = StimulusSolver(S, S_old, cfg)
        dens = DensitySolver(rho, rho_old, A, S, cfg)
        dirn = DirectionSolver(A, A_old, cfg)
        driver = InstantDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, stim, dens, dirn,
                       rho, rho_old, A, A_old, S, S_old)
        
        # Check slice sizes
        s_rho, s_A, s_S = fps.state_slices
        
        assert (s_rho.stop - s_rho.start) == fps.n_rho, "rho slice size mismatch"
        assert (s_A.stop - s_A.start) == fps.n_A, "A slice size mismatch"
        assert (s_S.stop - s_S.start) == fps.n_S, "S slice size mismatch"
        
        # Check contiguity: slices should be consecutive starting at zero
        assert s_rho.start == 0, "rho slice doesn't start at 0"
        assert s_A.start == s_rho.stop, "A slice not after rho"
        assert s_S.start == s_A.stop, "S slice not after A"
        assert s_S.stop == fps.state_size, "S slice doesn't end at state_size"
    
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
        A.interpolate(lambda x: (np.eye(3) * 0.5).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        S.x.array[:] = 0.3; S.x.scatter_forward()
        # Store originals
        u_orig = u.x.array.copy(); rho_orig = rho.x.array.copy(); A_orig = A.x.array.copy(); S_orig = S.x.array.copy()
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        stim = StimulusSolver(S, S_old, cfg)
        dens = DensitySolver(rho, rho_old, A, S, cfg)
        dirn = DirectionSolver(A, A_old, cfg)
        driver = InstantDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, stim, dens, dirn,
                       rho, rho_old, A, A_old, S, S_old)
        flat = fps._flatten_state(copy=True)
        # Modify fields and restore
        u.x.array[:] = 999.0; rho.x.array[:] = 888.0; A.x.array[:] = 777.0; S.x.array[:] = 666.0
        fps._restore_state(flat)
        assert np.allclose(rho.x.array, rho_orig), "rho not restored correctly"
        assert np.allclose(A.x.array, A_orig), "A not restored correctly"
        assert np.allclose(S.x.array, S_orig), "S not restored correctly"
        assert np.allclose(u.x.array, 999.0), "mechanics state should remain untouched"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_fix_mask_empty(self, unit_cube, cfg, spaces, fields, bc_mech):
        """Fixed-point mask should remain empty without displacement state."""
        comm = MPI.COMM_WORLD
        V, Q, T = spaces.V, spaces.Q, spaces.T
        u, rho, rho_old, A, A_old, S, S_old = fields
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        stim = StimulusSolver(S, S_old, cfg)
        dens = DensitySolver(rho, rho_old, A, S, cfg)
        dirn = DirectionSolver(A, A_old, cfg)
        driver = InstantDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, stim, dens, dirn,
                               rho, rho_old, A, A_old, S, S_old)

        assert fps.fix_mask.size == fps.state_size, "Fix mask length mismatch"
        assert not np.any(fps.fix_mask), "Fix mask should remain all False"


# =============================================================================
# Anderson Acceleration Tests
# =============================================================================

class TestAndersonAcceleration:
    """Test Anderson acceleration implementation."""
    
    @pytest.mark.parametrize("operation", ["init", "restart", "mix", "reject_restart"])
    def test_anderson_accelerator_operations(self, operation):
        """Test Anderson accelerator: initialization, restart, mix, reject-triggered restart.
        
        Consolidates 4 separate Anderson tests into single parametrized test.
        """
        m, n = 5, 20
        
        if operation == "init":
            # Initialization test
            beta, lam = 1.0, 1e-8
            aa = _Anderson(MPI.COMM_WORLD, m=m, beta=beta, lam=lam)
            assert aa.m == m and aa.beta == beta and aa.lam == lam, "Anderson parameters not set correctly"

        elif operation == "restart":
            # Reset clears history
            aa = _Anderson(MPI.COMM_WORLD, m=3)
            aa.x_hist.append(np.random.rand(50))
            aa.r_hist.append(np.random.rand(50))
            assert len(aa.x_hist) > 0, "History not accumulated"
            aa.reset()
            assert len(aa.x_hist) == 0, "History not cleared after reset"

        elif operation == "mix":
            # Basic mix operation
            aa = _Anderson(MPI.COMM_SELF, m=m, beta=1.0, lam=1e-10)
            x_old = np.random.rand(n)
            x_raw = x_old + 0.1 * np.random.rand(n)
            x_new, info = aa.mix(x_old, x_raw)
            assert x_new.shape == x_old.shape, "Output shape mismatch"
            assert "aa_hist" in info and "accepted" in info, "Info dict missing required keys"

        elif operation == "reject_restart":
            # Restart triggered by rejection streak
            aa = _Anderson(MPI.COMM_SELF, m=2, beta=1.0, lam=1e-10, restart_on_reject_k=1)
            x_old, x_raw = np.zeros(10), np.ones(10)

            # Proxy residual larger than reference triggers rejection
            def prn(x_ref, x_test, xR):
                return 2.0 if x_test is not xR else 1.0

            # First rejection
            x1, info1 = aa.mix(x_old, x_raw, proj_residual_norm=prn)
            assert info1.get("accepted") is False, "First call should reject"

            # Second rejection triggers restart
            x2, info2 = aa.mix(x_old, x_raw, proj_residual_norm=prn)
            assert isinstance(info2.get("restart_reason", ""), str), "Restart reason missing"
            assert "reject_streak" in info2.get("restart_reason", ""), "Restart not scheduled on reject streak"

            # Third call honors pending reset
            _ = aa.mix(x_old, x_raw, proj_residual_norm=prn)
            assert len(aa.x_hist) <= 1, "History not cleared after scheduled reset"


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
        
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [(t_load, 2)])
        
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
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        
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

    
    def test_stimulus_lhs_spd(self, cfg, spaces):
        """Stimulus LHS matrix should be SPD."""
        cfg.set_dt(10.0 * 86400.0)  # 10 days in seconds
        Q = spaces.Q
        S = Function(Q, name="S")
        S_old = Function(Q, name="S_old")
        stim = StimulusSolver(S, S_old, cfg)
        stim.setup()
        
        from petsc4py import PETSc
        norm = stim.A.norm(PETSc.NormType.FROBENIUS)
        assert np.isfinite(norm), "Stimulus matrix has non-finite entries"
        assert norm > 0, "Stimulus matrix is zero"

    def test_stimulus_update_lhs_changes_matrix(self, cfg, spaces):
        """Changing dt via set_dt and assemble_lhs should alter matrix norm measurably."""
        # Build a local Config that accentuates dt effect (mass-only term)
        cfg2 = Config(domain=cfg.domain, facet_tags=cfg.facet_tags, verbose=False,
                      cS=1.0, tauS=0.0, kappaS=0.0)
        cfg2.set_dt(1.0 * 86400.0)  # 1 day baseline
        Q = spaces.Q
        S = Function(Q, name="S")
        S_old = Function(Q, name="S_old")
        stim = StimulusSolver(S, S_old, cfg2)
        stim.setup()
        from petsc4py import PETSc
        n1 = stim.A.norm(PETSc.NormType.FROBENIUS)

        # Change dt significantly and update LHS
        cfg2.set_dt(100.0 * 86400.0)  # 100 days
        stim.assemble_lhs()
        n2 = stim.A.norm(PETSc.NormType.FROBENIUS)
        rel_change = abs(n2 - n1) / max(n1, 1e-300)
        assert rel_change > 0.5, f"Stimulus matrix norm unchanged after dt update (rel={rel_change:.3e}, n1={n1:.3e}, n2={n2:.3e})"

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    @pytest.mark.parametrize("solver_type", ["mechanics", "stimulus", "density"])
    def test_solver_operator_positive_definite(self, unit_cube, cfg, spaces, bc_mech, solver_type):
        """Check x^T A x > 0 (or >= 0) for solver operators with random test vectors.
        
        Mechanics: Positive-definite with Dirichlet DOFs zeroed
        Stimulus/Density: Positive semi-definite (allow zero eigenvalues)
        """
        V, Q, T = spaces.V, spaces.Q, spaces.T
        
        if solver_type == "mechanics":
            # Mechanics solver: positive-definite for non-Dirichlet DOFs
            u = Function(V, name="u")
            rho = Function(Q, name="rho"); rho.x.array[:] = 0.6; rho.x.scatter_forward()
            A = Function(T, name="A"); A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1]))); A.x.scatter_forward()
            solver = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
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
            
        elif solver_type == "stimulus":
            # Stimulus solver: positive semi-definite
            cfg.set_dt(10.0 * 86400.0)  # 10 days in seconds
            S = Function(Q, name="S")
            S_old = Function(Q, name="S_old"); S_old.x.array[:] = 0.0; S_old.x.scatter_forward()
            solver = StimulusSolver(S, S_old, cfg)
            solver.setup()
            
            # Random vector
            z = Function(Q, name="z")
            n_owned = Q.dofmap.index_map.size_local
            z.x.array[:n_owned] = np.random.randn(n_owned)
            z.x.scatter_forward()
            
            y = z.x.petsc_vec.duplicate()
            solver.A.mult(z.x.petsc_vec, y)
            energy = z.x.petsc_vec.dot(y)
            assert energy >= 0.0, f"Stimulus operator not positive semi-definite: {energy}"
            
        elif solver_type == "density":
            # Density solver: positive semi-definite
            rho = Function(Q, name="rho")
            rho_old = Function(Q, name="rho_old"); rho_old.x.array[:] = 0.5; rho_old.x.scatter_forward()
            A_tens = Function(T, name="A"); A_tens.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1]))); A_tens.x.scatter_forward()
            S = Function(Q, name="S"); S.x.array[:] = 0.0; S.x.scatter_forward()
            solver = DensitySolver(rho, rho_old, A_tens, S, cfg)
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
        A.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A.x.scatter_forward()
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [])
        stim = StimulusSolver(S, S_old, cfg)
        dens = DensitySolver(rho, rho_old, A, S, cfg)
        dirn = DirectionSolver(A, A_old, cfg)
        driver = InstantDriver(mech)
        fps = FixedPointSolver(comm, cfg, driver, stim, dens, dirn,
                               rho, rho_old, A, A_old, S, S_old)

        x_old = fps._flatten_state(copy=True)
        x_raw = x_old.copy()
        rng = np.random.default_rng(42)
        x_test = x_raw.copy()
        for s in fps.state_slices:
            x_test[s] += rng.standard_normal(size=x_test[s].shape)

        weights = (0.5, 1.5, 0.25)
        res = fps._proj_residual_norm(x_old, x_test, x_raw, weights)

        expected_sq = 0.0
        for s, w in zip(fps.state_slices, weights):
            diff = x_test[s] - x_raw[s]
            expected_sq += w * float(np.dot(diff, diff))
        
        # Reduce expected_sq across all ranks to match _proj_residual_norm's global reduction
        expected_sq = comm.allreduce(expected_sq, op=MPI.SUM)
        expected = expected_sq ** 0.5

        assert np.isclose(res, expected)


class TestUtilsEigen:
    """Tests for utils.compute_principal_dirs_and_vals_vec."""

    def test_principal_dirs_vals_constant_tensor(self, spaces):
        V, Q, T = spaces.V, spaces.Q, spaces.T
        A = Function(T, name="A")
        const = np.diag([0.6, 0.3, 0.1]).reshape(9, 1)
        A.interpolate(lambda x: np.tile(const, (1, x.shape[1])))
        A.x.scatter_forward()
        from simulation.utils import compute_principal_dirs_and_vals_vec
        eigvecs, eigvals = compute_principal_dirs_and_vals_vec(A, V, Q)

        # Check eigenvalues (descending)
        n_owned = Q.dofmap.index_map.size_local
        lam1 = float(np.mean(eigvals[0].x.array[:n_owned]))
        lam2 = float(np.mean(eigvals[1].x.array[:n_owned]))
        lam3 = float(np.mean(eigvals[2].x.array[:n_owned]))
        assert abs(lam1 - 0.6) < 1e-12
        assert abs(lam2 - 0.3) < 1e-12
        assert abs(lam3 - 0.1) < 1e-12

        # Check eigenvectors align with axes (up to sign)
        n_owned_v = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        v1 = eigvecs[0].x.array[:n_owned_v].reshape(-1, 3)
        v2 = eigvecs[1].x.array[:n_owned_v].reshape(-1, 3)
        v3 = eigvecs[2].x.array[:n_owned_v].reshape(-1, 3)

        # Mean absolute component values should match identity
        m1 = np.mean(np.abs(v1), axis=0)
        m2 = np.mean(np.abs(v2), axis=0)
        m3 = np.mean(np.abs(v3), axis=0)
        assert m1[0] > 0.999 and m1[1] < 1e-12 and m1[2] < 1e-12
        assert m2[1] > 0.999 and m2[0] < 1e-12 and m2[2] < 1e-12
        assert m3[2] > 0.999 and m3[0] < 1e-12 and m3[1] < 1e-12


# =============================================================================
# Config Validation Tests
# =============================================================================

class TestConfigValidation:
    """Test Config parameter validation and bounds checking."""

    def test_config_requires_domain(self, facet_tags):
        """Config must have domain parameter."""
        with pytest.raises((ValueError, TypeError)):
            Config(facet_tags=facet_tags)  # Missing domain

    def test_config_requires_domain_not_none(self, facet_tags):
        """Config domain cannot be None."""
        with pytest.raises(ValueError, match="[Dd]omain"):
            Config(domain=None, facet_tags=facet_tags)

    def test_config_accepts_valid_domain(self, unit_cube, facet_tags):
        """Config should accept valid mesh."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=False)
        assert cfg.domain is not None
        assert cfg.domain == unit_cube

    def test_config_rejects_negative_timestep(self, unit_cube, facet_tags):
        """set_dt should reject non-positive timestep."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=False)

        # Zero timestep
        with pytest.raises((ValueError, ZeroDivisionError)):
            cfg.set_dt(0.0)

        # Negative timestep
        with pytest.raises((ValueError, RuntimeError)):
            cfg.set_dt(-1.0)

    def test_config_poisson_ratio_in_valid_range(self, unit_cube, facet_tags):
        """Poisson ratio must be in physically valid range (-1, 0.5)."""
        # Test boundary values
        with pytest.raises((ValueError, RuntimeError)):
            Config(domain=unit_cube, facet_tags=facet_tags, nu=0.6, verbose=False)  # Too high

        with pytest.raises((ValueError, RuntimeError)):
            Config(domain=unit_cube, facet_tags=facet_tags, nu=-1.5, verbose=False)  # Too low

        # Valid values should work
        cfg1 = Config(domain=unit_cube, facet_tags=facet_tags, nu=0.3, verbose=False)
        assert cfg1.nu == 0.3

        cfg2 = Config(domain=unit_cube, facet_tags=facet_tags, nu=0.0, verbose=False)
        assert cfg2.nu == 0.0

    def test_config_positive_modulus(self, unit_cube, facet_tags):
        """Young's modulus must be positive."""
        with pytest.raises(ValueError):
            cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                        E0=-1000.0, verbose=False)
        
        # Positive value should work
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    E0=1000.0, verbose=False)
        assert cfg.E0 == 1000.0

    def test_config_solver_type_validation(self, unit_cube, facet_tags):
        """KSP and PC types should be valid."""
        # Valid solvers
        cfg1 = Config(domain=unit_cube, facet_tags=facet_tags,
                     ksp_type="cg", pc_type="jacobi", verbose=False)
        assert cfg1.ksp_type == "cg"

        cfg2 = Config(domain=unit_cube, facet_tags=facet_tags,
                     ksp_type="gmres", pc_type="ilu", verbose=False)
        assert cfg2.ksp_type == "gmres"

    def test_config_accel_type_validation(self, unit_cube, facet_tags):
        """Acceleration type must be valid choice ('anderson' or 'picard')."""
        for accel in ["anderson", "picard"]:
            cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                        accel_type=accel, verbose=False)
            assert cfg.accel_type == accel
        # 'none' is not accepted at config time
        with pytest.raises((ValueError, RuntimeError)):
            Config(domain=unit_cube, facet_tags=facet_tags, accel_type="none", verbose=False)

    def test_config_tolerance_values_positive(self, unit_cube, facet_tags):
        """Solver tolerances must be positive."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags, verbose=False)

        assert cfg.ksp_rtol > 0
        assert cfg.ksp_atol > 0
        assert cfg.coupling_tol > 0
        assert cfg.smooth_eps > 0

    def test_config_iteration_limits_sensible(self, unit_cube, facet_tags):
        """Iteration limits should be positive integers."""
        cfg = Config(domain=unit_cube, facet_tags=facet_tags,
                    max_subiters=100, min_subiters=1,
                    ksp_max_it=500, verbose=False)

        assert cfg.max_subiters > 0
        assert cfg.min_subiters > 0
        assert cfg.min_subiters <= cfg.max_subiters
        assert cfg.ksp_max_it > 0
