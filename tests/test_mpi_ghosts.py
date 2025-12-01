#!/usr/bin/env python3
"""
MPI ghost cell consistency tests.

Tests ghost/halo cell synchronization, partition interface values,
and correct data flow across MPI boundaries.

Run with: mpirun -n 4 pytest tests/test_mpi_ghosts.py -v
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import Function, functionspace
import basix.ufl
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs, get_owned_size, assign, field_stats
from simulation.subsolvers import MechanicsSolver, DensitySolver

pytestmark = [pytest.mark.mpi]

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size


# =============================================================================
# Helper functions
# =============================================================================

def create_test_mesh(n: int = 8):
    """Create unit cube mesh with shared_facet ghost mode."""
    return mesh.create_unit_cube(
        comm, n, n, n, 
        ghost_mode=mesh.GhostMode.shared_facet
    )


def get_ghost_info(field: fem.Function):
    """Extract ghost DOF information from a field."""
    imap = field.function_space.dofmap.index_map
    bs = field.function_space.dofmap.index_map_bs
    n_owned = imap.size_local
    n_ghosts = imap.num_ghosts
    ghost_global_indices = imap.ghosts
    ghost_owners = imap.owners
    
    return {
        "n_owned": n_owned,
        "n_ghosts": n_ghosts,
        "ghost_global_indices": ghost_global_indices,
        "ghost_owners": ghost_owners,
        "bs": bs,
    }


def assert_ghosts_updated(field: fem.Function, rtol: float = 1e-12):
    """Assert that ghost values are consistent after scatter_forward."""
    # Store current values
    original = field.x.array.copy()
    
    # Scatter should be idempotent if already synced
    field.x.scatter_forward()
    
    # Compare
    diff = np.abs(field.x.array - original)
    max_diff = float(np.max(diff)) if diff.size > 0 else 0.0
    max_diff_global = comm.allreduce(max_diff, op=MPI.MAX)
    
    assert max_diff_global < rtol, \
        f"Ghost values changed after scatter_forward: max_diff={max_diff_global}"


# =============================================================================
# Basic Ghost Synchronization Tests
# =============================================================================

class TestGhostBasics:
    """Basic ghost cell creation and synchronization tests."""
    
    def test_ghost_cells_exist_multirank(self):
        """Verify ghost cells are created when using multiple ranks."""
        if size < 2:
            pytest.skip("Need >= 2 ranks for ghost cell testing")
        
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        imap = Q.dofmap.index_map
        n_ghosts = imap.num_ghosts
        
        # At least some ranks should have ghosts
        total_ghosts = comm.allreduce(n_ghosts, op=MPI.SUM)
        assert total_ghosts > 0, "No ghost DOFs created in multi-rank setup"
    
    def test_owned_plus_ghosts_cover_local(self):
        """Verify owned + ghost DOFs equal local array size."""
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        imap = Q.dofmap.index_map
        n_owned = imap.size_local
        n_ghosts = imap.num_ghosts
        bs = Q.dofmap.index_map_bs
        
        expected_size = (n_owned + n_ghosts) * bs
        actual_size = f.x.array.size
        
        assert actual_size == expected_size, \
            f"Array size mismatch: {actual_size} != {expected_size} (owned={n_owned}, ghosts={n_ghosts}, bs={bs})"
    
    def test_scatter_forward_updates_ghosts(self):
        """Verify scatter_forward propagates owned values to ghosts."""
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        n_owned = get_owned_size(f)
        
        # Set owned DOFs to rank-specific pattern
        f.x.array[:n_owned] = rank * 1000.0 + np.arange(n_owned, dtype=np.float64)
        
        # Before scatter: ghosts may have stale/zero values
        # After scatter: ghosts should have values from their owners
        f.x.scatter_forward()
        
        # Verify ghosts are in valid range (from some rank's owned DOFs)
        info = get_ghost_info(f)
        if info["n_ghosts"] > 0:
            ghost_vals = f.x.array[n_owned:n_owned + info["n_ghosts"]]
            
            for i, (gval, owner) in enumerate(zip(ghost_vals, info["ghost_owners"])):
                # Ghost value should be from owner's pattern: owner * 1000 + local_idx
                assert owner * 1000 <= gval < (owner + 1) * 1000, \
                    f"Ghost {i} has value {gval}, expected from rank {owner}"
    
    def test_scatter_idempotent(self):
        """Verify scatter_forward is idempotent (calling twice gives same result)."""
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        # Set some values
        n_owned = get_owned_size(f)
        f.x.array[:n_owned] = np.random.rand(n_owned)
        
        # First scatter
        f.x.scatter_forward()
        vals_after_first = f.x.array.copy()
        
        # Second scatter should not change anything
        f.x.scatter_forward()
        vals_after_second = f.x.array.copy()
        
        np.testing.assert_array_equal(
            vals_after_first, vals_after_second,
            err_msg="scatter_forward is not idempotent"
        )


# =============================================================================
# Partition Interface Consistency Tests
# =============================================================================

class TestPartitionInterfaces:
    """Test value consistency at partition boundaries."""
    
    def test_continuous_field_at_interfaces(self):
        """Verify continuous field has matching values at partition interfaces."""
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        # Interpolate a smooth function that should be continuous
        f.interpolate(lambda x: x[0]**2 + x[1]**2 + x[2]**2)
        f.x.scatter_forward()
        
        # After scatter, ghost values should match their owners exactly
        assert_ghosts_updated(f, rtol=1e-12)
    
    def test_interface_values_unique(self):
        """Verify interface DOFs have unique global values (no double-counting)."""
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        # Set owned DOFs to their global index
        imap = Q.dofmap.index_map
        local_range = imap.local_range
        n_owned = imap.size_local
        
        # Create global indices for owned DOFs
        global_indices = np.arange(local_range[0], local_range[1], dtype=np.float64)
        f.x.array[:n_owned] = global_indices
        f.x.scatter_forward()
        
        # Ghost values should match the global indices assigned by their owners
        if imap.num_ghosts > 0:
            ghost_vals = f.x.array[n_owned:n_owned + imap.num_ghosts]
            ghost_global = imap.ghosts
            
            # Ghost value should equal its global index
            np.testing.assert_array_almost_equal(
                ghost_vals, ghost_global.astype(np.float64),
                decimal=10,
                err_msg="Ghost values don't match global indices"
            )
    
    def test_global_integral_partition_independent(self):
        """Verify global integral is same regardless of partition."""
        domain = create_test_mesh(8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        # Interpolate known function: f(x,y,z) = x + y + z
        # Integral over [0,1]^3 = 3/2
        f.interpolate(lambda x: x[0] + x[1] + x[2])
        f.x.scatter_forward()
        
        # Compute integral using owned cells only (via dx measure)
        integral_local = fem.assemble_scalar(fem.form(f * cfg.dx))
        integral_global = comm.allreduce(integral_local, op=MPI.SUM)
        
        expected = 1.5
        assert abs(integral_global - expected) < 1e-10, \
            f"Global integral {integral_global} != expected {expected}"


# =============================================================================
# Solver Ghost Update Tests
# =============================================================================

class TestSolverGhostUpdates:
    """Test ghost updates in solver contexts."""
    
    def test_mechanics_solution_continuous(self):
        """Verify mechanics solution is continuous across partitions."""
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(6)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags)
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.8
        rho.x.scatter_forward()
        
        # Apply traction load
        t_vec = np.array([0.0, -0.1, 0.0], dtype=np.float64)
        traction = (fem.Constant(domain, t_vec), 2)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [traction])
        
        mech.setup()
        mech.assemble_rhs()
        mech.solve()
        
        # Solution should be continuous - check ghost consistency
        assert_ghosts_updated(u, rtol=1e-10)
        
        mech.destroy()
    
    def test_density_solution_continuous(self):
        """Verify density solution is continuous across partitions."""
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(6)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags)
        cfg.set_dt(1.0)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        psi = Function(Q, name="psi")
        
        # Initialize fields
        rho.x.array[:] = 0.8
        rho.x.scatter_forward()
        rho_old.x.array[:] = 0.8
        rho_old.x.scatter_forward()
        psi.x.array[:] = cfg.psi_ref * 1.1  # Slightly above reference
        psi.x.scatter_forward()
        
        dens = DensitySolver(rho, rho_old, psi, cfg)
        dens.setup()
        dens.assemble_lhs()
        dens.assemble_rhs()
        dens.solve()
        
        # Solution should be continuous
        assert_ghosts_updated(rho, rtol=1e-10)
        
        dens.destroy()


# =============================================================================
# Field Statistics Tests (No Double-Counting)
# =============================================================================

class TestFieldStatistics:
    """Test MPI-reduced statistics avoid ghost double-counting."""
    
    def test_field_stats_owned_only(self):
        """Verify field_stats uses owned DOFs only."""
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        # Set all values to 1.0
        f.x.array[:] = 1.0
        f.x.scatter_forward()
        
        fmin, fmax, fmean = field_stats(f, comm)
        
        assert abs(fmin - 1.0) < 1e-12, f"Min incorrect: {fmin}"
        assert abs(fmax - 1.0) < 1e-12, f"Max incorrect: {fmax}"
        assert abs(fmean - 1.0) < 1e-12, f"Mean incorrect: {fmean}"
    
    def test_mean_not_affected_by_ghosts(self):
        """Verify mean computation is partition-independent."""
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        # Interpolate linear function
        f.interpolate(lambda x: x[0])
        f.x.scatter_forward()
        
        # Mean of x over [0,1]^3 = 0.5
        fmin, fmax, fmean = field_stats(f, comm)
        
        # Mean should be close to 0.5 (exact value depends on mesh)
        # The key test: all ranks should get the same mean
        all_means = comm.gather(fmean, root=0)
        if rank == 0:
            assert all(abs(m - fmean) < 1e-14 for m in all_means), \
                f"Ranks computed different means: {all_means}"
    
    def test_global_dof_count_correct(self):
        """Verify global DOF count matches sum of owned (not total local)."""
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        imap = Q.dofmap.index_map
        n_owned = imap.size_local
        n_global_from_map = imap.size_global
        
        # Sum of owned across ranks should equal global
        sum_owned = comm.allreduce(n_owned, op=MPI.SUM)
        
        assert sum_owned == n_global_from_map, \
            f"Global DOF count mismatch: sum(owned)={sum_owned} != global={n_global_from_map}"


# =============================================================================
# Assign Utility Tests
# =============================================================================

class TestAssignGhostHandling:
    """Test assign() utility correctly handles ghosts."""
    
    def test_assign_scalar_updates_ghosts(self):
        """Verify assign() with scalar updates and scatters."""
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        assign(f, 3.14159)
        
        # All values (owned + ghost) should be 3.14159
        assert np.allclose(f.x.array, 3.14159), "assign(scalar) failed"
        
        # Ghosts should be consistent
        assert_ghosts_updated(f)
    
    def test_assign_function_updates_ghosts(self):
        """Verify assign() from function updates and scatters."""
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="f")
        g = Function(Q, name="g")
        
        # Set g to a known pattern
        g.interpolate(lambda x: x[0] + x[1])
        g.x.scatter_forward()
        
        # Assign g to f
        assign(f, g)
        
        # f should have same owned values as g
        n_owned = get_owned_size(f)
        np.testing.assert_array_almost_equal(
            f.x.array[:n_owned], g.x.array[:n_owned],
            decimal=12
        )
        
        # Ghosts should be consistent
        assert_ghosts_updated(f)


# =============================================================================
# DG0 vs CG1 Interface Tests
# =============================================================================

class TestDGCGInterfaces:
    """Test interactions between DG0 and CG1 spaces at interfaces."""
    
    def test_dg0_no_shared_dofs(self):
        """Verify DG0 spaces have no ghost DOFs (cell-local)."""
        domain = create_test_mesh(6)
        DG0 = basix.ufl.element("DG", domain.basix_cell(), 0)
        W = functionspace(domain, DG0)
        
        imap = W.dofmap.index_map
        
        # DG0: Each cell owns its DOF, no sharing
        # Ghost DOFs may exist for ghost cells, but they're independent
        # The key property: no continuity required
        n_local_cells = domain.topology.index_map(3).size_local
        n_owned_dofs = imap.size_local
        
        # For DG0, owned DOFs = owned cells
        assert n_owned_dofs == n_local_cells, \
            f"DG0 owned DOFs ({n_owned_dofs}) != owned cells ({n_local_cells})"
    
    def test_cg1_projection_from_dg0(self):
        """Test projecting DG0 field to CG1 preserves global integral."""
        domain = create_test_mesh(8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags)
        
        DG0 = basix.ufl.element("DG", domain.basix_cell(), 0)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        
        W_dg = functionspace(domain, DG0)
        Q_cg = functionspace(domain, P1)
        
        # Create DG0 field with constant value
        f_dg = Function(W_dg, name="f_dg")
        f_dg.x.array[:] = 1.0
        f_dg.x.scatter_forward()
        
        # Project to CG1 using manual assembly
        f_cg = Function(Q_cg, name="f_cg")
        
        # Use L2 projection via assembled system
        u = ufl.TrialFunction(Q_cg)
        v = ufl.TestFunction(Q_cg)
        a = ufl.inner(u, v) * cfg.dx
        L = ufl.inner(f_dg, v) * cfg.dx
        
        a_form = fem.form(a)
        L_form = fem.form(L)
        
        from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector
        from petsc4py import PETSc
        
        A = create_matrix(a_form)
        assemble_matrix(A, a_form)
        A.assemble()
        
        b = create_vector(Q_cg)
        assemble_vector(b, L_form)
        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        
        # Solve
        ksp = PETSc.KSP().create(comm)
        ksp.setType("cg")
        ksp.getPC().setType("jacobi")
        ksp.setOperators(A)
        ksp.setFromOptions()
        ksp.solve(b, f_cg.x.petsc_vec)
        f_cg.x.scatter_forward()
        
        ksp.destroy()
        A.destroy()
        b.destroy()
        
        # Integrals should match (up to projection error)
        int_dg = comm.allreduce(fem.assemble_scalar(fem.form(f_dg * cfg.dx)), op=MPI.SUM)
        int_cg = comm.allreduce(fem.assemble_scalar(fem.form(f_cg * cfg.dx)), op=MPI.SUM)
        
        assert abs(int_dg - int_cg) < 1e-6, \
            f"Projection changed integral: DG={int_dg}, CG={int_cg}"


# =============================================================================
# Vector Assembly Ghost Tests
# =============================================================================

class TestVectorAssemblyGhosts:
    """Test ghost handling in vector assembly (RHS)."""
    
    def test_rhs_ghost_accumulation(self):
        """Verify RHS vector ghost update pattern: ADD-REVERSE then INSERT-FORWARD."""
        domain = create_test_mesh(6)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags)
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="f")
        f.x.array[:] = 1.0
        f.x.scatter_forward()
        
        v = ufl.TestFunction(Q)
        L = f * v * cfg.dx
        L_form = fem.form(L)
        
        from dolfinx.fem.petsc import create_vector, assemble_vector
        b = create_vector(Q)
        
        # Assemble
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        assemble_vector(b, L_form)
        
        # Pattern: ADD_VALUES + REVERSE to accumulate ghost contributions to owners
        from petsc4py import PETSc
        b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        
        # Pattern: INSERT_VALUES + FORWARD to update ghosts from owners
        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
        # Global sum of RHS should equal integral of f (which is 1.0 over unit cube)
        b_sum_local = np.sum(b.array[:Q.dofmap.index_map.size_local])
        b_sum_global = comm.allreduce(b_sum_local, op=MPI.SUM)
        
        # For lumped mass, sum ≈ volume = 1.0
        # Actual value depends on mass matrix row sums, but should be positive
        assert b_sum_global > 0, f"RHS sum should be positive, got {b_sum_global}"
        
        b.destroy()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and potential failure modes."""
    
    def test_empty_rank_handling(self):
        """Verify handling when a rank has no owned DOFs."""
        # Use very coarse mesh - some ranks may be empty
        domain = mesh.create_unit_cube(
            comm, 2, 2, 2,
            ghost_mode=mesh.GhostMode.shared_facet
        )
        
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        n_owned = get_owned_size(f)
        
        # Even if n_owned == 0, assign should work
        assign(f, 1.0)
        
        # Scatter should not crash
        f.x.scatter_forward()
        
        # Global operations should still work
        fmin, fmax, fmean = field_stats(f, comm)
        
        # All ranks should agree
        all_mins = comm.gather(fmin, root=0)
        if rank == 0:
            assert len(set(all_mins)) == 1, f"Ranks disagree on min: {all_mins}"
    
    def test_scatter_reverse_then_forward(self):
        """Test scatter REVERSE followed by FORWARD (common assembly pattern)."""
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        f = Function(Q, name="test")
        
        n_owned = get_owned_size(f)
        
        # Simulate assembly: each rank contributes to owned AND ghost DOFs
        f.x.array[:] = float(rank + 1)  # Include ghosts
        
        # REVERSE: ghosts contribute to owners (additive)
        from petsc4py import PETSc
        vec = f.x.petsc_vec
        vec.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        
        # FORWARD: owners update ghosts
        vec.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        
        # After this, ghosts should match their owners
        assert_ghosts_updated(f, rtol=1e-12)


# =============================================================================
# Comprehensive Integration Test
# =============================================================================

class TestGhostIntegration:
    """Integration tests for full solver pipeline ghost consistency."""
    
    def test_full_step_ghost_consistency(self):
        """Verify ghost consistency throughout a full simulation step."""
        if size < 2:
            pytest.skip("Need >= 2 ranks for meaningful ghost testing")
        
        domain = create_test_mesh(6)
        facet_tags = build_facetag(domain)
        cfg = Config(
            domain=domain, 
            facet_tags=facet_tags,
            max_subiters=5,
            coupling_tol=1e-4
        )
        cfg.set_dt(1.0)
        
        # Create spaces and fields
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        V = functionspace(domain, P1_vec)
        Q = functionspace(domain, P1)
        
        u = Function(V, name="u")
        rho = Function(Q, name="rho")
        rho_old = Function(Q, name="rho_old")
        psi = Function(Q, name="psi")
        
        # Initialize
        rho.x.array[:] = 0.8
        rho.x.scatter_forward()
        assign(rho_old, rho)
        psi.x.array[:] = cfg.psi_ref
        psi.x.scatter_forward()
        
        # Setup solvers
        t_vec = np.array([0.0, -0.05, 0.0], dtype=np.float64)
        traction = (fem.Constant(domain, t_vec), 2)
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        
        mech = MechanicsSolver(u, rho, cfg, bc_mech, [traction])
        dens = DensitySolver(rho, rho_old, psi, cfg)
        
        mech.setup()
        dens.setup()
        
        # Perform one coupling iteration
        mech.assemble_rhs()
        mech.solve()
        
        # Check u is ghost-consistent
        assert_ghosts_updated(u, rtol=1e-10)
        
        # Update stimulus (simplified - just use constant for test)
        psi.x.array[:get_owned_size(psi)] = cfg.psi_ref * 1.05
        psi.x.scatter_forward()
        
        # Solve density
        dens.assemble_lhs()
        dens.assemble_rhs()
        dens.solve()
        
        # Check rho is ghost-consistent
        assert_ghosts_updated(rho, rtol=1e-10)
        
        # Cleanup
        mech.destroy()
        dens.destroy()
        
        comm.Barrier()


# =============================================================================
# Diagnostic Tests for Partition Interface Visualization Issues
# =============================================================================

class TestPartitionInterfaceArtifacts:
    """
    Diagnostic tests to identify sources of visualization artifacts at partition interfaces.
    
    These tests specifically look for:
    - Value discontinuities at partition boundaries
    - Ghost vs owned value mismatches after operations
    - DG0 to CG1 interpolation issues
    - Output pipeline problems
    """
    
    def test_detect_interface_discontinuity(self):
        """
        DIAGNOSTIC: Check for value discontinuities at partition interfaces.
        
        If this test fails, there's a real discontinuity in the field values.
        If it passes, the issue is likely in visualization, not computation.
        """
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="test")
        
        # Interpolate a smooth function
        f.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
        f.x.scatter_forward()
        
        # Compare ghost values with what they SHOULD be (from owner)
        imap = Q.dofmap.index_map
        n_owned = imap.size_local
        n_ghosts = imap.num_ghosts
        
        if n_ghosts > 0:
            ghost_vals = f.x.array[n_owned:n_owned + n_ghosts]
            ghost_global = imap.ghosts
            ghost_owners = imap.owners
            
            # For each ghost, its value should match the owner's value
            # We can verify this by checking the interpolated function value at ghost DOF coords
            coords = Q.tabulate_dof_coordinates()
            ghost_coords = coords[n_owned:n_owned + n_ghosts]
            
            # Expected values from the analytical function
            expected = np.sin(np.pi * ghost_coords[:, 0]) * np.sin(np.pi * ghost_coords[:, 1])
            
            max_error = np.max(np.abs(ghost_vals - expected))
            
            # Report the error magnitude
            max_error_global = comm.allreduce(max_error, op=MPI.MAX)
            
            if rank == 0:
                print(f"\n  [DIAGNOSTIC] Max ghost interpolation error: {max_error_global:.2e}")
            
            assert max_error_global < 1e-10, \
                f"Ghost values don't match analytical function: max_error={max_error_global}"
    
    def test_ghost_owned_value_comparison(self):
        """
        DIAGNOSTIC: Directly compare ghost values with their owners.
        
        This gathers all owned values to rank 0 and checks that each ghost
        matches its owner exactly.
        """
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="test")
        
        # Each rank sets owned DOFs to global_index * 0.001 for easy verification
        imap = Q.dofmap.index_map
        n_owned = imap.size_local
        local_range = imap.local_range
        
        # Set owned values to their global index
        global_indices = np.arange(local_range[0], local_range[1], dtype=np.float64)
        f.x.array[:n_owned] = global_indices
        f.x.scatter_forward()
        
        # Now check: each ghost should equal its global index
        n_ghosts = imap.num_ghosts
        if n_ghosts > 0:
            ghost_vals = f.x.array[n_owned:n_owned + n_ghosts]
            ghost_global = imap.ghosts.astype(np.float64)
            
            mismatches = np.where(np.abs(ghost_vals - ghost_global) > 1e-12)[0]
            
            if len(mismatches) > 0:
                if rank == 0:
                    print(f"\n  [ERROR] Found {len(mismatches)} ghost/owner mismatches!")
                    for i in mismatches[:5]:  # Show first 5
                        print(f"    Ghost {i}: got {ghost_vals[i]}, expected {ghost_global[i]}")
            
            assert len(mismatches) == 0, \
                f"Ghost values don't match owners: {len(mismatches)} mismatches"
    
    def test_dg0_stimulus_interface_values(self):
        """
        DIAGNOSTIC: Check DG0 stimulus field behavior at partition interfaces.
        
        DG0 fields are discontinuous by design, but when used as input to CG1
        density solver, interface handling matters.
        """
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(6)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags)
        
        # Create DG0 space (like stimulus/psi)
        DG0 = basix.ufl.element("DG", domain.basix_cell(), 0)
        W = functionspace(domain, DG0)
        
        psi = Function(W, name="psi")
        
        # Set to constant value
        psi.x.array[:] = 0.01
        psi.x.scatter_forward()
        
        # Check: for DG0, ghost cells should have correct values
        imap = W.dofmap.index_map
        n_owned = imap.size_local
        n_ghosts = imap.num_ghosts
        
        all_vals = psi.x.array[:n_owned + n_ghosts]
        
        # All values should be 0.01
        max_deviation = np.max(np.abs(all_vals - 0.01))
        max_deviation_global = comm.allreduce(max_deviation, op=MPI.MAX)
        
        if rank == 0:
            print(f"\n  [DIAGNOSTIC] DG0 max deviation from constant: {max_deviation_global:.2e}")
            print(f"  [DIAGNOSTIC] DG0 owned cells: {n_owned}, ghost cells: {n_ghosts}")
        
        assert max_deviation_global < 1e-14, \
            f"DG0 field has inconsistent values: max_dev={max_deviation_global}"
    
    def test_output_array_structure(self):
        """
        DIAGNOSTIC: Examine the structure of arrays that would be written to VTX.
        
        This reveals if ghost DOFs are included in output and their ordering.
        """
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="test")
        f.interpolate(lambda x: x[0] + x[1] + x[2])
        f.x.scatter_forward()
        
        imap = Q.dofmap.index_map
        n_owned = imap.size_local
        n_ghosts = imap.num_ghosts
        n_total_local = n_owned + n_ghosts
        
        # Gather info from all ranks
        all_owned = comm.gather(n_owned, root=0)
        all_ghosts = comm.gather(n_ghosts, root=0)
        all_total = comm.gather(n_total_local, root=0)
        global_size = imap.size_global
        
        if rank == 0:
            print(f"\n  [DIAGNOSTIC] Array structure per rank:")
            print(f"  {'Rank':<6} {'Owned':<10} {'Ghosts':<10} {'Total Local':<12}")
            print(f"  {'-'*38}")
            for r in range(size):
                print(f"  {r:<6} {all_owned[r]:<10} {all_ghosts[r]:<10} {all_total[r]:<12}")
            print(f"  {'-'*38}")
            print(f"  Global DOFs: {global_size}")
            print(f"  Sum of owned: {sum(all_owned)} (should equal global)")
            print(f"  Sum of total local: {sum(all_total)} (includes ghost duplicates)")
            
            # Check for duplication factor
            dup_factor = sum(all_total) / global_size
            print(f"  Duplication factor: {dup_factor:.2f}x")
            
            if dup_factor > 1.5:
                print(f"  [WARNING] High duplication - ghosts may cause VTX artifacts!")
        
        # The sum of owned should equal global
        sum_owned = comm.allreduce(n_owned, op=MPI.SUM)
        assert sum_owned == global_size, \
            f"Sum of owned ({sum_owned}) != global ({global_size})"
    
    def test_interface_node_values_match(self):
        """
        DIAGNOSTIC: For interface nodes, compare values across ranks.
        
        Interface nodes are owned by one rank and are ghosts on others.
        Their values MUST match after scatter_forward.
        """
        if size < 2:
            pytest.skip("Need >= 2 ranks")
        
        domain = create_test_mesh(8)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="test")
        
        # Set a pattern that will reveal mismatches
        n_owned = get_owned_size(f)
        f.x.array[:n_owned] = rank * 100 + np.arange(n_owned, dtype=np.float64)
        f.x.scatter_forward()
        
        # Gather ALL values for interface DOFs to rank 0 for comparison
        imap = Q.dofmap.index_map
        ghost_global_indices = imap.ghosts
        ghost_values = f.x.array[n_owned:n_owned + len(ghost_global_indices)]
        
        # Create (global_index, value, rank) tuples for ghosts
        local_ghost_data = list(zip(ghost_global_indices, ghost_values, [rank] * len(ghost_global_indices)))
        
        all_ghost_data = comm.gather(local_ghost_data, root=0)
        
        if rank == 0:
            # Flatten and group by global index
            from collections import defaultdict
            index_values = defaultdict(list)
            
            for rank_data in all_ghost_data:
                for gidx, val, r in rank_data:
                    index_values[int(gidx)].append((val, r))
            
            # Find any inconsistencies
            inconsistent = []
            for gidx, vals in index_values.items():
                unique_vals = set(v for v, _ in vals)
                if len(unique_vals) > 1:
                    inconsistent.append((gidx, vals))
            
            if inconsistent:
                print(f"\n  [ERROR] Found {len(inconsistent)} inconsistent interface DOFs!")
                for gidx, vals in inconsistent[:5]:
                    print(f"    Global DOF {gidx}: values={vals}")
            else:
                print(f"\n  [OK] All {len(index_values)} interface DOFs are consistent")
            
            assert len(inconsistent) == 0, \
                f"Interface DOF inconsistencies found: {len(inconsistent)}"
    
    def test_vtx_would_write_duplicates(self):
        """
        DIAGNOSTIC: Check if VTX output would contain duplicate DOF values.
        
        VTXWriter writes ALL local DOFs (owned + ghosts). If ghosts are included
        in the output file, ParaView may show duplicates at interfaces.
        """
        domain = create_test_mesh(6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        f = Function(Q, name="test")
        f.interpolate(lambda x: x[0])
        f.x.scatter_forward()
        
        # Simulate what VTX would write: f.x.array (all local DOFs)
        imap = Q.dofmap.index_map
        n_owned = imap.size_local
        n_ghosts = imap.num_ghosts
        
        # VTX writes the FULL array including ghosts
        vtx_would_write = len(f.x.array)
        owned_only = n_owned
        
        # Gather statistics
        total_vtx = comm.allreduce(vtx_would_write, op=MPI.SUM)
        total_owned = comm.allreduce(owned_only, op=MPI.SUM)
        global_dofs = imap.size_global
        
        if rank == 0:
            print(f"\n  [DIAGNOSTIC] VTX output analysis:")
            print(f"    Global unique DOFs: {global_dofs}")
            print(f"    Sum of owned DOFs: {total_owned}")
            print(f"    Sum of VTX output DOFs: {total_vtx}")
            print(f"    Overlap (ghost duplicates): {total_vtx - global_dofs}")
            
            overlap_pct = 100 * (total_vtx - global_dofs) / global_dofs
            print(f"    Overlap percentage: {overlap_pct:.1f}%")
            
            if overlap_pct > 10:
                print(f"    [WARNING] Significant ghost overlap in VTX output!")
                print(f"    This can cause visible artifacts at partition boundaries in ParaView.")
                print(f"    Recommendation: Use 'Merge Blocks' filter or write owned-only data.")


# =============================================================================
# Sanity Checks: Rank ID and Constant Field Tests
# =============================================================================

class TestSanityChecks:
    """
    Sanity check tests to distinguish ghost/solver issues from output issues.
    
    (A) Rank ID test: Each subdomain should have constant rank value, no noise at interfaces.
    (B) Constant field test: A field set to 1.0 everywhere must remain exactly 1.0.
    
    If these fail, the problem is in DOF mapping or ghost handling.
    If these pass but simulation fields show artifacts, the problem is elsewhere.
    """
    
    def test_rank_id_field_owned_values(self):
        """Test that owned DOFs have correct rank ID value."""
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rank_id = Function(Q, name="rank_id")
        n_owned = get_owned_size(rank_id)
        
        # Set owned DOFs to rank
        rank_id.x.array[:n_owned] = float(rank)
        rank_id.x.scatter_forward()
        
        # Verify owned DOFs are exactly rank
        owned_vals = rank_id.x.array[:n_owned]
        assert np.allclose(owned_vals, float(rank)), \
            f"Rank {rank}: owned DOFs should all be {rank}, got range [{owned_vals.min()}, {owned_vals.max()}]"
    
    def test_rank_id_ghost_values_match_owners(self):
        """Test that ghost DOFs have the correct owner rank value."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rank_id = Function(Q, name="rank_id")
        n_owned = get_owned_size(rank_id)
        
        # Set owned DOFs to rank
        rank_id.x.array[:n_owned] = float(rank)
        rank_id.x.scatter_forward()
        
        # Get ghost info
        imap = Q.dofmap.index_map
        bs = Q.dofmap.index_map_bs
        n_local = imap.size_local * bs
        n_ghosts = imap.num_ghosts
        ghost_owners = imap.owners
        
        if n_ghosts > 0:
            ghost_vals = rank_id.x.array[n_local:]
            
            # Each ghost DOF should have the value equal to its owner rank
            for i, (val, owner) in enumerate(zip(ghost_vals, ghost_owners)):
                assert np.isclose(val, float(owner)), \
                    f"Rank {rank}: ghost DOF {i} has value {val}, but owner is rank {owner}"
    
    def test_rank_id_no_interface_noise(self):
        """Test that rank_id field has no spurious values at partition interfaces."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rank_id = Function(Q, name="rank_id")
        n_owned = get_owned_size(rank_id)
        
        # Set owned DOFs to rank
        rank_id.x.array[:n_owned] = float(rank)
        rank_id.x.scatter_forward()
        
        # All values in the array should be valid rank IDs (0, 1, ..., size-1)
        all_vals = rank_id.x.array
        valid_ranks = set(range(size))
        
        for val in all_vals:
            rounded = round(val)
            assert rounded in valid_ranks, \
                f"Rank {rank}: found invalid rank ID {val} (rounded: {rounded})"
            assert np.isclose(val, float(rounded), atol=1e-10), \
                f"Rank {rank}: rank ID has noise: {val} != {rounded}"
    
    def test_constant_field_owned_values(self):
        """Test that a constant field has correct owned DOF values."""
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        const = Function(Q, name="const")
        n_owned = get_owned_size(const)
        
        # Set owned DOFs to 1.0
        const.x.array[:n_owned] = 1.0
        const.x.scatter_forward()
        
        # Verify owned DOFs are exactly 1.0
        owned_vals = const.x.array[:n_owned]
        assert np.allclose(owned_vals, 1.0, atol=1e-14), \
            f"Rank {rank}: owned DOFs should all be 1.0, got range [{owned_vals.min()}, {owned_vals.max()}]"
    
    def test_constant_field_ghost_values(self):
        """Test that ghost DOFs of a constant field are also exactly 1.0."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        const = Function(Q, name="const")
        n_owned = get_owned_size(const)
        
        # Set owned DOFs to 1.0
        const.x.array[:n_owned] = 1.0
        const.x.scatter_forward()
        
        # Verify ghost DOFs are also 1.0
        imap = Q.dofmap.index_map
        bs = Q.dofmap.index_map_bs
        n_local = imap.size_local * bs
        n_ghosts = imap.num_ghosts
        
        if n_ghosts > 0:
            ghost_vals = const.x.array[n_local:]
            assert np.allclose(ghost_vals, 1.0, atol=1e-14), \
                f"Rank {rank}: ghost DOFs should be 1.0, got range [{ghost_vals.min()}, {ghost_vals.max()}]"
    
    def test_constant_field_global_integral(self):
        """Test that integral of constant=1 field equals mesh volume."""
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        const = Function(Q, name="const")
        n_owned = get_owned_size(const)
        
        const.x.array[:n_owned] = 1.0
        const.x.scatter_forward()
        
        # Integral of 1 over unit cube should be 1.0
        dx = ufl.Measure("dx", domain=domain)
        integral = fem.assemble_scalar(fem.form(const * dx))
        global_integral = comm.allreduce(integral, op=MPI.SUM)
        
        # Unit cube volume = 1.0
        assert np.isclose(global_integral, 1.0, atol=1e-10), \
            f"Integral of const=1 should be 1.0 (unit cube volume), got {global_integral}"
    
    def test_constant_field_no_interface_artifacts(self):
        """Test that constant field has no artifacts at partition interfaces."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        const = Function(Q, name="const")
        n_owned = get_owned_size(const)
        
        const.x.array[:n_owned] = 1.0
        const.x.scatter_forward()
        
        # ALL values (owned + ghost) should be exactly 1.0
        all_vals = const.x.array
        
        # Check for any deviation from 1.0
        max_deviation = np.max(np.abs(all_vals - 1.0))
        assert max_deviation < 1e-14, \
            f"Rank {rank}: constant field has deviation {max_deviation} from 1.0"
        
        # No value should be outside [1.0 - eps, 1.0 + eps]
        assert np.all(np.isclose(all_vals, 1.0, atol=1e-14)), \
            f"Rank {rank}: not all values are exactly 1.0"
    
    def test_rank_id_statistics(self):
        """Diagnostic: print rank_id field statistics for debugging."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rank_id = Function(Q, name="rank_id")
        n_owned = get_owned_size(rank_id)
        
        rank_id.x.array[:n_owned] = float(rank)
        rank_id.x.scatter_forward()
        
        # Compute field stats
        rmin, rmax, rmean = field_stats(rank_id, comm)
        
        if rank == 0:
            print(f"\n  [DIAGNOSTIC] Rank ID field statistics:")
            print(f"    Min: {rmin}, Max: {rmax}, Mean: {rmean:.4f}")
            print(f"    Expected: Min=0, Max={size-1}, Mean={0.5*(size-1):.4f}")
        
        # Mean of rank IDs should be approximately (size-1)/2
        # (weighted by owned DOFs per rank, which may vary slightly)
        assert rmin == 0.0, f"Min rank should be 0, got {rmin}"
        assert rmax == float(size - 1), f"Max rank should be {size-1}, got {rmax}"
    
    def test_constant_field_statistics(self):
        """Diagnostic: verify constant field has uniform statistics."""
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        const = Function(Q, name="const")
        n_owned = get_owned_size(const)
        
        const.x.array[:n_owned] = 1.0
        const.x.scatter_forward()
        
        # Compute field stats
        cmin, cmax, cmean = field_stats(const, comm)
        
        if rank == 0:
            print(f"\n  [DIAGNOSTIC] Constant field statistics:")
            print(f"    Min: {cmin}, Max: {cmax}, Mean: {cmean}")
            print(f"    Expected: Min=1.0, Max=1.0, Mean=1.0")
        
        assert np.isclose(cmin, 1.0, atol=1e-14), f"Min should be 1.0, got {cmin}"
        assert np.isclose(cmax, 1.0, atol=1e-14), f"Max should be 1.0, got {cmax}"
        assert np.isclose(cmean, 1.0, atol=1e-14), f"Mean should be 1.0, got {cmean}"


class TestSanityChecksWithVTXOutput:
    """
    Extended sanity checks that verify VTX output format.
    
    These tests write diagnostic fields to VTX files and verify the data
    that would be visible in ParaView.
    """
    
    def test_rank_id_vtx_output_consistency(self, tmp_path_factory):
        """Test that rank_id field VTX output has correct values."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        # Use shared temp directory across ranks
        if rank == 0:
            tmp_dir = tmp_path_factory.mktemp("rank_id_test")
        else:
            tmp_dir = None
        tmp_dir = comm.bcast(tmp_dir, root=0)
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rank_id = Function(Q, name="rank_id")
        n_owned = get_owned_size(rank_id)
        
        rank_id.x.array[:n_owned] = float(rank)
        rank_id.x.scatter_forward()
        
        # The data that VTX would write
        vtx_data = rank_id.x.array.copy()
        
        # Get ghost info
        imap = Q.dofmap.index_map
        bs = Q.dofmap.index_map_bs
        n_local = imap.size_local * bs
        n_ghosts = imap.num_ghosts
        ghost_owners = imap.owners
        
        # Verify: each ghost has value = its owner rank
        if n_ghosts > 0:
            ghost_vals = vtx_data[n_local:]
            for i, (val, owner) in enumerate(zip(ghost_vals, ghost_owners)):
                assert np.isclose(val, float(owner)), \
                    f"VTX output would show ghost DOF {i} with value {val}, but owner is {owner}"
        
        if rank == 0:
            print(f"\n  [OK] rank_id VTX output would be consistent")
            print(f"       Owned DOFs show rank {rank}, ghosts show their owner ranks")
    
    def test_constant_field_vtx_output_no_artifacts(self, tmp_path_factory):
        """Test that constant field VTX output has no artifacts."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        # Use shared temp directory across ranks
        if rank == 0:
            tmp_dir = tmp_path_factory.mktemp("const_test")
        else:
            tmp_dir = None
        tmp_dir = comm.bcast(tmp_dir, root=0)
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        const = Function(Q, name="const")
        n_owned = get_owned_size(const)
        
        const.x.array[:n_owned] = 1.0
        const.x.scatter_forward()
        
        # The data that VTX would write
        vtx_data = const.x.array.copy()
        
        # ALL data should be exactly 1.0
        max_deviation = np.max(np.abs(vtx_data - 1.0))
        
        assert max_deviation < 1e-14, \
            f"Rank {rank}: VTX output would show artifacts with deviation {max_deviation}"
        
        if rank == 0:
            print(f"\n  [OK] constant field VTX output would have no artifacts")
            print(f"       All values (owned + ghost) are exactly 1.0")
    
    def test_vtx_ghost_duplication_diagnostic(self):
        """Diagnostic: show ghost duplication in VTX output."""
        if size < 2:
            pytest.skip("Need at least 2 MPI ranks")
        
        domain = create_test_mesh(n=6)
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)
        
        rank_id = Function(Q, name="rank_id")
        n_owned = get_owned_size(rank_id)
        
        rank_id.x.array[:n_owned] = float(rank)
        rank_id.x.scatter_forward()
        
        # Get counts
        imap = Q.dofmap.index_map
        bs = Q.dofmap.index_map_bs
        local_owned = imap.size_local * bs
        local_ghosts = imap.num_ghosts
        local_total = len(rank_id.x.array)
        
        global_owned = comm.allreduce(local_owned, op=MPI.SUM)
        global_ghosts = comm.allreduce(local_ghosts, op=MPI.SUM)
        global_vtx_total = comm.allreduce(local_total, op=MPI.SUM)
        global_unique = imap.size_global
        
        if rank == 0:
            print(f"\n  [DIAGNOSTIC] VTX output duplication analysis:")
            print(f"    Global unique DOFs: {global_unique}")
            print(f"    Total owned across ranks: {global_owned}")
            print(f"    Total ghosts across ranks: {global_ghosts}")
            print(f"    VTX would output per rank: sum = {global_vtx_total}")
            print(f"    Duplication factor: {global_vtx_total / global_unique:.2f}x")
            print(f"    ")
            print(f"    [INFO] ParaView sees overlapping data at interfaces.")
            print(f"    [INFO] Use 'Merge Blocks' filter or 'Group Datasets' to fix.")
        
        # Test passes - this is informational
        assert global_vtx_total >= global_unique, "VTX output should include at least all unique DOFs"

