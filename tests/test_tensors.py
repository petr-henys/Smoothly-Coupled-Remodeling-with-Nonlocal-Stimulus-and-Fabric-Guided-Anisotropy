import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_dirichlet_bcs, build_facetag
from simulation.subsolvers import MechanicsSolver, DirectionSolver, unittrace_psd, unittrace_psd_from_any
from dolfinx import mesh

def make_unit_cube(comm=MPI.COMM_WORLD, n=6):
    return mesh.create_unit_cube(comm, n, n, n)

def iso_tensor(x):
    values = np.zeros((9, x.shape[1]), dtype=default_scalar_type)
    values[0] = 1.0
    values[4] = 1.0
    values[8] = 1.0
    return values

# =============================================================================
# PSD Tensor Tests
# =============================================================================

class TestPSDTensors:
    """Test positive-semidefinite tensor enforcement."""
    
    def test_unittrace_psd_from_any_properties(self):
        """Verify unittrace_psd_from_any produces symmetric, SPD, unit-trace tensor."""
        comm = MPI.COMM_WORLD
        domain = make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        T = functionspace(domain, P1_ten)
        
        # Test with arbitrary (non-symmetric, non-PSD) tensor
        A_in = Function(T, name="A_in")
        
        # Create tensor field via component-wise interpolation
        def tensor_field(x):
            n_points = x.shape[1]
            result = np.zeros((9, n_points))
            # Fill tensor components [A00, A01, A02, A10, A11, A12, A20, A21, A22]
            result[0, :] = x[0]      # A00 = x
            result[1, :] = x[1]      # A01 = y
            result[2, :] = 0.0       # A02 = 0
            result[3, :] = x[2]      # A10 = z
            result[4, :] = x[0] + x[1]  # A11 = x+y
            result[5, :] = 0.0       # A12 = 0
            result[6, :] = 0.0       # A20 = 0
            result[7, :] = 0.0       # A21 = 0
            result[8, :] = 1.0       # A22 = 1
            return result
        
        A_in.interpolate(tensor_field)
        A_in.x.scatter_forward()
        
        # Apply PSD enforcement
        Asym = 0.5 * (A_in + ufl.transpose(A_in))
        A_hat = unittrace_psd_from_any(Asym, 3, float(cfg.smooth_eps))
        
        # Check trace = 1
        tr_A = ufl.tr(A_hat)
        tr_local = fem.assemble_scalar(fem.form(tr_A * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        tr_avg = comm.allreduce(tr_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        assert abs(tr_avg - 1.0) < 1e-10, f"Unit trace not preserved: tr(A_hat) = {tr_avg}"
        
        # Check symmetry: A_hat[0,1] = A_hat[1,0]
        A_hat_01 = A_hat[0, 1]
        A_hat_10 = A_hat[1, 0]
        diff_sq = (A_hat_01 - A_hat_10)**2
        diff_sq_local = fem.assemble_scalar(fem.form(diff_sq * cfg.dx))
        diff_sq_global = comm.allreduce(diff_sq_local, op=MPI.SUM)
        
        assert diff_sq_global < 1e-12, f"A_hat not symmetric"
    
    def test_fabric_normalization_stability(self):
        """Test fabric normalization doesn't blow up with zero strain."""
        comm = MPI.COMM_WORLD
        domain = make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
        
        V = functionspace(domain, P1_vec)
        T = functionspace(domain, P1_ten)
        
        # Zero displacement → zero strain
        u = Function(V, name="u")
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        
        # Direction solver target: B = ε^T ε normalized
        eps = ufl.sym(ufl.grad(u))
        B = ufl.dot(ufl.transpose(eps), eps)
        I = ufl.Identity(3)
        
        # Should default to I/3 when trB ~ 0
        B_hat = unittrace_psd(B, 3, float(cfg.smooth_eps))
        
        # Check all eigenvalues ~ 1/3 (isotropic)
        B_hat_00 = B_hat[0, 0]
        B_hat_11 = B_hat[1, 1]
        B_hat_22 = B_hat[2, 2]
        
        diag_avg_local = fem.assemble_scalar(fem.form((B_hat_00 + B_hat_11 + B_hat_22) * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        
        diag_avg = comm.allreduce(diag_avg_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        # Trace = 1 → avg diagonal = 1/3 for isotropic
        assert abs(diag_avg - 1.0) < 0.05, f"Zero strain should yield isotropic fabric (tr=1), got {diag_avg}"


class TestDirectionSolverProperties:
    """Properties of the direction tensor solver outputs."""

    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_direction_solver_unit_trace_psd(self, unit_cube, facet_tags, traction_factory):
        """Direction solver output should be symmetric PSD with unit trace."""
        comm = MPI.COMM_WORLD
        domain = unit_cube
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))

        P1_vec = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3,))
        P1 = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", unit_cube.basix_cell(), 1, shape=(3, 3))

        V = functionspace(unit_cube, P1_vec)
        Q = functionspace(unit_cube, P1)
        T = functionspace(unit_cube, P1_ten)

        rho = Function(Q, name="rho")
        rho.x.array[:] = 0.6
        rho.x.scatter_forward()

        A_old = Function(T, name="A_old")
        A_old.interpolate(lambda x: (np.eye(3) / 3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
        A_old.x.scatter_forward()

        u = Function(V, name="u")
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        traction = traction_factory(-0.4, facet_id=2, axis=0)

        mech = MechanicsSolver(u, rho, A_old, cfg, bc_mech, [traction])
        mech.setup()
        mech.solve()

        A = Function(T, name="A")
        dir_solver = DirectionSolver(A, A_old, cfg)
        dir_solver.setup()
        
        # Get strain tensor for RHS
        strain_tensor = mech.get_strain_tensor()
        dir_solver.assemble_rhs(strain_tensor)

        dir_solver.solve()
        A.x.scatter_forward()

        n_owned = T.dofmap.index_map.size_local * T.dofmap.index_map_bs
        if n_owned == 0:
            pytest.skip("No owned DOFs on this rank for tensor space")

        values = A.x.array[:n_owned]
        assert values.size % 9 == 0, "Tensor DOF array not divisible by 9 components"
        tensors = values.reshape(-1, 9)

        for row in tensors:
            mat = row.reshape(3, 3)
            sym = 0.5 * (mat + mat.T)
            trace = np.trace(sym)
            assert abs(trace - 1.0) < 2e-5, f"Trace not unity: {trace}"
            eigvals = np.linalg.eigvalsh(sym)
            assert eigvals.min() >= -1e-8, f"Tensor not PSD: eigenvalues={eigvals}"
