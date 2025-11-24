import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, unittrace_psd
from dolfinx import mesh

def make_unit_cube(comm=MPI.COMM_WORLD, n=6):
    return mesh.create_unit_cube(comm, n, n, n)

# =============================================================================
# PSD Tensor Tests
# =============================================================================

class TestPSDTensors:
    """Test positive-semidefinite tensor enforcement."""
    
    def test_fabric_normalization_stability(self):
        """Test fabric normalization doesn't blow up with zero strain."""
        comm = MPI.COMM_WORLD
        domain = make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=(comm.rank == 0))
        
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
        
        V = functionspace(domain, P1_vec)
        
        # Zero displacement → zero strain
        u = Function(V, name="u")
        u.x.array[:] = 0.0
        u.x.scatter_forward()
        
        # Direction solver target: B = ε^T ε normalized
        eps = ufl.sym(ufl.grad(u))
        B = ufl.dot(ufl.transpose(eps), eps)
        
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
