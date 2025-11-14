"""Debug test to understand why stimulus stays zero."""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, build_dirichlet_bcs
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitEnergyDriver
from tests.test_model import _DummyGaitLoader, _unit_cube


def test_driver_produces_nonzero_energy(tmp_path):
    """Check if GaitEnergyDriver produces non-zero energy expression."""
    comm = MPI.COMM_WORLD
    domain = _unit_cube(4)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True, results_dir=str(tmp_path))
    
    from dolfinx.fem import functionspace
    import basix
    
    gdim = domain.geometry.dim
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim, gdim))
    
    V = functionspace(domain, P1_vec)
    Q = functionspace(domain, P1)
    T = functionspace(domain, P1_ten)
    
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    A = fem.Function(T, name="dir_tensor")
    
    # Initialize fields
    rho.x.array[:] = cfg.rho0
    
    # Isotropic A
    def _A_const(x):
        n = x.shape[1]
        vals = (np.eye(gdim, dtype=np.float64) / gdim).reshape(gdim * gdim, 1)
        return np.tile(vals, (1, n))
    A.interpolate(_A_const)
    A.x.scatter_forward()
    
    # Setup mechanics
    bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    gait_loader = _DummyGaitLoader(V)
    neumann_bcs = [
        (gait_loader.t_hip, 2),
        (gait_loader.t_glmed, 2),
        (gait_loader.t_glmax, 2),
    ]
    
    mech = MechanicsSolver(u, rho, A, cfg, bc_mech, neumann_bcs)
    mech.setup()
    
    # Create driver
    driver = GaitEnergyDriver(mech, gait_loader, cfg)
    
    # Update snapshots to compute gait-averaged fields
    print(f"\n[Rank {comm.rank}] Updating snapshots...")
    driver.update_snapshots()
    
    # Build energy expression
    print(f"[Rank {comm.rank}] Building energy expression...")
    psi_expr = driver.energy_expr()
    print(f"[Rank {comm.rank}] Energy expression type: {type(psi_expr)}")
    
    # Integrate energy
    psi_form = fem.form(psi_expr * cfg.dx)
    psi_local = fem.assemble_scalar(psi_form)
    psi_global = comm.allreduce(psi_local, op=MPI.SUM)
    
    vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
    vol_global = comm.allreduce(vol_local, op=MPI.SUM)
    
    psi_avg = psi_global / vol_global
    
    print(f"[Rank {comm.rank}] psi_global={psi_global:.6e}, vol={vol_global:.6e}, psi_avg={psi_avg:.6e}")
    
    assert psi_avg > 1e-8, f"Average energy density too small: {psi_avg:.3e}"
    
    mech.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
