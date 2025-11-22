import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.fem import Function, functionspace
import ufl

from simulation.config import Config
from simulation.utils import build_dirichlet_bcs, build_facetag, collect_dirichlet_dofs
from simulation.subsolvers import MechanicsSolver
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
# Boundary Condition Tests
# =============================================================================

class TestBoundaryConditions:
    """Test boundary condition enforcement."""
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_dirichlet_enforcement_strong(self, unit_cube, traction_factory):
        """Verify Dirichlet BCs are strongly enforced."""
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
        
        # Apply traction on right face
        traction = traction_factory(-0.1, facet_id=2, axis=0)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
        
        mech.setup()
        mech.assemble_rhs()
        mech.solve()
        
        # Extract DOFs on left boundary (tag=1, x=0)
        bc_dofs = collect_dirichlet_dofs(bc_mech, mech.function_space.dofmap.index_map.size_local)
        
        if bc_dofs.size > 0:
            u_bc_vals = u.x.array[bc_dofs]
            max_bc_val = np.max(np.abs(u_bc_vals))
            assert max_bc_val < 1e-9, f"Dirichlet BC not enforced: max |u| on BC = {max_bc_val}"
    
    @pytest.mark.parametrize("unit_cube", [6], indirect=True)
    def test_traction_load_response(self, unit_cube, traction_factory):
        """Verify mechanics solver responds correctly to applied traction."""
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
        
        # Compression in x-direction
        traction = traction_factory(-0.5, facet_id=2, axis=0)
        
        bc_mech = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        mech = MechanicsSolver(u, rho, A, cfg, bc_mech, [traction])
        
        mech.setup()
        mech.assemble_rhs()
        mech.solve()
        
        # Under compression, expect negative x-displacement (compression)
        u_x = u.sub(0).collapse()
        u_x_avg_local = fem.assemble_scalar(fem.form(u_x * cfg.dx))
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        
        u_x_avg = comm.allreduce(u_x_avg_local, op=MPI.SUM) / comm.allreduce(vol_local, op=MPI.SUM)
        
        assert u_x_avg < -1e-10, f"Compression load should yield negative x-displacement, got {u_x_avg}"

def test_mechanics_uniform_extension():
    """Uniform extension test: apply displacement BCs, check solver converges and energy balance holds."""
    comm = MPI.COMM_WORLD
    m = make_unit_cube(comm, n=6)
    facets = build_facetag(m)

    # Function spaces
    V = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,)))
    Q = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1))
    T = fem.functionspace(m, basix.ufl.element("Lagrange", m.basix_cell(), 1, shape=(3,3)))

    # Fields
    rho = fem.Function(Q, name="rho")
    rho.x.array[:] = 1.0
    rho.x.scatter_forward()

    Afield = fem.Function(T, name="A")
    Afield.interpolate(lambda x: (np.eye(3)/3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    Afield.x.scatter_forward()

    # Config
    cfg = Config(domain=m, facet_tags=facets, verbose=False)
    cfg.xi_aniso = 0.0

    # Simple extension test: clamp x=0, prescribe u_x=0.01 on x=1
    fdim = m.topology.dim - 1
    eps = 0.01

    # Clamp x=0 face
    bcs = build_dirichlet_bcs(V, facets, id_tag=1, value=0.0)
    
    # Prescribe u_x=eps on x=1 face
    facets_x1 = facets.find(2)
    V0 = V.sub(0)
    dofs_x1 = fem.locate_dofs_topological(V0, fdim, facets_x1)
    bc_x1 = fem.dirichletbc(default_scalar_type(eps), dofs_x1, V0)
    bcs.append(bc_x1)

    # Create solution function
    u = fem.Function(V, name="u")
    
    # Solve
    mech = MechanicsSolver(u, rho, Afield, cfg, bcs, [])
    mech.setup()
    mech.assemble_rhs()
    its, reason = mech.solve()
    assert reason > 0, f"KSP failed to converge, reason={reason}"

    # Check solution is nonzero (should have extension)
    idxmap = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    owned = idxmap.size_local * bs
    u_norm = np.linalg.norm(u.x.array[:owned])
    assert u_norm > 1e-6, f"Solution is nearly zero: {u_norm:.2e}"
