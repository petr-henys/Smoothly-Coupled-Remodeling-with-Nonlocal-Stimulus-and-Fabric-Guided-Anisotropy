import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem

from simulation.projector import L2Projector

@pytest.fixture
def unit_cube():
    return mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)

@pytest.fixture
def V_scalar(unit_cube):
    return fem.functionspace(unit_cube, ("DG", 0))

@pytest.fixture
def V_vector(unit_cube):
    return fem.functionspace(unit_cube, ("DG", 0, (3,)))

@pytest.fixture
def V_tensor(unit_cube):
    return fem.functionspace(unit_cube, ("DG", 0, (3, 3)))

def test_project_constant_scalar(unit_cube, V_scalar):
    """Projecting a constant value should be exact (machine precision)."""
    proj = L2Projector(V_scalar, alpha=0.0)  # Pure L2 projection
    c = 3.14159
    u_out = proj.project(fem.Constant(unit_cube, c))
    
    arr = u_out.x.array
    assert np.allclose(arr, c, atol=1e-12)

def test_project_linear_scalar_dg0(unit_cube, V_scalar):
    """Projecting linear x onto DG0 gives cell averages."""
    proj = L2Projector(V_scalar, alpha=0.0)
    x = ufl.SpatialCoordinate(unit_cube)
    u_out = proj.project(x[0])
    
    # Bounds check: x in [0, 1]
    arr = u_out.x.array
    assert arr.min() >= -1e-12
    assert arr.max() <= 1.0 + 1e-12
    # It shouldn't be constant
    assert arr.std() > 0.1

def test_project_vector(unit_cube, V_vector):
    """Projecting a vector expression."""
    proj = L2Projector(V_vector, alpha=0.0)
    x = ufl.SpatialCoordinate(unit_cube)
    # v = (x, y, z)
    expr = ufl.as_vector((x[0], x[1], x[2]))
    u_out = proj.project(expr)
    
    # Check shape and bounds
    arr = u_out.x.array.reshape((-1, 3))
    assert arr.shape[1] == 3
    assert np.all(arr >= -1e-12)
    assert np.all(arr <= 1.0 + 1e-12)

def test_project_tensor_identity(unit_cube, V_tensor):
    """Projecting the identity tensor."""
    proj = L2Projector(V_tensor, alpha=0.0)
    I = ufl.Identity(3)
    u_out = proj.project(I)
    
    arr = u_out.x.array.reshape((-1, 9))
    # Identity flattened is [1,0,0, 0,1,0, 0,0,1]
    expected = np.array([1,0,0, 0,1,0, 0,0,1], dtype=float)
    
    # Check a few random cells
    for i in [0, -1]:
        assert np.allclose(arr[i], expected, atol=1e-12)

def test_smoothing_parameter_effect(unit_cube, V_scalar):
    """Test that alpha > 0 introduces smoothing (diffusion)."""
    # Step function at x=0.5
    x = ufl.SpatialCoordinate(unit_cube)
    step = ufl.conditional(ufl.lt(x[0], 0.5), 0.0, 1.0)
    
    # Pure L2 (alpha=0)
    p0 = L2Projector(V_scalar, alpha=0.0)
    u0 = p0.project(step)
    
    # Smoothed (alpha=1.0)
    p1 = L2Projector(V_scalar, alpha=1.0)
    u1 = p1.project(step)
    
    # With alpha=0, it's local averaging.
    # With alpha>0, it penalizes jumps, so it tries to be continuous.
    # The step function 0->1 will be smeared.
    
    assert not np.allclose(u0.x.array, u1.x.array, atol=1e-3)

def test_reuse_projector(unit_cube, V_scalar):
    """Reuse the same projector instance for multiple solves."""
    proj = L2Projector(V_scalar, alpha=0.0)
    
    # 1. Project constant 1
    u1 = proj.project(fem.Constant(unit_cube, 1.0))
    assert np.allclose(u1.x.array, 1.0, atol=1e-12)
    
    # 2. Project constant 2
    u2 = proj.project(fem.Constant(unit_cube, 2.0))
    assert np.allclose(u2.x.array, 2.0, atol=1e-12)
    
    # Ensure u1 was updated in place if we didn't pass a result function?
    # The default behavior of project(expr) is to return self._x (internal buffer) if result is None.
    # So u1 and u2 point to the same object.
    assert u1 is u2
    assert np.allclose(u1.x.array, 2.0, atol=1e-12)

def test_project_into_user_function(unit_cube, V_scalar):
    """Project into a user-provided function."""
    proj = L2Projector(V_scalar, alpha=0.0)
    u_user = fem.Function(V_scalar)
    u_user.x.array[:] = -1.0
    
    proj.project(fem.Constant(unit_cube, 5.0), result=u_user)
    assert np.allclose(u_user.x.array, 5.0, atol=1e-12)
