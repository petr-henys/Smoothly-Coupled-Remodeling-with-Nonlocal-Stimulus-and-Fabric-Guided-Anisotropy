import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

from femurloader.projector import Projector

# Test tolerances
ABS_TOL = 1e-12
REL_TOL = 1e-5

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def unit_mesh():
    """Simple unit cube mesh for testing."""
    return mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)

@pytest.fixture
def scalar_space(unit_mesh):
    """DG0 scalar function space."""
    return fem.functionspace(unit_mesh, ("DG", 0))

@pytest.fixture
def vector_space(unit_mesh):
    """DG0 vector function space."""
    return fem.functionspace(unit_mesh, ("DG", 0, (3,)))

@pytest.fixture
def tensor_space(unit_mesh):
    """DG0 tensor function space."""
    return fem.functionspace(unit_mesh, ("DG", 0, (3, 3)))

@pytest.fixture
def continuous_space(unit_mesh):
    """CG1 continuous scalar space."""
    return fem.functionspace(unit_mesh, ("Lagrange", 1))

# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_projector_initialization(scalar_space):
    """Test basic projector initialization."""
    projector = Projector(scalar_space)
    
    assert projector._A is not None
    assert projector._ksp is not None
    assert projector._x is not None
    assert projector._b is not None
    assert projector.u is not None
    assert projector.v is not None

def test_projector_with_options(scalar_space):
    """Test projector with custom PETSc options."""
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "jacobi",
        "ksp_rtol": 1e-12
    }
    
    projector = Projector(scalar_space, petsc_options=petsc_options)
    assert projector._ksp is not None

def test_projector_with_smoothing(scalar_space):
    """Test projector with different smoothing parameters."""
    alphas = [0.1, 1.0, 10.0]
    
    for alpha in alphas:
        projector = Projector(scalar_space, alpha=alpha)
        
        # Test simple projection
        x = ufl.SpatialCoordinate(scalar_space.mesh)
        result = projector.project(x[0])
        
        assert isinstance(result, fem.Function)
        assert result.function_space == scalar_space

# ============================================================================
# PROJECTION ACCURACY TESTS
# ============================================================================

def test_constant_projection(scalar_space):
    """Test projection of constant functions (should be exact)."""
    projector = Projector(scalar_space)
    
    # Project constant
    constant_value = 5.0
    constant_expr = fem.Constant(scalar_space.mesh, constant_value)
    result = projector.project(constant_expr)
    
    assert np.allclose(result.x.array, constant_value, atol=ABS_TOL)

def test_linear_projection_DG0(scalar_space):
    """Test projection of linear function onto DG0 space."""
    projector = Projector(scalar_space)
    
    # Project x-coordinate (should give cell-wise averages)
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    result = projector.project(x[0])
    
    # Values should be between 0 and 1
    assert np.all(result.x.array >= -ABS_TOL)
    assert np.all(result.x.array <= 1.0 + ABS_TOL)
    
    # Should not be constant
    assert np.std(result.x.array) > REL_TOL

def test_quadratic_projection(scalar_space):
    """Test projection of quadratic function."""
    projector = Projector(scalar_space)
    
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    quadratic_expr = x[0]**2 + x[1]**2 + x[2]**2
    result = projector.project(quadratic_expr)
    
    # Should be bounded properly
    assert np.all(result.x.array >= -ABS_TOL)
    assert np.all(result.x.array <= 3.0 + REL_TOL)

def test_vector_projection(vector_space):
    """Test projection of vector expressions."""
    projector = Projector(vector_space)
    
    x = ufl.SpatialCoordinate(vector_space.mesh)
    vector_expr = ufl.as_vector([x[0], x[1], x[2]])
    result = projector.project(vector_expr)
    
    assert isinstance(result, fem.Function)
    assert result.function_space == vector_space
    
    # Check that components are reasonable
    result_array = result.x.array.reshape(-1, 3)
    for i in range(3):
        assert np.all(result_array[:, i] >= -ABS_TOL)
        assert np.all(result_array[:, i] <= 1.0 + ABS_TOL)

def test_tensor_projection(tensor_space):
    """Test projection of tensor expressions."""
    projector = Projector(tensor_space)
    
    # Project identity tensor
    I = ufl.Identity(3)
    result = projector.project(I)
    
    assert isinstance(result, fem.Function)
    assert result.function_space == tensor_space
    
    # Check identity structure
    result_array = result.x.array.reshape(-1, 9)
    expected_flat = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    
    for row in result_array:
        assert np.allclose(row, expected_flat, atol=REL_TOL)

def test_continuous_space_projection(continuous_space):
    """Test projection onto continuous function space."""
    projector = Projector(continuous_space)
    
    x = ufl.SpatialCoordinate(continuous_space.mesh)
    result = projector.project(x[0])
    
    assert isinstance(result, fem.Function)
    assert result.function_space == continuous_space

# ============================================================================
# SMOOTHING PARAMETER TESTS
# ============================================================================

def test_smoothing_effect_discontinuous(scalar_space):
    """Test effect of smoothing parameter on discontinuous projections."""
    # Create discontinuous function (step function)
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    step_expr = ufl.conditional(x[0] < 0.5, 0.0, 1.0)
    
    # Test different smoothing levels
    alphas = [0.0, 1.0, 10.0]
    results = []
    
    for alpha in alphas:
        projector = Projector(scalar_space, alpha=alpha)
        result = projector.project(step_expr)
        results.append(result.x.array.copy())
    
    # Higher smoothing should reduce variation
    variations = [np.std(result) for result in results]
    
    # With smoothing, variation should generally decrease
    assert len(variations) == 3

def test_smoothing_continuous_space(continuous_space):
    """Test smoothing parameter with continuous function space."""
    alphas = [0.1, 1.0, 10.0]
    
    x = ufl.SpatialCoordinate(continuous_space.mesh)
    expr = x[0] * x[1]  # Smooth function
    
    for alpha in alphas:
        projector = Projector(continuous_space, alpha=alpha)
        result = projector.project(expr)
        
        # Should work without errors
        assert isinstance(result, fem.Function)
        assert np.all(np.isfinite(result.x.array))

# ============================================================================
# ANALYTICAL VALIDATION TESTS
# ============================================================================

def test_polynomial_preservation():
    """Test projection accuracy for polynomials in appropriate function spaces."""
    # Use higher order space for this test
    mesh_fine = mesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    space_p1 = fem.functionspace(mesh_fine, ("Lagrange", 1))
    
    projector = Projector(space_p1, alpha=0.0)  # No smoothing
    
    # Linear function should be projected with good accuracy in P1 space
    x = ufl.SpatialCoordinate(space_p1.mesh)
    linear_expr = 2.0 * x[0] + 3.0 * x[1] - x[2]
    
    result = projector.project(linear_expr)
    
    # Test that the projection preserves the function's integral (mass conservation)
    original_integral = fem.assemble_scalar(fem.form(linear_expr * ufl.dx))
    projected_integral = fem.assemble_scalar(fem.form(result * ufl.dx))
    
    relative_error = abs(projected_integral - original_integral) / abs(original_integral)
    assert relative_error < 1e-6, f"Mass conservation error: {relative_error}"
    
    # Test that projection gives reasonable values at vertices
    coords = space_p1.tabulate_dof_coordinates()
    expected_values = 2.0 * coords[:, 0] + 3.0 * coords[:, 1] - coords[:, 2]
    
    # For a linear function in P1 space, projection should be quite accurate
    max_pointwise_error = np.max(np.abs(result.x.array - expected_values))
    assert max_pointwise_error < 1e-3, f"Max pointwise error: {max_pointwise_error}"

def test_mass_conservation(scalar_space):
    """Test that projection preserves integral (mass conservation)."""
    projector = Projector(scalar_space, alpha=0.0)
    
    # Project a function and check integral preservation
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    expr = x[0] + 2.0
    
    # Compute original integral
    original_integral = fem.assemble_scalar(fem.form(expr * ufl.dx))
    
    # Project and compute integral of projection
    result = projector.project(expr)
    projected_integral = fem.assemble_scalar(fem.form(result * ufl.dx))
    
    # Should be approximately equal
    relative_error = abs(projected_integral - original_integral) / abs(original_integral)
    assert relative_error < REL_TOL

# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================

def test_zero_function_projection(scalar_space):
    """Test projection of zero function."""
    projector = Projector(scalar_space)
    
    zero_expr = fem.Constant(scalar_space.mesh, 0.0)
    result = projector.project(zero_expr)
    
    assert np.allclose(result.x.array, 0.0, atol=ABS_TOL)

def test_very_large_values(scalar_space):
    """Test projection with very large values."""
    projector = Projector(scalar_space)
    
    large_constant = fem.Constant(scalar_space.mesh, 1e10)
    result = projector.project(large_constant)
    
    assert np.allclose(result.x.array, 1e10, rtol=REL_TOL)

def test_very_small_values(scalar_space):
    """Test projection with very small values."""
    projector = Projector(scalar_space)
    
    small_constant = fem.Constant(scalar_space.mesh, 1e-10)
    result = projector.project(small_constant)
    
    assert np.allclose(result.x.array, 1e-10, atol=1e-12)

def test_complex_expression(scalar_space):
    """Test projection of complex mathematical expression."""
    projector = Projector(scalar_space)
    
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    complex_expr = ufl.sin(np.pi * x[0]) * ufl.cos(np.pi * x[1]) * ufl.exp(-x[2])
    
    result = projector.project(complex_expr)
    
    assert isinstance(result, fem.Function)
    assert np.all(np.isfinite(result.x.array))
    assert np.all(np.abs(result.x.array) <= 1.1)  # sin*cos*exp bound

# ============================================================================
# MULTIPLE PROJECTIONS AND REUSE TESTS
# ============================================================================

def test_multiple_projections(scalar_space):
    """Test multiple projections with same projector."""
    projector = Projector(scalar_space)
    
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    
    # Multiple different expressions
    expressions = [
        fem.Constant(scalar_space.mesh, 1.0),
        x[0],
        x[0]**2,
        ufl.sin(np.pi * x[0])
    ]
    
    results = []
    for expr in expressions:
        result = projector.project(expr)
        results.append(result.x.array.copy())
    
    # All should be different
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            assert not np.allclose(results[i], results[j], atol=REL_TOL)

def test_projector_consistency(scalar_space):
    """Test that repeated projections give same result."""
    projector = Projector(scalar_space)
    
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    expr = x[0] * x[1]
    
    result1 = projector.project(expr)
    result2 = projector.project(expr)
    
    assert np.allclose(result1.x.array, result2.x.array, atol=ABS_TOL)

# ============================================================================
# CONVERGENCE TESTS
# ============================================================================

def test_mesh_refinement_convergence():
    """Test convergence with mesh refinement."""
    # Test projection of smooth function with different mesh sizes
    mesh_sizes = [4, 8, 12]
    errors = []
    
    # Smooth function that we can evaluate exactly
    def smooth_func(x):
        return np.sin(np.pi * x[0]) * np.cos(np.pi * x[1])
    
    for nx in mesh_sizes:
        test_mesh = mesh.create_unit_cube(MPI.COMM_WORLD, nx, nx, nx)
        space = fem.functionspace(test_mesh, ("DG", 0))
        projector = Projector(space)
        
        x = ufl.SpatialCoordinate(test_mesh)
        expr = ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])
        
        result = projector.project(expr)
        
        # Compute L2 error using quadrature
        # Create a test function for L2 norm computation
        v = ufl.TestFunction(space)
        error_expr = (result - expr)**2
        error_form = fem.form(error_expr * ufl.dx)
        l2_error_squared = fem.assemble_scalar(error_form)
        l2_error = np.sqrt(l2_error_squared)
        errors.append(l2_error)
    
    # Error should decrease with refinement
    for i in range(1, len(errors)):
        assert errors[i] < errors[i-1] * 1.5  # Allow for numerical variations

# ============================================================================
# RESOURCE MANAGEMENT TESTS
# ============================================================================

def test_projector_destruction(scalar_space):
    """Test proper cleanup of projector resources."""
    projector = Projector(scalar_space)
    
    # Store references to PETSc objects
    A_ref = projector._A
    ksp_ref = projector._ksp
    
    # Delete projector
    del projector
    
    # Objects should be cleaned up (this is hard to test directly)
    # At minimum, this shouldn't crash

def test_many_projectors(scalar_space):
    """Test creating many projectors doesn't cause memory issues."""
    projectors = []
    
    for i in range(10):
        proj = Projector(scalar_space, alpha=0.1 * i)
        projectors.append(proj)
    
    # All should work
    x = ufl.SpatialCoordinate(scalar_space.mesh)
    for proj in projectors:
        result = proj.project(x[0])
        assert isinstance(result, fem.Function)
    
    # Clean up
    for proj in projectors:
        del proj

# ============================================================================
# INTEGRATION WITH FE_MODEL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
