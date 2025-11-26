"""Utilities for creating traction and pressure functions."""

from dolfinx import fem
import ufl
import numpy as np
from dolfinx.fem.petsc import LinearProblem

def create_traction_function(V, ds, value, blur_radius):
    """Create a vector-valued function for traction boundary conditions.
    
    Args:
        V: Vector function space
        ds: UFL Measure for the boundary (ds(tag))
        value: Vector value [x, y, z]
        blur_radius: Radius for diffusive smoothing of the boundary condition.
    """
    mesh = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dx = ufl.Measure("dx", domain=mesh)
    
    # Regularization parameter (diffusion coefficient)
    alpha = blur_radius**2
    
    val = fem.Constant(mesh, np.array(value, dtype=np.float64))
    
    # Screened Poisson equation: -alpha*div(grad(u)) + u = 0
    # This ensures the solution decays into the volume (boundary layer).
    # Boundary condition: u ~ val on ds
    # Variational form: alpha*(grad(u), grad(v))_dx + (u, v)_dx + penalty*(u - val, v)_ds = 0
    
    penalty = 1.0e6
    
    a = alpha * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.inner(u, v) * dx + penalty * ufl.inner(u, v) * ds
    L = penalty * ufl.inner(val, v) * ds
    
    # Use a unique prefix based on memory address or random to avoid collision if called multiple times
    # But we don't have tag ID here easily. Use a generic prefix.
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "gamg"}, petsc_options_prefix="traction_proj")
    f = problem.solve()
    f.name = "traction"
    
    return f

def create_pressure_function(V, ds, value, blur_radius):
    """Create a vector-valued function for pressure boundary conditions (Normal * value).
    
    Args:
        V: Vector function space
        ds: UFL Measure for the boundary (ds(tag))
        value: Scalar pressure magnitude.
        blur_radius: Radius for diffusive smoothing.
    """
    mesh = V.mesh
    n = ufl.FacetNormal(mesh)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    dx = ufl.Measure("dx", domain=mesh)
    
    alpha = blur_radius**2
    penalty = 1.0e6
    
    # Minimizing: alpha*||grad(u)||^2_dx + ||u||^2_dx + penalty*||u - (-value*n)||^2_ds
    a = alpha * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.inner(u, v) * dx + penalty * ufl.inner(u, v) * ds
    L = penalty * ufl.inner(-value * n, v) * ds
    
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "gamg"}, petsc_options_prefix="pressure_proj")
    f = problem.solve()
    f.name = "pressure"
    
    return f


