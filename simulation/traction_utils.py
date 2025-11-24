"""Utilities for creating traction and pressure functions."""

from dolfinx import fem
import basix.ufl
import ufl
import numpy as np
from mpi4py import MPI
from dolfinx.fem.petsc import LinearProblem

def create_traction_function(V, meshtags, tag, value, blur_radius=5.0):
    """Create a vector-valued function for traction boundary conditions.
    
    Args:
        V: Vector function space
        meshtags: MeshTags for boundaries
        tag: Surface tag ID
        value: Vector value [x, y, z]
        blur_radius: Ignored in this simplified implementation
    """
    f = fem.Function(V, name=f"traction_{tag}")
    val = np.array(value, dtype=np.float64)
    
    # Set constant value everywhere (masked by ds(tag) in solver)
    # This is efficient and correct for the solver integration
    f.x.array[:] = np.tile(val, len(f.x.array)//3)
    
    return f

def create_pressure_function(V, meshtags, tag, value, blur_radius=5.0):
    """Create a vector-valued function for pressure boundary conditions (Normal * value).
    
    Args:
        V: Vector function space
        meshtags: MeshTags for boundaries
        tag: Surface tag ID
        value: Scalar pressure magnitude (positive = compression/inward?)
               Usually Pressure P means traction t = -P * n.
               If value is magnitude of pressure, then t = -value * n.
        blur_radius: Ignored
    """
    mesh = V.mesh
    n = ufl.FacetNormal(mesh)
    
    # We want t = -value * n
    # If value is positive pressure (compression), t points inwards (-n).
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=meshtags, subdomain_id=tag)
    dx = ufl.Measure("dx", domain=mesh)
    
    # Regularization parameter to extend smoothly into domain
    alpha = 0.1 
    
    a = alpha * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + ufl.inner(u, v) * ds
    L = ufl.inner(-value * n, v) * ds
    
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "cg", "pc_type": "gamg"}, petsc_options_prefix=f"pressure_proj_{tag}")
    f = problem.solve()
    f.name = f"pressure_{tag}"
    
    return f
