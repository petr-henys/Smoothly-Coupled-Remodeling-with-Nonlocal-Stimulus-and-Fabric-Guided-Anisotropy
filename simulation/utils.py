"""Utility functions: nullspace, projection, Dirichlet BCs, field operations, memory tracking."""

from typing import List, Tuple
import resource

from dolfinx import fem, la, mesh, default_scalar_type
from dolfinx.fem import FunctionSpace
from petsc4py import PETSc
import numpy as np
from mpi4py import MPI
import ufl

dtype = PETSc.ScalarType


def build_nullspace(V: FunctionSpace):
    """Build PETSc nullspace for 3D elasticity (3 translations + 3 rotations)."""
    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)]
    b = [b.array for b in basis]

    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    for i in range(3):
        b[i][dofs[i]] = 1.0

    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm) for x in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)


def build_facetag(m: mesh.Mesh) -> mesh.MeshTags:
    """Create boundary facet tags for unit-cube domains (MPI-safe)."""
    boundaries = [
        (1, lambda x: np.isclose(x[0], 0)),
        (2, lambda x: np.isclose(x[0], 1)),
        (3, lambda x: np.isclose(x[1], 0)),
        (4, lambda x: np.isclose(x[1], 1)),
    ]
    fdim = m.topology.dim - 1
    facet_indices, facet_markers = [], []
    for marker, locator in boundaries:
        facets = mesh.locate_entities(m, fdim, locator)
        facet_indices.append(facets)
        facet_markers.append(np.full_like(facets, marker))
    facet_indices = np.hstack(facet_indices).astype(np.int32)
    facet_markers = np.hstack(facet_markers).astype(np.int32)
    sorted_facets = np.argsort(facet_indices)
    m.topology.create_connectivity(m.topology.dim - 1, m.topology.dim)
    return mesh.meshtags(
        m, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
    )

def build_dirichlet_bcs(
    V: fem.FunctionSpace, facet_tags: mesh.MeshTags, id_tag: int, value: float = 0.0
) -> List[fem.DirichletBC]:
    """Homogeneous Dirichlet BCs on all components of V for facets tagged id_tag."""
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(id_tag)
    bcs = []
    for i in range(V.mesh.geometry.dim):
        Vi = V.sub(i)
        dofs = fem.locate_dofs_topological(Vi, fdim, facets)
        bcs.append(fem.dirichletbc(default_scalar_type(value), dofs, Vi))
    return bcs

def assign(f: fem.Function, v) -> None:
    """Assign scalar or array to owned DOFs and scatter forward."""
    owned = f.function_space.dofmap.index_map.size_local * f.function_space.dofmap.index_map_bs
    if isinstance(v, fem.Function):
        f.x.array[:owned] = v.x.array[:owned]
    else:
        arr = np.asarray(v, dtype=f.x.array.dtype)
        if arr.size == 1:
            f.x.array[:owned] = arr.item()
        else:
            if arr.size != owned:
                raise ValueError(f"assign: size mismatch (got {arr.size}, need {owned})")
            f.x.array[:owned] = arr.ravel()
    f.x.scatter_forward()

def get_owned_size(field: fem.Function) -> int:
    """Count of locally owned scalar DOFs."""
    return int(field.function_space.dofmap.index_map.size_local * field.function_space.dofmap.index_map_bs)

def collect_dirichlet_dofs(bcs, n_owned: int) -> np.ndarray:
    """Unique owned DOF indices from list of DirichletBC objects."""
    chunks = []
    for bc in bcs:
        idx, first_ghost = bc.dof_indices()
        owned = idx[:first_ghost]
        if owned.size:
            chunks.append(owned.astype(np.int64, copy=False))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def _global_dot(comm: MPI.Comm, a: np.ndarray, b: np.ndarray) -> float:
    """MPI-global dot product."""
    return comm.allreduce(float(a @ b), op=MPI.SUM)

def _global_norm(comm: MPI.Comm, v: np.ndarray) -> float:
    """MPI-global L2 norm."""
    return _global_dot(comm, v, v) ** 0.5

def current_memory_mb() -> float:
    """Current process resident memory (RSS) in MB."""
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return mem_kb / 1024.0

def spectral_decomposition_3x3(A):
    """
    Compute eigenvalues of a 3x3 symmetric tensor A using Cardano's formula.
    Returns (lambda1, lambda2, lambda3).
    """
    # Invariants
    I1 = ufl.tr(A)
    I2 = 0.5 * (I1**2 - ufl.tr(A*A))
    I3 = ufl.det(A)
    
    # Depressed cubic coefficients
    p = I2 - I1**2 / 3.0
    q = -2.0 * I1**3 / 27.0 + I1 * I2 / 3.0 - I3
    
    # Trigonometric solution
    # Ensure argument for acos is in [-1, 1]
    eps = 1e-16
    p_safe = ufl.min_value(p, -eps) 
    
    r = ufl.sqrt(-p_safe / 3.0)
    phi_arg = 3.0 * q / (2.0 * p_safe) * ufl.sqrt(-3.0 / p_safe)
    phi_arg_clamped = ufl.max_value(-1.0, ufl.min_value(1.0, phi_arg))
    phi = ufl.acos(phi_arg_clamped) / 3.0
    
    eig1 = 2.0 * r * ufl.cos(phi) + I1 / 3.0
    eig2 = 2.0 * r * ufl.cos(phi + 2.0 * ufl.pi / 3.0) + I1 / 3.0
    eig3 = 2.0 * r * ufl.cos(phi + 4.0 * ufl.pi / 3.0) + I1 / 3.0
    
    return eig1, eig2, eig3

def matrix_function_3x3(A, func):
    """
    Compute f(A) for a scalar function f using Lagrange interpolation (Sylvester's formula).
    f(A) = sum f(li) * product_{j!=i} (A - lj I) / (li - lj)
    """
    l1, l2, l3 = spectral_decomposition_3x3(A)
    f1, f2, f3 = func(l1), func(l2), func(l3)
    
    # Regularized denominators to handle repeated eigenvalues
    eps = 1e-5
    
    def safe_denom(d):
        # If d is small, return eps with sign of d (or 1 if d=0)
        return ufl.conditional(ufl.lt(abs(d), eps), eps, d)
        
    d12 = safe_denom(l1 - l2)
    d13 = safe_denom(l1 - l3)
    d23 = safe_denom(l2 - l3)
    
    I = ufl.Identity(3)
    
    P1 = (A - l2*I) * (A - l3*I) / (d12 * d13)
    P2 = (A - l1*I) * (A - l3*I) / (-d12 * d23)
    P3 = (A - l1*I) * (A - l2*I) / (-d13 * -d23)
    
    return f1 * P1 + f2 * P2 + f3 * P3

def matrix_exp(A):
    """Matrix exponential exp(A)."""
    return matrix_function_3x3(A, ufl.exp)

def matrix_ln(A):
    """Matrix logarithm ln(A)."""
    return matrix_function_3x3(A, ufl.ln)


def unittrace_psd(B, dim: int, eps: float):
    """Project PSD tensor B to unit-trace via B + εI."""
    I = ufl.Identity(dim)
    M = B + eps * I
    return M / ufl.tr(M)


# --- Smooth regularization helpers (C^∞ approximations) ---

def smooth_abs(x, eps: float):
    """C^∞ approximation of |x|."""
    return ufl.sqrt(x * x + eps * eps)


def smooth_plus(x, eps: float):
    """C^∞ approximation of max(x, 0)."""
    sabs = smooth_abs(x, eps)
    return 0.5 * (x + sabs)


def smooth_max(x, xmin, eps: float):
    """C^∞ approximation of max(x, xmin)."""
    dx = x - xmin
    return xmin + 0.5 * (dx + ufl.sqrt(dx * dx + eps * eps))


def smooth_heaviside(x, eps: float):
    """C^∞ approximation of step function H(x)."""
    return 0.5 * (1.0 + x / ufl.sqrt(x * x + eps * eps))
