from typing import List, Tuple

from mpi4py import MPI
from dolfinx import fem, la, mesh, default_scalar_type
from dolfinx.fem import FunctionSpace
from petsc4py import PETSc
import numpy as np
import ufl

dtype = PETSc.ScalarType

def build_nullspace(V: FunctionSpace):
    """6-vector PETSc nullspace for 3D elasticity (rigid-body modes)."""
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
    """Tag unit-cube facets: x=0→1, x=1→2, y=0→3, y=1→4."""
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
    """Homogeneous Dirichlet BCs on tagged facets."""
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(id_tag)
    bcs = []
    for i in range(V.mesh.geometry.dim):
        Vi = V.sub(i)
        dofs = fem.locate_dofs_topological(Vi, fdim, facets)
        bcs.append(fem.dirichletbc(default_scalar_type(value), dofs, Vi))
    return bcs

def assign(f: fem.Function, v, *, scatter: bool = True) -> None:
    """Assign value to owned DOFs of a Function.

    Args:
        f: Target function.
        v: Source value (scalar, array-like, or Function).
        scatter: If True, update ghosts via scatter_forward().
    """
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
    if scatter:
        f.x.scatter_forward()

def get_owned_size(field: fem.Function) -> int:
    """Number of locally owned DOFs."""
    return int(field.function_space.dofmap.index_map.size_local * field.function_space.dofmap.index_map_bs)


def field_stats(field: fem.Function, comm: MPI.Comm) -> Tuple[float, float, float]:
    """Global min, max, mean of owned DOFs."""
    n_owned = get_owned_size(field)
    local_data = field.x.array[:n_owned]
    
    if local_data.size > 0:
        local_min = float(local_data.min())
        local_max = float(local_data.max())
        local_sum = float(local_data.sum())
    else:
        local_min = float("inf")
        local_max = float("-inf")
        local_sum = 0.0
    
    global_min = comm.allreduce(local_min, op=MPI.MIN)
    global_max = comm.allreduce(local_max, op=MPI.MAX)
    global_sum = comm.allreduce(local_sum, op=MPI.SUM)
    global_count = comm.allreduce(n_owned, op=MPI.SUM)
    
    global_mean = global_sum / global_count if global_count > 0 else 0.0
    return global_min, global_max, global_mean


def collect_dirichlet_dofs(bcs, n_owned: int) -> np.ndarray:
    """Unique owned DOF indices from BCs."""
    chunks = []
    for bc in bcs:
        idx, first_ghost = bc.dof_indices()
        owned = idx[:first_ghost]
        if owned.size:
            chunks.append(owned.astype(np.int64, copy=False))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))

def smooth_abs(x, eps=1e-4):
    """C¹ approximation of |x|: sqrt(x² + eps²) - eps."""
    return ufl.sqrt(x**2 + eps**2) - eps

def smooth_plus(x, eps=1e-4):
    """C¹ approximation of max(x, 0)."""
    return 0.5 * (x + smooth_abs(x, eps))

def smooth_max(x, y, eps=1e-4):
    """C¹ approximation of max(x, y)."""
    return 0.5 * (x + y + smooth_abs(x - y, eps))


def smoothstep01(t):
    """Cubic smoothstep on [0,1] with hard clamping.

    Returns 0 for t<=0, 1 for t>=1, and t^2(3-2t) in between.
    """
    t_clamped = ufl.conditional(ufl.le(t, 0.0), 0.0, ufl.conditional(ufl.ge(t, 1.0), 1.0, t))
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)


# ---------------------------------------------------------------------------
# UFL tensor utilities
# ---------------------------------------------------------------------------

def clamp(x, a, b):
    """UFL clamp: max(a, min(x, b))."""
    return ufl.conditional(ufl.lt(x, a), a, ufl.conditional(ufl.gt(x, b), b, x))


def symm(X):
    """Symmetric part of a tensor: (X + X^T) / 2."""
    return 0.5 * (X + ufl.transpose(X))


def eigenvalues_sym3(X, *, eps_p: float = 1e-18, eps_r: float = 1e-12, tol: float = 1e-14):
    """Eigenvalues of a symmetric 3×3 tensor via invariant formula (robust, no eigenvectors)."""
    Xs = symm(X)
    I = ufl.Identity(3)

    q = ufl.tr(Xs) / 3.0
    B = Xs - q * I

    p2 = ufl.tr(ufl.dot(B, B)) / 6.0
    # Scale-aware isotropy detection (relative to tensor magnitude)
    scale2 = ufl.max_value(q * q + p2, 1.0)
    iso = ufl.lt(p2, tol * scale2)

    p = ufl.sqrt(ufl.max_value(p2, eps_p * scale2))
    r = ufl.det(B) / (2.0 * p * p * p)
    r_clamped = clamp(r, -1.0 + eps_r, 1.0 - eps_r)

    phi = ufl.acos(r_clamped) / 3.0
    two_pi_over_3 = 2.0 * ufl.pi / 3.0

    l1_raw = q + 2.0 * p * ufl.cos(phi)
    l2_raw = q + 2.0 * p * ufl.cos(phi + two_pi_over_3)
    l3_raw = q + 2.0 * p * ufl.cos(phi + 2.0 * two_pi_over_3)

    l1 = ufl.conditional(iso, q, l1_raw)
    l2 = ufl.conditional(iso, q, l2_raw)
    l3 = ufl.conditional(iso, q, l3_raw)
    return l1, l2, l3


def projectors_sylvester(X, l1, l2, l3, *, eps_d: float = 1e-12, tol: float = 1e-14):
    """Spectral projectors for a symmetric 3×3 tensor using Sylvester formula (robust denominators)."""
    Xs = symm(X)
    I = ufl.Identity(3)

    q = ufl.tr(Xs) / 3.0
    B = Xs - q * I
    p2 = ufl.tr(ufl.dot(B, B)) / 6.0
    scale2 = ufl.max_value(q * q + p2, 1.0)
    iso = ufl.lt(p2, tol * scale2)
    eps_d_scaled = eps_d * scale2

    def _sign(a):
        return ufl.conditional(ufl.ge(a, 0.0), 1.0, -1.0)

    def _safe_denom(a):
        abs_a = ufl.sqrt(a * a)
        return _sign(a) * ufl.max_value(abs_a, eps_d_scaled)

    def _cond_tensor(A_true, A_false):
        return ufl.as_tensor([[ufl.conditional(iso, A_true[i, j], A_false[i, j]) for j in range(3)] for i in range(3)])

    X_l2 = Xs - l2 * I
    X_l3 = Xs - l3 * I
    X_l1 = Xs - l1 * I

    P1_raw = ufl.dot(X_l2, X_l3) / _safe_denom((l1 - l2) * (l1 - l3))
    P2_raw = ufl.dot(X_l1, X_l3) / _safe_denom((l2 - l1) * (l2 - l3))
    P3_raw = ufl.dot(X_l1, X_l2) / _safe_denom((l3 - l1) * (l3 - l2))

    P1_raw = symm(P1_raw)
    P2_raw = symm(P2_raw)
    P3_raw = symm(P3_raw)

    I3 = I / 3.0
    P1 = _cond_tensor(I3, P1_raw)
    P2 = _cond_tensor(I3, P2_raw)
    P3 = _cond_tensor(I3, P3_raw)
    return P1, P2, P3

