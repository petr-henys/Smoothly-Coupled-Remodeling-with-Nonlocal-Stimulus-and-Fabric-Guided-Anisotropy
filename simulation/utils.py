from typing import List, Tuple

from mpi4py import MPI
from dolfinx import fem, la, mesh, default_scalar_type
from dolfinx.fem import FunctionSpace
from petsc4py import PETSc
import numpy as np
import ufl

dtype = PETSc.ScalarType

def build_nullspace(V: FunctionSpace):
    """Constructs rigid-body mode nullspace (3 translations, 3 rotations) for 3D elasticity."""
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
    """Tags unit cube boundaries for testing (x=0,1, y=0,1)."""
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
    """Creates homogeneous Dirichlet boundary conditions on specified facets."""
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(id_tag)
    bcs = []
    for i in range(V.mesh.geometry.dim):
        Vi = V.sub(i)
        dofs = fem.locate_dofs_topological(Vi, fdim, facets)
        bcs.append(fem.dirichletbc(default_scalar_type(value), dofs, Vi))
    return bcs

def assign(f: fem.Function, v, *, scatter: bool = True) -> None:
    """Assigns values to owned DOFs and optionally scatters to ghosts."""
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
    """Returns the number of locally owned degrees of freedom."""
    return int(field.function_space.dofmap.index_map.size_local * field.function_space.dofmap.index_map_bs)


def field_stats(field: fem.Function, comm: MPI.Comm) -> Tuple[float, float, float]:
    """Computes global min, max, and mean of owned DOFs across all ranks."""
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
    """Return unique owned DOF indices from boundary conditions."""
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
    """C¹ approximation of |x|."""
    return ufl.sqrt(x**2 + eps**2) - eps

def smooth_plus(x, eps=1e-4):
    """C¹ approximation of max(x, 0)."""
    return 0.5 * (x + smooth_abs(x, eps))

def smooth_max(x, y, eps=1e-4):
    """C¹ approximation of max(x, y)."""
    return 0.5 * (x + y + smooth_abs(x - y, eps))


def smooth_min(x, y, eps=1e-4):
    """C¹ approximation of min(x, y)."""
    return 0.5 * (x + y - smooth_abs(x - y, eps))


def smooth_clamp(x, min_val, max_val, eps=1e-4):
    """C¹ approximation of clamp(x, min_val, max_val)."""
    return smooth_min(smooth_max(x, min_val, eps), max_val, eps)



def hard_max(x, y):
    """Pointwise max using UFL's built-in max_value (non-smooth)."""
    return ufl.max_value(x, y)


def hard_min(x, y):
    """Pointwise min using UFL's built-in min_value (non-smooth)."""
    return ufl.min_value(x, y)


def hard_clamp(x, min_val, max_val):
    """Pointwise clamp using hard min/max (guaranteed range, non-smooth at bounds)."""
    return hard_min(hard_max(x, min_val), max_val)


def smoothstep01(t, eps=1e-4):
    """Cubic smoothstep on [0,1] with optional smooth clamping width.

    Notes:
      - We intentionally clamp t to [0,1] (smoothly) before evaluating the polynomial.
      - Pass the global smoothing width (cfg.numerics.smooth_eps) to keep behavior consistent.
    """
    t_clamped = smooth_clamp(t, 0.0, 1.0, eps)
    return t_clamped * t_clamped * (3.0 - 2.0 * t_clamped)


# ---------------------------------------------------------------------------
# UFL tensor utilities
# ---------------------------------------------------------------------------

def clamp(x, a, b, eps=1e-4):
    """Backward-compatible alias for smooth_clamp.

    Prefer calling smooth_clamp(...) explicitly in new code.
    """
    return smooth_clamp(x, a, b, eps)


def symm(X):
    """Symmetric part: (X + X^T) / 2."""
    return 0.5 * (X + ufl.transpose(X))


def eigenvalues_sym3(X, *, eps_p: float = 1e-18, eps_r: float = 1e-12, tol: float = 1e-14):
    """Eigenvalues of symmetric 3×3 tensor via invariant formula."""
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
    r_clamped = hard_clamp(r, -1.0 + eps_r, 1.0 - eps_r)

    phi = ufl.acos(r_clamped) / 3.0
    two_pi_over_3 = 2.0 * ufl.pi / 3.0

    l1_raw = q + 2.0 * p * ufl.cos(phi)
    l2_raw = q + 2.0 * p * ufl.cos(phi + two_pi_over_3)
    l3_raw = q + 2.0 * p * ufl.cos(phi + 2.0 * two_pi_over_3)

    l1 = ufl.conditional(iso, q, l1_raw)
    l2 = ufl.conditional(iso, q, l2_raw)
    l3 = ufl.conditional(iso, q, l3_raw)
    return l1, l2, l3


def projectors_sylvester(X, l1, l2, l3, *, eps_d: float = 1e-12, tol: float = 1e-14, tol_deg: float = 1e-8):
    """
    Computes spectral projectors for 3x3 symmetric tensor using Sylvester's formula (robust to degeneracy).

    Key behavior (smooth, no hard isotropy switch in the output):
      - isotropic (all three ~ equal): P1=P2=P3=I/3 (via weights -> w_iso≈1)
      - transversely isotropic (any pair ~ equal): the two projectors in the degenerate subspace are set to 0.5*(I-P_unique)
        so the split is stable and basis-invariant.
      - fully anisotropic: standard Sylvester projectors, with a small correction that enforces P1+P2+P3 = I.
    """
    Xs = symm(X)
    I = ufl.Identity(3)

    # Scale-aware safety for Sylvester denominators (units: eigenvalue^2)
    q = ufl.tr(Xs) / 3.0
    B = Xs - q * I
    p2 = ufl.tr(ufl.dot(B, B)) / 6.0
    scale2 = ufl.max_value(q * q + p2, 1.0)
    eps_d_scaled = eps_d * scale2

    def _sign(a):
        return ufl.conditional(ufl.ge(a, 0.0), 1.0, -1.0)

    def _safe_denom(a):
        abs_a = ufl.sqrt(a * a)  # |a| (piecewise-smooth)
        return _sign(a) * ufl.max_value(abs_a, eps_d_scaled)

    # Raw Sylvester projectors (may be ill-conditioned if eigenvalues are (nearly) degenerate)
    X_l2 = Xs - l2 * I
    X_l3 = Xs - l3 * I
    X_l1 = Xs - l1 * I

    P1_raw = ufl.dot(X_l2, X_l3) / _safe_denom((l1 - l2) * (l1 - l3))
    P2_raw = ufl.dot(X_l1, X_l3) / _safe_denom((l2 - l1) * (l2 - l3))
    P3_raw = ufl.dot(X_l1, X_l2) / _safe_denom((l3 - l1) * (l3 - l2))

    P1_raw = symm(P1_raw)
    P2_raw = symm(P2_raw)
    P3_raw = symm(P3_raw)

    # Enforce partition of unity in the fully anisotropic regime (helps keep downstream formulas well-behaved)
    Psum = P1_raw + P2_raw + P3_raw
    P_fix = (I - Psum) / 3.0
    P1_full = symm(P1_raw + P_fix)
    P2_full = symm(P2_raw + P_fix)
    P3_full = symm(P3_raw + P_fix)

    # Smooth degeneracy indicators: s_ij ~ 1 if li≈lj, ~0 if well-separated
    gap_eps2 = (tol_deg * tol_deg) * scale2  # units: eigenvalue^2
    d12_2 = (l1 - l2) * (l1 - l2)
    d23_2 = (l2 - l3) * (l2 - l3)
    d13_2 = (l1 - l3) * (l1 - l3)

    s12 = gap_eps2 / (d12_2 + gap_eps2)
    s23 = gap_eps2 / (d23_2 + gap_eps2)
    s13 = gap_eps2 / (d13_2 + gap_eps2)

    # Partition weights (sum to 1): full / pair-degenerate / (double-degenerate -> treated as iso here)
    w_full = (1.0 - s12) * (1.0 - s23) * (1.0 - s13)
    w12 = s12 * (1.0 - s23) * (1.0 - s13)
    w23 = s23 * (1.0 - s12) * (1.0 - s13)
    w13 = s13 * (1.0 - s12) * (1.0 - s23)
    w_iso = 1.0 - (w_full + w12 + w23 + w13)

    I3 = I / 3.0

    # TI collapses (unique index is the one NOT in the degenerate pair)
    # pair (2,3): unique is 1
    P1_T23 = P1_full
    P2_T23 = 0.5 * (I - P1_full)
    P3_T23 = 0.5 * (I - P1_full)

    # pair (1,2): unique is 3
    P3_T12 = P3_full
    P1_T12 = 0.5 * (I - P3_full)
    P2_T12 = 0.5 * (I - P3_full)

    # pair (1,3): unique is 2
    P2_T13 = P2_full
    P1_T13 = 0.5 * (I - P2_full)
    P3_T13 = 0.5 * (I - P2_full)

    # Blended projectors (smooth through degeneracy)
    P1_out = w_full * P1_full + w23 * P1_T23 + w12 * P1_T12 + w13 * P1_T13 + w_iso * I3
    P2_out = w_full * P2_full + w23 * P2_T23 + w12 * P2_T12 + w13 * P2_T13 + w_iso * I3
    P3_out = w_full * P3_full + w23 * P3_T23 + w12 * P3_T12 + w13 * P3_T13 + w_iso * I3

    return symm(P1_out), symm(P2_out), symm(P3_out)


def compute_mean_element_length(m: mesh.Mesh) -> float:
    """Computes the mean element length h = mean(V_e^(1/3))."""
    # Create DG0 space for cell-wise quantities
    V_dg = fem.functionspace(m, ("DG", 0))

    # Expression for cell volume
    vol_expr = fem.Expression(ufl.CellVolume(m), V_dg.element.interpolation_points)
    vol_fn = fem.Function(V_dg)
    vol_fn.interpolate(vol_expr)

    # Get owned cell volumes
    # DG0 dofs correspond one-to-one with cells (usually).
    map_bs = V_dg.dofmap.index_map_bs
    num_owned = V_dg.dofmap.index_map.size_local

    # For DG0, block size is 1.
    local_volumes = vol_fn.x.array[: num_owned * map_bs]

    # Compute local sum of lengths (h ~ V^(1/3))
    local_h_sum = np.sum(np.cbrt(local_volumes))
    local_count = local_volumes.size

    # Global reduction
    comm = m.comm
    total_h_sum = comm.allreduce(local_h_sum, op=MPI.SUM)
    total_count = comm.allreduce(local_count, op=MPI.SUM)

    return total_h_sum / total_count if total_count > 0 else 0.0
