from dolfinx import fem, la, mesh, default_scalar_type

from petsc4py import PETSc
import numpy as np
from typing import List, Tuple
from mpi4py import MPI
import resource

dtype = PETSc.ScalarType

def build_nullspace(V: fem.FunctionSpace) -> PETSc.NullSpace:
    """Build PETSc near-nullspace for 3D linear elasticity (6 RBMs).

    Strictly 3D: 3 translations + 3 rotations.
    """
    gdim = V.mesh.geometry.dim
    if gdim != 3:
        raise NotImplementedError("Nullspace is implemented for 3D only.")

    index_map = V.dofmap.index_map
    bs = V.dofmap.index_map_bs
    length0 = index_map.size_local

    basis = [la.vector(index_map, bs=bs, dtype=dtype) for _ in range(6)]
    b = [vec.array for vec in basis]

    # Component dofs (parent indices)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Translations
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Rotations
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = +x0
    b[4][dofs[0]] = +x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = +x1
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)

    basis_petsc = [
        PETSc.Vec().createWithArray(bi[: bs * length0], bsize=bs, comm=V.mesh.comm)
        for bi in b
    ]
    return PETSc.NullSpace().create(vectors=basis_petsc)

def compute_principal_dirs_and_vals_vec(
    A_func: fem.Function,
    V_vec: fem.FunctionSpace,
    Q_sca: fem.FunctionSpace,
) -> Tuple[List[fem.Function], List[fem.Function]]:
    """Nodewise eigen-decomposition of tensor field A via NumPy.

    Returns a pair of lists (eigvecs, eigvals) sorted by descending eigenvalue.
    """
    # Ensure ghosts are up-to-date before reading arrays
    A_func.x.scatter_forward()

    T = A_func.function_space
    msh = T.mesh
    gdim = msh.geometry.dim

    nT = T.dofmap.index_map.size_local
    bsT = T.dofmap.index_map_bs

    nV = V_vec.dofmap.index_map.size_local
    bsV = V_vec.dofmap.index_map_bs
    nQ = Q_sca.dofmap.index_map.size_local
    bsQ = Q_sca.dofmap.index_map_bs

    A_all = A_func.x.array
    A_owned_flat = A_all[: nT * bsT]
    A_owned = A_owned_flat.reshape(nT, gdim, gdim, order="C")
    A_owned = 0.5 * (A_owned + np.swapaxes(A_owned, 1, 2))

    w, V = np.linalg.eigh(A_owned)
    order = np.argsort(w, axis=1)[:, ::-1]
    w_sorted = np.take_along_axis(w, order, axis=1)
    V_sorted = np.take_along_axis(V, order[:, np.newaxis, :], axis=2)

    eigvec_funcs = [fem.Function(V_vec, name=f"A_eigvec_{k+1}") for k in range(gdim)]
    eigval_funcs = [fem.Function(Q_sca, name=f"A_eigval_{k+1}") for k in range(gdim)]

    for k in range(gdim):
        vecs_k = V_sorted[:, :, k]
        arr = eigvec_funcs[k].x.array
        arr[:] = 0.0
        arr[: nV * bsV] = vecs_k.reshape(-1)
        eigvec_funcs[k].x.scatter_forward()

        vals_k = w_sorted[:, k]
        arr = eigval_funcs[k].x.array
        arr[:] = 0.0
        arr[: nQ * bsQ] = vals_k
        eigval_funcs[k].x.scatter_forward()

    for vf in eigvec_funcs:
        vf.x.scatter_forward()
    for lf in eigval_funcs:
        lf.x.scatter_forward()

    return eigvec_funcs, eigval_funcs

def build_facetag(m: mesh.Mesh) -> mesh.MeshTags:
    """Create facet tags for unit-cube-like domains (MPI-safe)."""
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
    """Homogeneous Dirichlet on facets with tag id_tag (MPI-safe, component-wise)."""
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(id_tag)
    bcs = []
    # Use direct subspace dofs; collapse/tuple indexing not needed
    for i in range(V.mesh.geometry.dim):
        Vi = V.sub(i)
        dofs = fem.locate_dofs_topological(Vi, fdim, facets)
        bcs.append(fem.dirichletbc(default_scalar_type(value), dofs, Vi))
    return bcs

def assign(f: fem.Function, v) -> None:
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
    """Return count of locally owned scalar DOFs (including block size)."""
    return int(field.function_space.dofmap.index_map.size_local * field.function_space.dofmap.index_map_bs)

def collect_dirichlet_dofs(bcs, n_owned: int) -> np.ndarray:
    """Return unique owned Dirichlet DOFs (DOLFINx 0.9+)."""
    chunks = []
    for bc in bcs:
        # DOLFINx 0.9+ returns (indices, first_ghost)
        idx, first_ghost = bc.dof_indices()
        owned = idx[:first_ghost]
        if owned.size:
            chunks.append(owned.astype(np.int64, copy=False))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def _global_dot(comm: MPI.Comm, a: np.ndarray, b: np.ndarray) -> float:
    """MPI global dot product over local vectors."""
    return comm.allreduce(float(a @ b), op=MPI.SUM)


def _global_norm(comm: MPI.Comm, v: np.ndarray) -> float:
    return _global_dot(comm, v, v) ** 0.5

def current_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return mem_kb / 1024.0
