from typing import List
import resource

from dolfinx import fem, la, mesh, default_scalar_type
from dolfinx.fem import FunctionSpace
from petsc4py import PETSc
import numpy as np
import ufl

dtype = PETSc.ScalarType

def build_nullspace(V: FunctionSpace):
    """Build 6-vector PETSc nullspace for 3D elasticity (rigid-body modes)."""
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
    """Tag unit-cube boundary facets: x=0→1, x=1→2, y=0→3, y=1→4."""
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
    """Create homogeneous Dirichlet BCs on facets with given tag."""
    fdim = V.mesh.topology.dim - 1
    facets = facet_tags.find(id_tag)
    bcs = []
    for i in range(V.mesh.geometry.dim):
        Vi = V.sub(i)
        dofs = fem.locate_dofs_topological(Vi, fdim, facets)
        bcs.append(fem.dirichletbc(default_scalar_type(value), dofs, Vi))
    return bcs

def assign(f: fem.Function, v) -> None:
    """Assign scalar/array/Function to owned DOFs, then scatter."""
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
    """Count of locally owned DOFs."""
    return int(field.function_space.dofmap.index_map.size_local * field.function_space.dofmap.index_map_bs)

def collect_dirichlet_dofs(bcs, n_owned: int) -> np.ndarray:
    """Unique owned DOF indices from Dirichlet BCs."""
    chunks = []
    for bc in bcs:
        idx, first_ghost = bc.dof_indices()
        owned = idx[:first_ghost]
        if owned.size:
            chunks.append(owned.astype(np.int64, copy=False))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(chunks))


def current_memory_mb() -> float:
    """Process RSS memory in MB."""
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return mem_kb / 1024.0

def smooth_abs(x, eps=1e-4):
    """Smooth |x| approximation: sqrt(x² + eps²) - eps."""
    return ufl.sqrt(x**2 + eps**2) - eps

def smooth_plus(x, eps=1e-4):
    """Smooth max(x, 0) approximation."""
    return 0.5 * (x + smooth_abs(x, eps))

def smooth_max(x, y, eps=1e-4):
    """Smooth max(x, y) approximation."""
    return 0.5 * (x + y + smooth_abs(x - y, eps))


