import numpy as np
from dolfinx import mesh
from mpi4py import MPI

def make_unit_cube(comm: MPI.Comm, n: int = 6):
    """Create a tiny 3D unit cube mesh."""
    return mesh.create_unit_cube(comm, n, n, n, cell_type=mesh.CellType.hexahedron, ghost_mode=mesh.GhostMode.shared_facet)

def iso_tensor(x):
    """Isotropic unit-trace tensor I/3."""
    base = (np.eye(3) / 3.0).flatten()[:, None]
    return np.tile(base, (1, x.shape[1]))

def fiber_tensor(x):
    """Anisotropic unit-trace tensor with fiber in x-direction."""
    mat = np.array([[0.92, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]], dtype=float)
    return np.tile(mat.flatten()[:, None], (1, x.shape[1]))
