import numpy as np
import ants
from scipy.spatial import KDTree
from dolfinx import fem
from mpi4py import MPI

from simulation.logger import get_logger


def idw(image, mesh, threshold=0., power=2, k_neighbors=8, method='nodes', verbose=True):
    """Interpolate image intensities onto a DOLFINx mesh via inverse-distance weighting.

    Args:
        image: ANTs image.
        mesh: Target DOLFINx mesh.
        threshold: Ignore voxels below this intensity.
        power: IDW power parameter.
        k_neighbors: Number of nearest voxels to use.
        method: `"nodes"` (CG1) or `"centroids"` (DG0).
        verbose: Whether to log basic diagnostics on rank 0.

    Returns:
        DOLFINx function with interpolated values.
    """
    logger = get_logger(mesh.comm, name="IntensityMapper")

    # Determine function space based on method
    if method == 'nodes':
        V = fem.functionspace(mesh, ("Lagrange", 1))
    elif method == 'centroids':
        V = fem.functionspace(mesh, ("DG", 0))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nodes' or 'centroids'.")

    u = fem.Function(V)
    u.name = "interpolated_intensity"

    # Get interpolation points (DOF coordinates)
    points = V.tabulate_dof_coordinates()[:, :3]

    ct_array = image.numpy()
    origin = np.array(image.origin)
    spacing = np.array(image.spacing)

    # Check bounding boxes
    mesh_coords = mesh.geometry.x
    if mesh_coords.shape[0] > 0:
        local_min = mesh_coords.min(axis=0)
        local_max = mesh_coords.max(axis=0)
    else:
        local_min = np.full(3, np.inf)
        local_max = np.full(3, -np.inf)

    comm = mesh.comm
    global_min = np.zeros(3, dtype=float)
    global_max = np.zeros(3, dtype=float)

    comm.Allreduce(local_min, global_min, op=MPI.MIN)
    comm.Allreduce(local_max, global_max, op=MPI.MAX)

    image_min = origin
    image_max = origin + (np.array(ct_array.shape) - 1) * spacing

    if comm.rank == 0:
        logger.info(f"Mesh bbox:    {global_min} - {global_max}")
        logger.info(f"Image bbox:   {image_min} - {image_max}")

        if np.any(global_min < image_min) or np.any(global_max > image_max):
            logger.warning("Mesh extends outside image bounds!")
    
    # Optimization: Filter voxels to local mesh bounding box + buffer
    # This reduces memory usage and KDTree build time on each rank
    buffer = 10.0 * np.max(spacing) # 10 voxels buffer
    
    # Local bounding box (already computed as local_min, local_max)
    search_min = local_min - buffer
    search_max = local_max + buffer
    
    # Create 3D grid of voxel centers
    # We can optimize this by only generating the grid within the search bounds
    # Convert physical bounds to voxel indices
    
    idx_min = np.floor((search_min - origin) / spacing).astype(int)
    idx_max = np.ceil((search_max - origin) / spacing).astype(int)
    
    # Clamp to image bounds
    idx_min = np.maximum(idx_min, 0)
    idx_max = np.minimum(idx_max, np.array(ct_array.shape))
    
    # If the local mesh is completely outside the image (shouldn't happen with valid overlap), handle gracefully
    if np.any(idx_min >= idx_max):
        u.x.array[:] = 0.0
        return u

    # Extract sub-volume for this rank
    sub_ct = ct_array[idx_min[0]:idx_max[0], idx_min[1]:idx_max[1], idx_min[2]:idx_max[2]]
    
    grid_x, grid_y, grid_z = np.mgrid[
        idx_min[0]:idx_max[0], 
        idx_min[1]:idx_max[1], 
        idx_min[2]:idx_max[2]
    ]
    
    indices = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel())).T
    voxel_centers = indices * spacing + origin

    voxel_values = sub_ct.ravel()
    valid_mask = voxel_values >= threshold
    valid_voxel_centers = voxel_centers[valid_mask]
    valid_voxel_values = voxel_values[valid_mask]

    if len(valid_voxel_values) == 0:
        u.x.array[:] = 0.0
        return u

    # Create KD-tree for efficient nearest neighbor search
    tree = KDTree(valid_voxel_centers)
    distances, indices = tree.query(points, k=k_neighbors)
    distances[distances == 0] = 1e-6  # Prevent division by zero

    valid_voxel_neighbors = valid_voxel_values[indices]
    weights = 1.0 / distances**power
    weighted_sum = np.sum(weights * valid_voxel_neighbors, axis=1)
    sum_of_weights = np.sum(weights, axis=1)

    interpolated_values = weighted_sum / sum_of_weights
    
    u.x.array[:] = interpolated_values
    
    return u


if __name__ == "__main__":

    from femur import FEBio2Dolfinx, FemurPaths

    mesh = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB).mesh_dolfinx
    image = ants.image_read('anatomy/raw/proximal_femur/template_new3.nii.gz')

    interpolated_function = idw(image, mesh, threshold=0.2, power=1, k_neighbors=16, method='centroids')
    
    if mesh.comm.rank == 0:
        print(f"Interpolation complete. Function range: [{interpolated_function.x.array.min()}, {interpolated_function.x.array.max()}]")

    # Save as VTK using PyVista
    import pyvista as pv
    from dolfinx import plot

    # Create a CG1 function space for topology extraction, as DG0 spaces don't have vertex-based topology
    V_plot = fem.functionspace(mesh, ("Lagrange", 1))
    topology, cell_types, geometry = plot.vtk_mesh(V_plot)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)

    # Attach data
    values = interpolated_function.x.array.real
    
    if values.size == grid.n_points:
        grid.point_data["Intensity"] = values
    elif values.size == grid.n_cells:
        grid.cell_data["Intensity"] = values
    else:
        if mesh.comm.rank == 0:
            print(f"Warning: Data size {values.size} does not match grid (Points: {grid.n_points}, Cells: {grid.n_cells})")

    # Save separate files for each rank to avoid write conflicts
    filename = f"interpolated_intensity_{mesh.comm.rank:03d}.vtk"
    grid.save(filename)
    print(f"Rank {mesh.comm.rank}: Saved {filename}")
