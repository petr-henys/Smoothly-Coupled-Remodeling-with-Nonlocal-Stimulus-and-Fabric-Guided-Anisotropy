"""CT-to-mesh intensity mapping utilities.

Provides IDW interpolation, ANTs warp transforms, and cohort statistics export.
"""

import numpy as np
from pathlib import Path
import ants
import database as db
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

from dolfinx import fem
from dolfinx.io import VTXWriter
from mpi4py import MPI

from simulation.logger import get_logger


def _owned_dofs(V: fem.FunctionSpace) -> int:
    dofmap = V.dofmap
    return int(dofmap.index_map.size_local * dofmap.index_map_bs)


def save_intensity_stats_vtx(
    mesh,
    I: np.ndarray,
    out_path: Path,
    *,
    method: str = "nodes",
    engine: str = "bp4",
) -> None:
    """Save per-DOF cohort intensity statistics as a VTX (ADIOS2) dataset.

    Args:
        mesh: DOLFINx mesh.
        I: Array shaped (n_patients, n_local_dofs_with_ghosts) as returned by
           `collect_intensities`.
        out_path: Output path (typically ends with `.bp`).
        method: Must match the discretization used for `I` (currently `nodes` → CG1).
        engine: ADIOS2 engine for VTXWriter.
    """
    logger = get_logger(mesh.comm, name="MorphoMapper.Stats")

    if method == "nodes":
        V = fem.functionspace(mesh, ("Lagrange", 1))
    elif method == "centroids":
        V = fem.functionspace(mesh, ("DG", 0))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nodes' or 'centroids'.")
    n_owned = _owned_dofs(V)

    if I.ndim != 2:
        raise ValueError(f"Expected I to be 2D (n_patients, n_dofs); got shape={I.shape}")
    if I.shape[1] < n_owned:
        raise ValueError(
            f"I has too few DOFs for this mesh/space: I.shape[1]={I.shape[1]} < n_owned={n_owned}"
        )

    I_owned = I[:, :n_owned]
    mean_vals = np.mean(I_owned, axis=0)
    median_vals = np.median(I_owned, axis=0)
    two_sigma_vals = 2.0 * np.std(I_owned, axis=0, ddof=0)
    mean_minus_2sigma_vals = mean_vals - two_sigma_vals
    mean_plus_2sigma_vals = mean_vals + two_sigma_vals

    f_mean = fem.Function(V)
    f_mean.name = "I_mean"
    f_median = fem.Function(V)
    f_median.name = "I_median"
    f_mean_minus_2sigma = fem.Function(V)
    f_mean_minus_2sigma.name = "I_mean_minus_2sigma"
    f_mean_plus_2sigma = fem.Function(V)
    f_mean_plus_2sigma.name = "I_mean_plus_2sigma"

    # Assign owned DOFs only; then scatter to fill ghosts consistently.
    f_mean.x.array[:n_owned] = mean_vals
    f_median.x.array[:n_owned] = median_vals
    f_mean_minus_2sigma.x.array[:n_owned] = mean_minus_2sigma_vals
    f_mean_plus_2sigma.x.array[:n_owned] = mean_plus_2sigma_vals
    f_mean.x.scatter_forward()
    f_median.x.scatter_forward()
    f_mean_minus_2sigma.x.scatter_forward()
    f_mean_plus_2sigma.x.scatter_forward()

    # VTXWriter creates a `.bp` directory on rank 0.
    writer = VTXWriter(
        mesh.comm,
        str(out_path),
        [f_mean, f_median, f_mean_minus_2sigma, f_mean_plus_2sigma],
        engine=engine,
    )
    writer.write(0.0)
    writer.close()

    if mesh.comm.rank == 0:
        logger.info(f"Wrote intensity stats VTX: {out_path}")


def save_intensity_stats_checkpoint(
    mesh,
    I: np.ndarray,
    out_path: Path,
    *,
    method: str = "nodes",
) -> None:
    """Save per-DOF cohort intensity statistics to an adios4dolfinx checkpoint.

    VTX output is great for ParaView, but cannot be read back reliably for
    analysis; this checkpoint format can.
    """
    import adios4dolfinx as adx

    logger = get_logger(mesh.comm, name="MorphoMapper.Stats")

    if method == "nodes":
        V = fem.functionspace(mesh, ("Lagrange", 1))
    elif method == "centroids":
        V = fem.functionspace(mesh, ("DG", 0))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nodes' or 'centroids'.")

    n_owned = _owned_dofs(V)
    if I.ndim != 2:
        raise ValueError(f"Expected I to be 2D (n_patients, n_dofs); got shape={I.shape}")
    if I.shape[1] < n_owned:
        raise ValueError(
            f"I has too few DOFs for this mesh/space: I.shape[1]={I.shape[1]} < n_owned={n_owned}"
        )

    I_owned = I[:, :n_owned]
    mean_vals = np.mean(I_owned, axis=0)
    median_vals = np.median(I_owned, axis=0)
    two_sigma_vals = 2.0 * np.std(I_owned, axis=0, ddof=0)

    f_mean = fem.Function(V)
    f_mean.name = "I_mean"
    f_median = fem.Function(V)
    f_median.name = "I_median"
    f_mean_minus_2sigma = fem.Function(V)
    f_mean_minus_2sigma.name = "I_mean_minus_2sigma"
    f_mean_plus_2sigma = fem.Function(V)
    f_mean_plus_2sigma.name = "I_mean_plus_2sigma"

    f_mean.x.array[:n_owned] = mean_vals
    f_median.x.array[:n_owned] = median_vals
    f_mean_minus_2sigma.x.array[:n_owned] = mean_vals - two_sigma_vals
    f_mean_plus_2sigma.x.array[:n_owned] = mean_vals + two_sigma_vals
    f_mean.x.scatter_forward()
    f_median.x.scatter_forward()
    f_mean_minus_2sigma.x.scatter_forward()
    f_mean_plus_2sigma.x.scatter_forward()

    out_path = Path(out_path)
    if mesh.comm.rank == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.comm.Barrier()

    adx.write_mesh(out_path, mesh)
    adx.write_function(out_path, f_mean, time=0.0)
    adx.write_function(out_path, f_median, time=0.0)
    adx.write_function(out_path, f_mean_minus_2sigma, time=0.0)
    adx.write_function(out_path, f_mean_plus_2sigma, time=0.0)

    if mesh.comm.rank == 0:
        logger.info(f"Wrote intensity stats checkpoint: {out_path}")


def rescale_to_density(
    intensity: fem.Function,
    rho_min: float = 0.1,
    rho_max: float = 2.0,
    *,
    intensity_min: float | None = None,
    intensity_max: float | None = None,
) -> fem.Function:
    """Linearly rescale intensity to density range [rho_min, rho_max]."""
    comm = intensity.function_space.mesh.comm

    if intensity_min is None or intensity_max is None:
        local_min = float(intensity.x.array.min()) if len(intensity.x.array) > 0 else np.inf
        local_max = float(intensity.x.array.max()) if len(intensity.x.array) > 0 else -np.inf

        global_min = comm.allreduce(local_min, op=MPI.MIN)
        global_max = comm.allreduce(local_max, op=MPI.MAX)
    else:
        global_min = float(intensity_min)
        global_max = float(intensity_max)

    rho = fem.Function(intensity.function_space)
    rho.name = "density"

    if global_max > global_min:
        normalized = (intensity.x.array - global_min) / (global_max - global_min)
        rho.x.array[:] = rho_min + normalized * (rho_max - rho_min)
    else:
        rho.x.array[:] = 0.5 * (rho_min + rho_max)

    return rho


def export_vtx(function: fem.Function, output_dir: Path, filename: str = "field", *, engine: str = "bp4") -> None:
    """Export a DOLFINx function to VTX checkpoint format for ParaView visualization."""
    comm = function.function_space.mesh.comm
    output_dir = Path(output_dir)

    if comm.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    vtx_path = output_dir / f"{filename}.bp"
    writer = VTXWriter(comm, str(vtx_path), [function], engine=engine)
    writer.write(0.0)
    writer.close()

    if comm.rank == 0:
        get_logger(comm, name="MorphoMapper").info(f"Exported VTX checkpoint to: {vtx_path}")

def warp_fwd(points, warp_file_fwd, aff_file_fwd):
    """
    Apply forward warping transformation to a set of points using ANTs.
    
    Args:
        points (numpy.ndarray): Array of 3D points to transform (N x 3)
        warp_file_fwd (str): Path to the forward warp transformation file (.nii.gz)
        aff_file_fwd (str): Path to the forward affine transformation file (.mat)
    
    Returns:
        numpy.ndarray: Transformed points in the target space
    """
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return ants.apply_transforms_to_points(dim=3, points=points,
                             transformlist=[warp_file_fwd, aff_file_fwd], 
                             whichtoinvert=[False, False]).values

def warp_inv(points, warp_file_inv, aff_file_fwd):
    """
    Apply inverse warping transformation to a set of points using ANTs.
    
    Args:
        points (numpy.ndarray): Array of 3D points to transform (N x 3)
        warp_file_inv (str): Path to the inverse warp transformation file (.nii.gz)
        aff_file_fwd (str): Path to the forward affine transformation file (.mat)
    
    Returns:
        numpy.ndarray: Inversely transformed points in the original space
    """
    points = pd.DataFrame(points, columns=['x', 'y', 'z'])
    return ants.apply_transforms_to_points(dim=3, points=points,
                             transformlist=[aff_file_fwd, warp_file_inv], 
                             whichtoinvert=[True, False]).values

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

def collect_intensities(
    patient_info,
    database_registration,
    mesh,
    *,
    method: str = "nodes",
    threshold: float = 100.0,
    k_neighbors: int = 128,
    power: float = 1.0,
):
    """Collect per-DOF intensity from warped CT images for all patients.

    Returns:
        Array of shape (n_patients, n_dofs).
    """
    def process_patient(patient_id, mesh):
        """Helper function to process individual patient intensities"""
        patient_folder = database_registration / patient_id
        warped_img = patient_folder / 'warped.nii.gz'
        return idw(ants.image_read(str(warped_img)), mesh, 
                  threshold=threshold, k_neighbors=k_neighbors, power=power,
                  method=method).x.array[:]

    i_array = []
    for patient_id in tqdm(patient_info, desc="Processing Patients"):
        i_array.append(process_patient(patient_id, mesh))
    
    return np.asarray(i_array)

if __name__ == '__main__':
    database_directory = Path('/mnt/database/IOR_femurs')
    database_raw = database_directory / 'raw'
    database_registration = database_directory / 'derived' / 'registrations'
    database_fields = database_directory / 'derived' / 'fields'

    patient_info = db.collect_patient_info(database_raw)
    from femur import FEBio2Dolfinx, FemurPaths

    mesh = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB).mesh_dolfinx

    mapping_method = "nodes"  # or "centroids"
    I = collect_intensities(patient_info, database_registration, mesh, method=mapping_method)
    stats_dir = Path("results/ct_density")
    # Avoid clobbering in MPI runs (np.save is not collective).
    comm = mesh.comm
    # Save cohort statistics:
    # - VTX for ParaView visualization
    # - adios4dolfinx checkpoint for analysis (MPI-independent readback)
    if comm.rank == 0:
        stats_dir.mkdir(parents=True, exist_ok=True)
    comm.Barrier()

    save_intensity_stats_vtx(
        mesh,
        I,
        stats_dir / "population_stats_viz.bp",
        method=mapping_method,
        engine="bp4",
    )
    save_intensity_stats_checkpoint(
        mesh,
        I,
        stats_dir / "population_stats_checkpoint.bp",
        method=mapping_method,
    )
    