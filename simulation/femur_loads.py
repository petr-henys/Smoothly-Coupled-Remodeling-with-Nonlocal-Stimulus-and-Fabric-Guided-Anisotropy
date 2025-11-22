from __future__ import annotations


import sys
from pathlib import Path

# Add repository root to path to allow importing simulation package
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
import pyvista as pv
from mpi4py import MPI
from scipy.interpolate import RBFInterpolator, splprep, splev, interp1d
from scipy.spatial import cKDTree

from simulation.logger import get_logger

if TYPE_CHECKING:
    from .femur_css import FemurCSS

# Module-level logger for standalone functions
_logger = get_logger(MPI.COMM_WORLD, verbose=True, name="femur_loads")
def build_load(gait_data, force_vector):
    gait_cycle = gait_data[:, 0]
    gait_values = gait_data[:, 1]
    full_data = np.zeros((len(gait_cycle), 4))
    full_data[:, 0] = gait_cycle
    full_data[:, 1] = force_vector[0] * gait_values
    full_data[:, 2] = force_vector[1] * gait_values
    full_data[:, 3] = force_vector[2] * gait_values
    return full_data

def vector_from_angles(magnitude: float, alpha_sag: float = 0.0, alpha_front: float = 0.0) -> np.ndarray:
    a_sag, a_front = np.deg2rad([alpha_sag, alpha_front])
    t_sag, t_front = np.tan(a_sag), np.tan(a_front)
    
    denom = np.sqrt(1.0 + t_sag**2 + t_front**2)
    y = magnitude / denom
    vector = np.array([t_sag * y, y, t_front * y])
    _logger.debug(f"vector_from_angles → mag={magnitude}, sag={alpha_sag}, front={alpha_front}, vec={vector}")
    return vector

def gait_interpolator(gait_vs_force: np.ndarray) -> np.ndarray:
    """
    Returns a cubic interpolator for gait vs force data.
    Input: (N,4) array of [gait, Fx, Fy, Fz].
    Output: callable interpolator: gait -> [Fx, Fy, Fz].
    """
    if not isinstance(gait_vs_force, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if gait_vs_force.ndim != 2 or gait_vs_force.shape[1] != 4:
        raise ValueError("Provide an (N,4) array of [gait, Fx, Fy, Fz].")
    if gait_vs_force.shape[0] < 2:
        raise ValueError("At least two data points are required for interpolation.")

    # Sort by gait cycle/time
    pts = gait_vs_force[gait_vs_force[:, 0].argsort()]
    # +x: anterior->Orthoload y+
    # +z: medial->Orthoload x+
    # +y: superior->Orthoload z+
    t_interp = interp1d(pts[:, 0], pts[:, 1:], axis=0, kind='cubic', fill_value='extrapolate')
    return t_interp

def orthoload2ISB(points: np.ndarray) -> np.ndarray:

    return points[:, [0, 2, 3, 1]]

class GaussianSurfaceLoad:

    def __init__(self, femur_mesh: pv.PolyData, femur_css: FemurCSS, use_cell_data: bool = True, verbose: bool = True):
        self.css = femur_css
        self.head_center_world = femur_css.fhc
        self.head_radius = femur_css.head_radius
        self._use_cell_data = use_cell_data
        self._interp: Optional[RBFInterpolator] = None
        self._gait_interpolator: Optional[interp1d] = None
        
        # Prepare surface mesh
        # create a class‐named logger for this instance
        self.logger = get_logger(MPI.COMM_WORLD, verbose=verbose, name="GaussianSurfaceLoad")

        self.logger.info(f"Init {self.__class__.__name__} (use_cell_data={use_cell_data})")
        self._setup_mesh(femur_mesh)

    def _setup_mesh(self, femur_mesh: pv.PolyData) -> None:
        """Initialize world and CSS surface meshes."""
        # World mesh
        surf = femur_mesh.extract_surface().compute_cell_sizes(length=False, area=True)
        self.mesh_world = surf
        self.centers_world = surf.cell_centers().points

        # CSS mesh
        surf_css = self.css.forward_transform(surf).compute_cell_sizes(length=False, area=True)
        self.mesh_css = surf_css
        self.centers_css = surf_css.cell_centers().points
        self.areas = surf_css.cell_data['Area']
        self.logger.debug(f"Meshes ready: world_cells={self.mesh_world.n_cells}, css_cells={self.mesh_css.n_cells}")

    def _resolve_force_vector(self, force_vector_css: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
        if force_vector_css is None:
            raise ValueError("Force vector must be provided")
        
        F_css = np.asarray(force_vector_css, dtype=float)

        F_world = self.css.css_to_world_vector(F_css)
        if (norm_world := np.linalg.norm(F_world)) < 1e-8:
            raise ValueError("World force vector magnitude too small")

        return F_world, norm_world

    def _create_traction_mesh(self, traction_vectors: np.ndarray) -> pv.PolyData:
        """Create mesh with traction data and setup interpolator."""
        mesh = self.mesh_world.copy()
        mesh.cell_data.clear()
        mesh.cell_data['traction'] = traction_vectors
        # Setup interpolator
        if self._use_cell_data:
            points, values = self.centers_world, traction_vectors
        else:
            mesh = mesh.cell_data_to_point_data(pass_cell_data=False)
            points, values = mesh.points, mesh.point_data['traction']

        # if need to interpolate surface data, we need linear kernel 
        # and smoothing=0.0 and neighbors=1. works well with dolfinx functions
        # otherwise we get blurred results
        self._interp = RBFInterpolator(points, values, kernel='linear',
                                       smoothing=0.0, neighbors=1)
        self.logger.debug(f"Creating traction mesh ({'cell' if self._use_cell_data else 'point'} data)")
        return mesh

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """
        Interpolate traction at given world-space points.
        """
        if self._interp is None:
            raise RuntimeError("Apply load first before interpolation")
        base = self._interp(points)
        return base
    
    def check_equilibrium(self, traction_vector, F_tot) -> np.ndarray:
        """Check if the total force from traction matches expected total force."""
        if traction_vector.ndim != 2 or traction_vector.shape[1] != 3:
            raise ValueError("Traction vector must be Nx3 array")
        # (Ncells,)
        F = (traction_vector * self.areas[:, None]).sum(axis=0)
        F_total_comp = np.linalg.norm(F)
        # Use relative tolerance to handle floating point precision issues
        rel_tol = max(1e-10, abs(F_tot) * 1e-12)
        if not np.isclose(F_total_comp, F_tot, rtol=1e-10, atol=rel_tol):
            self.logger.warning(f"Force equilibrium check: computed_force={F_total_comp:.6f}, expected={F_tot:.6f}, diff={abs(F_total_comp-F_tot):.2e}")
        else:
            self.logger.debug(f"Equilibrium check: computed_force={F_total_comp:.3f}, expected={F_tot:.3f}")
    
    def _compute_traction(self, weights: np.ndarray, F_norm: float, unit_force: np.ndarray, flip: bool=False) -> np.ndarray:
        """Compute traction vectors and handle force flipping and equilibrium checks."""
        norm_factor = np.sum(weights * self.areas)
        if norm_factor <= 0:
            raise RuntimeError("Invalid normalization")

        traction_magnitudes = F_norm * weights / norm_factor
        traction_vectors = traction_magnitudes[:, None] * unit_force

        if flip:
            traction_vectors = -traction_vectors
            self.logger.info("Force direction flipped")

        self.check_equilibrium(traction_vectors, F_norm)
        return traction_vectors

class HIPJointLoad(GaussianSurfaceLoad):
    """Hip joint load using ray-traced Gaussian distribution."""

    def apply_gaussian_load(self, force_vector_css: Optional[np.ndarray] = None, sigma_deg: float = 10.0, 
                          flip: bool = False) -> pv.PolyData:
        self.logger.info("Applying HIPJointLoad Gaussian")
        # resolve vector but do not flip here (keeps ray‐trace location correct)
        F_world, F_norm = self._resolve_force_vector(force_vector_css)
        unit_force = F_world / F_norm

        # Ray trace to find impact center
        start = self.head_center_world
        end = start + unit_force * (self.head_radius * 1.1)
        self.logger.debug(f"Ray trace start={start}, end={end}")
        _, hits = self.mesh_world.ray_trace(start, end, first_point=True)
        if not hits:
            raise RuntimeError("Ray cast missed surface")
        center_css = self.centers_css[hits[0]]
        self.logger.debug(f"Ray hits indices={hits}, center_css_idx={hits[0]}")

        # Compute Gaussian weights
        sigma_len = np.deg2rad(sigma_deg) * self.head_radius
        distances = np.linalg.norm(self.centers_css - center_css, axis=1)
        weights = np.exp(-0.5 * (distances / sigma_len) ** 2)

        traction_vectors = self._compute_traction(weights, F_norm, unit_force, flip)
        return self._create_traction_mesh(traction_vectors)

class MuscleLoad(GaussianSurfaceLoad):
    """Muscle load distributed along attachment spline."""

    def set_attachment_points(self, points: np.ndarray, degree: int = 3, smooth: float = 0.1) -> None:
        """Define muscle attachment curve from points."""
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")

        # Fit spline
        k = min(degree, len(pts) - 1)
        self._tck, _ = splprep(pts.T, k=k, s=smooth)
        
        # Build curve cache
        u = np.linspace(0, 1, 200)
        curve = np.array(splev(u, self._tck)).T
        
        # Segment lengths for weighting
        seg_lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        self._seg_lengths = np.append(seg_lengths, seg_lengths[-1])
        
        self._curve = curve
        self._tree = cKDTree(curve)
        self.logger.info(f"Muscle curve set: pts={pts.shape[0]}, spline_degree={k}, smooth={smooth}")

    def apply_gaussian_load(self, force_vector_css: Optional[np.ndarray] = None, sigma: float = 3.0, 
                          flip: bool = False) -> pv.PolyData:
        self.logger.info("Applying MuscleLoad Gaussian")
        if not hasattr(self, '_tree'):
            raise RuntimeError("Set attachment points first")

        F_world, F_norm = self._resolve_force_vector(force_vector_css)
        if flip:
            F_world = -F_world
            self.logger.info("Force direction flipped")
        unit_force = F_world / F_norm

        # Find distances to curve and apply Gaussian weighting
        distances, indices = self._tree.query(self.centers_world)
        weights = np.exp(-0.5 * (distances / sigma) ** 2) * self._seg_lengths[indices]
        traction_vectors = self._compute_traction(weights, F_norm, unit_force, flip)
        return self._create_traction_mesh(traction_vectors)