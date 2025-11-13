
from typing import List, Tuple, Callable
import sys
from pathlib import Path

# Add repository root to path to allow importing simulation package
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import basix
import numpy as np
from dolfinx import fem, plot

from simulation.config import Config
from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths, GaitPaths
from simulation.femur_loads import (
    HIPJointLoad, gait_interpolator, orthoload2ISB, MuscleLoad, build_load
)
from simulation.process_gait_data import (
    parse_hip_file, load_xy_datasets, segment_curves_grid, rescale_curve
)

from simulation.febio_parser import FEBio2Dolfinx
import pyvista as pv


class FemurRemodellerGait:
    """Gait-cycle integrator: accumulates multi-load strain energy for remodelling."""
    """Concrete implementation of GaitQuadrature using femurloader for gait-phase-dependent loads.
    
    Parameters
    ----------
    t_hip, t_glmed, t_glmax : fem.Function
        Traction vector functions (updated by this loader)
    hip, gl_med, gl_max : HIPJointLoad | MuscleLoad
        femurloader objects with apply_gaussian_load(...) and __call__(points)->(N,3).
    hip_gait, glmed_gait, glmax_gait : Callable[[float], np.ndarray]
        Interpolators that accept gait percentage in [0,100] and return force vectors in CSS.
    n_samples : int
        Number of quadrature points over gait cycle
    load_scale : float
        Load magnitude multiplier
    """

    def __init__(
        self,
        t_hip: fem.Function,
        t_glmed: fem.Function,
        t_glmax: fem.Function,
        hip,
        gl_med,
        gl_max,
        hip_gait: Callable[[float], np.ndarray],
        glmed_gait: Callable[[float], np.ndarray],
        glmax_gait: Callable[[float], np.ndarray],
        n_samples: int = 9,
        load_scale: float = 1.0
    ):
        self.t_hip = t_hip
        self.t_glmed = t_glmed
        self.t_glmax = t_glmax
        
        self.hip = hip
        self.gl_med = gl_med
        self.gl_max = gl_max
        
        self.hip_gait = hip_gait
        self.glmed_gait = glmed_gait
        self.glmax_gait = glmax_gait
        
        self.n_samples = n_samples
        self.load_scale = load_scale
    
    
    def get_quadrature(self) -> List[Tuple[float, float]]:
        """Return trapezoid quadrature over gait cycle [0, 100]%."""
        ps = np.linspace(0.0, 100.0, self.n_samples)
        ws = np.ones(self.n_samples) / (self.n_samples - 1)
        ws[0] *= 0.5
        ws[-1] *= 0.5
        ws = ws / ws.sum()
        return [(float(p), float(w)) for p, w in zip(ps, ws)]
    
    def update_loads(self, phase_percent: float) -> None:
        """Update traction functions to given gait phase [%]."""
        F_hip = self.hip_gait(phase_percent)
        F_glmed = self.glmed_gait(phase_percent)
        F_glmax = self.glmax_gait(phase_percent)
        
        # Apply loads to create interpolators (in Pascals)
        self.hip.apply_gaussian_load(force_vector_css=F_hip, sigma_deg=10.0, flip=True)
        self.gl_med.apply_gaussian_load(force_vector_css=F_glmed, sigma=3.0, flip=False)
        self.gl_max.apply_gaussian_load(force_vector_css=F_glmax, sigma=3.0, flip=False)
        
        scale = self.load_scale
        # Interpolate into DOLFINx functions
        self.t_hip.interpolate(lambda x: self.hip(x.T).T * scale)
        self.t_glmed.interpolate(lambda x: self.gl_med(x.T).T * scale)
        self.t_glmax.interpolate(lambda x: self.gl_max(x.T).T * scale)


def setup_femur_gait_loading(V: fem.FunctionSpace, BW_kg: float = 75.0, n_samples: int = 9
                             ) -> "FemurRemodellerGait":
    """
    
    This function shows HOW to set up loading. Users must adapt this
    to their own geometry, loading conditions, and data sources.
    """
    # Load PyVista mesh and build coordinate system
    vtk_path = FemurPaths.FEMUR_MESH_VTK
    pv_mesh = pv.read(str(vtk_path))
    head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
    le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
    css = FemurCSS(pv_mesh, head_line, le_me_line, side='left')

    # Hip joint reaction force (from OrthoLoad database)
    hip = HIPJointLoad(pv_mesh, css, use_cell_data=False)
    hip_data = parse_hip_file(GaitPaths.HIP99_WALKING)["data"]
    hip_gait = gait_interpolator(orthoload2ISB(hip_data))

    # Muscle forces (from Amiri 2020 dataset)
    F_mag = BW_kg * 9.81
    muscle_data = load_xy_datasets(GaitPaths.AMIRI_EXCEL, flip_y=True)["Dataset_WN"]
    curves = segment_curves_grid(muscle_data, 4, 9)

    # Gluteus medius
    gl_med = MuscleLoad(pv_mesh, css, use_cell_data=False)
    gl_med.set_attachment_points(load_json_points(FemurPaths.GL_MED_JSON))
    curve = rescale_curve(curves[0], x_scale=(0, 100), y_scale=(-1, 0.))
    load_vec = np.array([1.1, 1.87, 0.89]) * F_mag
    gl_med_gait = gait_interpolator(build_load(curve, load_vec))

    # Gluteus maximus
    gl_max = MuscleLoad(pv_mesh, css, use_cell_data=False)
    gl_max.set_attachment_points(load_json_points(FemurPaths.GL_MAX_JSON))
    curve = rescale_curve(curves[3], x_scale=(0, 100), y_scale=(-1, 0.))
    load_vec = np.array([-0.3, 1.27, 0.39]) * F_mag
    gl_max_gait = gait_interpolator(build_load(curve, load_vec))

    t_hip = fem.Function(V, name="t_hip")
    t_glmed = fem.Function(V, name="t_glmed")
    t_glmax = fem.Function(V, name="t_glmax")
    
    return FemurRemodellerGait(
        t_hip=t_hip, t_glmed=t_glmed, t_glmax=t_glmax,
        hip=hip, gl_med=gl_med, gl_max=gl_max, hip_gait=hip_gait,
        glmed_gait=gl_med_gait, glmax_gait=gl_max_gait,
        n_samples=n_samples,
    )
if __name__ == "__main__":
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB, scale=1000.0)  # Convert m to mm
    mdl.save_mesh_vtk("tt.vtk")
    domain = mdl.mesh_dolfinx
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, P1_vec)
    cfg = Config(domain=domain)
    gait_loader = setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)
    topology, cells, geometry = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cells, geometry)
    folder = Path("gait_load_outputs")
    folder.mkdir(exist_ok=True)
    for phase, weight in gait_loader.get_quadrature():
        gait_loader.update_loads(phase)
        t_hip_vals = gait_loader.t_hip.x.array.reshape((-1, 3))
        grid["t_hip"] = t_hip_vals
        grid.save(f"{folder}/t_hip_phase_{int(phase):03d}.vtk")
