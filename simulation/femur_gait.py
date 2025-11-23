
from typing import List, Tuple, Callable, Optional
import sys
from pathlib import Path
from mpi4py import MPI

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
    """Gait-cycle integrator: accumulates multi-load strain energy for remodelling.

    Units:
    - Coordinates: mm (domain + PyVista mesh)
    - Forces: N (body-mass * g)
    - Tractions: MPa (since 1 N/mm² = 1 MPa)
    """
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
    mass_tonnes : float (handled upstream)
        Body mass in tonnes (0.075 t ≈ 75 kg) used to scale database forces.
    """

    def __init__(
        self,
        t_hip: fem.Function,
        t_glmed: fem.Function,
        t_glmax: fem.Function,
        hip,
        gl_med,
        gl_max,
        hip_gait: Optional[Callable[[float], np.ndarray]],
        glmed_gait: Optional[Callable[[float], np.ndarray]],
        glmax_gait: Optional[Callable[[float], np.ndarray]],
        n_samples: int = 9,
        load_scale: float = 1.0,
        verbose: bool = True
    ):
        if n_samples < 2:
            raise ValueError("n_samples must be at least 2 for trapezoidal quadrature.")
        if load_scale < 0:
            raise ValueError("load_scale must be non-negative.")

        self.t_hip = t_hip
        self.t_glmed = t_glmed
        self.t_glmax = t_glmax
        
        self.hip = hip
        self.gl_med = gl_med
        self.gl_max = gl_max
        
        self.hip_gait = hip_gait
        self.glmed_gait = glmed_gait
        self.glmax_gait = glmax_gait
        
        self.n_samples = int(n_samples)
        self.load_scale = float(load_scale)
        self.coord_scale = 1.0  # Both DOLFINx mesh and PyVista mesh in mm
        self.verbose = verbose
    
    
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
        comm = self.t_hip.function_space.mesh.comm
        rank = comm.Get_rank()
        scale = self.load_scale

        if rank == 0:
            F_hip = self.hip_gait(phase_percent)
            F_glmed = self.glmed_gait(phase_percent)
            F_glmax = self.glmax_gait(phase_percent)
            
            # Apply loads to create interpolators (in MPa = N/mm²)
            self.hip.apply_gaussian_load(force_vector_css=F_hip, sigma_deg=10.0, flip=True)
            self.gl_med.apply_gaussian_load(force_vector_css=F_glmed, sigma=3.0, flip=True)
            self.gl_max.apply_gaussian_load(force_vector_css=F_glmax, sigma=3.0, flip=True)
        
        # Interpolate into DOLFINx functions using distributed strategy
        self._apply_load(self.t_hip, self.hip if rank == 0 else None, scale)
        self._apply_load(self.t_glmed, self.gl_med if rank == 0 else None, scale)
        self._apply_load(self.t_glmax, self.gl_max if rank == 0 else None, scale)

        # Simple diagnostics: warn if all tractions are numerically zero
        for name, t in ("t_hip", self.t_hip), ("t_glmed", self.t_glmed), ("t_glmax", self.t_glmax):
            local_max = np.max(np.abs(t.x.array)) if t.x.array.size > 0 else 0.0
            global_max = comm.allreduce(local_max, op=MPI.MAX)
            
            if rank == 0 and global_max < 1e-14 and self.verbose:
                print(f"[FemurRemodellerGait] Warning: {name} is zero at phase {phase_percent:.1f}% (load_scale={scale}).", flush=True)

    def _apply_load(self, target_func: fem.Function, loader, scale: float):
        """
        Apply load from a loader (only available on rank 0) to a distributed dolfinx function.
        """
        comm = target_func.function_space.mesh.comm
        rank = comm.Get_rank()
        
        # Broadcast the interpolator from rank 0
        interp = None
        if rank == 0:
            if loader is not None and hasattr(loader, '_interp'):
                interp = loader._interp
        
        interp = comm.bcast(interp, root=0)
        
        if interp is None:
             # If loader was not ready or None, do nothing
             return

        V = target_func.function_space
        
        # Identify owned dofs
        n_local = V.dofmap.index_map.size_local
        bs = V.dofmap.index_map_bs
        
        # Get coordinates of owned nodes
        # tabulate_dof_coordinates returns (num_nodes, 3)
        # We assume the first n_local nodes correspond to owned dofs.
        local_coords = V.tabulate_dof_coordinates()[:n_local]
        
        # Compute values locally
        # local_coords is (N, 3)
        vals = interp(local_coords) * scale
        
        # Assign to function (owned dofs only)
        # target_func.x.array is flat. Owned dofs are the first n_local * bs elements.
        target_func.x.array[:n_local * bs] = vals.reshape(-1)
        
        # Update ghosts
        target_func.x.scatter_forward()

    def save_loads_to_vtx(self, output_dir: Path) -> None:
        """Export traction fields for all quadrature phases to VTX (ADIOS2)."""
        from dolfinx.io import VTXWriter
        
        comm = self.t_hip.function_space.mesh.comm
        if comm.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        
        if self.verbose and comm.rank == 0:
            print(f"[FemurRemodellerGait] Exporting load phases to {output_dir}...", flush=True)

        with VTXWriter(comm, output_dir / "loads_gait.bp", [self.t_hip, self.t_glmed, self.t_glmax], engine="BP4") as vtx:
            for phase, _ in self.get_quadrature():
                self.update_loads(phase)
                vtx.write(phase)


def setup_femur_gait_loading(V: fem.FunctionSpace, mass_tonnes: float = 0.075, n_samples: int = 9,
                             load_scale: float = 1.0, verbose: bool = True,
                             debug_export_dir: Optional[Path] = None) -> "FemurRemodellerGait":
    """
    
    This function shows HOW to set up loading. Users must adapt this
    to their own geometry, loading conditions, and data sources.

    Parameters
    ----------
    mass_tonnes : float
        Body mass in tonnes (0.075 t ≈ 75 kg) used to scale gait/muscle forces.
    n_samples : int
        Number of trapezoidal samples over the gait cycle.
    load_scale : float
        Additional global multiplier applied to tractions.
    debug_export_dir : Path, optional
        If provided, exports the load fields for all phases to this directory (VTX/BP4).
    """
    if mass_tonnes <= 0:
        raise ValueError("mass_tonnes must be positive (tonnes).")
    comm = V.mesh.comm
    rank = comm.Get_rank()

    t_hip = fem.Function(V, name="t_hip")
    t_glmed = fem.Function(V, name="t_glmed")
    t_glmax = fem.Function(V, name="t_glmax")

    hip = None
    gl_med = None
    gl_max = None
    hip_gait = None
    gl_med_gait = None
    gl_max_gait = None

    if rank == 0:
        # Load PyVista mesh and build coordinate system
        vtk_path = FemurPaths.FEMUR_MESH_VTK
        pv_mesh = pv.read(str(vtk_path))
        head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
        le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
        css = FemurCSS(pv_mesh, head_line, le_me_line, side='left', verbose=verbose)

        # Hip joint reaction force (from OrthoLoad database)
        hip = HIPJointLoad(pv_mesh, css, use_cell_data=False, verbose=verbose)
        hip_data = parse_hip_file(GaitPaths.HIP99_WALKING)["data"]
        hip_gait = gait_interpolator(orthoload2ISB(hip_data))

        # Muscle forces (from Amiri 2020 dataset)
        g = 9.81  # m/s^2; produces forces in Newtons, traction becomes MPa via N/mm²
        mass_kg = mass_tonnes * 1000.0  # tonne → kg for F = m·a
        F_mag = mass_kg * g
        muscle_data = load_xy_datasets(GaitPaths.AMIRI_EXCEL, flip_y=True)["Dataset_WN"]
        curves = segment_curves_grid(muscle_data, 4, 9)

        # Gluteus medius
        gl_med = MuscleLoad(pv_mesh, css, use_cell_data=False, verbose=verbose)
        gl_med.set_attachment_points(load_json_points(FemurPaths.GL_MED_JSON))
        curve = rescale_curve(curves[0], x_scale=(0, 100), y_scale=(-1, 0.))
        load_vec = np.array([1.1, 1.87, 0.89]) * F_mag
        gl_med_gait = gait_interpolator(build_load(curve, load_vec))

        # Gluteus maximus
        gl_max = MuscleLoad(pv_mesh, css, use_cell_data=False, verbose=verbose)
        gl_max.set_attachment_points(load_json_points(FemurPaths.GL_MAX_JSON))
        curve = rescale_curve(curves[3], x_scale=(0, 100), y_scale=(-1, 0.))
        load_vec = np.array([-0.3, 1.27, 0.39]) * F_mag
        gl_max_gait = gait_interpolator(build_load(curve, load_vec))

    loader = FemurRemodellerGait(
        t_hip=t_hip, t_glmed=t_glmed, t_glmax=t_glmax,
        hip=hip, gl_med=gl_med, gl_max=gl_max, hip_gait=hip_gait,
        glmed_gait=gl_med_gait, glmax_gait=gl_max_gait,
        n_samples=n_samples, load_scale=load_scale, verbose=verbose
    )
    
    if debug_export_dir is not None:
        loader.save_loads_to_vtx(debug_export_dir)
        
    return loader
if __name__ == "__main__":
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    mdl.save_mesh_vtk("tt.vtk")
    domain = mdl.mesh_dolfinx
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, P1_vec)
    cfg = Config(domain=domain)
    gait_loader = setup_femur_gait_loading(V, mass_tonnes=0.075, n_samples=9)
    topology, cells, geometry = plot.vtk_mesh(V)
    grid = pv.UnstructuredGrid(topology, cells, geometry)
    folder = Path("gait_load_outputs")
    folder.mkdir(exist_ok=True)
    for phase, weight in gait_loader.get_quadrature():
        gait_loader.update_loads(phase)
        t_hip_vals = gait_loader.t_hip.x.array.reshape((-1, 3))
        grid["t_hip"] = t_hip_vals
        grid.save(f"{folder}/t_hip_phase_{int(phase):03d}.vtk")
