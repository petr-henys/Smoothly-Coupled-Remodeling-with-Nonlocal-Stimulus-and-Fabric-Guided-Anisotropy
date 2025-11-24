"""
Remodeling drivers: translate mechanics to stimulus ψ(u) and structure M(u).
"""

from typing import Protocol, Dict, List, Optional, Tuple, Any
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl
import pyvista as pv

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.traction_utils import create_traction_function, create_pressure_function
from simulation.logger import get_logger
from simulation.femur_css import FemurCSS, load_json_points
from simulation.paths import FemurPaths

class RemodelingDriver(Protocol):
    """Protocol for drivers that provide mechanical fields to remodeling PDEs."""

    def stimulus_expr(self) -> ufl.core.expr.Expr: ...
    def invalidate(self) -> None: ...
    def update_snapshots(self) -> Optional[Dict]: ...
    def setup(self) -> None: ...
    def destroy(self) -> None: ...
    def update_stiffness(self) -> None: ...
    def get_stimulus_stats(self) -> Dict[str, float]: ...


class SimplifiedGaitDriver:
    """
    Simplified remodeling driver that uses a fixed set of discrete load cases (stages)
    instead of a continuous gait cycle.
    
    Implements the RemodelingDriver protocol.
    """

    def __init__(
        self, 
        mech: MechanicsSolver, 
        t_hip: fem.Function,
        t_glmed: fem.Function,
        load_stages: List[Dict],
        config: Config
    ):
        """
        Args:
            mech: The mechanics solver instance.
            t_hip: The fem.Function used for Hip traction in the solver's Neumann BCs.
            t_glmed: The fem.Function used for GL Medius traction in the solver's Neumann BCs.
            load_stages: A list of dictionaries, each defining a load stage:
                         {
                             "name": "Heel Strike",
                             "weight": 1.0, # Relative frequency/duration
                             "hip_tag": 3, # Surface tag for hip load
                             "hip_magnitude": 1158.0, # Pressure magnitude (MPa)
                             "gl_tag": 4, # Surface tag for gluteus load
                             "gl_magnitude": 351.0, # Traction magnitude (MPa)
                             "gl_vector_css": [x, y, z] # Direction vector in CSS
                         }
            config: Simulation configuration.
        """
        self.mech = mech
        self.t_hip = t_hip
        self.t_glmed = t_glmed
        self.cfg = config
        self.comm = self.mech.u.function_space.mesh.comm
        self.logger = get_logger(self.comm, verbose=(self.cfg.verbose is True), name="SimplifiedDriver")

        self.stages = load_stages
        
        # Validate stages
        total_weight = sum(s.get("weight", 1.0) for s in self.stages)
        if total_weight <= 0:
            raise ValueError("Total weight of load stages must be positive.")
        
        # Normalize weights
        self.weights = [s.get("weight", 1.0) / total_weight for s in self.stages]
        
        # Log the configuration for verification
        if self.comm.rank == 0:
            self.logger.info("SimplifiedGaitDriver Configuration:")
            for i, s in enumerate(self.stages):
                h_tag = s.get("hip_tag", 3)
                g_tag = s.get("gl_tag", 4)
                self.logger.info(f"  Stage {i+1}: Hip Tag={h_tag}, Gluteus Tag={g_tag}")

        # Parameters for stimulus
        self.m_exp = float(config.n_power)
        self.n_cycles = float(config.gait_cycles_per_day)
        self.k_stimulus = 1.0

        # Snapshots for displacement field
        V = self.mech.u.function_space
        self.u_snap = [fem.Function(V, name=f"u_snap_{i}") for i in range(len(self.stages))]

        # Precompute load arrays (in World coordinates)
        self.stage_loads = self._precompute_loads()

        # UFL Expressions
        self.psi_expr: Optional[ufl.core.expr.Expr] = None
        self._build_expressions()
        
        # Stats
        self.V_stats = fem.functionspace(self.mech.u.function_space.mesh, ("DG", 0))
        self.psi_stats = fem.Function(self.V_stats)

    def setup(self) -> None:
        self.mech.setup()

    def destroy(self) -> None:
        self.mech.destroy()

    def update_stiffness(self) -> None:
        self.mech.assemble_lhs()

    def invalidate(self) -> None:
        dirty = False
        if abs(self.m_exp - float(self.cfg.n_power)) > 1e-9:
            self.m_exp = float(self.cfg.n_power)
            dirty = True
        if abs(self.n_cycles - float(self.cfg.gait_cycles_per_day)) > 1e-9:
            self.n_cycles = float(self.cfg.gait_cycles_per_day)
            dirty = True
        if dirty:
            self._build_expressions()

    def update_snapshots(self) -> Dict:
        """Solve mechanics for each stage and update snapshots."""
        times = []
        iters = []

        for idx, stage_data in enumerate(self.stage_loads):
            start = MPI.Wtime()
            
            # 1. Apply loads
            # stage_data is (hip_array, glmed_array)
            self.t_hip.x.array[:] = stage_data[0]
            self.t_glmed.x.array[:] = stage_data[1]
            
            # Update ghosts (though solver usually handles this via scatter, 
            # but we modified local values directly)
            self.t_hip.x.scatter_forward()
            self.t_glmed.x.scatter_forward()
            
            # 2. Solve
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            
            elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
            times.append(float(elapsed))
            iters.append(float(its))

            # 3. Store snapshot
            self.u_snap[idx].x.array[:] = self.mech.u.x.array

        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": sum(times)
        }

    def stimulus_expr(self) -> ufl.core.expr.Expr:
        return self.psi_expr

    def get_stimulus_stats(self) -> Dict[str, float]:
        # Similar to GaitDriver
        psi_expr_compiled = fem.Expression(self.psi_expr, self.V_stats.element.interpolation_points)
        self.psi_stats.interpolate(psi_expr_compiled)
        
        psi_int = self.comm.allreduce(fem.assemble_scalar(fem.form(self.psi_stats * self.cfg.dx)), op=MPI.SUM)
        vol = self.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * self.cfg.dx)), op=MPI.SUM)
        psi_avg = psi_int / vol if vol > 0 else 0.0
        
        local_vals = self.psi_stats.x.array
        local_min = np.min(local_vals) if local_vals.size > 0 else float('inf')
        local_max = np.max(local_vals) if local_vals.size > 0 else float('-inf')
        
        psi_min = self.comm.allreduce(local_min, op=MPI.MIN)
        psi_max = self.comm.allreduce(local_max, op=MPI.MAX)
        
        return {"psi_avg": psi_avg, "psi_min": psi_min, "psi_max": psi_max, "psi_median": 0.0}

    def _precompute_loads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate the traction field arrays for each stage.
        Uses create_traction_function to generate the fields (including blurring),
        then extracts the arrays.
        """
        rank = self.comm.Get_rank()
        
        # Initialize CSS for transformation
        css = None
        if rank == 0:
            try:
                pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
                head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
                le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
                css = FemurCSS(pv_mesh, head_line, le_me_line, side='left')
            except Exception as e:
                self.logger.warning(f"Could not initialize CSS: {e}. Assuming World coordinates.")
        
        # We need the meshtags to identify surfaces
        # Assuming the domain in mech.u has meshtags attached or we can get them.
        # The Config object usually has facet_tags.
        meshtags = self.cfg.facet_tags
        V = self.mech.u.function_space
        
        precomputed = []
        
        for stage in self.stages:
            # Extract parameters
            hip_tag = stage.get("hip_tag", 3)
            hip_mag = stage.get("hip_magnitude", 0.0)
            
            gl_tag = stage.get("gl_tag", 4)
            gl_mag = stage.get("gl_magnitude", 0.0)
            gl_vec_css = np.array(stage.get("gl_vector_css", [0, 0, 1]), dtype=float)
            
            # Transform Gluteus vector
            gl_vec_world = None
            if rank == 0:
                if css:
                    gl_vec_world = css.css_to_world_vector(gl_vec_css)
                else:
                    gl_vec_world = gl_vec_css
            gl_vec_world = self.comm.bcast(gl_vec_world, root=0)
            
            # Scale Gluteus vector by magnitude
            # Assuming gl_vec_css is a direction (unit vector), we multiply by magnitude.
            # If it's already a force vector, magnitude might be redundant or a scaler.
            # The user provided "gl_magnitude" and "TRACTION_CSS".
            # We assume TRACTION_CSS is direction.
            # Normalize direction just in case?
            norm = np.linalg.norm(gl_vec_world)
            if norm > 1e-6:
                gl_vec_world = gl_vec_world / norm * gl_mag
            else:
                gl_vec_world = gl_vec_world * 0.0
            
            # Create Hip Pressure Function
            # Uses create_pressure_function (normal * magnitude)
            t_hip_func = create_pressure_function(V, meshtags, hip_tag, hip_mag, blur_radius=5.0)
            
            # Create Gluteus Traction Function
            t_glmed_func = create_traction_function(V, meshtags, gl_tag, gl_vec_world, blur_radius=5.0)
            
            # Store arrays
            precomputed.append((
                t_hip_func.x.array.copy(),
                t_glmed_func.x.array.copy()
            ))
            
        return precomputed

    def _build_expressions(self) -> None:
        # Same logic as GaitDriver
        rho = self.mech.rho
        rho_max = self.cfg.rho_max
        E_max = self.cfg.E0
        
        rho_safe = ufl.max_value(rho, self.cfg.smooth_eps)
        
        def smoothstep(x, edge0, edge1):
            t = ufl.max_value(0.0, ufl.min_value(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        w = smoothstep(rho_safe, self.cfg.rho_trab_max, self.cfg.rho_cort_min)
        k_var = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w
        
        E_field = E_max * (rho_safe**k_var)

        psi_summation = 0.0
        total_weight = 0.0
        
        for u_i, weight in zip(self.u_snap, self.weights):
            sig_i = self.mech.sigma(u_i, rho)
            e_i = self.mech.get_strain_tensor(u_i)
            U_i = 0.5 * ufl.inner(sig_i, e_i)
            U_safe = ufl.max_value(U_i, 0.0)
            
            sigma_continuum = ufl.sqrt(2.0 * E_field * U_safe + self.cfg.smooth_eps)
            tissue_scaling = (rho_max / rho_safe)**self.k_stimulus
            sigma_tissue = tissue_scaling * sigma_continuum
            
            n_i = weight * self.n_cycles
            psi_summation += n_i * (sigma_tissue ** self.m_exp)
            total_weight += weight

        self.psi_expr = ufl.max_value(psi_summation, 0.0) ** (1.0 / self.m_exp)
