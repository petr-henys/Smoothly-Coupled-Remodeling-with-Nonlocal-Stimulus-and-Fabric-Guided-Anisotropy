"""Gait drivers: solve mechanics for load stages and compute daily stimulus."""

from typing import Dict, List, Optional, Tuple
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

class SimplifiedGaitDriver:
    """
    Simplified remodeling driver using discrete load cases (stages).
    """

    def __init__(
        self, 
        mech: MechanicsSolver, 
        t_hip: fem.Function,
        t_glmed: fem.Function,
        load_stages: List[Dict],
        config: Config
    ):
        self.mech = mech
        self.t_hip = t_hip
        self.t_glmed = t_glmed
        self.cfg = config
        self.comm = self.mech.u.function_space.mesh.comm
        self.logger = get_logger(self.comm, name="Driver", log_file=self.cfg.log_file)

        self.stages = load_stages
        
        total_weight = sum(s["weight"] for s in self.stages)
        if total_weight <= 0:
            raise ValueError("Total weight of load stages must be positive.")
        self.weights = [s["weight"] / total_weight for s in self.stages]
        
        if self.comm.rank == 0:
            self.logger.info("SimplifiedGaitDriver Configuration:")
            for i, s in enumerate(self.stages):
                self.logger.info(f"  Stage {i+1}: Hip Tag={s['hip_tag']}, Gluteus Tag={s['gl_tag']}")

        self.m_exp = float(config.n_power)
        self.n_cycles = float(config.gait_cycles_per_day)
        self.k_stimulus = float(config.k_stimulus)

        V = self.mech.u.function_space
        self.u_snap = [fem.Function(V, name=f"u_snap_{i}") for i in range(len(self.stages))]

        self.stage_loads = self._precompute_loads()
        self._save_load_stages()

        self.psi_expr: Optional[ufl.core.expr.Expr] = None
        self._build_expressions()
        
        self.V_stats = fem.functionspace(self.mech.u.function_space.mesh, ("DG", 0))
        self.psi_stats = fem.Function(self.V_stats)

    def setup(self) -> None:
        self.mech.setup()

    def destroy(self) -> None:
        self.mech.destroy()

    def update_stiffness(self) -> None:
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """
        Solve mechanics for each stage and update snapshots.
        
        Updates the coefficients (u_snap) of the prebuilt UFL expression 
        without rebuilding the expression graph.
        """
        times = []
        iters = []

        for idx, stage_data in enumerate(self.stage_loads):
            start = MPI.Wtime()
            
            # Update BCs (coefficients in L_form)
            self.t_hip.x.array[:] = stage_data[0]
            self.t_glmed.x.array[:] = stage_data[1]
            
            self.t_hip.x.scatter_forward()
            self.t_glmed.x.scatter_forward()
            
            # Solve mechanics (updates self.mech.u)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()
            
            elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
            times.append(float(elapsed))
            iters.append(float(its))

            # Update snapshot coefficient (used in psi_expr)
            self.u_snap[idx].x.array[:] = self.mech.u.x.array
            self.u_snap[idx].x.scatter_forward()

        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": sum(times)
        }

    def stimulus_expr(self) -> ufl.core.expr.Expr:
        return self.psi_expr

    def get_stimulus_stats(self) -> Dict[str, float]:
        psi_expr_compiled = fem.Expression(self.psi_expr, self.V_stats.element.interpolation_points)
        self.psi_stats.interpolate(psi_expr_compiled)
        
        psi_int = self.comm.allreduce(fem.assemble_scalar(fem.form(self.psi_stats * self.cfg.dx)), op=MPI.SUM)
        vol = self.comm.allreduce(fem.assemble_scalar(fem.form(1.0 * self.cfg.dx)), op=MPI.SUM)
        psi_avg = psi_int / vol if vol > 0 else 0.0
        
        # Use only owned DOFs to avoid ghost double-counting
        n_owned = self.V_stats.dofmap.index_map.size_local * self.V_stats.dofmap.index_map_bs
        local_vals = self.psi_stats.x.array[:n_owned]
        local_min = np.min(local_vals) if local_vals.size > 0 else float('inf')
        local_max = np.max(local_vals) if local_vals.size > 0 else float('-inf')
        
        psi_min = self.comm.allreduce(local_min, op=MPI.MIN)
        psi_max = self.comm.allreduce(local_max, op=MPI.MAX)
        
        # Median (approximate via gather to rank 0) - only owned DOFs
        all_data = self.comm.gather(local_vals, root=0)
        psi_median = 0.0
        if self.comm.rank == 0:
            full_data = np.concatenate(all_data)
            if full_data.size > 0:
                psi_median = float(np.median(full_data))
        psi_median = self.comm.bcast(psi_median, root=0)
        
        return {"psi_avg": psi_avg, "psi_min": psi_min, "psi_max": psi_max, "psi_median": psi_median}

    def _precompute_loads(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate traction field arrays for each stage."""
        rank = self.comm.Get_rank()
        
        css = None
        if rank == 0:
            pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
            head_line = load_json_points(FemurPaths.HEAD_LINE_JSON)
            le_me_line = load_json_points(FemurPaths.LE_ME_LINE_JSON)
            css = FemurCSS(pv_mesh, head_line, le_me_line, side='left')
        
        meshtags = self.cfg.facet_tags
        V = self.mech.u.function_space
        
        precomputed = []
        
        for stage in self.stages:
            hip_tag = stage["hip_tag"]
            hip_mag = stage["hip_magnitude"]
            
            gl_tag = stage["gl_tag"]
            gl_mag = stage["gl_magnitude"]
            gl_vec_css = np.array(stage["gl_vector_css"], dtype=float)
            
            gl_vec_world = None
            if rank == 0:
                gl_vec_world = css.css_to_world_vector(gl_vec_css)
            gl_vec_world = self.comm.bcast(gl_vec_world, root=0)
            
            norm = np.linalg.norm(gl_vec_world)
            if norm > 1e-6:
                gl_vec_world = gl_vec_world / norm * gl_mag
            else:
                gl_vec_world = gl_vec_world * 0.0
            
            ds_hip = ufl.Measure("ds", domain=V.mesh, subdomain_data=meshtags, subdomain_id=hip_tag)
            t_hip_func = create_pressure_function(V, ds_hip, hip_mag, blur_radius=5.0)
            
            ds_gl = ufl.Measure("ds", domain=V.mesh, subdomain_data=meshtags, subdomain_id=gl_tag)
            t_glmed_func = create_traction_function(V, ds_gl, gl_vec_world, blur_radius=5.0)
            
            precomputed.append((
                t_hip_func.x.array.copy(),
                t_glmed_func.x.array.copy()
            ))
            
        return precomputed

    def _build_expressions(self) -> None:
        """
        Prebuild the UFL expression for the stimulus.
        
        Constructs the graph linking snapshots (u_snap) and density (rho) to stimulus (psi).
        """
        rho = self.mech.rho
        rho_max = self.cfg.rho_max
        E_max = self.cfg.E0
        
        rho_safe = ufl.max_value(rho, self.cfg.smooth_eps)
        
        def smoothstep(x, edge0, edge1):
            t = ufl.max_value(0.0, ufl.min_value(1.0, (x - edge0) / (edge1 - edge0)))
            return t * t * (3.0 - 2.0 * t)

        w = smoothstep(rho_safe, self.cfg.rho_trab_max, self.cfg.rho_cort_min)
        k_var = self.cfg.n_trab * (1.0 - w) + self.cfg.n_cort * w
        
        # Normalize density for stiffness
        rho_rel = rho_safe / rho_max
        E_field = E_max * (rho_rel**k_var)

        psi_summation = 0.0
        
        for u_i, weight in zip(self.u_snap, self.weights):
            sig_i = self.mech.sigma(u_i, rho)
            e_i = self.mech.get_strain_tensor(u_i)
            U_i = 0.5 * ufl.inner(sig_i, e_i)
            U_safe = ufl.max_value(U_i, 0.0)
            
            sigma_continuum = ufl.sqrt(2.0 * E_field * U_safe + self.cfg.smooth_eps)
            
            # Tissue stress scaling: sigma_tissue = (rho_max / rho) * sigma_continuum
            # This assumes rho_max is the tissue density.
            tissue_scaling = (rho_max / rho_safe)**self.k_stimulus
            sigma_tissue = tissue_scaling * sigma_continuum
            
            n_i = weight * self.n_cycles
            psi_summation += n_i * (sigma_tissue ** self.m_exp)

        self.psi_expr = ufl.max_value(psi_summation, 0.0) ** (1.0 / self.m_exp)

    def _save_load_stages(self) -> None:
        """Save precomputed load stages to VTX for visualization."""
        from dolfinx.io import VTXWriter
        from pathlib import Path
        
        output_dir = Path(self.cfg.results_dir)
        if self.comm.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()
        
        file_path = output_dir / "load_stages.bp"
        
        # Create writer for t_hip and t_glmed
        writer = VTXWriter(self.comm, str(file_path), [self.t_hip, self.t_glmed], engine="bp4")
        
        for i, (hip_vals, gl_vals) in enumerate(self.stage_loads):
            # Update functions
            self.t_hip.x.array[:] = hip_vals
            self.t_glmed.x.array[:] = gl_vals
            self.t_hip.x.scatter_forward()
            self.t_glmed.x.scatter_forward()
            
            # Write with time = stage index
            writer.write(float(i))
            
        writer.close()
        
        # Reset functions to zero to be safe
        self.t_hip.x.array[:] = 0.0
        self.t_glmed.x.array[:] = 0.0
        self.t_hip.x.scatter_forward()
        self.t_glmed.x.scatter_forward()
        
        if self.comm.rank == 0:
            self.logger.info(f"Saved {len(self.stage_loads)} load stages to {file_path}")
