"""Gait driver: solves mechanics for load stages and computes daily stimulus."""

from typing import Dict, List
import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.logger import get_logger


class GaitDriver:
    """Solves mechanics for discrete load stages, accumulates weighted stimulus (SED)."""

    def __init__(
        self,
        mech: MechanicsSolver,
        mag_hip: fem.Constant,
        vec_glmed: fem.Constant,
        load_stages: List[Dict],
        config: Config,
        css_transformer: object,
    ):
        self.mech = mech
        self.mag_hip = mag_hip
        self.vec_glmed = vec_glmed
        self.cfg = config
        self.stages = load_stages
        self.css = css_transformer

        mesh = self.mech.u.function_space.mesh
        self.comm = mesh.comm
        self.rank = self.comm.rank
        self.logger = get_logger(self.comm, name="Driver", log_file=self.cfg.log_file)

        # Normalize weights
        total_weight = sum(s["weight"] for s in self.stages)
        if total_weight <= 0:
            raise ValueError("Total weight of load stages must be positive.")
        self.weights = np.array([s["weight"] / total_weight for s in self.stages])


        # Stimulus field (Strain Energy Density - SED)
        # Using DG0 (element-wise constant) is standard for SED to avoid smoothing artifacts
        self.V_psi = fem.functionspace(mesh, ("DG", 0))
        self.psi = fem.Function(self.V_psi, name="Stimulus_SED")
        
        # Temp buffer for single-stage SED
        self._psi_local = fem.Function(self.V_psi)
        
        # Optimization: Pre-compile the SED expression to avoid recompilation in loops
        self._weight_val = fem.Constant(mesh, 0.0)
        self._sed_expr = self._build_sed_expression()

        if self.rank == 0:
            self.logger.info(f"GaitDriver: {len(self.stages)} stages initialized.")

    def setup(self) -> None:
        """Initialize mechanics solver."""
        self.mech.setup()

    def destroy(self) -> None:
        """Release solver resources."""
        self.mech.destroy()

    def update_stiffness(self) -> None:
        """Reassemble mechanics stiffness matrix."""
        self.mech.assemble_lhs()

    def update_snapshots(self) -> Dict:
        """Solve mechanics for each stage and accumulate Strain Energy Density."""
        times = []
        iters = []

        # 1. Reset accumulated stimulus to zero
        self.psi.x.array[:] = 0.0

        # 2. Loop through load stages
        for stage, weight in zip(self.stages, self.weights):
            start = MPI.Wtime()

            # A. Solve Equilibrium
            self._apply_stage_loads(stage)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()

            elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
            times.append(float(elapsed))
            iters.append(int(its))

            # B. Calculate SED for this stage
            # Update the weight constant (Weight)
            self._weight_val.value = weight
            
            # Interpolate current stage SED into temp buffer
            self._psi_local.interpolate(self._sed_expr)
            
            # Accumulate into total stimulus (Vector addition, very fast)
            self.psi.x.array[:] += self._psi_local.x.array[:]

        # 3. Synchronize ghosts after accumulation
        self.psi.x.scatter_forward()

        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": sum(times),
        }

    def stimulus_field(self) -> fem.Function:
        """Return the computed stimulus function (Psi)."""
        return self.psi

    def get_stimulus_stats(self) -> Dict[str, float]:
        """Compute MPI-reduced stimulus statistics."""
        # Calculate on owned dofs to avoid double counting ghosts
        local_vals = self.psi.x.array[:self.psi.function_space.dofmap.index_map.size_local]

        n_local = local_vals.size
        if n_local > 0:
            local_min = float(np.min(local_vals))
            local_max = float(np.max(local_vals))
            local_sum = float(np.sum(local_vals))
        else:
            local_min = float("inf")
            local_max = float("-inf")
            local_sum = 0.0

        psi_min = self.comm.allreduce(local_min, op=MPI.MIN)
        psi_max = self.comm.allreduce(local_max, op=MPI.MAX)
        total_sum = self.comm.allreduce(local_sum, op=MPI.SUM)
        total_count = self.comm.allreduce(n_local, op=MPI.SUM)

        psi_avg = total_sum / total_count if total_count > 0 else 0.0

        return {
            "psi_avg": psi_avg,
            "psi_min": psi_min,
            "psi_max": psi_max,
        }

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _apply_stage_loads(self, stage: Dict) -> None:
        """Update load constants for a stage."""
        hip_mag = float(stage["hip_magnitude"])
        gl_mag = float(stage["gl_magnitude"])
        gl_vec_css = np.asarray(stage["gl_vector_css"], dtype=np.float64)

        # Transform gluteus vector from CSS to world (rank 0 only, then broadcast)
        gl_vec_world = np.empty(3, dtype=np.float64)
        if self.rank == 0:
            if self.css is not None:
                gl_vec_world[:] = self.css.css_to_world_vector(gl_vec_css)
            else:
                gl_vec_world[:] = gl_vec_css
        self.comm.Bcast(gl_vec_world, root=0)

        norm = np.linalg.norm(gl_vec_world)
        if norm > 1e-12:
            gl_vec_world *= gl_mag / norm
        else:
            gl_vec_world[:] = 0.0

        self.mag_hip.value = hip_mag
        self.vec_glmed.value[:] = gl_vec_world

    def _build_sed_expression(self) -> fem.Expression:
        """
        Build UFL expression for Strain Energy Density (SED).
        Psi = 0.5 * sigma : eps
        According to Bensel et al. (2024), Eq 1[cite: 114].
        """
        u = self.mech.u
        rho = self.mech.rho
        
        # Get stress and strain from mechanics solver
        sig = self.mech.sigma(u, rho)
        eps = self.mech.eps(u)
        
        # Strain Energy Density: Psi = 1/2 * inner(sigma, epsilon)
        # Note: Ensure sigma and eps correspond to the same density state
        Psi = 0.5 * ufl.inner(sig, eps)
        
        # Apply weighting for daily accumulation
        # Use max(0, Psi) to ensure numerical stability, though SED >= 0 physically
        weighted_Psi = self._weight_val * ufl.max_value(Psi, 0.0)

        # Compile expression for the Stimulus Function Space (DG0)
        return fem.Expression(weighted_Psi, self.V_psi.element.interpolation_points)