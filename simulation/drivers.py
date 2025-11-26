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
    """Solves mechanics for discrete load stages, accumulates weighted stimulus."""

    def __init__(
        self,
        mech: MechanicsSolver,
        mag_hip: fem.Constant,
        vec_glmed: fem.Constant,
        load_stages: List[Dict],
        config: Config,
        css_transformer: object,
    ):
        """Initialize gait driver.
        
        Parameters
        ----------
        mech : MechanicsSolver
            Mechanics solver instance.
        mag_hip : fem.Constant
            Hip load magnitude constant (scalar pressure).
        vec_glmed : fem.Constant
            Gluteus medius traction vector constant.
        load_stages : List[Dict]
            Stage defs with: hip_magnitude, gl_magnitude, gl_vector_css, weight.
        config : Config
            Simulation configuration.
        css_transformer : object
            Object with css_to_world_vector(v) method, or None for identity transform.
        """
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

        # Physical parameters
        self.m_exp = float(config.n_power)
        self.n_cycles = float(config.gait_cycles_per_day)
        self.k_stimulus = float(config.k_stimulus)

        # Stimulus field (DG0 - cell-constant)
        self.V_psi = fem.functionspace(mesh, ("DG", 0))
        self.psi = fem.Function(self.V_psi, name="psi")
        self._psi_temp = fem.Function(self.V_psi)  # Reusable temp buffer
        self._n_owned = self.V_psi.dofmap.index_map.size_local

        # Compile stimulus expression
        self._weight_const = fem.Constant(mesh, 0.0)
        self._stimulus_expr = self._build_stimulus_expression()

        if self.rank == 0:
            self.logger.info(f"GaitDriver: {len(self.stages)} stages, m={self.m_exp:.1f}")

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
        """Solve mechanics for each stage and accumulate stimulus."""
        times = []
        iters = []

        # Reset stimulus (owned DOFs only - ghosts updated at end)
        self.psi.x.array[:self._n_owned] = 0.0

        for stage, weight in zip(self.stages, self.weights):
            start = MPI.Wtime()

            self._apply_stage_loads(stage)
            self.mech.assemble_rhs()
            its, _ = self.mech.solve()

            elapsed = self.comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
            times.append(float(elapsed))
            iters.append(int(its))

            # Accumulate: psi += weight * n_cycles * sigma_tissue^m
            self._weight_const.value = weight * self.n_cycles
            self._accumulate_stimulus()

        # Finalize: psi = (sum)^(1/m)
        self._finalize_stimulus()

        return {
            "phase_iters": iters,
            "phase_times": times,
            "total_time": sum(times),
        }

    def stimulus_expr(self) -> fem.Function:
        """Return computed stimulus field."""
        return self.psi

    def get_stimulus_stats(self) -> Dict[str, float]:
        """Compute MPI-reduced stimulus statistics."""
        local_vals = self.psi.x.array[:self._n_owned]

        # Local aggregates
        n_local = local_vals.size
        if n_local > 0:
            local_min = float(np.min(local_vals))
            local_max = float(np.max(local_vals))
            local_sum = float(np.sum(local_vals))
        else:
            local_min = float("inf")
            local_max = float("-inf")
            local_sum = 0.0

        # Global reductions
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

        # Normalize and scale by magnitude
        norm = np.linalg.norm(gl_vec_world)
        if norm > 1e-12:
            gl_vec_world *= gl_mag / norm
        else:
            gl_vec_world[:] = 0.0

        self.mag_hip.value = hip_mag
        self.vec_glmed.value[:] = gl_vec_world

    def _build_stimulus_expression(self) -> fem.Expression:
        """Build UFL expression for single-stage stimulus contribution."""
        rho = self.mech.rho
        u = self.mech.u
        cfg = self.cfg
        eps = cfg.smooth_eps

        # Safe density and relative density
        rho_safe = ufl.max_value(rho, eps)
        rho_rel = rho_safe / cfg.rho_max

        # Variable stiffness exponent (trabecular -> cortical smoothstep)
        t = ufl.max_value(
            0.0,
            ufl.min_value(
                1.0,
                (rho_safe - cfg.rho_trab_max) / (cfg.rho_cort_min - cfg.rho_trab_max),
            ),
        )
        w = t * t * (3.0 - 2.0 * t)
        k_var = cfg.n_trab * (1.0 - w) + cfg.n_cort * w

        # Elastic modulus
        E_field = cfg.E0 * (rho_rel**k_var)

        # Strain energy density
        sig = self.mech.sigma(u, rho)
        eps_tensor = self.mech.eps(u)
        U = ufl.max_value(0.5 * ufl.inner(sig, eps_tensor), 0.0)

        # Continuum stress from energy: sigma = sqrt(2 * E * U)
        sigma_continuum = ufl.sqrt(2.0 * E_field * U + eps)

        # Tissue-level stress (porosity scaling)
        tissue_scaling = (cfg.rho_max / rho_safe) ** self.k_stimulus
        sigma_tissue = tissue_scaling * sigma_continuum

        # Weighted contribution
        contribution = self._weight_const * (sigma_tissue**self.m_exp)

        return fem.Expression(contribution, self.V_psi.element.interpolation_points)

    def _accumulate_stimulus(self) -> None:
        """Add current stage contribution to stimulus field (owned DOFs)."""
        self._psi_temp.interpolate(self._stimulus_expr)
        self.psi.x.array[:self._n_owned] += self._psi_temp.x.array[:self._n_owned]

    def _finalize_stimulus(self) -> None:
        """Apply power law to accumulated stimulus and sync ghosts."""
        vals = self.psi.x.array[:self._n_owned]
        np.maximum(vals, 0.0, out=vals)
        np.power(vals, 1.0 / self.m_exp, out=vals)
        self.psi.x.scatter_forward()
