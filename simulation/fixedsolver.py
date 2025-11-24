"""Fixed-point solver for coupled mechanics-density problem."""

from __future__ import annotations
from typing import Optional, Dict, List

import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.utils import assign, current_memory_mb
from simulation.logger import get_logger

class FixedPointSolver:
    """
    Simplified fixed-point solver for 2-field coupling (Mechanics <-> Density).
    Mechanics is handled by the driver (which may involve multiple load cases).
    Density is handled by the density solver.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        cfg: Config,
        driver, # SimplifiedGaitDriver
        densolver, # DensitySolver
        rho: fem.Function, 
        rho_old: fem.Function
    ):
        self.comm = comm
        self.cfg = cfg
        self.driver = driver
        self.densolver = densolver
        self.rho = rho
        self.rho_old = rho_old
        
        # Fix verbose logic: if "progressbar", we want clean console (WARNING level)
        verbose = self.cfg.verbose
        is_verbose = False if verbose == "progressbar" else bool(verbose)
        self.logger = get_logger(self.comm, verbose=is_verbose, name="FixedPoint", log_file=self.cfg.log_file)
        
        # Stats
        self.mech_time_total = 0.0
        self.dens_time_total = 0.0
        self.mech_iters_total = 0
        self.subiter_metrics: List[Dict] = []
        
        self.solver_stats = {
            "mech": {"time": 0.0, "iters": 0, "reason": 0},
            "dens": {"time": 0.0, "iters": 0, "reason": 0}
        }

    def run(self, *, progress=None, task_id=None) -> None:
        """Execute fixed-point loop."""
        tol = float(self.cfg.coupling_tol)
        max_subiters = int(self.cfg.max_subiters)
        min_subiters = int(self.cfg.min_subiters)
        
        self.mech_time_total = 0.0
        self.dens_time_total = 0.0
        self.mech_iters_total = 0
        self.subiter_metrics = []
        
        # Reset solver stats for this step
        self.solver_stats["mech"] = {"time": 0.0, "iters": 0, "reason": 0}
        self.solver_stats["dens"] = {"time": 0.0, "iters": 0, "reason": 0}

        rho_prev_iter = fem.Function(self.rho.function_space)
        
        if progress is not None and task_id is not None:
            progress.reset(task_id, total=max_subiters)
            progress.start_task(task_id)

        for itr in range(1, max_subiters + 1):
            assign(rho_prev_iter, self.rho)
            
            # 1. Mechanics (Driver updates snapshots)
            # The driver solves mechanics for all stages and updates u_snap
            self.driver.update_stiffness() # Update E(rho)
            mech_stats = self.driver.update_snapshots()
            
            self.mech_time_total += mech_stats["total_time"]
            self.mech_iters_total += sum(mech_stats["phase_iters"])
            
            self.solver_stats["mech"]["time"] += mech_stats["total_time"]
            self.solver_stats["mech"]["iters"] += int(sum(mech_stats["phase_iters"]))
            
            # 2. Stimulus
            # Driver provides the stimulus expression based on updated snapshots
            psi_expr = self.driver.stimulus_expr()
            
            # 3. Density
            # Update driving force in density solver
            t0 = MPI.Wtime()
            self.densolver.update_driving_force(psi_expr)
            # Re-assemble LHS because reaction_coeff depends on S_driving, which changed.
            # Note: dt is constant within this loop, but S changes.
            self.densolver.assemble_lhs()
            self.densolver.assemble_rhs()
            dens_iters, dens_reason = self.densolver.solve()
            
            elapsed = self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)
            self.dens_time_total += elapsed
            
            self.solver_stats["dens"]["time"] += elapsed
            self.solver_stats["dens"]["iters"] += dens_iters
            self.solver_stats["dens"]["reason"] = dens_reason

            # 4. Convergence Check
            # Norm of (rho - rho_prev_iter)
            diff_local = self.rho.x.array - rho_prev_iter.x.array
            diff_norm_sq = self.comm.allreduce(np.dot(diff_local, diff_local), op=MPI.SUM)
            rho_norm_sq = self.comm.allreduce(np.dot(self.rho.x.array, self.rho.x.array), op=MPI.SUM)
            
            rel_error = np.sqrt(diff_norm_sq) / max(np.sqrt(rho_norm_sq), 1e-10)
            
            # Log
            if self.comm.rank == 0 and self.cfg.verbose:
                self.logger.info(f"   Substep {itr}: rel_err={rel_error:.3e}")
            
            if progress is not None and task_id is not None:
                progress.update(task_id, advance=1, info=f"rel_err={rel_error:.3e}")

            # Record metrics
            rec = {
                "iter": itr,
                "proj_res": float(rel_error), # Using rel_error as proxy for projected residual
                "mech_time": float(mech_stats["total_time"]),
                "dens_time": float(elapsed),
                "mech_iters": int(sum(mech_stats["phase_iters"])),
                "dens_iters": int(dens_iters)
            }
            self.subiter_metrics.append(rec)
            
            if rel_error < tol and itr >= min_subiters:
                break
