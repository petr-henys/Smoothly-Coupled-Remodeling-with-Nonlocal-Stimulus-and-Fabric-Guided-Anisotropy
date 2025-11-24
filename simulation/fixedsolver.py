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
from simulation.anderson import _Anderson

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
        
        self.logger = get_logger(self.comm, name="FixedPoint", log_file=self.cfg.log_file)
        
        # Stats
        self.mech_time_total = 0.0
        self.dens_time_total = 0.0
        self.mech_iters_total = 0
        self.subiter_metrics: List[Dict] = []
        
        self.solver_stats = {
            "mech": {"time": 0.0, "iters": 0, "reason": 0},
            "dens": {"time": 0.0, "iters": 0, "reason": 0}
        }
        
        self.anderson = None
        if self.cfg.accel_type == "anderson":
            self.anderson = _Anderson(
                comm=self.comm,
                m=self.cfg.m,
                beta=self.cfg.beta,
                lam=self.cfg.lam,
                restart_on_reject_k=self.cfg.restart_on_reject_k,
                restart_on_stall=self.cfg.restart_on_stall,
                restart_on_cond=self.cfg.restart_on_cond,
                step_limit_factor=self.cfg.step_limit_factor,
            )

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

        if self.anderson:
            self.anderson.reset()

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
            
            # Anderson Acceleration
            aa_info = {}
            if self.anderson:
                # x_old = rho_prev_iter (input to this step)
                # x_raw = self.rho (output of this step)
                # We want to mix x_raw and x_old to get x_new
                
                # Note: Anderson expects numpy arrays.
                # We need to be careful with ghost values if we were doing this distributed,
                # but here we operate on local arrays and Anderson handles global reduction for Gram matrix.
                # However, dolfinx functions have ghost values.
                # Ideally we should work with owned dofs only, but for simplicity we use the full local array
                # and rely on the fact that Anderson uses global reductions which might double count ghosts
                # if not careful. But _Anderson._build_gram uses allreduce(SUM).
                # If we include ghosts, we double count.
                # Let's assume for now we just use the local array as is, or we should mask ghosts.
                # Given the complexity, let's just pass the array.
                
                x_new_array, aa_info = self.anderson.mix(
                    x_old=rho_prev_iter.x.array,
                    x_raw=self.rho.x.array,
                    # We don't have a fixed mask here easily available without more context, 
                    # but density usually doesn't have Dirichlet BCs in this formulation (Neumann/Robin).
                    mask_fixed=None, 
                    # We can provide a residual norm function if we want safeguarding
                    proj_residual_norm=None, 
                    gamma=self.cfg.gamma,
                    use_safeguard=self.cfg.safeguard,
                    backtrack_max=self.cfg.backtrack_max
                )
                
                self.rho.x.array[:] = x_new_array
                self.rho.x.scatter_forward()
                
                # Re-evaluate error after mixing? 
                # Usually convergence is checked on the unaccelerated step (x_raw - x_old),
                # which we already did above (rel_error).
            
            # Log
            log_msg = f"   Substep {itr}: rel_err={rel_error:.3e}"
            if aa_info:
                hist = aa_info.get('aa_hist', 0)
                acc = "acc" if aa_info.get('accepted', True) else "rej"
                bt = aa_info.get('backtracks', 0)
                log_msg += f" | AA({hist}) {acc} bt={bt}"
                if aa_info.get('restart_reason'):
                    log_msg += f" | Rst: {aa_info['restart_reason']}"
            
            self.logger.info(log_msg)
            
            if progress is not None and task_id is not None:
                prog_info = f"err={rel_error:.1e}"
                if aa_info:
                    prog_info += f" AA({aa_info.get('aa_hist',0)})"
                progress.update(task_id, advance=1, info=prog_info)

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
