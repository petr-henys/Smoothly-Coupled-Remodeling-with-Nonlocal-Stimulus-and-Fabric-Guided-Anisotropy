"""Block Gauss-Seidel with Anderson acceleration."""

from __future__ import annotations
from typing import Dict, List

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.utils import assign, get_owned_size
from simulation.logger import get_logger
from simulation.anderson import _Anderson

class FixedPointSolver:
    """
    Fixed-point iteration: mechanics → SED → density → Anderson mixing.
    Converges when ||ρ_new - ρ_old|| / ||ρ|| < coupling_tol.
    """

    def __init__(
        self, 
        comm: MPI.Comm, 
        cfg: Config,
        driver,
        densolver,
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

        
    def proj_residual_norm(self, x_old_vec: np.ndarray,
                            x_trial_vec: np.ndarray,
                            x_ref_vec: np.ndarray) -> float:
        """Relative step: ||x_trial - x_old|| / ||x_ref|| (global L2)."""
        diff = x_trial_vec - x_old_vec
        diff_loc = float(np.dot(diff, diff))
        ref_loc  = float(np.dot(x_ref_vec, x_ref_vec))
        diff_glob = self.comm.allreduce(diff_loc, op=MPI.SUM)
        ref_glob  = self.comm.allreduce(ref_loc,  op=MPI.SUM)

        if ref_glob <= 1e-30:
            return np.sqrt(diff_glob)
        return np.sqrt(diff_glob / ref_glob)

    def run(self, *, progress=None, task_id=None) -> bool:
        """Run fixed-point loop. Returns True if converged."""
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
            # Store previous iterate (skip scatter - rho is already synced from previous iter or init)
            assign(rho_prev_iter, self.rho, scatter=False)
            
            # 1. Mechanics
            # Driver solves equilibrium and updates its internal fields
            # rho is already synced, no need to scatter in assemble_lhs
            self.driver.update_stiffness() 
            mech_stats = self.driver.update_snapshots()
            # After this: u and psi are synced by driver
            
            self.mech_time_total += mech_stats["total_time"]
            self.mech_iters_total += sum(mech_stats["phase_iters"])
            
            self.solver_stats["mech"]["time"] += mech_stats["total_time"]
            self.solver_stats["mech"]["iters"] += int(sum(mech_stats["phase_iters"]))
            
            # 2. Density
            # psi was already scattered by driver, no need to scatter again
            t0 = MPI.Wtime()
            self.densolver.assemble_lhs()
            self.densolver.assemble_rhs()
            dens_iters, dens_reason = self.densolver.solve()
            # After solve: rho is synced by _solve()
            
            elapsed = self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)
            self.dens_time_total += elapsed
            
            self.solver_stats["dens"]["time"] += elapsed
            self.solver_stats["dens"]["iters"] += dens_iters
            self.solver_stats["dens"]["reason"] = dens_reason

            # 3. Anderson Acceleration (modifies owned DOFs, needs scatter)
            n_owned = get_owned_size(self.rho)
            aa_info = {}
            if self.anderson:
                x_new_owned, aa_info = self.anderson.mix(
                    x_old=rho_prev_iter.x.array[:n_owned],
                    x_raw=self.rho.x.array[:n_owned],
                    mask_fixed=None,
                    proj_residual_norm=self.proj_residual_norm,
                    gamma=self.cfg.gamma,
                    use_safeguard=self.cfg.safeguard,
                    backtrack_max=self.cfg.backtrack_max
                )
                assign(self.rho, x_new_owned, scatter=True)
            # Note: if no Anderson, rho is already synced from _solve()
            
            # 4. Convergence Check
            diff_owned = self.rho.x.array[:n_owned] - rho_prev_iter.x.array[:n_owned]
            diff_norm_sq = self.comm.allreduce(np.dot(diff_owned, diff_owned), op=MPI.SUM)
            rho_owned = self.rho.x.array[:n_owned]
            rho_norm_sq = self.comm.allreduce(np.dot(rho_owned, rho_owned), op=MPI.SUM)
            
            rel_error = np.sqrt(diff_norm_sq) / max(np.sqrt(rho_norm_sq), 1e-10)
            
            log_msg = f"   Substep {itr}: rel_err={rel_error:.3e}"
            if aa_info:
                hist = aa_info.get('aa_hist', 0)
                acc = "acc" if aa_info.get('accepted', True) else "rej"
                bt = aa_info.get('backtracks', 0)
                log_msg += f" | AA({hist}) {acc}"
                if bt > 0:
                    log_msg += f" bt={bt}"
                if aa_info.get('restart_reason'):
                    log_msg += f" | Rst: {aa_info['restart_reason']}"
            
            self.logger.debug(log_msg)
            
            if progress is not None and task_id is not None:
                prog_info = f"err={rel_error:.1e}"
                if aa_info:
                    prog_info += f" AA({aa_info.get('aa_hist',0)})"
                progress.update(task_id, advance=1, info=prog_info)

            rec = {
                "iter": itr,
                "proj_res": float(rel_error),
                "mech_time": float(mech_stats["total_time"]),
                "dens_time": float(elapsed),
                "mech_iters": int(sum(mech_stats["phase_iters"])),
                "dens_iters": int(dens_iters)
            }
            self.subiter_metrics.append(rec)
            
            if rel_error < tol and itr >= min_subiters:
                return True
        
        return False