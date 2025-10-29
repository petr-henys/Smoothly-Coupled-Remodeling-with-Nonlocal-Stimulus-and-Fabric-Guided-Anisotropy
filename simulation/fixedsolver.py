from __future__ import annotations

from typing import Sequence, Optional

import numpy as np
from mpi4py import MPI
from dolfinx import fem
import ufl

from simulation.config import Config
from simulation.subsolvers import (MechanicsSolver, StimulusSolver,
                        DensitySolver, DirectionSolver)
from simulation.utils import assign, get_owned_size, collect_dirichlet_dofs
from simulation.utils import _global_dot, current_memory_mb
from simulation.logger import get_logger
from simulation.anderson import _Anderson


# -----------------------------------------------------------------------------
# Fixed-point orchestrator
# -----------------------------------------------------------------------------

class FixedPointSolver:
    def __init__(self, comm: MPI.Comm, cfg: Config,
                 mechsolver: MechanicsSolver,
                 stimsolver: StimulusSolver,
                 densolver: DensitySolver,
                 dirsolver: DirectionSolver,
                 u: fem.Function,
                 rho: fem.Function, rho_old: fem.Function,
                 A: fem.Function, A_old: fem.Function,
                 S: fem.Function, S_old: fem.Function):
        self.comm = comm
        self.cfg = cfg

        self.mech = mechsolver
        self.stim = stimsolver
        self.den  = densolver
        self.dir  = dirsolver

        self.u = u
        self.rho = rho
        self.rho_old = rho_old
        self.A = A
        self.A_old = A_old
        self.S = S
        self.S_old = S_old

        # public counters for outer statistics
        self.total_gs_iters = 0
        self.gs_steps = 0

        # Determine local sizes (order: u, rho, A, S)
        self.n_u  = get_owned_size(u)
        self.n_rho = get_owned_size(rho)
        self.n_A  = get_owned_size(A)
        self.n_S  = get_owned_size(S)

        self._build_state_slices()

        # Reusable buffer for current local state
        self._state_buffer = np.empty(self._state_size, dtype=self.u.x.array.dtype)

        # Dirichlet mask for mechanics block (local)
        self.dirichlet_dofs_u = collect_dirichlet_dofs(self.mech.bcs, n_owned=self.n_u)
        self._mask_u = np.zeros(self.n_u, dtype=bool)
        if self.dirichlet_dofs_u.size:
            self._mask_u[self.dirichlet_dofs_u] = True
        # cache boolean to avoid repeated .any() scans on a large boolean array
        self._mask_u_any = bool(self.dirichlet_dofs_u.size)

        # Global fix mask (same length as flattened state)
        self._fix_mask = np.zeros(self._state_size, dtype=bool)
        s_u, _, _, _ = self._state_slices
        # BUGFIX: chained boolean indexing creates a copy; write via absolute indices instead
        if self._mask_u_any:
            idx_u = np.nonzero(self._mask_u)[0]
            self._fix_mask[s_u.start + idx_u] = True

        # cumulative timings for the last external step (one call to run())
        self.mech_time_total = 0.0
        self.stim_time_total = 0.0
        self.dens_time_total = 0.0
        self.dir_time_total  = 0.0
        # Per-subiteration metrics (filled by run when enabled)
        self.subiter_metrics: list[dict] = []
        # Average memory per timestep (computed from subiter_metrics)
        self.avg_memory_mb: Optional[float] = None

        # Logger controlled by cfg.verbose
        self.logger = get_logger(self.comm, verbose=bool(getattr(self.cfg, "verbose", True)), name="FixedPoint")
        self.telemetry = getattr(self.cfg, "telemetry", None)
        if self.telemetry is not None:
            self.telemetry.register_csv(
                "subiterations",
                [
                    "step",
                    "iter",
                    "time_days",
                    "rel_change",
                    "proj_res",
                    "r_norm",
                    "r_proxy_norm",
                    "aa_hist",
                    "accepted",
                    "backtracks",
                    "restart",
                    "condH",
                    "mech_time",
                    "stim_time",
                    "dens_time",
                    "dir_time",
                    "mech_reason",
                    "stim_reason",
                    "dens_reason",
                    "dir_reason",
                    "mech_iters",
                    "stim_iters",
                    "dens_iters",
                    "dir_iters",
                    "memory_mb",
                    "rhoJ",
                    "J_gs",
                ],
                gz=False,
                filename="subiterations.csv",
            )

    # -- packing/unpacking state --

    def _build_state_slices(self) -> None:
        offs = [0]
        offs.append(offs[-1] + self.n_u)
        offs.append(offs[-1] + self.n_rho)
        offs.append(offs[-1] + self.n_A)
        offs.append(offs[-1] + self.n_S)
        self._state_size = offs[-1]
        self._state_slices = (slice(offs[0], offs[1]),   # u
                              slice(offs[1], offs[2]),   # rho
                              slice(offs[2], offs[3]),   # A
                              slice(offs[3], offs[4]))   # S

    def _flatten_state(self, copy: bool = True) -> np.ndarray:
        """Flatten current fields to a 1D numpy array (local part only)."""
        s_u, s_rho, s_A, s_S = self._state_slices
        buf = self._state_buffer
        buf[s_u]  = self.u.x.array[:self.n_u]
        buf[s_rho] = self.rho.x.array[:self.n_rho]
        buf[s_A]  = self.A.x.array[:self.n_A]
        buf[s_S]  = self.S.x.array[:self.n_S]
        return buf.copy() if copy else buf

    def _restore_state(self, flat: np.ndarray) -> None:
        s_u, s_rho, s_A, s_S = self._state_slices
        assign(self.u, flat[s_u])
        assign(self.rho, flat[s_rho])
        assign(self.A, flat[s_A])
        assign(self.S, flat[s_S])

    # -- timing helper --
    def _elapsed_max(self, t0: float) -> float:
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)

    # -- optional helper for preconditioner maintenance (no AA reset) --
    @staticmethod
    def _maybe_update_precond(solver, threshold: float | int | None) -> bool:
        """
        If solver exposes 'last_iters' and exceeds threshold, call update_precond().
        Returns True if update_precond() was called (purely informational).
        """
        touched = False
        if getattr(solver, "last_iters", None) is not None and threshold is not None:
            try:
                if solver.last_iters > threshold and hasattr(solver, "update_precond"):
                    solver.update_precond()
                    touched = True
            except Exception:
                # Preconditioner updates are best-effort; never let them break the coupling loop.
                pass
        return touched

    # -- one GS sweep over the four fields --
    def _gauss_seidel_sweep(self) -> tuple[float, float, float, float, int, int, int, int, int, int, int, int]:
        """
        Perform one Gauss-Seidel sweep over all fields.
        
        Returns:
            (mech_time, stim_time, dens_time, dir_time, 
             mech_reason, stim_reason, dens_reason, dir_reason,
             mech_iters, stim_iters, dens_iters, dir_iters)
        """
        mech_time_total = stim_time_total = dens_time_total = dir_time_total = 0.0

        # Mechanics
        t0 = MPI.Wtime()
        self.mech.update_stiffness()
        mech_iters, mech_reason = self.mech.solve(self.u)
        mech_time_total += self._elapsed_max(t0)
        self._maybe_update_precond(self.mech, self.cfg.precond_threshold_update)

        # Stimulus
        t0 = MPI.Wtime()
        psi_density = 0.5 * ufl.inner(self.mech.sigma(self.u, self.rho), self.mech.eps(self.u))
        self.stim.update_rhs(psi_density)
        stim_iters, stim_reason = self.stim.solve(self.S)
        stim_time_total += self._elapsed_max(t0)
        self._maybe_update_precond(self.stim, self.cfg.precond_threshold_update)

        # Density
        t0 = MPI.Wtime()
        self.den.update_system()
        dens_iters, dens_reason = self.den.solve(self.rho)
        dens_time_total += self._elapsed_max(t0)
        self._maybe_update_precond(self.den, self.cfg.precond_threshold_update)

        # Direction
        t0 = MPI.Wtime()
        self.dir.update_rhs(self.mech, self.u)
        dir_iters, dir_reason = self.dir.solve(self.A)
        dir_time_total += self._elapsed_max(t0)
        self._maybe_update_precond(self.dir, self.cfg.precond_threshold_update)

        return (mech_time_total, stim_time_total, dens_time_total, dir_time_total,
                mech_reason, stim_reason, dens_reason, dir_reason,
                mech_iters, stim_iters, dens_iters, dir_iters)

    # -- norms & stopping --

    def _relative_residual_blockwise(self, x_old: np.ndarray, x_new: np.ndarray) -> float:
        """Max over blocks of ||x_new - x_old|| / ||x_new|| (global 2-norms)."""
        rel_vals: list[float] = []
        s_u, s_rho, s_A, s_S = self._state_slices

        # Mechanics block (zero Dirichlet)
        d_u = x_new[s_u] - x_old[s_u]
        if self._mask_u_any:
            d_u = d_u.copy()
            d_u[self._mask_u] = 0.0
            x_u_ref = x_new[s_u].copy()
            x_u_ref[self._mask_u] = 0.0
        else:
            x_u_ref = x_new[s_u]
        num_u = _global_dot(self.comm, d_u, d_u)
        den_u = _global_dot(self.comm, x_u_ref, x_u_ref) + 1e-32
        rel_vals.append((num_u / den_u) ** 0.5)

        # Remaining blocks
        for s in (s_rho, s_A, s_S):
            d = x_new[s] - x_old[s]
            num = _global_dot(self.comm, d, d)
            den = _global_dot(self.comm, x_new[s], x_new[s]) + 1e-32
            rel_vals.append((num / den) ** 0.5)

        return max(rel_vals)

    # Weighted projected-residual norm used as *merit/stopping* metric
    def _proj_residual_norm(self,
                            x_old: np.ndarray,
                            x_test: np.ndarray,
                            x_raw: np.ndarray,
                            weights: Sequence[float]) -> float:
        """
        ||P (x_test - x_old)||_W    if x_test is x_raw  (Picard residual)
        ||P (x_test - x_raw)||_W    if x_test is a candidate mixed iterate (proxy)
        where P projects out Dirichlet DOFs in the mechanics block.
        """
        s_u, s_rho, s_A, s_S = self._state_slices

        def block_norm(slc: slice, mask_u: Optional[np.ndarray], w: float, is_mech: bool) -> float:
            a = x_test[slc]
            if is_mech:
                # mechanics: compare to x_old for Picard, to x_raw for proxy
                base = x_old[slc] if (x_test is x_raw) else x_raw[slc]
                diff = a - base
                if mask_u is not None and self._mask_u_any:
                    diff = diff.copy()
                    diff[mask_u] = 0.0
            else:
                # other blocks: Picard residual (x_raw-x_old) vs. proxy (x_test-x_raw)
                base = x_old[slc] if (x_test is x_raw) else x_raw[slc]
                diff = a - base
            return w * _global_dot(self.comm, diff, diff)

        w_u, w_rho, w_A, w_S = weights
        total = 0.0
        total += block_norm(s_u, self._mask_u, w_u, True)
        total += block_norm(s_rho, None, w_rho, False)
        total += block_norm(s_A, None, w_A, False)
        total += block_norm(s_S, None, w_S, False)
        return max(total, 0.0) ** 0.5

    def rank0(self) -> bool:
        return self.comm.rank == 0
    
        # --- Diagnostics: one-step GS map and interaction gains (MPI-safe) --------
    def _one_sweep_map(self, flat_in: np.ndarray) -> np.ndarray:
        """
        Perform exactly ONE Gauss–Seidel sweep on a COPY of the state and return the flattened result.
        Temporarily disables preconditioner maintenance so diagnostics do not mutate solver state.
        The original state and config are restored afterwards.
        """
        # Backup current state
        x_backup = self._flatten_state(copy=True)

        # Temporarily freeze preconditioner maintenance (if present)
        old_thr = getattr(self.cfg, 'precond_threshold_update', None)
        try:
            # Freeze to prevent rebuilds inside diagnostics
            try:
                self.cfg.precond_threshold_update = None
            except Exception:
                pass

            # Run on a copy of the state
            self._restore_state(flat_in)
            self._gauss_seidel_sweep()
            out = self._flatten_state(copy=True)

        finally:
            # Restore cfg and state regardless of inner failures
            try:
                self.cfg.precond_threshold_update = old_thr
            except Exception:
                pass
            self._restore_state(x_backup)

        return out

    def _block_norm(self, vec: np.ndarray, block: int, *, project_u_dirichlet: bool = True) -> float:
        s_u, s_rho, s_A, s_S = self._state_slices
        slices = (s_u, s_rho, s_A, s_S)
        part = vec[slices[block]].copy()
        # Project mechanics Dirichlet DOFs to avoid boundary contamination of the norm
        if project_u_dirichlet and block == 0 and getattr(self, "_mask_u", None) is not None:
            if self._mask_u.size:
                part[self._mask_u] = 0.0
        return float((_global_dot(self.comm, part, part)) ** 0.5)

    def compute_interaction_gains(self, eps: float = 1e-3, project_u_dirichlet: bool = True
                                  ) -> tuple[np.ndarray, float]:
        """
        Empirically estimate the 4x4 block interaction-gain matrix J for the GS map G at current state x*.
        Returns (J, rho), where J[i,j] ≈ || G_i(x* + δ_j) - G_i(x*) || / ||δ_j|| and rho is spectral radius(J).
        Uses one base sweep + 4 perturbed sweeps. Leaves solver state unchanged. Deterministic in MPI.
        """
        # Base point and its image under one sweep
        x0 = self._flatten_state(copy=True)
        Gx0 = self._one_sweep_map(x0)

        J = np.zeros((4, 4), dtype=float)
        s_u, s_rho, s_A, s_S = self._state_slices
        slices = (s_u, s_rho, s_A, s_S)

        for j, sj in enumerate(slices):
            # Deterministic perturbation: direction = current block state (not RNG)
            dloc = x0[sj].copy()
            # Project mechanics Dirichlet DOFs out of the perturbation, if requested
            if project_u_dirichlet and j == 0 and getattr(self, "_mask_u", None) is not None:
                if self._mask_u.size:
                    dloc[self._mask_u] = 0.0

            # If block is (near) zero, use ones as a direction
            # Global norm via MPI (consistent scaling)
            base_norm = float((_global_dot(self.comm, dloc, dloc)) ** 0.5)

            if base_norm < 1e-20:
                # Fallback: unit vector of ones
                dloc = np.ones_like(dloc)
                base_norm = float((_global_dot(self.comm, dloc, dloc)) ** 0.5)

            # Target perturbation magnitude in block j
            target = eps * (1.0 + base_norm)
            # Local L2 for per-rank rescale (final norm checked globally)
            loc_norm = max(1e-300, float(np.linalg.norm(dloc)))
            dloc *= (target / loc_norm)

            # Insert into full vector
            delta = np.zeros_like(x0)
            delta[sj] = dloc

            # Map via single sweep
            Gx_pert = self._one_sweep_map(x0 + delta)
            diff = Gx_pert - Gx0

            # Denominator: global norm of perturbation in block j (after projection)
            denom = self._block_norm(delta, j, project_u_dirichlet=project_u_dirichlet) + 1e-300

            # Fill column j
            for i in range(4):
                numer = self._block_norm(diff, i, project_u_dirichlet=project_u_dirichlet)
                J[i, j] = numer / denom

        # Spectral radius
        try:
            eigvals = np.linalg.eigvals(J)
            rho = float(np.max(np.abs(eigvals)))
        except np.linalg.LinAlgError:
            rho = float("nan")

        return J, rho

    # -- main loop --

    def run(
        self,
        *,
        time_days: Optional[float] = None,
        step_index: Optional[int] = None,
    ) -> None:
        """
        Execute the inner fixed-point Gauss–Seidel + acceleration loop.
        All parameters are read from Config (cfg).
        """
        # Read configuration
        accel_type = str(self.cfg.accel_type).lower()
        m = int(self.cfg.m)
        beta = float(self.cfg.beta)
        lam = float(self.cfg.lam)

        # Use global DOF counts to balance contributions across blocks (mechanics uses *free* DOFs)
        n_u_free_loc = self.n_u - int(self._mask_u.sum()) if self._mask_u_any else self.n_u
        n_u_free = self.comm.allreduce(n_u_free_loc, op=MPI.SUM)
        n_rho_g = self.comm.allreduce(self.n_rho, op=MPI.SUM)
        n_A_g = self.comm.allreduce(self.n_A, op=MPI.SUM)
        n_S_g = self.comm.allreduce(self.n_S, op=MPI.SUM)
        # The merit uses sqrt(sum_i w_i * ||diff_i||^2); choosing w_i = 1/N_i yields an RMS-per-DOF scaling
        weights = (
            1.0 / max(int(n_u_free), 1),
            1.0 / max(int(n_rho_g), 1),
            1.0 / max(int(n_A_g), 1),
            1.0 / max(int(n_S_g), 1),
        )
        tol = float(self.cfg.coupling_tol)
        gamma = float(self.cfg.gamma)
        safeguard = bool(self.cfg.safeguard)
        backtrack_max = int(self.cfg.backtrack_max)

        # Restart heuristics
        restart_on_reject_k = int(self.cfg.restart_on_reject_k)
        restart_on_stall = float(self.cfg.restart_on_stall)
        restart_on_cond = float(self.cfg.restart_on_cond)
        step_limit_factor = float(self.cfg.step_limit_factor)

        # Subiteration bounds
        max_subiters = int(self.cfg.max_subiters)
        min_subiters = int(self.cfg.min_subiters)

        self.verbose = bool(self.cfg.verbose)

        # Choose accelerator
        if accel_type == "anderson":
            accelerator = _Anderson(self.comm, m=m, beta=beta, lam=lam,
                                    restart_on_reject_k=restart_on_reject_k,
                                    restart_on_stall=restart_on_stall,
                                    restart_on_cond=restart_on_cond,
                                    step_limit_factor=step_limit_factor)
        elif accel_type in ("none", "picard"):
            accelerator = None
        else:
            raise ValueError(f"Unknown accel_type={accel_type!r}")

        # Outer time steps/loads provided by caller; we only iterate substeps here
        used_iters = 0
        accelerator.reset() if accelerator is not None else None

        # Pack current state
        x_k = self._flatten_state(copy=True)

        # Reset per-step cumulative timings
        self.mech_time_total = 0.0
        self.stim_time_total = 0.0
        self.dens_time_total = 0.0
        self.dir_time_total  = 0.0

        log_enabled = self.rank0() and self.verbose
        # Reset metrics storage for this run

        current_step = self.gs_steps
        self.subiter_metrics = []

        for itr in range(1, max_subiters + 1):
            # Perform one GS sweep to get x_raw = G(x_k)
            (mech_t, stim_t, dens_t, dir_t, 
             mech_reason, stim_reason, dens_reason, dir_reason,
             mech_iters, stim_iters, dens_iters, dir_iters) = self._gauss_seidel_sweep()
            self.mech_time_total += mech_t
            self.stim_time_total += stim_t
            self.dens_time_total += dens_t
            self.dir_time_total  += dir_t

            # Measure memory (MPI-aware: sum across all ranks for total consumed)
            memory_mb_local = current_memory_mb()
            memory_mb = self.comm.allreduce(memory_mb_local, op=MPI.SUM)

            x_raw = self._flatten_state(copy=True)

            # Merit function (closure with weights)
            merit = lambda x_test: self._proj_residual_norm(x_k, x_test, x_raw, weights)

            # Decide on mixed iterate
            if accelerator is None:
                x_next = x_raw
                info = {"accepted": True, "backtracks": 0, "aa_hist": 0,
                        "r_norm": merit(x_raw), "r_proxy_norm": 0.0, "restart_reason": ""}
            else:
                x_next, info = accelerator.mix(
                    x_old=x_k, x_raw=x_raw, mask_fixed=self._fix_mask,
                    proj_residual_norm=lambda x_ref, x_test, xR: self._proj_residual_norm(x_ref, x_test, xR, weights),
                    gamma=gamma, use_safeguard=safeguard, backtrack_max=backtrack_max
                )

        
            # Restore state and compute *relative* iterate change (for logging)
            self._restore_state(x_next)
            relres = self._relative_residual_blockwise(x_k, x_next)

            is_picard_local = int(np.allclose(x_next, x_raw))
            is_picard = bool(self.comm.allreduce(is_picard_local, op=MPI.MIN))
            proj_norm = merit(x_raw) if is_picard else self._proj_residual_norm(x_k, x_next, x_raw, weights)


            used_iters = itr

            # Optional per-subiteration coupling diagnostics (heavy)
            J_gs = None
            rhoJ = None
            if getattr(self.cfg, "coupling_each_iter", False):
                # Evaluate around the chosen iterate (state already restored)
                J_packed, rhoJ = self.compute_interaction_gains(
                    eps=getattr(self.cfg, "coupling_eps", 1e-3),
                    project_u_dirichlet=True
                )
                # Reorder to GS sweep order: (u, S, rho, A)
                p = np.array([0, 3, 1, 2], dtype=int)
                J_gs = J_packed[np.ix_(p, p)]

            if log_enabled:
                msg_parts = [
                    f"      Substep {itr}: rel-change = {relres:.3e}",
                    f"proj-res = {proj_norm:.3e}",
                ]
                if accelerator is not None:
                    msg_parts.append(f"AA_hist={info['aa_hist']}")
                    msg_parts.append(f"accepted={'Y' if info['accepted'] else 'N'}")
                    if info.get('restart_reason', ''):
                        msg_parts.append(f"restart={info['restart_reason']}")
                    if info.get('backtracks', 0):
                        msg_parts.append(f"bt={info['backtracks']}")
                    if info.get('condH') is not None:
                        msg_parts.append(f"condH~{info['condH']:.2e}")

                if getattr(self.cfg, "coupling_each_iter", False) and (J_gs is not None):
                    msg_parts.append(f"rho(J) = {rhoJ:.3e}")

                self.logger.info("   " + " | ".join(msg_parts))

            # Store per-subiteration metrics (always collected)
            rec = {
                "iter": itr,
                "rel_change": float(relres),
                "proj_res": float(proj_norm),
                "aa_hist": int(info.get('aa_hist', 0)),
                "accepted": bool(info.get('accepted', True)),
                "backtracks": int(info.get('backtracks', 0)),
                "restart": str(info.get('restart_reason', '')),
                "condH": float(info.get('condH')) if info.get('condH') is not None else float('nan'),
                # Additional Anderson diagnostics
                "r_norm": float(info.get('r_norm')) if info.get('r_norm') is not None else float('nan'),
                "r_proxy_norm": float(info.get('r_proxy_norm')) if info.get('r_proxy_norm') is not None else float('nan'),
                "mech_time": float(mech_t),
                "stim_time": float(stim_t),
                "dens_time": float(dens_t),
                "dir_time": float(dir_t),
                # KSP convergence reasons (negative = failed)
                "mech_reason": int(mech_reason),
                "stim_reason": int(stim_reason),
                "dens_reason": int(dens_reason),
                "dir_reason": int(dir_reason),
                # KSP iterations per subsolver
                "mech_iters": int(mech_iters),
                "stim_iters": int(stim_iters),
                "dens_iters": int(dens_iters),
                "dir_iters": int(dir_iters),
                # Memory usage (max across ranks)
                "memory_mb": float(memory_mb),
            }
            # Prefer externally supplied step index if provided
            rec["step"] = int(step_index) if step_index is not None else current_step
            if time_days is not None:
                rec["time_days"] = float(time_days)
            if J_gs is not None:
                rec["J_gs"] = np.asarray(J_gs).tolist()
                rec["rhoJ"] = float(rhoJ)
            self.subiter_metrics.append(rec)
            if self.telemetry is not None:
                self.telemetry.record("subiterations", rec, csv_event=True)

            # Stopping by *projected residual*
            if proj_norm < tol and itr >= min_subiters:
                break

            # Prepare next iteration
            x_k = x_next

        self.total_gs_iters += used_iters
        self.gs_steps += 1

        # Compute average memory for this timestep
        if self.subiter_metrics:
            memory_values = [m["memory_mb"] for m in self.subiter_metrics if "memory_mb" in m]
            if memory_values:
                self.avg_memory_mb = float(np.mean(memory_values))
            else:
                self.avg_memory_mb = None
        else:
            self.avg_memory_mb = None

        return None
