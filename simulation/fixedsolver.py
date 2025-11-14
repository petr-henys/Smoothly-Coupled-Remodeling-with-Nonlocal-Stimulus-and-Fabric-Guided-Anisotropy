"""Fixed-point solver orchestrating coupled PDE subsolvers with Anderson acceleration."""

from __future__ import annotations
from typing import Sequence, Optional

import numpy as np
from mpi4py import MPI
from dolfinx import fem

from simulation.config import Config
from simulation.subsolvers import MechanicsSolver, StimulusSolver, DensitySolver, DirectionSolver, unittrace_psd
from simulation.utils import assign, get_owned_size, _global_dot, current_memory_mb
from simulation.logger import get_logger
from simulation.anderson import _Anderson
from simulation.drivers import StrainDriver


class FixedPointSolver:
    """Orchestrate Gauss-Seidel iteration over four coupled PDEs with Anderson or Picard."""
    def __init__(self, comm: MPI.Comm, cfg: Config,
                 driver: StrainDriver,
                 stimsolver: StimulusSolver,
                 densolver: DensitySolver,
                 dirsolver: DirectionSolver,
                 rho: fem.Function, rho_old: fem.Function,
                 A: fem.Function, A_old: fem.Function,
                 S: fem.Function, S_old: fem.Function):
        self.comm = comm
        self.cfg = cfg

        self.driver = driver
        self.stim = stimsolver
        self.den  = densolver
        self.dir  = dirsolver

        self.mech: MechanicsSolver | None = getattr(self.driver, "mech", None)
        if self.mech is None:
            raise ValueError("StrainDriver must expose MechanicsSolver via 'mech'")

        self.rho = rho
        self.rho_old = rho_old
        self.A = A
        self.A_old = A_old
        self.S = S
        self.S_old = S_old

        self.total_gs_iters = 0
        self.gs_steps = 0

        # Local DOF counts
        self.n_rho = get_owned_size(rho)
        self.n_A  = get_owned_size(A)
        self.n_S  = get_owned_size(S)

        self._build_state_slices()

        self.state_buffer = np.empty(self.state_size, dtype=self.rho.x.array.dtype)

        # Global fix mask (kept for API compatibility, currently all False)
        self.fix_mask = np.zeros(self.state_size, dtype=bool)

        # Cumulative timings per timestep
        self.mech_time_total = 0.0
        self.stim_time_total = 0.0
        self.dens_time_total = 0.0
        self.dir_time_total  = 0.0
        self.subiter_metrics: list[dict] = []
        self.avg_memory_mb: Optional[float] = None

        self.logger = get_logger(self.comm, verbose=bool(getattr(self.cfg, "verbose", True)), name="FixedPoint")
        self.telemetry = getattr(self.cfg, "telemetry", None)
        if self.telemetry is not None:
            self.telemetry.register_csv(
                "subiterations",
                [
                    "step",
                    "iter",
                    "time_days",
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
                    "stim_reason",
                    "dens_reason",
                    "dir_reason",
                    "stim_iters",
                    "dens_iters",
                    "dir_iters",
                    "memory_mb",
                    "memory_mb_max",
                    "energy_Wint",
                    "energy_Wext",
                    "energy_res_rel",
                    "power_res_abs",
                    "power_res_rel",
                    "mass_res_abs",
                    "mass_res_rel",
                    "trace_A_avg",
                    "trace_Mhat_avg",
                    "trace_res",
                    "rhoJ",
                    "J_gs",
                ],
                filename="subiterations.csv",
            )

    def _build_state_slices(self) -> None:
        """Build slice indices for flattened (ρ, A, S) state vector."""
        offs = [0]
        offs.append(offs[-1] + self.n_rho)
        offs.append(offs[-1] + self.n_A)
        offs.append(offs[-1] + self.n_S)
        self.state_size = offs[-1]
        self.state_slices = (
            slice(offs[0], offs[1]),  # rho
            slice(offs[1], offs[2]),  # A
            slice(offs[2], offs[3])   # S
        )

    def _flatten_state(self, copy: bool = True) -> np.ndarray:
        """Flatten current fields (ρ, A, S) to 1D array (local DOFs)."""
        s_rho, s_A, s_S = self.state_slices
        buf = self.state_buffer
        buf[s_rho] = self.rho.x.array[:self.n_rho]
        buf[s_A]  = self.A.x.array[:self.n_A]
        buf[s_S]  = self.S.x.array[:self.n_S]
        return buf.copy() if copy else buf

    def _restore_state(self, flat: np.ndarray) -> None:
        """Unpack flattened state back to field functions."""
        s_rho, s_A, s_S = self.state_slices
        assign(self.rho, flat[s_rho])
        assign(self.A, flat[s_A])
        assign(self.S, flat[s_S])

    def _elapsed_max(self, t0: float) -> float:
        """Max wall time across ranks since t0."""
        return self.comm.allreduce(MPI.Wtime() - t0, op=MPI.MAX)

    def _gauss_seidel_sweep(self) -> tuple[float, float, float, float, int, int, int, int, int, int, int, int]:
        """One GS sweep: mechanics + S → ρ → A (sequential subsolver calls)."""
        mech_time_total = stim_time_total = dens_time_total = dir_time_total = 0.0

        t0 = MPI.Wtime()
        self.mech.assemble_lhs()  # K(ρ, A) reassembled; solves happen inside driver.update_snapshots()
        mech_time_total += self._elapsed_max(t0)

        # Stimulus (energy-driven via driver)
        t0 = MPI.Wtime()
        self.driver.update_snapshots()
        psi_expr = self.driver.energy_expr()
        self.stim.assemble_rhs(psi_expr)
        stim_iters, stim_reason = self.stim.solve()
        stim_time_total += self._elapsed_max(t0)

        # Density
        t0 = MPI.Wtime()
        self.den.assemble_lhs()
        self.den.assemble_rhs()
        dens_iters, dens_reason = self.den.solve()
        dens_time_total += self._elapsed_max(t0)

        # Direction (structure tensor via driver)
        t0 = MPI.Wtime()
        self.driver.update_snapshots()
        M_expr = self.driver.structure_expr()
        self.dir.assemble_rhs(M_expr)
        dir_iters, dir_reason = self.dir.solve()
        dir_time_total += self._elapsed_max(t0)

        return (stim_time_total, dens_time_total, dir_time_total,
                stim_reason, dens_reason, dir_reason,
                stim_iters, dens_iters, dir_iters)

    def _proj_residual_norm(self, x_old: np.ndarray, x_test: np.ndarray,
                           x_raw: np.ndarray, weights: Sequence[float]) -> float:
        """Weighted residual ||x_test - x_base||_W with per-block weights."""
        if len(weights) != len(self.state_slices):
            raise ValueError("weights must match number of state blocks")

        is_picard = (x_test is x_raw)
        total = 0.0
        for s, w in zip(self.state_slices, weights):
            base = x_old[s] if is_picard else x_raw[s]
            diff = x_test[s] - base
            total += float(w) * _global_dot(self.comm, diff, diff)

        return float(max(total, 0.0) ** 0.5)

    def rank0(self) -> bool:
        """Check if current rank is 0."""
        return self.comm.rank == 0

    def _one_sweep_map(self, flat_in: np.ndarray) -> np.ndarray:
        """Evaluate GS map on state copy: x ↦ G(x)."""
        x_backup = self._flatten_state(copy=True)
        self._restore_state(flat_in)
        self._gauss_seidel_sweep()
        out = self._flatten_state(copy=True)
        self._restore_state(x_backup)
        return out

    def _block_norm(self, vec: np.ndarray, block: int) -> float:
        """Global L2 norm of a state block (ρ, A, or S)."""
        part = vec[self.state_slices[block]]
        return float((_global_dot(self.comm, part, part)) ** 0.5)

    def compute_interaction_gains(self, eps: float = 1e-3) -> tuple[np.ndarray, float]:
        """Finite-difference Jacobian J of GS map G over (ρ, A, S); returns (J, ρ(J))."""
        x0 = self._flatten_state(copy=True)
        Gx0 = self._one_sweep_map(x0)

        n_blocks = len(self.state_slices)
        J = np.zeros((n_blocks, n_blocks), dtype=float)

        for j, sj in enumerate(self.state_slices):
            dloc = x0[sj].copy()
            base_norm = float((_global_dot(self.comm, dloc, dloc)) ** 0.5)

            if base_norm < 1e-20:
                dloc = np.ones_like(dloc)
                base_norm = float((_global_dot(self.comm, dloc, dloc)) ** 0.5)

            target = eps * (1.0 + base_norm)
            dloc *= (target / max(base_norm, 1e-300))

            delta = np.zeros_like(x0)
            delta[sj] = dloc

            Gx_pert = self._one_sweep_map(x0 + delta)
            diff = Gx_pert - Gx0

            denom = self._block_norm(delta, j) + 1e-300

            for i in range(n_blocks):
                numer = self._block_norm(diff, i)
                J[i, j] = numer / denom

        eigvals = np.linalg.eigvals(J)
        rho = float(np.max(np.abs(eigvals)))

        return J, rho

    def run(self, *, time_days: Optional[float] = None, step_index: Optional[int] = None) -> None:
        """Inner fixed-point loop: GS + Anderson acceleration until coupling_tol met."""
        # Read configuration
        accel_type = str(self.cfg.accel_type).lower()
        m = int(self.cfg.m)
        beta = float(self.cfg.beta)
        lam = float(self.cfg.lam)

        # Compute global DOF counts for weight balancing
        n_rho_g = self.comm.allreduce(self.n_rho, op=MPI.SUM)
        n_A_g = self.comm.allreduce(self.n_A, op=MPI.SUM)
        n_S_g = self.comm.allreduce(self.n_S, op=MPI.SUM)
        weights = (
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
        accelerator = None
        if accel_type == "anderson":
            accelerator = _Anderson(self.comm, m=m, beta=beta, lam=lam,
                                    restart_on_reject_k=restart_on_reject_k,
                                    restart_on_stall=restart_on_stall,
                                    restart_on_cond=restart_on_cond,
                                    step_limit_factor=step_limit_factor,
                                    verbose=self.verbose)
        elif accel_type != "picard":
            raise ValueError(f"accel_type must be 'anderson' or 'picard', got {accel_type!r}")

        used_iters = 0
        if accelerator is not None:
            accelerator.reset()

        # Invalidate driver cache (forces rebuild of gait-averaged expressions with current state)
        self.driver.invalidate()

        x_k = self._flatten_state(copy=True)

        mask_for_mix = self.fix_mask if np.any(self.fix_mask) else None

        # Reset per-step cumulative timings
        self.mech_time_total = 0.0
        self.stim_time_total = 0.0
        self.dens_time_total = 0.0
        self.dir_time_total  = 0.0

        log_enabled = self.rank0() and self.verbose
        current_step = self.gs_steps
        self.subiter_metrics = []

        for itr in range(1, max_subiters + 1):
            # Perform one GS sweep: x_raw = G(x_k)
            (mech_t, stim_t, dens_t, dir_t,
             stim_reason, dens_reason, dir_reason,
             stim_iters, dens_iters, dir_iters) = self._gauss_seidel_sweep()
            self.mech_time_total += mech_t
            self.stim_time_total += stim_t
            self.dens_time_total += dens_t
            self.dir_time_total  += dir_t

            memory_mb_local = current_memory_mb()
            memory_mb = self.comm.allreduce(memory_mb_local, op=MPI.SUM)
            memory_mb_max = self.comm.allreduce(memory_mb_local, op=MPI.MAX)
            memory_mb_avg = memory_mb / max(self.comm.size, 1)

            x_raw = self._flatten_state(copy=True)
            # --- Diagnostics: all subsolvers have conservation/balance checks ---
            W_int, W_ext, energy_rel = self.mech.energy_balance()
            psi_density_expr = self.driver.energy_expr()
            power_abs, power_rel = self.stim.power_balance_residual(psi_density_expr)
            mass_abs, mass_rel = self.den.mass_balance_residual()
            B = self.driver.structure_expr()
            gdim = self.mech.u.function_space.mesh.geometry.dim
            Mhat_expr = unittrace_psd(B, gdim, eps=self.cfg.smooth_eps)
            trA_avg, trMhat_avg, trace_res = self.dir.trace_balance_residual(Mhat_expr)


            # Mix iterate (Anderson or Picard)
            if accelerator is None:
                x_next = x_raw
                info = {
                    "accepted": True, 
                    "backtracks": 0, 
                    "aa_hist": 0,
                    "r_norm": self._proj_residual_norm(x_k, x_raw, x_raw, weights), 
                    "r_proxy_norm": 0.0, 
                    "restart_reason": ""
                }
            else:
                x_next, info = accelerator.mix(
                    x_old=x_k, x_raw=x_raw, mask_fixed=mask_for_mix,
                    proj_residual_norm=lambda x_ref, x_test, xR: self._proj_residual_norm(x_ref, x_test, xR, weights),
                    gamma=gamma, use_safeguard=safeguard, backtrack_max=backtrack_max
                )

            self._restore_state(x_next)
            # Always stop by Picard residual (||x_raw - x_k||_W), AA step norm is logged separately
            proj_norm = self._proj_residual_norm(x_k, x_raw, x_raw, weights)
            used_iters = itr

            # Optional coupling diagnostics
            J_gs = None
            rhoJ = None
            if getattr(self.cfg, "coupling_each_iter", False):
                J_gs, rhoJ = self.compute_interaction_gains(
                    eps=getattr(self.cfg, "coupling_eps", 1e-3)
                )

            if log_enabled:
                msg_parts = [f"      Substep {itr}: proj-res = {proj_norm:.3e}"]
                if accelerator is not None:
                    msg_parts.append(f"AA_hist={info['aa_hist']}")
                    msg_parts.append(f"accepted={'Y' if info['accepted'] else 'N'}")
                    restart_reason = info.get('restart_reason', '')
                    if restart_reason:
                        msg_parts.append(f"restart={restart_reason}")
                    backtracks = info.get('backtracks', 0)
                    if backtracks:
                        msg_parts.append(f"bt={backtracks}")
                    condH = info.get('condH')
                    if condH is not None:
                        msg_parts.append(f"condH~{condH:.2e}")

                if getattr(self.cfg, "coupling_each_iter", False) and J_gs is not None:
                    msg_parts.append(f"rho(J) = {rhoJ:.3e}")

                self.logger.info("   " + " | ".join(msg_parts))

            # Store metrics
            rec = {
                "iter": itr,
                "proj_res": float(proj_norm),
                "aa_hist": int(info.get('aa_hist', 0)),
                "accepted": bool(info.get('accepted', True)),
                "backtracks": int(info.get('backtracks', 0)),
                "restart": str(info.get('restart_reason', '')),
                "condH": float(info['condH']) if info.get('condH') is not None else float('nan'),
                "r_norm": float(info['r_norm']) if info.get('r_norm') is not None else float('nan'),
                "r_proxy_norm": float(info['r_proxy_norm']) if info.get('r_proxy_norm') is not None else float('nan'),
                "energy_Wint": float(W_int),
                "energy_Wext": float(W_ext),
                "energy_res_rel": float(energy_rel),
                "power_res_abs": float(power_abs),
                "power_res_rel": float(power_rel),
                "mass_res_abs": float(mass_abs),
                "mass_res_rel": float(mass_rel),
                "trace_A_avg": float(trA_avg),
                "trace_Mhat_avg": float(trMhat_avg),
                "trace_res": float(trace_res),
                "mech_time": float(mech_t),
                "stim_time": float(stim_t),
                "dens_time": float(dens_t),
                "dir_time": float(dir_t),
                "stim_reason": int(stim_reason),
                "dens_reason": int(dens_reason),
                "dir_reason": int(dir_reason),
                "stim_iters": int(stim_iters),
                "dens_iters": int(dens_iters),
                "dir_iters": int(dir_iters),
                "memory_mb": float(memory_mb),
                "memory_mb_sum": float(memory_mb),
                "memory_mb_avg": float(memory_mb_avg),
                "memory_mb_max": float(memory_mb_max),
            }
            
            if step_index is not None:
                rec["step"] = int(step_index)
            else:
                rec["step"] = current_step
                
            if time_days is not None:
                rec["time_days"] = float(time_days)
                
            if J_gs is not None:
                rec["J_gs"] = np.asarray(J_gs).tolist()
                rec["rhoJ"] = float(rhoJ)
                
            self.subiter_metrics.append(rec)
            
            if self.telemetry is not None:
                self.telemetry.record("subiterations", rec, csv_event=True)

            # Stopping criterion
            if proj_norm < tol and itr >= min_subiters:
                break

            x_k = x_next

        self.total_gs_iters += used_iters
        self.gs_steps += 1

        # Compute average memory
        memory_values = [m["memory_mb"] for m in self.subiter_metrics if "memory_mb" in m]
        if memory_values:
            self.avg_memory_mb = float(np.mean(memory_values))
        else:
            self.avg_memory_mb = None
