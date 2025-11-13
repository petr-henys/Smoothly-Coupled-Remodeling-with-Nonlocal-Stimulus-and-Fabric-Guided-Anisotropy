"""Debug stimulus evolution."""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.model import Remodeller
from tests.test_model import _DummyGaitLoader


def test_stimulus_debug(tmp_path, monkeypatch):
    """Debug why stimulus stays zero."""
    import simulation.storage as storage_mod
    import simulation.model as model_mod
    from pathlib import Path as _P

    class _DummyVTXWriter:
        def __init__(self, comm, path, fields, engine="bp4"):
            if comm.rank == 0:
                _P(path).mkdir(parents=True, exist_ok=True)
        def write(self, t):
            return None
        def close(self):
            return None

    monkeypatch.setattr(storage_mod, "VTXWriter", _DummyVTXWriter, raising=True)

    def _factory(V, *args, **kwargs):
        return _DummyGaitLoader(V)

    monkeypatch.setattr(model_mod, "setup_femur_gait_loading", _factory, raising=True)

    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 5, 5, 5, ghost_mode=mesh.GhostMode.shared_facet)
    facet_tags = build_facetag(domain)
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True, results_dir=str(tmp_path), max_subiters=8)

    with Remodeller(cfg) as rem:
        # Check initial stimulus
        S_initial = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        print(f"\n[Rank {comm.rank}] Initial stimulus max: {S_initial:.3e}")
        
        # Before step
        print(f"[Rank {comm.rank}] rho0={cfg.rho0}, rS_gain={cfg.rS}, psi_ref={cfg.psi_ref}, dt={cfg.dt}")
        
        rem.step(dt=86400.0)  # 1 day in seconds
        
        # Check driver energy
        psi_expr = rem.driver.energy_expr()
        psi_form = fem.form(psi_expr * cfg.dx)
        psi_local = fem.assemble_scalar(psi_form)
        psi_global = comm.allreduce(psi_local, op=MPI.SUM)
        vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
        vol_global = comm.allreduce(vol_local, op=MPI.SUM)
        psi_avg_driver = psi_global / vol_global
        
        print(f"[Rank {comm.rank}] Driver energy avg: {psi_avg_driver:.3e}")
        
        # Check mechanics energy
        psi_mech = rem.mechsolver.average_strain_energy()
        print(f"[Rank {comm.rank}] Mechanics energy avg: {psi_mech:.3e}")
        
        # Check stimulus
        S_max = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        S_mean = comm.allreduce(np.mean(rem.S.x.array), op=MPI.SUM) / comm.size
        
        print(f"[Rank {comm.rank}] After step: S_max={S_max:.3e}, S_mean={S_mean:.3e}")
        
        # Check config values
        DAY_TO_SEC = 86400.0
        print(f"[Rank {comm.rank}] Config: rS_gain={cfg.rS:.3e} [1/(MPa·day)], rS_sec={cfg.rS/DAY_TO_SEC:.3e} [1/(MPa·s)], psi_ref={cfg.psi_ref:.3e} [MPa]")
        
        # Check stimulus RHS directly
        psi_expr_check = rem.driver.energy_expr()
        rem.stimsolver.assemble_rhs(psi_expr_check)
        b_max = comm.allreduce(np.max(np.abs(rem.stimsolver.b.array)), op=MPI.MAX)
        b_sum = comm.allreduce(np.sum(rem.stimsolver.b.array), op=MPI.SUM)
        print(f"[Rank {comm.rank}] Stimulus RHS: b_max={b_max:.3e}, b_sum={b_sum:.3e}")
        
        # Manually solve stimulus
        iters, reason = rem.stimsolver.solve()
        print(f"[Rank {comm.rank}] Manual stimulus solve: iters={iters}, reason={reason}")
        
        S_after_solve = comm.allreduce(np.max(np.abs(rem.S.x.array)), op=MPI.MAX)
        print(f"[Rank {comm.rank}] After manual solve: S_max={S_after_solve:.3e}")
        
        # Check matrix properties
        A = rem.stimsolver.A
        info = A.getInfo()
        print(f"[Rank {comm.rank}] Matrix: rows={A.getSize()[0]}, nz={info['nz_used']:.0f}")
        
        # Check if matrix is essentially zero or identity
        mat_norm = A.norm()
        print(f"[Rank {comm.rank}] Matrix Frobenius norm: {mat_norm:.3e}")
        
        # Check coefficients
        DAY_TO_SEC = 86400.0
        print(f"[Rank {comm.rank}] Coeffs: cS={cfg.cS:.3e} [MPa·day], tauS={cfg.tauS:.3e} [1/day], kappaS={cfg.kappaS:.3e} [mm²/day], dt={cfg.dt:.3e} [s]")
        print(f"[Rank {comm.rank}] In seconds: cS_sec={cfg.cS/DAY_TO_SEC:.3e}, tauS_sec={cfg.tauS/DAY_TO_SEC:.3e}, kappaS_sec={cfg.kappaS/DAY_TO_SEC:.3e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
