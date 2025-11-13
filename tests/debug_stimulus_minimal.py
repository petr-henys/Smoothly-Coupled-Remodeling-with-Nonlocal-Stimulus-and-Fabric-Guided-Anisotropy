"""Minimal test to debug stimulus solver with day-based time."""

from dolfinx import mesh, fem
from mpi4py import MPI
import numpy as np
import ufl

from simulation.config import Config
from simulation.utils import build_facetag, assign
from simulation.subsolvers import StimulusSolver

domain = mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)
facet_tags = build_facetag(domain)
cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)

# Create stimulus functions
Q = fem.functionspace(domain, ("Lagrange", 1))
S = fem.Function(Q, name="S")
S_old = fem.Function(Q, name="S_old")
assign(S, 0.0)
assign(S_old, 0.0)

# Create solver
stimsolver = StimulusSolver(S, S_old, cfg)
stimsolver.setup()

# Assemble with constant psi
psi_const = fem.Constant(domain, 1.0)  # 1 MPa energy density
stimsolver.assemble_lhs()
stimsolver.assemble_rhs(psi_const)

# Check RHS
b_arr = stimsolver.b.array
b_max = np.max(np.abs(b_arr))
b_sum = np.sum(b_arr)
print(f"RHS: max={b_max:.3e}, sum={b_sum:.3e}")

# Solve
stimsolver.solve()

# Check solution
S_arr = S.x.array
S_max = np.max(np.abs(S_arr))
S_mean = np.mean(S_arr)
print(f"Solution: max={S_max:.3e}, mean={S_mean:.3e}")

# Expected source term: rS_gain * (psi - psi_ref) * dt
source_expected = cfg.rS_gain * (1.0 - cfg.psi_ref) * cfg.dt
print(f"Expected source term: {source_expected:.3e}")

# Equilibrium estimate: (cS/dt + tauS) * S ~= rS_gain * (psi - psi_ref)
S_equilibrium = cfg.rS_gain * (1.0 - cfg.psi_ref) / (cfg.cS/cfg.dt + cfg.tauS)
print(f"Equilibrium S estimate: {S_equilibrium:.3e}")
