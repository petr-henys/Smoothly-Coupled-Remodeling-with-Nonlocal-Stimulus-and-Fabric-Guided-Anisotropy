"""Tests for physical unit consistency and scaling laws."""

import pytest
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from simulation.config import Config
from simulation.drivers import GaitEnergyDriver
from simulation.femur_gait import FemurRemodellerGait

class MockMechanicsSolver:
    """Mock mechanics solver that returns energy density dependent on u."""
    def __init__(self, u):
        self.u = u
        self.comm = u.function_space.mesh.comm

    def get_strain_energy_density(self, u=None):
        # Return expression dependent on u so it updates when u_snap updates
        # Let psi = u[0]^2
        uu = self.u if u is None else u
        return uu[0]**2

    def get_strain_tensor(self, u=None):
        # Return identity for structure tensor testing
        return ufl.Identity(3)

    def assemble_rhs(self):
        pass

    def solve(self):
        return 1, 1

class MockGaitLoader:
    """Mock gait loader."""
    def __init__(self, V):
        self.t_hip = fem.Function(V)
        self.t_glmed = fem.Function(V)
        self.t_glmax = fem.Function(V)
        self.V = V
        self.target_psi = 0.0

    def get_quadrature(self):
        # Single phase for simplicity
        return [(0.0, 1.0)]

    def update_loads(self, phase):
        pass
    
    def set_target_psi(self, psi):
        self.target_psi = psi

def test_gait_driver_daily_dose_scaling():
    """Verify GaitEnergyDriver correctly scales energy by cycles/day."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    # Config parameters
    psi_ref = 20.0  # MPa (Target Daily Dose)
    N_cyc = 1000.0  # cycles/day
    n_power = 1.0   # linear
    
    cfg = Config(domain=domain, facet_tags=None)
    cfg.psi_ref = psi_ref
    cfg.gait_cycles_per_day = N_cyc
    cfg.n_power = n_power
    
    # 1. Test Equilibrium Case
    # If psi_mech = psi_ref / N_cyc (for n=1), then psi_expr should equal psi_ref
    psi_mech_eq = psi_ref / N_cyc
    
    u = fem.Function(V)
    mech = MockMechanicsSolver(u)
    loader = MockGaitLoader(V)
    
    # We need to inject logic to set u based on target psi during update_snapshots
    # We can subclass GaitEnergyDriver or just monkeypatch the solve method?
    # Actually, MockMechanicsSolver.solve is called. We can update u there.
    
    def solve_side_effect():
        # Set u such that u[0]^2 = target_psi
        # u[0] = sqrt(target_psi)
        val = np.sqrt(loader.target_psi)
        u.x.array[:] = 0.0
        # We need to set x-component. 
        # V is vector space. dofs are blocked? 
        # For P1 vector, dofs are usually ordered by node, then component? Or component then node?
        # dolfinx default is usually blocked if block size is set.
        # Let's just set all values to val/sqrt(3) so magnitude squared is val^2?
        # No, psi = u[0]^2. So just set u[0] component.
        # Easier: set all u to val. Then u[0]^2 is val^2.
        # Wait, u[0] in UFL means x-component.
        # Let's set x-component of all nodes to sqrt(target_psi).
        
        bs = V.dofmap.index_map_bs
        # Assuming bs=3
        u.x.array[0::bs] = val
        u.x.scatter_forward()
        return 1, 1
        
    mech.solve = solve_side_effect
    
    driver = GaitEnergyDriver(mech, loader, cfg)
    
    # Set target for equilibrium
    loader.set_target_psi(psi_mech_eq)
    driver.update_snapshots()
    
    # Evaluate psi_expr
    psi_expr = driver.energy_expr()
    psi_val = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
    vol = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
    psi_avg = comm.allreduce(psi_val, op=MPI.SUM) / comm.allreduce(vol, op=MPI.SUM)
    
    assert abs(psi_avg - psi_ref) < 1e-9, f"Equilibrium scaling failed: got {psi_avg}, expected {psi_ref}"

    # 2. Test General Scaling
    # If psi_mech = 2 * psi_mech_eq
    # psi_expr should be 2 * psi_ref (for n=1)
    loader.set_target_psi(2.0 * psi_mech_eq)
    driver.update_snapshots()
    
    psi_val = fem.assemble_scalar(fem.form(driver.energy_expr() * cfg.dx))
    psi_avg = comm.allreduce(psi_val, op=MPI.SUM) / comm.allreduce(vol, op=MPI.SUM)
    
    assert abs(psi_avg - 2.0 * psi_ref) < 1e-9, f"Linear scaling failed: got {psi_avg}, expected {2*psi_ref}"

def test_gait_driver_exponent_scaling():
    """Verify GaitEnergyDriver handles power law exponent n correctly."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 2, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    psi_ref = 20.0
    N_cyc = 100.0
    n_power = 2.0
    
    cfg = Config(domain=domain, facet_tags=None)
    cfg.psi_ref = psi_ref
    cfg.gait_cycles_per_day = N_cyc
    cfg.n_power = n_power
    
    # Equilibrium condition for n=2:
    # psi_mech = psi_ref * (1/N_cyc)^(1/n)
    psi_mech_eq = psi_ref * (1.0 / N_cyc)**(1.0 / n_power)
    
    u = fem.Function(V)
    mech = MockMechanicsSolver(u)
    loader = MockGaitLoader(V)
    
    def solve_side_effect():
        val = np.sqrt(loader.target_psi)
        bs = V.dofmap.index_map_bs
        u.x.array[0::bs] = val
        u.x.scatter_forward()
        return 1, 1
    mech.solve = solve_side_effect
    
    driver = GaitEnergyDriver(mech, loader, cfg)
    
    loader.set_target_psi(psi_mech_eq)
    driver.update_snapshots()
    
    psi_expr = driver.energy_expr()
    psi_val = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
    vol = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
    psi_avg = comm.allreduce(psi_val, op=MPI.SUM) / comm.allreduce(vol, op=MPI.SUM)
    
    assert abs(psi_avg - psi_ref) < 1e-9, f"Power law equilibrium failed: got {psi_avg}, expected {psi_ref}"


def test_traction_units_conversion():
    """Verify that force [N] on mesh [mm] results in traction [MPa]."""
    # This logic is inside GaussianSurfaceLoad._compute_traction
    # traction = F_norm * weights / sum(weights * areas)
    
    # Let's simulate this calculation
    F_norm_N = 1000.0  # 1000 N
    area_mm2 = 200.0   # 200 mm^2
    
    # Uniform weights
    weights = np.ones(1)
    areas = np.array([area_mm2])
    
    norm_factor = np.sum(weights * areas)
    traction_mag = F_norm_N * weights / norm_factor
    
    # Expected traction = Force / Area = 1000 / 200 = 5 MPa
    assert abs(traction_mag[0] - 5.0) < 1e-9
    
    # This confirms that if the mesh is in mm (areas in mm^2) and force in N,
    # the resulting traction is in N/mm^2 = MPa.
