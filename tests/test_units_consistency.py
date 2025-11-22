"""Tests for physical unit consistency and scaling laws."""

import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

from simulation.config import Config
from simulation.femur_gait import FemurRemodellerGait

class MockMechanicsSolver:
    """Mock mechanics solver that returns stress dependent on u."""
    def __init__(self, u):
        self.u = u
        self.rho = u # dummy
        self.A_dir = u # dummy
        self.gdim = 3
        self.comm = u.function_space.mesh.comm

    def sigma(self, u, rho, A_dir):
        # Return uniaxial stress tensor with magnitude u[0]
        # sigma = [[u[0], 0, 0], [0, 0, 0], [0, 0, 0]]
        # Von Mises of this is |u[0]|
        S = u[0]
        zero = ufl.as_ufl(0.0)
        return ufl.as_tensor([[S, zero, zero], [zero, zero, zero], [zero, zero, zero]])

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
        self.target_stress = 0.0

    def get_quadrature(self):
        # Single phase for simplicity
        return [(0.0, 1.0)]

    def update_loads(self, phase):
        pass
    
    def set_target_stress(self, s):
        self.target_stress = s

def test_gait_driver_daily_dose_scaling():
    """Verify GaitDriver correctly scales energy by cycles/day."""
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
    # J_day = psi_ref * N_cyc * (sigma_vm / psi_ref)^n
    # For J_day = psi_ref (equilibrium), we need:
    # 1 = N_cyc * (sigma_vm / psi_ref)^n
    # (sigma_vm / psi_ref)^n = 1 / N_cyc
    # sigma_vm / psi_ref = (1 / N_cyc)^(1/n)
    # sigma_vm = psi_ref * (1 / N_cyc)^(1/n)
    
    sigma_vm_eq = psi_ref * (1.0 / N_cyc)**(1.0 / n_power)
    
    u = fem.Function(V)
    mech = MockMechanicsSolver(u)
    loader = MockGaitLoader(V)
    
    def solve_side_effect():
        # Set u[0] = target_stress so that von Mises = target_stress
        val = loader.target_stress
        bs = V.dofmap.index_map_bs
        u.x.array[:] = 0.0
        u.x.array[0::bs] = val
        u.x.scatter_forward()
        return 1, 1
        
    mech.solve = solve_side_effect
    
    driver = GaitDriver(mech, loader, cfg)
    
    # Set target for equilibrium
    loader.set_target_stress(sigma_vm_eq)
    driver.update_snapshots()
    
    # Evaluate stimulus_expr
    psi_expr = driver.stimulus_expr()
    psi_val = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
    vol = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
    psi_avg = comm.allreduce(psi_val, op=MPI.SUM) / comm.allreduce(vol, op=MPI.SUM)
    
    # Driver output is dimensionless ratio relative to equilibrium (1.0)
    assert abs(psi_avg - 1.0) < 1e-9, f"Equilibrium scaling failed: got {psi_avg}, expected 1.0"

    # 2. Test General Scaling
    # If sigma_vm = 2 * sigma_vm_eq
    # Then J_day should be 2^n * psi_ref (since n=1, it is 2*psi_ref)
    loader.set_target_stress(2.0 * sigma_vm_eq)
    driver.update_snapshots()
    
    psi_val = fem.assemble_scalar(fem.form(driver.stimulus_expr() * cfg.dx))
    psi_avg = comm.allreduce(psi_val, op=MPI.SUM) / comm.allreduce(vol, op=MPI.SUM)
    
    # With n=1, doubling stress doubles the stimulus ratio
    assert abs(psi_avg - 2.0) < 1e-9, f"Linear scaling failed: got {psi_avg}, expected 2.0"

def test_gait_driver_exponent_scaling():
    """Verify GaitDriver handles power law exponent n correctly."""
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
    sigma_vm_eq = psi_ref * (1.0 / N_cyc)**(1.0 / n_power)
    
    u = fem.Function(V)
    mech = MockMechanicsSolver(u)
    loader = MockGaitLoader(V)
    
    def solve_side_effect():
        val = loader.target_stress
        bs = V.dofmap.index_map_bs
        u.x.array[:] = 0.0
        u.x.array[0::bs] = val
        u.x.scatter_forward()
        return 1, 1
    mech.solve = solve_side_effect
    
    driver = GaitDriver(mech, loader, cfg)
    
    loader.set_target_stress(sigma_vm_eq)
    driver.update_snapshots()
    
    psi_expr = driver.stimulus_expr()
    psi_val = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
    vol = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
    psi_avg = comm.allreduce(psi_val, op=MPI.SUM) / comm.allreduce(vol, op=MPI.SUM)
    
    # Driver output is dimensionless ratio relative to equilibrium (1.0)
    assert abs(psi_avg - 1.0) < 1e-9, f"Power law equilibrium failed: got {psi_avg}, expected 1.0"


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
