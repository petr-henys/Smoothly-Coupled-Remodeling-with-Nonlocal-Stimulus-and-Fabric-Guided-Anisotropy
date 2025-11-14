"""Test accumulated strain energy computation using femur geometry and gait loads."""
import pytest
import numpy as np


import basix
from mpi4py import MPI
from dolfinx import fem

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.femur_gait import setup_femur_gait_loading
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
from simulation.drivers import GaitEnergyDriver
from simulation.utils import build_dirichlet_bcs


@pytest.fixture(scope="module")
def femur_setup():
    """Create femur mesh, function spaces, and config."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags
    
    # Create function spaces
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)
    
    # Create config
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)
    
    return domain, facet_tags, V, Q, cfg


@pytest.fixture(scope="module")
def gait_loader(femur_setup):
    """Create gait loader with hip and muscle loads."""
    _, _, V, _, cfg = femur_setup
    return setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)


@pytest.fixture
def mechanics_solver(femur_setup, gait_loader):
    """Create MechanicsSolver with femur geometry and gait loading."""
    domain, facet_tags, V, Q, cfg = femur_setup
    
    # Create field functions
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    A_dir = fem.Function(fem.functionspace(domain, 
                         basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                         name="A")
    
    # Initialize with uniform density and isotropic fabric
    rho.x.array[:] = 1.0  # Normalized density
    A_dir.x.array[:] = 0.0
    for i in range(3):
        A_dir.x.array[i::9] = 1.0/3.0  # Isotropic fabric (1/3 * I)
    
    # Build boundary conditions (fix distal end - tag 1)
    dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    
    # Neumann BCs from gait loader (applied on femur surface - tag 2)
    # Note: All gait loads (hip, glmed, glmax) are applied on femur_surface
    neumann_bcs = [
        (gait_loader.t_hip, 2),     # Hip joint on femur surface
        (gait_loader.t_glmed, 2),   # Glut med on femur surface
        (gait_loader.t_glmax, 2),   # Glut max on femur surface
    ]
    
    # Create and setup solver
    solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
    solver.setup()
    
    return solver


class TestAccumulatedStrainEnergy:
    """Test accumulated strain energy computation over gait cycle using GaitEnergyDriver."""
    
    def test_energy_driver_initialization(self, mechanics_solver, gait_loader):
        """GaitEnergyDriver should initialize without errors."""
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        assert driver.mech is mechanics_solver
        assert driver.gait is gait_loader
        assert len(driver.phases) == gait_loader.n_samples
        assert len(driver.weights) == gait_loader.n_samples
    
    def test_energy_expr_builds(self, mechanics_solver, gait_loader):
        """Energy expression should build successfully."""
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()
        assert psi_expr is not None, "Energy expression should be created"
    
    def test_energy_positivity(self, mechanics_solver, gait_loader):
        """Accumulated energy should be positive when integrated."""
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()
        
        # Integrate energy over domain
        cfg = mechanics_solver.cfg
        psi_total_local = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        comm = cfg.domain.comm
        psi_total = comm.allreduce(psi_total_local, op=MPI.SUM)
        
        assert psi_total > 0.0, f"Total strain energy should be positive, got {psi_total}"
    
    def test_energy_increases_with_load_magnitude(self, mechanics_solver, gait_loader):
        """Strain energy should increase with load magnitude."""
        # Store original load scale
        original_scale = gait_loader.load_scale
        comm = mechanics_solver.cfg.domain.comm
        
        # Compute with base load
        gait_loader.load_scale = 1.0
        driver_base = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver_base.update_snapshots()
        psi_expr_base = driver_base.energy_expr()
        psi_base_local = fem.assemble_scalar(fem.form(psi_expr_base * mechanics_solver.cfg.dx))
        psi_base = comm.allreduce(psi_base_local, op=MPI.SUM)
        
        # Compute with 2x load
        gait_loader.load_scale = 2.0
        driver_double = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver_double.update_snapshots()
        psi_expr_double = driver_double.energy_expr()
        psi_double_local = fem.assemble_scalar(fem.form(psi_expr_double * mechanics_solver.cfg.dx))
        psi_double = comm.allreduce(psi_double_local, op=MPI.SUM)
        
        # Restore original scale
        gait_loader.load_scale = original_scale
        
        # Energy should scale approximately as load^2 (linear elasticity)
        # Driver energy uses (psi/psi_ref)^n formulation, hence load^(2*n_power)
        ratio = psi_double / psi_base
        expected = 2.0 ** (2.0 * mechanics_solver.cfg.n_power)
        assert 0.5 * expected < ratio < 1.5 * expected, (
            "Energy should scale with load^(2*n_power); "
            f"expected≈{expected:.2f}, ratio={ratio:.2f}"
        )
    
    def test_energy_components_contribution(self, mechanics_solver, gait_loader):
        """Verify that multiple gait phases contribute to accumulated energy."""
        # Get individual phase energies (integrals over domain)
        quadrature = gait_loader.get_quadrature()
        phase_energies = []
        comm = mechanics_solver.cfg.domain.comm
        cfg = mechanics_solver.cfg
        
        for phase, weight in quadrature:
            gait_loader.update_loads(phase)
            mechanics_solver.assemble_rhs()
            mechanics_solver.solve()
            psi_density = mechanics_solver.get_strain_energy_density(mechanics_solver.u)
            psi_norm = (psi_density / cfg.psi_ref) ** cfg.n_power
            psi_local = fem.assemble_scalar(fem.form(psi_norm * cfg.dx))
            psi_total = comm.allreduce(psi_local, op=MPI.SUM)
            phase_energies.append((phase, weight, psi_total))
        
        # All phases should have positive energy
        for phase, weight, psi in phase_energies:
            assert psi > 0.0, f"Phase {phase}% should have positive energy, got {psi}"
        
        # Verify manual accumulation matches GaitEnergyDriver
        # Both compute weighted sum of energy integrals
        manual_accumulated = sum(w * psi for _, w, psi in phase_energies)
        
        driver = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver.update_snapshots()
        psi_expr = driver.energy_expr()
        method_local = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        method_accumulated = comm.allreduce(method_local, op=MPI.SUM)
        
        np.testing.assert_allclose(manual_accumulated, method_accumulated, rtol=1e-6,
            err_msg="Manual and method accumulation should match")
    
    def test_quadrature_weights_sum_to_one(self, gait_loader):
        """Verify that quadrature weights sum to 1 (proper normalization)."""
        quadrature = gait_loader.get_quadrature()
        total_weight = sum(w for _, w in quadrature)
        np.testing.assert_allclose(total_weight, 1.0, rtol=1e-10,
            err_msg="Quadrature weights should sum to 1.0")
    
    def test_peak_stance_has_maximum_energy(self, mechanics_solver, gait_loader):
        """Peak stance phase should have highest strain energy."""
        quadrature = gait_loader.get_quadrature()
        phase_energies = {}
        
        for phase, weight in quadrature:
            gait_loader.update_loads(phase)
            mechanics_solver.assemble_rhs()
            mechanics_solver.solve()
            psi_phase = mechanics_solver.average_strain_energy()
            phase_energies[phase] = psi_phase
        
        # Find peak energy phase
        max_phase = max(phase_energies, key=phase_energies.get)
        max_energy = phase_energies[max_phase]
        
        # Peak should be during stance phase (first half of gait cycle, 0-50%)
        # This is where hip joint reaction forces are highest
        assert 0.0 <= max_phase <= 60.0, \
            f"Peak energy should be during stance phase (0-60%), found at {max_phase}%"
        
        # Peak should be significantly higher than minimum
        min_energy = min(phase_energies.values())
        ratio = max_energy / min_energy
        assert ratio > 1.5, \
            f"Peak energy should be >1.5x minimum, got ratio={ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
