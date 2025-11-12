"""Test accumulated strain energy computation using femur geometry and gait loads."""
import pytest
import numpy as np
import basix
from dolfinx import fem
from pathlib import Path
import sys

# Add parent directory for imports
femurloader_dir = Path(__file__).parent.parent.parent / "femurloader"
if str(femurloader_dir.parent) not in sys.path:
    sys.path.insert(0, str(femurloader_dir.parent))

from femurloader.febio_parser import FEBio2Dolfinx
from femurloader.paths import FemurPaths
from femurloader.femur_remodeller_gait import setup_femur_gait_loading
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
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
    return setup_femur_gait_loading(V, cfg, BW_kg=75.0, n_samples=9)


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
    """Test accumulated strain energy computation over gait cycle."""
    
    def test_accumulated_energy_is_positive(self, mechanics_solver, gait_loader):
        """Accumulated strain energy should be positive."""
        psi_accumulated = mechanics_solver.accumulated_strain_energy_gait(gait_loader)
        assert psi_accumulated > 0.0, \
            f"Accumulated strain energy should be positive, got {psi_accumulated}"
    
    def test_accumulated_energy_physical_range(self, mechanics_solver, gait_loader):
        """Accumulated strain energy should be in physically reasonable range.
        
        For bone tissue under gait loading:
        - Typical strain energy density: 0.01 - 1.0 MPa (10^4 - 10^6 Pa)
        - Daily accumulation (multiple cycles) can be higher
        """
        psi_accumulated = mechanics_solver.accumulated_strain_energy_gait(gait_loader)
        
        # Expect range 10^3 - 10^7 Pa (0.001 - 10 MPa) for accumulated daily stimulus
        assert 1e3 < psi_accumulated < 1e7, \
            f"Accumulated energy should be 10^3-10^7 Pa, got {psi_accumulated:.2e} Pa"
    
    def test_energy_increases_with_load_magnitude(self, femur_setup, gait_loader):
        """Strain energy should increase with load magnitude."""
        domain, facet_tags, V, Q, cfg = femur_setup
        
        # Create solver with low load
        u1 = fem.Function(V, name="u1")
        rho1 = fem.Function(Q, name="rho1")
        A_dir1 = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A1")
        rho1.x.array[:] = 1.0
        A_dir1.x.array[:] = 0.0
        for i in range(3):
            A_dir1.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs1 = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs1 = [
            (gait_loader.t_hip, 2),
            (gait_loader.t_glmed, 2),
            (gait_loader.t_glmax, 2),
        ]
        solver1 = MechanicsSolver(u1, rho1, A_dir1, cfg, dirichlet_bcs1, neumann_bcs1)
        solver1.setup()
        
        # Store original load scale
        original_scale = gait_loader.load_scale
        
        # Compute with base load
        gait_loader.load_scale = 1.0
        psi_base = solver1.accumulated_strain_energy_gait(gait_loader)
        
        # Compute with 2x load
        gait_loader.load_scale = 2.0
        psi_double = solver1.accumulated_strain_energy_gait(gait_loader)
        
        # Restore original scale
        gait_loader.load_scale = original_scale
        
        # Energy should scale approximately as load^2 (linear elasticity)
        # Allow some tolerance due to numerical effects
        ratio = psi_double / psi_base
        assert 3.0 < ratio < 5.0, \
            f"Energy should scale ~4x with 2x load, got ratio={ratio:.2f}"
    
    def test_energy_components_contribution(self, mechanics_solver, gait_loader):
        """Verify that multiple gait phases contribute to accumulated energy."""
        # Get individual phase energies
        quadrature = gait_loader.get_quadrature()
        phase_energies = []
        
        for phase, weight in quadrature:
            gait_loader.update_loads(phase)
            mechanics_solver.assemble_rhs()
            mechanics_solver.solve()
            psi_phase = mechanics_solver.average_strain_energy()
            phase_energies.append((phase, weight, psi_phase))
        
        # All phases should have positive energy
        for phase, weight, psi in phase_energies:
            assert psi > 0.0, f"Phase {phase}% should have positive energy, got {psi}"
        
        # Verify manual accumulation matches method
        manual_accumulated = sum(w * psi for _, w, psi in phase_energies)
        method_accumulated = mechanics_solver.accumulated_strain_energy_gait(gait_loader)
        
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
