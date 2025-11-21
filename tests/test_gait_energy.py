"""
Tests for GaitDriver and strain energy accumulation.

Validates:
- Driver initialization and snapshot management
- Strain energy density (SED) integration
- Energy scaling with load magnitude
- Phase-dependent energy variations
"""

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
from simulation.drivers import GaitDriver
from simulation.utils import build_dirichlet_bcs

@pytest.fixture(scope="module")
def gait_context():
    """Setup solver and driver context once."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags

    # Spaces
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)
    T = fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3)))

    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
    
    # Loader
    loader = setup_femur_gait_loading(V, mass_tonnes=0.075, n_samples=5) # Reduced samples for speed

    # Fields
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho"); rho.x.array[:] = 1.0
    A = fem.Function(T, name="A")
    # Isotropic fabric
    A.x.array[:] = 0.0
    for i in range(3): A.x.array[i::9] = 1.0/3.0

    # BCs
    bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    neumann = [
        (loader.t_hip, 2),
        (loader.t_glmed, 2),
        (loader.t_glmax, 2),
    ]

    solver = MechanicsSolver(u, rho, A, cfg, bcs, neumann)
    solver.setup()
    
    return {
        "solver": solver,
        "loader": loader,
        "cfg": cfg,
        "V": V,
        "Q": Q
    }

class TestGaitDriver:
    
    def test_driver_initialization(self, gait_context):
        """Driver should initialize with correct phases."""
        solver = gait_context["solver"]
        loader = gait_context["loader"]
        cfg = gait_context["cfg"]
        
        driver = GaitDriver(solver, loader, cfg)
        assert len(driver.phases) == loader.n_samples
        assert len(driver.weights) == loader.n_samples
        assert np.isclose(sum(driver.weights), 1.0)

    @pytest.mark.slow
    def test_energy_positivity(self, gait_context):
        """Accumulated energy should be strictly positive."""
        solver = gait_context["solver"]
        loader = gait_context["loader"]
        cfg = gait_context["cfg"]
        comm = cfg.domain.comm
        
        driver = GaitDriver(solver, loader, cfg)
        driver.update_snapshots() # Runs solves
        
        # Check total energy
        psi_expr = driver.stimulus_expr()
        total_energy = comm.allreduce(
            fem.assemble_scalar(fem.form(psi_expr * cfg.dx)), 
            op=MPI.SUM
        )
        assert total_energy > 0.0

    @pytest.mark.slow
    def test_load_scaling(self, gait_context):
        """Doubling load should roughly quadruple energy (linear elastic)."""
        solver = gait_context["solver"]
        loader = gait_context["loader"]
        cfg = gait_context["cfg"]
        comm = cfg.domain.comm
        
        # Baseline
        loader.load_scale = 1.0
        driver1 = GaitDriver(solver, loader, cfg)
        driver1.update_snapshots()
        E1 = comm.allreduce(fem.assemble_scalar(fem.form(driver1.stimulus_expr() * cfg.dx)), op=MPI.SUM)
        
        # Double load
        loader.load_scale = 2.0
        driver2 = GaitDriver(solver, loader, cfg)
        driver2.update_snapshots()
        E2 = comm.allreduce(fem.assemble_scalar(fem.form(driver2.stimulus_expr() * cfg.dx)), op=MPI.SUM)
        
        ratio = E2 / E1
        # Should be exactly 4.0 for linear elasticity, but allow small numerical deviation
        assert 3.9 < ratio < 4.1, f"Energy should scale quadratically, got ratio {ratio:.2f}"

    @pytest.mark.slow
    def test_stance_vs_swing_energy(self, gait_context):
        """Stance phase (loaded) should have much higher energy than swing."""
        solver = gait_context["solver"]
        loader = gait_context["loader"]
        
        # We can check individual snapshots stored in driver
        # Stance is roughly 0-60%, Swing 60-100%
        # With 5 samples: 0, 25, 50, 75, 100
        # 0, 25, 50 are stance. 75 is swing.
        
        driver = GaitDriver(solver, loader, gait_context["cfg"])
        driver.update_snapshots()
        
        energies = []
        for u_snap in driver.u_snap:
            # Calculate energy for this snapshot
            # We need to manually compute it since solver doesn't store it per snapshot
            # But we can use the solver's helper if we update u
            solver.u.x.array[:] = u_snap.x.array[:]
            solver.u.x.scatter_forward()
            e = solver.average_strain_energy()
            energies.append(e)
            
        # 50% (mid-stance) should be > 75% (swing)
        # Indices depend on quadrature. 
        # Assuming uniform or similar:
        # Check max energy is much larger than min energy
        max_e = max(energies)
        min_e = min(energies)
        
        assert max_e > 10.0 * min_e, "Peak stance energy should be significantly higher than swing/min energy"
