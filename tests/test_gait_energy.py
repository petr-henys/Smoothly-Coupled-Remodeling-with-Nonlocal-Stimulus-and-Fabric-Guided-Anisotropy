"""Tests for gait cycle strain energy accumulation.

Tests the GaitEnergyDriver class for computing accumulated strain energy
over gait cycles, including:
- Driver initialization and setup
- Energy expression building
- Energy positivity and scaling with load magnitude
- Phase-by-phase energy contributions
- Peak stance energy identification

Related test files:
- `test_gait_forces.py`: Force validation and quadrature
- `test_femur_mechanics.py`: Deformation and reaction forces
- `test_gait_geometry.py`: Coordinate system validation
"""

import numpy as np
import pytest
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
def femur_mechanics_setup():
    """Create femur mesh, function spaces, config, and shared gait loader."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)

    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)
    gait_loader = setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)

    return domain, facet_tags, V, Q, cfg, gait_loader


@pytest.fixture
def gait_loader(femur_mechanics_setup):
    """Return gait loader built on the femur mechanics space."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup
    from simulation.femur_gait import setup_femur_gait_loading
    return setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)


@pytest.fixture
def mechanics_solver(femur_mechanics_setup, gait_loader):
    """Create MechanicsSolver with femur geometry and gait loading."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup

    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    A_dir = fem.Function(
        fem.functionspace(domain, basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
        name="A",
    )

    rho.x.array[:] = 1.0
    A_dir.x.array[:] = 0.0
    for i in range(3):
        A_dir.x.array[i::9] = 1.0 / 3.0

    dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    neumann_bcs = [
        (gait_loader.t_hip, 2),
        (gait_loader.t_glmed, 2),
        (gait_loader.t_glmax, 2),
    ]

    solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
    solver.setup()

    return solver


@pytest.mark.slow
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

        cfg = mechanics_solver.cfg
        psi_total_local = fem.assemble_scalar(fem.form(psi_expr * cfg.dx))
        comm = cfg.domain.comm
        psi_total = comm.allreduce(psi_total_local, op=MPI.SUM)

        assert psi_total > 0.0, f"Total strain energy should be positive, got {psi_total}"

    def test_energy_increases_with_load_magnitude(self, mechanics_solver, gait_loader):
        """Strain energy should increase with load magnitude."""
        comm = mechanics_solver.cfg.domain.comm

        gait_loader.load_scale = 1.0
        driver_base = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver_base.update_snapshots()
        psi_expr_base = driver_base.energy_expr()
        psi_base_local = fem.assemble_scalar(fem.form(psi_expr_base * mechanics_solver.cfg.dx))
        psi_base = comm.allreduce(psi_base_local, op=MPI.SUM)

        gait_loader.load_scale = 2.0
        driver_double = GaitEnergyDriver(mechanics_solver, gait_loader, mechanics_solver.cfg)
        driver_double.update_snapshots()
        psi_expr_double = driver_double.energy_expr()
        psi_double_local = fem.assemble_scalar(fem.form(psi_expr_double * mechanics_solver.cfg.dx))
        psi_double = comm.allreduce(psi_double_local, op=MPI.SUM)

        ratio = psi_double / psi_base
        expected = 4.0
        assert 0.5 * expected < ratio < 1.5 * expected, (
            "Energy should scale approximately with load²; " f"expected≈{expected:.2f}, ratio={ratio:.2f}"
        )

    def test_energy_components_contribution(self, mechanics_solver, gait_loader):
        """Verify that gait-averaged energy includes all phases and is positive."""
        cfg = mechanics_solver.cfg
        comm = cfg.domain.comm

        driver = GaitEnergyDriver(mechanics_solver, gait_loader, cfg)
        driver.update_snapshots()

        phase_energies = []
        for u_i, weight in zip(driver.u_snap, driver.weights):
            psi_i = mechanics_solver.get_strain_energy_density(u_i)
            psi_loc = fem.assemble_scalar(fem.form(psi_i * cfg.dx))
            psi_tot = comm.allreduce(psi_loc, op=MPI.SUM)
            phase_energies.append(psi_tot)
        for idx, psi in enumerate(phase_energies):
            assert psi > 0.0, f"Gait phase {idx} should have positive energy, got {psi}"

        psi_expr_driver = driver.energy_expr()
        psi_driver_loc = fem.assemble_scalar(fem.form(psi_expr_driver * cfg.dx))
        psi_driver = comm.allreduce(psi_driver_loc, op=MPI.SUM)

        assert psi_driver > 0.0, f"Driver energy should be positive, got {psi_driver}"

        avg_phase_energy = sum(w * e for w, e in zip(driver.weights, phase_energies))
        expected_scale = cfg.gait_cycles_per_day
        ratio = psi_driver / (avg_phase_energy * expected_scale)

        assert 0.1 < ratio < 10.0, (
            f"Driver daily energy {psi_driver:.2e} should be ~{expected_scale:.0f}× "
            f"weighted average phase energy {avg_phase_energy:.2e}, got ratio {ratio:.2f}"
        )

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

        max_phase = max(phase_energies, key=phase_energies.get)
        max_energy = phase_energies[max_phase]

        assert 0.0 <= max_phase <= 60.0, f"Peak energy should be during stance phase (0-60%), found at {max_phase}%"

        min_energy = min(phase_energies.values())
        ratio = max_energy / min_energy
        assert ratio > 1.5, f"Peak energy should be >1.5x minimum, got ratio={ratio:.2f}"
