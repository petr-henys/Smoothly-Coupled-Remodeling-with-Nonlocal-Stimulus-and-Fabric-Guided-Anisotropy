"""Tests for gait geometry and coordinate system validation.

Tests coordinate system consistency between DOLFINx and PyVista:
- Millimeter scaling verification
- Coordinate bounds matching
- Geometry range consistency

Related test files:
- `test_gait_energy.py`: Strain energy accumulation
- `test_gait_forces.py`: Force validation
- `test_femur_mechanics.py`: Deformation and reaction forces
"""

import numpy as np
import pytest
import basix
import pyvista as pv
from mpi4py import MPI
from dolfinx import fem

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.femur_gait import setup_femur_gait_loading
from simulation.config import Config


@pytest.fixture(scope="module")
def femur_mechanics_setup():
    """Create femur mesh, function spaces, and config."""
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


@pytest.fixture(scope="module")
def femur_geometry_setup(femur_mechanics_setup):
    """Geometry setup for coordinate tests."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup
    cfg.E0 = 17e3
    unit_scale = 1.0
    return domain, facet_tags, V, Q, cfg, unit_scale


@pytest.fixture
def gait_loader(femur_mechanics_setup):
    """Return gait loader for coordinate tests."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup
    from simulation.femur_gait import setup_femur_gait_loading

    return setup_femur_gait_loading(V, BW_kg=75.0, n_samples=9)


@pytest.mark.slow
class TestCoordinateScaling:
    """Verify DOLFINx mesh and femurloader use consistent coordinate systems."""

    def test_dolfinx_mesh_in_millimeters(self, femur_geometry_setup):
        """Verify DOLFINx mesh coordinates are in millimeters (not meters)."""
        domain, _, _, _, _, _ = femur_geometry_setup
        geom = domain.geometry.x

        max_coord = np.max(np.abs(geom))
        assert 10.0 < max_coord < 500.0, f"Mesh coords should be in mm (expected 10-500, got {max_coord})"

    def test_femurloader_mesh_in_millimeters(self, femur_geometry_setup):
        """Verify femurloader PyVista mesh is in millimeters."""
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))

        max_coord = np.max(np.abs(pv_mesh.points))
        assert 10.0 < max_coord < 500.0, f"PyVista mesh should be in mm (expected 10-500, got {max_coord})"

    def test_coord_scale_is_unity(self, gait_loader):
        """Verify coord_scale=1.0 (no conversion needed)."""
        assert gait_loader.coord_scale == 1.0, "coord_scale should be 1.0 since both DOLFINx and femurloader use mm"

    def test_geometry_bounds_match(self):
        """Verify DOLFINx and PyVista geometries have same bounds."""
        mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
        domain = mdl.mesh_dolfinx

        dolfinx_geom = domain.geometry.x
        comm = domain.comm
        if dolfinx_geom.size == 0:
            local_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            local_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        else:
            local_min = np.min(dolfinx_geom, axis=0).astype(np.float64)
            local_max = np.max(dolfinx_geom, axis=0).astype(np.float64)

        global_min = np.zeros_like(local_min)
        comm.Allreduce(local_min, global_min, op=MPI.MIN)
        
        global_max = np.zeros_like(local_max)
        comm.Allreduce(local_max, global_max, op=MPI.MAX)

        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        pv_geom = pv_mesh.points

        dolfinx_range = global_max - global_min
        pv_range = np.ptp(pv_geom.copy(), axis=0)

        np.testing.assert_allclose(
            dolfinx_range, pv_range, rtol=0.10, err_msg="DOLFINx and PyVista geometry ranges should match"
        )
