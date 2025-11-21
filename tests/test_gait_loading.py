"""
Tests for gait loading configuration, geometry, and force validation.

Combines validation of:
- Coordinate systems (mm vs m)
- Gait cycle force magnitudes and directions
- Quadrature integration properties
- Traction field application
"""

import pytest
import numpy as np
import basix
import pyvista as pv
from mpi4py import MPI
from dolfinx import fem

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.femur_gait import setup_femur_gait_loading
from simulation.config import Config

# Skip if mesh file not found (e.g. in CI without data)
import os
MESH_EXISTS = os.path.exists(FemurPaths.FEMUR_MESH_FEB)

@pytest.mark.skipif(not MESH_EXISTS, reason="Femur mesh file not found")
@pytest.fixture(scope="module")
def femur_context():
    """Load femur mesh and setup gait loader once for all tests."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags

    # Create function spaces
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    V = fem.functionspace(domain, P1_vec)
    
    # Setup config and loader
    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)
    # Use standard mass for validation
    loader = setup_femur_gait_loading(V, mass_tonnes=0.075, n_samples=9)
    
    return {
        "domain": domain,
        "facet_tags": facet_tags,
        "V": V,
        "cfg": cfg,
        "loader": loader,
        "mdl": mdl # Keep model for geometry checks
    }

@pytest.mark.skipif(not MESH_EXISTS, reason="Femur mesh file not found")
class TestGaitGeometry:
    """Validate coordinate systems and mesh scaling."""

    def test_mesh_units_millimeters(self, femur_context):
        """DOLFINx mesh should be in millimeters (coords > 10)."""
        domain = femur_context["domain"]
        # Check max coordinate magnitude
        max_coord = np.max(np.abs(domain.geometry.x))
        # Femur is roughly 400-500mm long
        assert 100.0 < max_coord < 1000.0, f"Mesh coordinates likely not in mm, max={max_coord}"

    def test_loader_scale_unity(self, femur_context):
        """Loader should not scale coordinates if mesh is already in mm."""
        loader = femur_context["loader"]
        assert loader.coord_scale == 1.0

    def test_geometry_consistency(self, femur_context):
        """DOLFINx mesh bounds should match PyVista source bounds."""
        domain = femur_context["domain"]
        comm = domain.comm
        
        # DOLFINx bounds
        x = domain.geometry.x
        local_min = np.min(x, axis=0) if x.size > 0 else np.full(3, np.inf)
        local_max = np.max(x, axis=0) if x.size > 0 else np.full(3, -np.inf)
        
        glob_min = np.zeros(3)
        glob_max = np.zeros(3)
        comm.Allreduce(local_min, glob_min, op=MPI.MIN)
        comm.Allreduce(local_max, glob_max, op=MPI.MAX)
        
        # PyVista bounds (read directly from source)
        # Only rank 0 needs to read and check, but for simplicity we let all read or just rank 0
        if comm.rank == 0:
            pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
            pv_bounds = np.array(pv_mesh.bounds).reshape(3, 2).T # [min, max]
            pv_min = pv_bounds[0]
            pv_max = pv_bounds[1]
            
            # Check range matches
            dolfin_range = glob_max - glob_min
            pv_range = pv_max - pv_min
            
            np.testing.assert_allclose(dolfin_range, pv_range, rtol=0.05, 
                                     err_msg="Geometry extent mismatch between DOLFINx and VTK")

@pytest.mark.skipif(not MESH_EXISTS, reason="Femur mesh file not found")
class TestGaitForces:
    """Validate physiological force magnitudes and integration."""

    def test_quadrature_properties(self, femur_context):
        """Quadrature weights should sum to 1 and cover cycle."""
        loader = femur_context["loader"]
        quad = loader.get_quadrature()
        phases, weights = zip(*quad)
        
        assert np.isclose(sum(weights), 1.0), "Weights must sum to 1.0"
        assert min(phases) == 0.0 and max(phases) == 100.0, "Cycle must cover 0-100%"
        assert len(quad) == loader.n_samples

    @pytest.mark.parametrize("load_type, expected_range_bw", [
        ("hip", (2.3, 4.8)),      # Hip contact force ~2.5-4.5 BW
        ("glmed", (0.3, 2.8)),    # Gluteus medius ~1-2 BW
        ("glmax", (0.1, 2.0))     # Gluteus maximus ~0.5-1.5 BW
    ])
    def test_peak_force_magnitude(self, femur_context, load_type, expected_range_bw):
        """Peak integrated force should be within physiological range (in BW)."""
        loader = femur_context["loader"]
        cfg = femur_context["cfg"]
        comm = cfg.domain.comm
        
        # Find peak phase from interpolator first (fast)
        phases = np.linspace(0, 100, 21)
        max_mag = 0.0
        peak_phase = 0.0
        
        # Get the interpolator function
        if load_type == "hip":
            func = loader.hip_gait
        elif load_type == "glmed":
            func = loader.glmed_gait
        elif load_type == "glmax":
            func = loader.glmax_gait
            
        for p in phases:
            f_vec = func(p)
            mag = np.linalg.norm(f_vec)
            if mag > max_mag:
                max_mag = mag
                peak_phase = p
                
        # Now integrate FE traction at peak phase
        loader.update_loads(peak_phase)
        
        # Select traction function
        if load_type == "hip":
            traction = loader.t_hip
        elif load_type == "glmed":
            traction = loader.t_glmed
        elif load_type == "glmax":
            traction = loader.t_glmax
            
        # Integrate: F = int(t) dA
        # We need to integrate the vector traction. 
        # Since we can't easily assemble vector form to vector global, 
        # we integrate each component.
        import ufl
        F_integrated = np.zeros(3)
        for i in range(3):
            form = fem.form(traction[i] * cfg.ds(2)) # ds(2) is Neumann surface
            local_val = fem.assemble_scalar(form)
            F_integrated[i] = comm.allreduce(local_val, op=MPI.SUM)
            
        F_mag = np.linalg.norm(F_integrated)
        BW = 75.0 * 9.81 # 75kg * g
        ratio = F_mag / BW
        
        assert expected_range_bw[0] <= ratio <= expected_range_bw[1], \
            f"{load_type} peak force {ratio:.2f} BW out of range {expected_range_bw}"

    def test_traction_field_nonzero(self, femur_context):
        """Traction fields should have non-zero support on the mesh."""
        loader = femur_context["loader"]
        comm = femur_context["domain"].comm
        
        # Update to a phase where all forces are likely active (e.g. 20% - stance)
        loader.update_loads(20.0)
        
        for name, func in [("hip", loader.t_hip), ("glmed", loader.t_glmed)]:
            # Check L2 norm of the field
            # We can just check max value in array
            local_max = np.max(np.abs(func.x.array)) if func.x.array.size > 0 else 0.0
            global_max = comm.allreduce(local_max, op=MPI.MAX)
            
            assert global_max > 1e-6, f"{name} traction field is effectively zero (max={global_max})"

