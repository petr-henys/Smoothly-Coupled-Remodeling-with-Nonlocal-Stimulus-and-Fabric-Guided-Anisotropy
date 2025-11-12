"""
Pytest configuration and shared fixtures/utilities for the test suite.

Responsibilities:
- Auto-restart with MPI if running serial (for VS Code Test Explorer)
- Ensure project root is importable (adjust sys.path once)
- Silence stdout/stderr on non-root MPI ranks to avoid duplicated output
- Provide deterministic randomness across tests
- Provide common mesh/config/function-space/field fixtures to reduce duplication
- Provide fixtures for femur-specific testing (geometry, gait data, FEBio parsing)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Dict, Any

import numpy as np
import pytest


# =============================================================================
# CHECK OPTIONAL DEPENDENCIES
# =============================================================================
HAS_PYVISTA = False
HAS_MPI = False
HAS_DOLFINX = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pv = None

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    MPI = None

try:
    from dolfinx import mesh, fem
    HAS_DOLFINX = True
except ImportError:
    mesh = None
    fem = None


# =============================================================================
# STANDARD TEST PARAMETERS - Consistent across test suite
# =============================================================================
# Mesh sizes for parametrized tests
DEFAULT_MESH_SIZES = [6, 8]     # Standard parametrization
SMOKE_MESH_SIZE = 4              # Quick smoke tests
TINY_MESH_SIZE = 2               # Minimal mesh (may have empty ranks)

# Tolerance presets
class TolerancePresets:
    """Standard tolerance configurations for different test types."""
    
    # VERIFICATION-GRADE (default for correctness tests)
    STANDARD = {
        'coupling_tol': 5e-8,
        'ksp_rtol': 5e-10,
        'ksp_atol': 5e-13,
        'max_subiters': 60,
    }
    
    # SMOKE TESTS (fast, relaxed)
    SMOKE = {
        'coupling_tol': 5e-4,
        'ksp_rtol': 5e-7,
        'ksp_atol': 5e-10,
        'max_subiters': 12,
    }
    
    # PERFORMANCE BENCHMARKS (realistic)
    BENCHMARK = {
        'coupling_tol': 5e-6,
        'ksp_rtol': 5e-9,
        'ksp_atol': 5e-12,
        'max_subiters': 40,
    }
    
    # NUMERICAL TEST TOLERANCES
    ABS_TOL = 1e-12
    REL_TOL = 1e-5
    GEOMETRIC_TOL = 1e-6  # For geometric operations
    FORCE_TOL = 1e-1  # For force equilibrium checks (Newtons)

# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register markers and silence non-rank-0 output."""
    if MPI is not None and MPI.COMM_WORLD.rank != 0:
        sys.stdout = sys.stderr = open(os.devnull, 'w')
    
    # Register markers
    for marker, desc in [
        ("smoke", "fast sanity checks for basic functionality"),
        ("mpi", "tests that are intended for MPI environments"),
        ("slow", "tests that may take longer to run"),
        ("performance", "tests focused on performance characteristics"),
        ("integration", "end-to-end or multi-component tests"),
        ("unit", "fast unit-level tests"),
        ("febio", "tests requiring FEBio file parsing"),
        ("gait_data", "tests requiring gait data processing"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {desc}")


# Add workspace root to sys.path (where simulation/ folder is located)
tests_dir = os.path.dirname(__file__)
workspace_root = os.path.abspath(os.path.join(tests_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


@pytest.fixture(autouse=True)
def _seed_numpy_for_determinism():
    """Seed numpy RNG for deterministic test behavior."""
    import numpy as np
    np.random.seed(1234)


# =============================================================================
# Shared computational fixtures
# =============================================================================

class Spaces(NamedTuple):
    V: object  # fem.FunctionSpace (vector)
    Q: object  # fem.FunctionSpace (scalar)
    T: object  # fem.FunctionSpace (second-order tensor)


class Fields(NamedTuple):
    u: object
    rho: object
    rho_old: object
    A: object
    A_old: object
    S: object
    S_old: object


@pytest.fixture
def unit_cube(request) -> object:
    """Create a unit cube mesh with optional resolution via parametrization.
    
    Usage: @pytest.mark.parametrize("unit_cube", [6, 8], indirect=True)
    """
    from dolfinx import mesh
    n = int(getattr(request, "param", 8))
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.shared_facet)


@pytest.fixture
def facet_tags(unit_cube) -> object:
    from simulation.utils import build_facetag
    return build_facetag(unit_cube)


@pytest.fixture
def cfg(unit_cube, facet_tags) -> object:
    from simulation.config import Config
    return Config(domain=unit_cube, facet_tags=facet_tags, verbose=False)


@pytest.fixture
def spaces(unit_cube) -> Spaces:
    from dolfinx import fem
    import basix
    domain = unit_cube
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3,))
    P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))
    return Spaces(
        V=fem.functionspace(domain, P1_vec),
        Q=fem.functionspace(domain, P1),
        T=fem.functionspace(domain, P1_ten)
    )


@pytest.fixture
def bc_mech(spaces, facet_tags) -> list:
    from simulation.utils import build_dirichlet_bcs
    return build_dirichlet_bcs(spaces.V, facet_tags, id_tag=1, value=0.0)


@pytest.fixture
def fields(spaces) -> Fields:
    """Create a set of default-initialized fields commonly used across tests."""
    from dolfinx import fem
    import numpy as np
    V, Q, T = spaces.V, spaces.Q, spaces.T
    
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    rho_old = fem.Function(Q, name="rho_old")
    A = fem.Function(T, name="A")
    A_old = fem.Function(T, name="A_old")
    S = fem.Function(Q, name="S")
    S_old = fem.Function(Q, name="S_old")
    
    # Default values
    rho.x.array[:] = 0.5
    rho.x.scatter_forward()
    A.interpolate(lambda x: (np.eye(3) / 3.0).flatten()[:, None] * np.ones((1, x.shape[1])))
    A.x.scatter_forward()
    S.x.array[:] = 0.0
    S.x.scatter_forward()
    
    return Fields(u=u, rho=rho, rho_old=rho_old, A=A, A_old=A_old, S=S, S_old=S_old)


@pytest.fixture
def traction_factory(cfg):
    """Factory to build a constant traction vector on a given facet.
    
    Usage: traction = traction_factory(value=-0.3, facet_id=2, axis=0)
    """
    def _make(value: float, facet_id: int = 2, axis: int = 0):
        from dolfinx import fem
        import numpy as np
        vec = np.zeros(3, dtype=float)
        vec[axis] = float(value)
        return (fem.Constant(cfg.domain, vec), int(facet_id))
    return _make


@pytest.fixture
def shared_tmpdir():
    """Create a shared temporary directory for MPI tests.
    
    In MPI environments, each rank creates a separate tempdir. This fixture
    broadcasts rank 0's directory to all ranks for shared file I/O.
    Returns a Path object for easy path manipulation.
    """
    from pathlib import Path
    
    if not HAS_MPI or MPI is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
        return
    
    comm = MPI.COMM_WORLD
    
    # Rank 0 creates the temp directory, others set to None
    tmpdir_obj = None
    if comm.rank == 0:
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
    else:
        tmpdir = None
    
    # Broadcast path to all ranks
    tmpdir = comm.bcast(tmpdir, root=0)
    
    try:
        yield Path(tmpdir)
    finally:
        # Barrier before cleanup to ensure all ranks finished using the directory
        comm.Barrier()
        # Only rank 0 cleans up (it's the only one with tmpdir_obj)
        if comm.rank == 0 and tmpdir_obj is not None:
            tmpdir_obj.cleanup()


@pytest.fixture
def mean_value_factory(cfg):
    """Factory to compute global mean value of a scalar UFL expression.
    
    Usage: mean = mean_value_factory(); result = mean(expr)
    """
    from dolfinx import fem

    def _mean(expr, *, dx=None) -> float:
        dxm = dx if dx is not None else cfg.dx
        local = fem.assemble_scalar(fem.form(expr * dxm))
        vol_local = fem.assemble_scalar(fem.form(1.0 * dxm))
        comm = cfg.domain.comm if hasattr(cfg, "domain") else (MPI.COMM_WORLD if HAS_MPI and MPI is not None else None)
        if comm is None or not HAS_MPI or MPI is None:
            tot = local
            vol = vol_local
        else:
            tot = comm.allreduce(local, op=MPI.SUM)
            vol = comm.allreduce(vol_local, op=MPI.SUM)
        return float(tot / max(vol, 1e-300))

    return _mean


@pytest.fixture
def cfg_factory(unit_cube, facet_tags):
    """Factory to create Config with preset tolerance configurations.
    
    Usage:
        cfg = cfg_factory('standard')  # verification-grade
        cfg = cfg_factory('smoke')     # fast, relaxed
        cfg = cfg_factory('benchmark') # realistic performance
        cfg = cfg_factory()            # default = standard
    """
    from simulation.config import Config
    
    def _make_config(preset='standard', **overrides):
        """Create Config with specified preset and optional overrides."""
        preset_lower = preset.lower()
        
        if preset_lower == 'smoke':
            params = TolerancePresets.SMOKE.copy()
        elif preset_lower == 'benchmark':
            params = TolerancePresets.BENCHMARK.copy()
        else:  # 'standard' or default
            params = TolerancePresets.STANDARD.copy()
        
        # Apply any user overrides
        params.update(overrides)
        
        # Create config with preset parameters
        return Config(
            domain=unit_cube,
            facet_tags=facet_tags,
            verbose=params.pop('verbose', False),
            **params
        )
    
    return _make_config


# =============================================================================
# FEMUR-SPECIFIC FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def simple_unit_cube():
    """Simple unit cube mesh for basic testing (module-scoped)."""
    from dolfinx import mesh
    return mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)


@pytest.fixture(scope="module")
def sample_femur_mesh():
    """Create a simple femur-like mesh for testing.
    
    Creates a cylinder (shaft) with a sphere on top (head).
    """
    import pyvista as pv
    
    cylinder = pv.Cylinder(
        center=(0, -50, 0),
        direction=(0, 1, 0),
        radius=15,
        height=100
    )
    sphere = pv.Sphere(center=(0, 0, 0), radius=25)
    femur = cylinder + sphere
    return femur


@pytest.fixture
def fine_femur_mesh():
    """Higher resolution femur mesh for convergence tests."""
    import pyvista as pv
    
    cylinder = pv.Cylinder(
        center=(0, -50, 0),
        direction=(0, 1, 0),
        radius=15,
        height=100,
        resolution=50
    )
    sphere = pv.Sphere(center=(0, 0, 0), radius=25, theta_resolution=30, phi_resolution=30)
    femur = cylinder + sphere
    return femur


# =============================================================================
# GEOMETRIC DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_head_line():
    """Sample head line points for femoral head fitting."""
    return np.array([
        [-20.0, 0.0, 0.0],
        [20.0, 0.0, 0.0]
    ])


@pytest.fixture
def sample_le_me_line():
    """Sample lateral/medial epicondyle line."""
    return np.array([
        [0.0, -100.0, -30.0],  # Lateral epicondyle
        [0.0, -100.0, 30.0]    # Medial epicondyle
    ])


@pytest.fixture
def sample_muscle_points():
    """Sample muscle attachment points for testing."""
    t = np.linspace(0, 1, 5)
    points = np.zeros((5, 3))
    points[:, 0] = 10 * t  # anterior-posterior
    points[:, 1] = -20 - 30 * t  # superior-inferior 
    points[:, 2] = 15 * np.sin(np.pi * t)  # medial-lateral
    return points


# =============================================================================
# GAIT DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_gait_data():
    """Sample gait cycle data for testing."""
    gait_cycle = np.linspace(0, 100, 51)  # 0-100% gait cycle
    gait_values = np.sin(2 * np.pi * gait_cycle / 100)  # Sinusoidal pattern
    return np.column_stack([gait_cycle, gait_values])


@pytest.fixture
def sample_force_vector():
    """Sample force vector for testing."""
    return np.array([100.0, -500.0, 200.0])  # Realistic hip force components


@pytest.fixture
def realistic_gait_force_data():
    """Realistic gait force data with all components."""
    gait_cycle = np.linspace(0, 100, 51)
    # Realistic hip joint force patterns
    fx = 200 * np.sin(2 * np.pi * gait_cycle / 100)  # Anterior-posterior
    fy = -1000 * (1 + 0.5 * np.sin(2 * np.pi * gait_cycle / 100))  # Superior-inferior
    fz = 150 * np.cos(2 * np.pi * gait_cycle / 100)  # Medial-lateral
    
    return np.column_stack([gait_cycle, fx, fy, fz])


# =============================================================================
# FILE FIXTURES (TEMPORARY)
# =============================================================================

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for testing."""
    data = {
        "markups": [{
            "controlPoints": [
                {"position": [10.0, 20.0, 30.0]},
                {"position": [15.0, 25.0, 35.0]},
                {"position": [20.0, 30.0, 40.0]}
            ]
        }]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    temp_path.unlink()


@pytest.fixture
def temp_vtk_file():
    """Create a temporary VTK file path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.vtk', delete=False) as f:
        temp_path = Path(f.name)
    
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# =============================================================================
# FEBIO XML FIXTURES
# =============================================================================

@pytest.fixture
def minimal_febio_xml():
    """Minimal valid FEBio XML content for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Part1">
            <elem id="1">1,2,3,4</elem>
        </Elements>
        <Surface name="Surface1">
            <tri3 id="1">1,2,3</tri3>
        </Surface>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Part1" mat="Material1"/>
    </MeshDomains>
    <Material>
        <material id="1" name="Material1">
            <E>1000.0</E>
            <v>0.3</v>
        </material>
    </Material>
</febio_spec>
"""
    return xml_content


@pytest.fixture
def complex_febio_xml():
    """More complex FEBio XML with multiple element groups and surfaces."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<febio_spec version="4.0">
    <Mesh>
        <Nodes name="AllNodes">
            <node id="1">0.0,0.0,0.0</node>
            <node id="2">1.0,0.0,0.0</node>
            <node id="3">0.0,1.0,0.0</node>
            <node id="4">0.0,0.0,1.0</node>
            <node id="5">2.0,0.0,0.0</node>
            <node id="6">1.0,1.0,0.0</node>
            <node id="7">1.0,0.0,1.0</node>
        </Nodes>
        <Elements type="tet4" name="Bone">
            <elem id="1">1,2,3,4</elem>
            <elem id="2">2,5,6,7</elem>
        </Elements>
        <Elements type="tet4" name="Cartilage">
            <elem id="3">2,3,6,7</elem>
        </Elements>
        <Surface name="TopSurface">
            <tri3 id="1">3,4,6</tri3>
        </Surface>
        <Surface name="BottomSurface">
            <tri3 id="2">1,2,3</tri3>
        </Surface>
        <DiscreteSet name="Ligament1">
            <delem>1,2</delem>
            <delem>2,3</delem>
        </DiscreteSet>
    </Mesh>
    <MeshDomains>
        <SolidDomain name="Bone" mat="BoneMaterial"/>
        <SolidDomain name="Cartilage" mat="CartilageMaterial"/>
    </MeshDomains>
    <Material>
        <material id="1" name="BoneMaterial">
            <E>10000.0</E>
            <v>0.3</v>
        </material>
        <material id="2" name="CartilageMaterial">
            <E>500.0</E>
            <v>0.45</v>
        </material>
    </Material>
</febio_spec>
"""
    return xml_content


@pytest.fixture
def temp_febio_file(minimal_febio_xml):
    """Create a temporary FEBio file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.feb', delete=False) as f:
        f.write(minimal_febio_xml)
        temp_path = Path(f.name)
    
    yield temp_path
    temp_path.unlink()


# =============================================================================
# EXCEL/CSV DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_excel_data_dict():
    """Sample data structure mimicking Excel XY datasets."""
    return {
        "Dataset_1": np.array([
            [0, 100],
            [25, 150],
            [50, 200],
            [75, 150],
            [100, 100]
        ]),
        "Dataset_2": np.array([
            [0, 50],
            [25, 75],
            [50, 100],
            [75, 75],
            [100, 50]
        ])
    }


@pytest.fixture
def sample_hip_file_content():
    """Sample HIP file content for gait data parsing."""
    content = """Orthoload Database - Hip Joint Forces
Subject: Walking Average
Peak Resultant Force: F = 2500N

Cycle [%]	Fx [N]	Fy [N]	Fz [N]	F [N]	Time [s]
0	100	-1000	200	1030.78	0.0
10	150	-1200	250	1241.90	0.1
20	200	-1500	300	1534.51	0.2
30	180	-1300	280	1340.75	0.3
40	120	-1100	220	1126.94	0.4
50	100	-1000	200	1030.78	0.5
60	90	-950	180	971.88	0.6
70	110	-1050	210	1074.63	0.7
80	130	-1150	240	1178.51	0.8
90	140	-1250	260	1282.81	0.9
100	100	-1000	200	1030.78	1.0
"""
    return content


@pytest.fixture
def temp_hip_file(sample_hip_file_content):
    """Create a temporary HIP file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.HIP', delete=False) as f:
        f.write(sample_hip_file_content)
        temp_path = Path(f.name)
    
    yield temp_path
    temp_path.unlink()


# =============================================================================
# FUNCTION SPACE FIXTURES (DOLFINx)
# =============================================================================

@pytest.fixture
def scalar_dg0_space(simple_unit_cube):
    """DG0 scalar function space."""
    from dolfinx import fem
    return fem.functionspace(simple_unit_cube, ("DG", 0))


@pytest.fixture
def vector_dg0_space(simple_unit_cube):
    """DG0 vector function space."""
    from dolfinx import fem
    return fem.functionspace(simple_unit_cube, ("DG", 0, (3,)))


@pytest.fixture
def tensor_dg0_space(simple_unit_cube):
    """DG0 tensor function space."""
    from dolfinx import fem
    return fem.functionspace(simple_unit_cube, ("DG", 0, (3, 3)))


@pytest.fixture
def continuous_p1_space(simple_unit_cube):
    """CG1 continuous scalar space."""
    from dolfinx import fem
    return fem.functionspace(simple_unit_cube, ("Lagrange", 1))


# =============================================================================
# COMMON PARAMETER SETS FOR PARAMETRIZED TESTS
# =============================================================================

# Common parameter sets for parametrized tests
VALID_SIDES = ["left", "right"]
INVALID_SIDES = ["middle", "both", "center", ""]

FORCE_DIRECTIONS = [
    np.array([1, 0, 0]),    # Anterior
    np.array([0, -1, 0]),   # Inferior
    np.array([0, 0, 1]),    # Medial
    np.array([1, -1, 1]),   # Combined
]

SIGMA_VALUES = [1.0, 5.0, 10.0, 20.0]
