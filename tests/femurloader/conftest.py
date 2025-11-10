"""
Shared fixtures and configuration for femurloader tests.

This module provides common test fixtures, test data generators, and
configuration for the femurloader test suite.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest


# =============================================================================
# ADD REPOSITORY ROOT TO sys.path
# =============================================================================
# Add repository root to sys.path so femurloader can be imported
tests_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(tests_dir, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Check if optional dependencies are available (lazy imports)
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


# ============================================================================
# TEST TOLERANCES
# ============================================================================

class TolerancePresets:
    """Standard tolerance presets for numerical tests."""
    ABS_TOL = 1e-12
    REL_TOL = 1e-5
    GEOMETRIC_TOL = 1e-6  # For geometric operations
    FORCE_TOL = 1e-1  # For force equilibrium checks (Newtons)


# ============================================================================
# MESH FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def simple_unit_cube():
    """Simple unit cube mesh for basic testing."""
    if not HAS_DOLFINX or not HAS_MPI:
        pytest.skip("DOLFINx or MPI not installed")
    return mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)


@pytest.fixture(scope="module")
def sample_femur_mesh():
    """Create a simple femur-like mesh for testing.
    
    Creates a cylinder (shaft) with a sphere on top (head).
    """
    if not HAS_PYVISTA:
        pytest.skip("PyVista not installed")
    
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
    if not HAS_PYVISTA:
        pytest.skip("PyVista not installed")
    
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


# ============================================================================
# GEOMETRIC DATA FIXTURES
# ============================================================================

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


# ============================================================================
# GAIT DATA FIXTURES
# ============================================================================

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


# ============================================================================
# FILE FIXTURES (TEMPORARY)
# ============================================================================

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


# ============================================================================
# FEBIO XML FIXTURES
# ============================================================================

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


# ============================================================================
# EXCEL/CSV DATA FIXTURES
# ============================================================================

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


# ============================================================================
# FUNCTION SPACE FIXTURES (DOLFINx)
# ============================================================================

@pytest.fixture
def scalar_dg0_space(simple_unit_cube):
    """DG0 scalar function space."""
    if not HAS_DOLFINX:
        pytest.skip("DOLFINx not installed")
    return fem.functionspace(simple_unit_cube, ("DG", 0))


@pytest.fixture
def vector_dg0_space(simple_unit_cube):
    """DG0 vector function space."""
    if not HAS_DOLFINX:
        pytest.skip("DOLFINx not installed")
    return fem.functionspace(simple_unit_cube, ("DG", 0, (3,)))


@pytest.fixture
def tensor_dg0_space(simple_unit_cube):
    """DG0 tensor function space."""
    if not HAS_DOLFINX:
        pytest.skip("DOLFINx not installed")
    return fem.functionspace(simple_unit_cube, ("DG", 0, (3, 3)))


@pytest.fixture
def continuous_p1_space(simple_unit_cube):
    """CG1 continuous scalar space."""
    if not HAS_DOLFINX:
        pytest.skip("DOLFINx not installed")
    return fem.functionspace(simple_unit_cube, ("Lagrange", 1))


# ============================================================================
# PARAMETRIZE HELPERS
# ============================================================================

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

# ============================================================================
# TEST MARKERS
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "febio: tests requiring FEBio file parsing"
    )
    config.addinivalue_line(
        "markers", "gait_data: tests requiring gait data processing"
    )
    config.addinivalue_line(
        "markers", "mpi: tests requiring MPI (auto-restart with mpirun)"
    )
    config.addinivalue_line(
        "markers", "slow: slow-running tests (skip with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests combining multiple components"
    )
