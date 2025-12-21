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

import os
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple, Dict

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
    if MPI is not None:
        comm = MPI.COMM_WORLD
        
        # Silence non-root ranks
        if comm.rank != 0:
            sys.stdout = sys.stderr = open(os.devnull, 'w')
            
        # Set unique basetemp for each rank to avoid cleanup race conditions
        if comm.size > 1 and not config.option.basetemp:
            # Use a unique temporary directory for each rank
            # This prevents "Directory not empty" errors during cleanup
            # We use a fixed path structure to avoid accumulating random directories
            base_tmp = Path(tempfile.gettempdir()) / f"pytest-mpi-{os.getuid()}" / f"rank-{comm.rank}"
            base_tmp.mkdir(parents=True, exist_ok=True)
            config.option.basetemp = str(base_tmp)
    
    # Register markers
    config.addinivalue_line("markers", "mpi: tests that are intended for MPI environments")
    config.addinivalue_line("markers", "smoke: quick smoke tests for basic functionality")
    config.addinivalue_line("markers", "integration: integration tests spanning multiple components")
    config.addinivalue_line("markers", "slow: tests that take significant time to complete")
    config.addinivalue_line("markers", "performance: performance benchmarks and profiling tests")


# Add workspace root to sys.path (where simulation/ folder is located)
tests_dir = os.path.dirname(__file__)
workspace_root = os.path.abspath(os.path.join(tests_dir, ".."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)


@pytest.fixture(autouse=True, scope="session")
def _seed_numpy_for_determinism():
    """Seed numpy RNG once for deterministic test behavior."""
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
    n = int(getattr(request, "param", 6))
    return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, ghost_mode=mesh.GhostMode.shared_facet)


@pytest.fixture
def facet_tags(unit_cube) -> object:
    from simulation.utils import build_facetag
    return build_facetag(unit_cube)


@pytest.fixture
def cfg(unit_cube, facet_tags) -> object:
    from simulation.config import Config
    return Config.from_flat_kwargs(
        domain=unit_cube, facet_tags=facet_tags,
        n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2
    )


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
    rho.x.array[:] = 0.8  # Updated default density (physical)
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
def mean_value_factory(cfg):
    """Factory to compute global mean value of a scalar UFL expression.
    
    Usage: mean = mean_value_factory; result = mean(expr)
    """
    from dolfinx import fem
    from mpi4py import MPI

    def _mean(expr, *, dx=None) -> float:
        dxm = dx if dx is not None else cfg.dx
        local = fem.assemble_scalar(fem.form(expr * dxm))
        vol_local = fem.assemble_scalar(fem.form(1.0 * dxm))
        comm = cfg.domain.comm if hasattr(cfg, "domain") else MPI.COMM_WORLD
        tot = comm.allreduce(local, op=MPI.SUM)
        vol = comm.allreduce(vol_local, op=MPI.SUM)
        return float(tot / max(vol, 1e-300))

    return _mean


@pytest.fixture
def shared_tmpdir():
    """Create a shared temporary directory for MPI tests.
    
    In MPI environments, each rank creates a separate tempdir. This fixture
    broadcasts rank 0's directory to all ranks for shared file I/O.
    Returns a Path object for easy path manipulation.
    """
    from pathlib import Path
    
    if not HAS_MPI or MPI is None:
        path = tempfile.mkdtemp(prefix="remodeller_tests_")
        yield Path(path)
        return

    comm = MPI.COMM_WORLD

    # Rank 0 creates a persistent temp directory, broadcast to others
    if comm.rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="remodeller_tests_")
    else:
        tmpdir = None

    tmpdir = comm.bcast(tmpdir, root=0)
    yield Path(tmpdir)





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
        
        # Create config with preset parameters (using backward-compatible factory)
        if 'verbose' in params:
            params.pop('verbose')
            
        return Config.from_flat_kwargs(
            domain=unit_cube,
            facet_tags=facet_tags,
            n_trab=2.0,
            n_cort=1.2,
            rho_trab_max=0.8,
            rho_cort_min=1.2,
            **params
        )
    
    return _make_config


# =============================================================================
# Femur-specific fixtures
# =============================================================================

@pytest.fixture(scope="module")
def femur_setup():
    """Create femur mesh and function spaces (mm geometry, MPa stresses)."""
    from simulation.paths import FemurPaths
    from simulation.febio_parser import FEBio2Dolfinx
    from simulation.config import Config
    import basix
    from dolfinx import fem

    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)
    cfg = Config.from_flat_kwargs(
        domain=domain, facet_tags=facet_tags,
        n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2
    )
    return domain, facet_tags, V, Q, cfg


# =============================================================================
# Tensor helpers
# =============================================================================

@pytest.fixture
def iso_tensor_factory():
    """Factory for isotropic unit-trace tensor I/3."""
    def _make(x):
        import numpy as np
        base = (np.eye(3) / 3.0).flatten()[:, None]
        return np.tile(base, (1, x.shape[1]))
    return _make


@pytest.fixture
def fiber_tensor_factory():
    """Factory for anisotropic unit-trace tensor with fiber in x-direction."""
    def _make(x):
        import numpy as np
        mat = np.array([[0.92, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]], dtype=float)
        return np.tile(mat.flatten()[:, None], (1, x.shape[1]))
    return _make


@pytest.fixture
def dummy_load(spaces, cfg):
    """Create dummy Loader and LoadingCase for unit tests."""
    from dolfinx import fem
    import numpy as np
    from simulation.loader import LoadingCase
    
    class MockLoader:
        """Mock Loader for testing without femur-specific dependencies."""
        def __init__(self, V, load_tag: int = 2):
            self.V = V
            self.load_tag = load_tag
            self.traction = fem.Function(V, name="Traction")
            self._cache = {}
        
        def precompute_loading_cases(self, cases):
            """Precompute and cache traction arrays for all loading cases."""
            for case in cases:
                traction_vec = np.array([0.0, -0.1, 0.05], dtype=np.float64)
                n_dofs = self.traction.x.array.size // 3
                traction_array = np.tile(traction_vec, n_dofs)
                self._cache[case.name] = {"traction": traction_array.copy()}
        
        def set_loading_case(self, case_name: str) -> None:
            """Apply cached traction for named case."""
            cached = self._cache[case_name]
            self.traction.x.array[:] = cached["traction"]
            self.traction.x.scatter_forward()
    
    loader = MockLoader(spaces.V, load_tag=2)
    loading_case = LoadingCase(name="test_case", day_cycles=1.0, hip=None, muscles=[])
    
    return {
        "loader": loader,
        "loading_cases": [loading_case],
    }
