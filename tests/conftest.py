"""
Pytest configuration and shared fixtures/utilities for the test suite.

Responsibilities:
- Auto-restart with MPI if running serial (for VS Code Test Explorer)
- Ensure project root is importable (adjust sys.path once)
- Silence stdout/stderr on non-root MPI ranks to avoid duplicated output
- Provide deterministic randomness across tests
- Provide common mesh/config/function-space/field fixtures to reduce duplication
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import NamedTuple

import pytest


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

# =============================================================================

def pytest_configure(config: pytest.Config) -> None:
    """Register markers and silence non-rank-0 output."""
    # Silence non-root ranks immediately
    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank != 0:
            sys.stdout = sys.stderr = open(os.devnull, 'w')
    except Exception:
        pass
    
    # Register markers
    for marker, desc in [
        ("mpi", "tests that are intended for MPI environments"),
        ("slow", "tests that may take longer to run"),
        ("performance", "tests focused on performance characteristics"),
        ("integration", "end-to-end or multi-component tests"),
        ("unit", "fast unit-level tests"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {desc}")


def pytest_runtest_setup(item):
    """Disable pytest.skip() during test discovery for VS Code compatibility.
    
    VS Code Test Explorer runs discovery with --collect-only, but tests that call
    pytest.skip() during setup still get marked as skipped. We prevent this by
    making pytest.skip() a no-op during collection phase.
    """
    if item.config.option.collectonly:
        # Replace pytest.skip with a no-op during discovery
        import pytest as pytest_module
        pytest_module.skip = lambda msg="", allow_module_level=False: None



# Add repository root to sys.path
tests_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(tests_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


@pytest.fixture(autouse=True)
def _seed_numpy_for_determinism():
    """Seed numpy RNG for deterministic test behavior."""
    try:
        import numpy as np
        np.random.seed(1234)
    except Exception:
        pass


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
    from mpi4py import MPI
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
    
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
        return
    
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
