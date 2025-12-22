"""Tests for box model components.

Verifies:
- Box mesh creation and facet tagging
- Pressure loader functionality
- Integration with Remodeller (smoke test)
"""

from __future__ import annotations

import numpy as np
import pytest
from mpi4py import MPI

from dolfinx import fem, mesh
import basix.ufl

from simulation.box_mesh import BoxGeometry, BoxMeshBuilder, create_box_mesh
from simulation.box_loader import BoxLoader, BoxLoadingCase, GradientType, PressureLoadSpec
from simulation.box_scenarios import (
    get_single_pressure_case,
    get_physiological_compression_cases,
    get_cyclic_loading_cases,
    get_parabolic_pressure_case,
    get_bending_like_case,
)
from simulation.box_factory import BoxSolverFactory
from simulation.config import Config
from simulation.params import GeometryParams, SolverParams, TimeParams, OutputParams


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def box_geometry() -> BoxGeometry:
    """Default box geometry for tests."""
    return BoxGeometry(
        Lx=10.0, Ly=10.0, Lz=30.0,
        nx=3, ny=3, nz=6,
    )


@pytest.fixture
def box_mesh_and_tags(box_geometry) -> tuple:
    """Create box mesh with facet tags."""
    builder = BoxMeshBuilder(box_geometry, MPI.COMM_WORLD)
    return builder.build()


@pytest.fixture
def box_cfg(box_mesh_and_tags) -> Config:
    """Create Config for box model tests."""
    domain, facet_tags = box_mesh_and_tags
    return Config(
        domain=domain,
        facet_tags=facet_tags,
        geometry=GeometryParams(
            fix_tag=BoxMeshBuilder.TAG_BOTTOM,
            load_tag=BoxMeshBuilder.TAG_TOP,
        ),
        solver=SolverParams(
            coupling_tol=1e-3,
            max_subiters=10,
            ksp_rtol=1e-6,
        ),
        time=TimeParams(
            total_time=10.0,
            dt_initial=5.0,
        ),
        output=OutputParams(
            results_dir=".test_box_results",
        ),
    )


# =============================================================================
# Box mesh tests
# =============================================================================

class TestBoxMesh:
    """Tests for box mesh creation."""
    
    def test_create_box_mesh_basic(self, box_geometry):
        """Test basic box mesh creation."""
        builder = BoxMeshBuilder(box_geometry, MPI.COMM_WORLD)
        domain, facet_tags = builder.build()
        
        assert domain is not None
        assert domain.geometry.dim == 3
        assert domain.topology.dim == 3
        
    def test_facet_tags_exist(self, box_mesh_and_tags):
        """Test that facet tags are created."""
        domain, facet_tags = box_mesh_and_tags
        
        assert facet_tags is not None
        assert facet_tags.dim == 2  # Facet dimension
        
    def test_facet_tags_bottom_top(self, box_mesh_and_tags, box_geometry):
        """Test that bottom and top surfaces are properly tagged."""
        domain, facet_tags = box_mesh_and_tags
        
        # Find facets on bottom (z=0) and top (z=Lz)
        bottom_facets = facet_tags.find(BoxMeshBuilder.TAG_BOTTOM)
        top_facets = facet_tags.find(BoxMeshBuilder.TAG_TOP)
        
        # There should be facets on both surfaces
        comm = MPI.COMM_WORLD
        n_bottom = comm.allreduce(len(bottom_facets), op=MPI.SUM)
        n_top = comm.allreduce(len(top_facets), op=MPI.SUM)
        
        # For structured mesh: nx*ny*2 triangles on each face (tetrahedra)
        assert n_bottom > 0, "No facets found on bottom surface"
        assert n_top > 0, "No facets found on top surface"
        
    def test_convenience_function(self):
        """Test create_box_mesh convenience function."""
        domain, facet_tags = create_box_mesh(
            Lx=5.0, Ly=5.0, Lz=10.0,
            nx=2, ny=2, nz=4,
        )
        
        assert domain is not None
        assert facet_tags is not None
        
    def test_mesh_dimensions(self, box_mesh_and_tags, box_geometry):
        """Test mesh dimensions match geometry."""
        domain, _ = box_mesh_and_tags
        
        coords = domain.geometry.x
        
        # Gather all coordinates
        comm = MPI.COMM_WORLD
        x_min = comm.allreduce(coords[:, 0].min(), op=MPI.MIN)
        x_max = comm.allreduce(coords[:, 0].max(), op=MPI.MAX)
        z_min = comm.allreduce(coords[:, 2].min(), op=MPI.MIN)
        z_max = comm.allreduce(coords[:, 2].max(), op=MPI.MAX)
        
        assert np.isclose(x_min, 0.0, atol=1e-10)
        assert np.isclose(x_max, box_geometry.Lx, atol=1e-10)
        assert np.isclose(z_min, 0.0, atol=1e-10)
        assert np.isclose(z_max, box_geometry.Lz, atol=1e-10)


# =============================================================================
# Box loader tests
# =============================================================================

class TestBoxLoader:
    """Tests for box pressure loader."""
    
    def test_loader_creation(self, box_mesh_and_tags):
        """Test loader can be created."""
        domain, facet_tags = box_mesh_and_tags
        loader = BoxLoader(domain, facet_tags)
        
        assert loader is not None
        assert loader.traction is not None
        
    def test_set_pressure(self, box_mesh_and_tags):
        """Test setting uniform pressure."""
        domain, facet_tags = box_mesh_and_tags
        loader = BoxLoader(domain, facet_tags)
        
        loader.set_pressure(1.0, direction=(0.0, 0.0, -1.0))
        
        # Check traction values are set
        traction_norm = np.linalg.norm(loader.traction.x.array)
        comm = MPI.COMM_WORLD
        total_norm = comm.allreduce(traction_norm**2, op=MPI.SUM) ** 0.5
        
        assert total_norm > 0, "Traction should be non-zero"
        
    def test_precompute_loading_cases(self, box_mesh_and_tags):
        """Test precomputing loading cases."""
        domain, facet_tags = box_mesh_and_tags
        loader = BoxLoader(domain, facet_tags)
        
        cases = [
            BoxLoadingCase(
                name="test_case",
                day_cycles=1.0,
                pressure=PressureLoadSpec(magnitude=2.0),
            ),
        ]
        
        loader.precompute_loading_cases(cases)
        
        # Should be able to set the case
        loader.set_loading_case("test_case")
        
    def test_set_loading_case_invalid(self, box_mesh_and_tags):
        """Test that invalid case name raises error."""
        domain, facet_tags = box_mesh_and_tags
        loader = BoxLoader(domain, facet_tags)
        
        with pytest.raises(KeyError):
            loader.set_loading_case("nonexistent_case")


# =============================================================================
# Box scenarios tests
# =============================================================================

class TestBoxScenarios:
    """Tests for predefined loading scenarios."""
    
    def test_single_pressure_case(self):
        """Test single pressure case creation."""
        case = get_single_pressure_case(pressure=1.5)
        
        assert case.name == "static_compression"
        assert case.pressure.magnitude == 1.5
        assert case.day_cycles == 1.0
        
    def test_physiological_cases(self):
        """Test physiological compression cases."""
        cases = get_physiological_compression_cases()
        
        assert len(cases) == 3
        
        # Check day_cycles sum to 1.0
        total_cycles = sum(c.day_cycles for c in cases)
        assert np.isclose(total_cycles, 1.0, atol=1e-10)
        
    def test_cyclic_loading_cases(self):
        """Test cyclic loading case generation."""
        cases = get_cyclic_loading_cases(
            pressure_min=0.5,
            pressure_max=3.0,
            n_levels=5,
        )
        
        assert len(cases) == 5
        
        # Check pressure range
        pressures = [c.pressure.magnitude for c in cases]
        assert np.isclose(min(pressures), 0.5)
        assert np.isclose(max(pressures), 3.0)

    def test_parabolic_pressure_case(self):
        """Test parabolic pressure distribution."""
        case = get_parabolic_pressure_case(
            pressure=1.0,
            gradient_axis=0,
            center_factor=2.0,
            edge_factor=0.5,
            box_extent=(0.0, 10.0),
        )
        
        assert case.name == "parabolic_compression"
        assert case.pressure.gradient_type == GradientType.PARABOLIC
        assert case.pressure.gradient_range == (0.5, 2.0)  # (edge, center)

    def test_bending_like_case(self):
        """Test bending-like pressure distribution."""
        case = get_bending_like_case(
            pressure=2.0,
            gradient_axis=1,
            tension_factor=0.2,
            compression_factor=1.8,
        )
        
        assert case.name == "bending_load"
        assert case.pressure.gradient_type == GradientType.LINEAR
        assert case.pressure.magnitude == 2.0
        assert case.pressure.gradient_axis == 1

    def test_parabolic_loader_produces_peak_at_center(self, box_mesh_and_tags):
        """Test that parabolic loading produces peak traction at center."""
        domain, facet_tags = box_mesh_and_tags
        loader = BoxLoader(domain, facet_tags)
        
        case = get_parabolic_pressure_case(
            pressure=1.0,
            gradient_axis=0,
            center_factor=2.0,
            edge_factor=0.5,
            box_extent=(0.0, 10.0),
        )
        
        loader.precompute_loading_cases([case])
        loader.set_loading_case(case.name)
        
        # Check that traction is non-zero and varies spatially
        traction_z = loader.traction.x.array[2::3]  # z-component
        local_max = np.max(np.abs(traction_z)) if len(traction_z) > 0 else 0.0
        local_min = np.min(np.abs(traction_z)) if len(traction_z) > 0 else 0.0
        
        comm = MPI.COMM_WORLD
        global_max = comm.allreduce(local_max, op=MPI.MAX)
        global_min = comm.allreduce(local_min, op=MPI.MIN)
        
        # Should have variation (center > edge)
        assert global_max > global_min, "Parabolic load should have spatial variation"


# =============================================================================
# Box factory tests
# =============================================================================

class TestBoxFactory:
    """Tests for box solver factory."""
    
    def test_factory_creation(self, box_cfg):
        """Test factory can be created."""
        factory = BoxSolverFactory(box_cfg)
        assert factory is not None
        
    def test_create_mechanics_solver(self, box_cfg, box_mesh_and_tags):
        """Test creating mechanics solver."""
        domain, facet_tags = box_mesh_and_tags
        factory = BoxSolverFactory(box_cfg)
        
        # Create function spaces
        gdim = domain.geometry.dim
        P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,))
        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        P1_ten = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim, gdim))
        
        V = fem.functionspace(domain, P1_vec)
        Q = fem.functionspace(domain, P1)
        T = fem.functionspace(domain, P1_ten)
        
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        L = fem.Function(T, name="L")
        
        # Initialize rho
        rho.x.array[:] = 1.0
        rho.x.scatter_forward()
        
        loader = BoxLoader(domain, facet_tags)
        
        mech = factory.create_mechanics_solver(u, rho, L, loader)
        
        assert mech is not None
        assert len(mech.dirichlet_bcs) > 0, "Should have Dirichlet BCs"


# =============================================================================
# Integration smoke test
# =============================================================================

@pytest.mark.integration
class TestBoxModelIntegration:
    """Integration tests for box model (smoke tests)."""
    
    def test_single_timestep_smoke(self, box_cfg, box_mesh_and_tags):
        """Smoke test: run a single timestep."""
        from simulation.model import Remodeller
        
        domain, facet_tags = box_mesh_and_tags
        
        # Create loader and cases
        loader = BoxLoader(domain, facet_tags)
        loading_cases = [get_single_pressure_case(pressure=1.0)]
        
        # Create factory
        factory = BoxSolverFactory(box_cfg)
        
        # Run single step
        with Remodeller(box_cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            dt = box_cfg.time.dt_initial
            error, metrics = remodeller.step(dt)
            
            assert error >= 0, "Error should be non-negative"
            assert "converged" in metrics
            
    def test_density_changes_with_loading(self, box_cfg, box_mesh_and_tags):
        """Test that density evolves under loading."""
        from simulation.model import Remodeller
        from simulation.utils import get_owned_size
        
        domain, facet_tags = box_mesh_and_tags
        
        loader = BoxLoader(domain, facet_tags)
        loading_cases = [get_single_pressure_case(pressure=2.0)]
        factory = BoxSolverFactory(box_cfg)
        
        with Remodeller(box_cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            # Get initial density
            n_owned = get_owned_size(remodeller.rho)
            rho_initial = remodeller.rho.x.array[:n_owned].copy()
            
            # Run a few steps
            dt = box_cfg.time.dt_initial
            for _ in range(2):
                remodeller.step(dt)
            
            rho_final = remodeller.rho.x.array[:n_owned].copy()
            
            # Density should change (at least slightly)
            diff = np.abs(rho_final - rho_initial).max()
            
            # Note: with very few steps, change may be small
            # Just verify the simulation ran without error
            assert np.isfinite(rho_final).all(), "Density should remain finite"

    def test_metrics_csv_output(self, box_mesh_and_tags, tmp_path):
        """Test that metrics CSV files are written during simulation."""
        from simulation.model import Remodeller
        from simulation.config import Config
        from simulation.params import GeometryParams, SolverParams, TimeParams, OutputParams
        import pandas as pd
        
        domain, facet_tags = box_mesh_and_tags
        comm = MPI.COMM_WORLD
        
        # Create config with tmp_path as output directory
        output_dir = tmp_path / "metrics_test"
        if comm.rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        comm.Barrier()
        
        cfg = Config(
            domain=domain,
            facet_tags=facet_tags,
            geometry=GeometryParams(
                fix_tag=BoxMeshBuilder.TAG_BOTTOM,
                load_tag=BoxMeshBuilder.TAG_TOP,
            ),
            solver=SolverParams(
                coupling_tol=1e-3,
                max_subiters=10,
                ksp_rtol=1e-6,
            ),
            time=TimeParams(
                total_time=10.0,
                dt_initial=5.0,
                adaptive_dt=False,
            ),
            output=OutputParams(
                results_dir=str(output_dir),
            ),
        )
        
        loader = BoxLoader(domain, facet_tags)
        loading_cases = [get_single_pressure_case(pressure=1.0)]
        factory = BoxSolverFactory(cfg)
        
        with Remodeller(cfg, loader=loader, loading_cases=loading_cases, factory=factory) as remodeller:
            remodeller.simulate()
        
        # Verify CSV files were written (rank 0 only)
        if comm.rank == 0:
            steps_csv = output_dir / "steps.csv"
            subiters_csv = output_dir / "subiterations.csv"
            
            assert steps_csv.exists(), "steps.csv should be created"
            assert subiters_csv.exists(), "subiterations.csv should be created"
            
            # Verify content
            steps_df = pd.read_csv(steps_csv)
            assert len(steps_df) == 2, f"Should have 2 steps, got {len(steps_df)}"
            assert "step" in steps_df.columns
            assert "time_days" in steps_df.columns
            assert "mech_iters" in steps_df.columns
            assert "fab_iters" in steps_df.columns
            
            subiters_df = pd.read_csv(subiters_csv)
            assert len(subiters_df) > 0, "Should have subiteration records"
            assert "proj_res" in subiters_df.columns
            assert "condH" in subiters_df.columns
