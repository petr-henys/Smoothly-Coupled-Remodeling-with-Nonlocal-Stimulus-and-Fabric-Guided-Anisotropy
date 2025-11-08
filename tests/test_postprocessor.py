"""
Comprehensive tests for postprocessor.py (SimulationLoader and SweepLoader).

Test coverage:
- SimulationLoader initialization and error handling
- Configuration and metadata loading
- Metrics loading (steps.csv, subiterations.csv)
- Field loading
- Field statistics computation
- SweepLoader initialization and filtering
- Cross-loader consistency
- MPI-safety and caching behavior
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
import basix.ufl

from postprocessor import SimulationLoader, SweepLoader
from analysis.utils import save_function_npz


# =============================================================================
# Fixtures for test data generation
# =============================================================================

@pytest.fixture
def mock_run_directory(tmp_path):
    """Create a complete mock simulation run directory with all expected files.
    
    Directory structure:
        tmp_path/
            config.json
            run_summary.json
            steps.csv
            subiterations.csv
            u.npz, rho.npz, S.npz, A.npz
    """
    run_dir = tmp_path / "test_run_12345678"
    run_dir.mkdir()
    
    # Config
    config = {
        "dt_days": 50.0,
        "total_time_days": 200.0,
        "accel_type": "anderson",
        "coupling_tol": 1e-6,
        "mesh_size": 33,
        "verbose": True,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Run summary
    run_summary = {
        "success": True,
        "total_steps": 4,
        "total_time_seconds": 123.45,
        "avg_subiters": 8.5,
        "final_residual": 3.2e-7,
    }
    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)
    
    # Steps CSV
    steps_data = {
        "step": [0, 1, 2, 3],
        "time_days": [0.0, 50.0, 100.0, 150.0],
        "dt_days": [50.0, 50.0, 50.0, 50.0],
        "num_subiters": [10, 8, 9, 7],
        "memory_mb": [1234.5, 1245.3, 1256.1, 1267.8],
    }
    pd.DataFrame(steps_data).to_csv(run_dir / "steps.csv", index=False)
    
    # Subiterations CSV
    subiters_rows = []
    for step in range(4):
        for iter_num in range(steps_data["num_subiters"][step]):
            subiters_rows.append({
                "step": step,
                "iter": iter_num,
                "time_days": steps_data["time_days"][step],
                "proj_res": 1e-4 * 0.5**iter_num,
                "rel_change": 1e-3 * 0.6**iter_num,
                "mech_iters": 5 + iter_num % 3,
                "stim_iters": 4 + iter_num % 2,
                "dens_iters": 3 + iter_num % 2,
                "dir_iters": 4 + iter_num % 3,
                "accepted": 1 if iter_num > 0 else 0,
                "restart": 0,
                "condH": 2.5 + 0.1 * iter_num,
                "mech_time": 0.05,
                "stim_time": 0.03,
                "dens_time": 0.02,
                "dir_time": 0.04,
                "memory_mb": 1234.5 + step * 10 + iter_num * 0.5,
            })
    pd.DataFrame(subiters_rows).to_csv(run_dir / "subiterations.csv", index=False)
    
    return run_dir


@pytest.fixture
def mock_run_with_fields(mock_run_directory):
    """Add field NPZ files to mock run directory."""
    comm = MPI.COMM_WORLD
    
    # Create tiny mesh for field generation
    domain = mesh.create_unit_cube(
        comm, 3, 3, 3, ghost_mode=mesh.GhostMode.shared_facet
    )
    
    # Create function spaces
    V = fem.functionspace(
        domain, basix.ufl.element("P", domain.topology.cell_name(), 1, shape=(3,))
    )
    Q = fem.functionspace(
        domain, basix.ufl.element("P", domain.topology.cell_name(), 1)
    )
    T = fem.functionspace(
        domain, basix.ufl.element("P", domain.topology.cell_name(), 1, shape=(3, 3))
    )
    
    # Create mock fields
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    S = fem.Function(Q, name="S")
    A = fem.Function(T, name="A")
    
    # Set some non-trivial values
    u.x.array[:] = np.random.randn(len(u.x.array)) * 0.001
    rho.x.array[:] = 0.5 + 0.3 * np.random.rand(len(rho.x.array))
    S.x.array[:] = 1.0 + 0.2 * np.random.randn(len(S.x.array))
    A.x.array[:] = np.random.randn(len(A.x.array)) * 0.1
    
    # Ensure A is symmetric (simplified)
    for i in range(0, len(A.x.array), 9):
        A.x.array[i+1] = A.x.array[i+3]  # A[0,1] = A[1,0]
        A.x.array[i+2] = A.x.array[i+6]  # A[0,2] = A[2,0]
        A.x.array[i+5] = A.x.array[i+7]  # A[1,2] = A[2,1]
    
    # Save fields
    save_function_npz(u, mock_run_directory / "u.npz", comm)
    save_function_npz(rho, mock_run_directory / "rho.npz", comm)
    save_function_npz(S, mock_run_directory / "S.npz", comm)
    save_function_npz(A, mock_run_directory / "A.npz", comm)
    
    return mock_run_directory, domain, (V, Q, T)


@pytest.fixture
def mock_sweep_directory(tmp_path):
    """Create mock sweep directory with multiple runs."""
    sweep_dir = tmp_path / "test_sweep"
    sweep_dir.mkdir()
    
    # Create multiple run directories with varying parameters
    runs = [
        {"hash": "aaaa1111", "dt_days": 25.0, "accel_type": "picard"},
        {"hash": "bbbb2222", "dt_days": 50.0, "accel_type": "picard"},
        {"hash": "cccc3333", "dt_days": 25.0, "accel_type": "anderson"},
        {"hash": "dddd4444", "dt_days": 50.0, "accel_type": "anderson"},
    ]
    
    for run in runs:
        run_dir = sweep_dir / run["hash"]
        run_dir.mkdir()
        
        # Minimal config for each run
        config = {
            "dt_days": run["dt_days"],
            "accel_type": run["accel_type"],
            "coupling_tol": 1e-6,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        # Minimal steps CSV
        steps = pd.DataFrame({
            "step": [0, 1],
            "time_days": [0.0, run["dt_days"]],
            "dt_days": [run["dt_days"], run["dt_days"]],
        })
        steps.to_csv(run_dir / "steps.csv", index=False)
        
        # Minimal subiterations CSV
        subiters = pd.DataFrame({
            "step": [0, 0, 1, 1],
            "iter": [0, 1, 0, 1],
            "time_days": [0.0, 0.0, run["dt_days"], run["dt_days"]],
            "proj_res": [1e-4, 1e-7, 1e-4, 1e-7],
        })
        subiters.to_csv(run_dir / "subiterations.csv", index=False)
    
    # Create sweep summary
    summary = pd.DataFrame([
        {"output_dir": r["hash"], "dt_days": r["dt_days"], "accel_type": r["accel_type"]}
        for r in runs
    ])
    summary.to_csv(sweep_dir / "sweep_summary.csv", index=False)
    
    return sweep_dir


# =============================================================================
# SimulationLoader Tests
# =============================================================================

@pytest.mark.unit
def test_simulation_loader_initialization(mock_run_directory):
    """Test basic initialization and directory validation."""
    comm = MPI.COMM_WORLD
    
    # Valid directory
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    assert loader.output_dir == Path(mock_run_directory)
    assert loader.comm == comm
    
    # Invalid directory
    with pytest.raises(FileNotFoundError, match="Simulation directory not found"):
        SimulationLoader("/nonexistent/path", comm, verbose=False)


@pytest.mark.unit
def test_config_loading(mock_run_directory):
    """Test configuration loading and parameter access."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    # Load full config
    config = loader.get_config()
    assert isinstance(config, dict)
    assert config["dt_days"] == 50.0
    assert config["accel_type"] == "anderson"
    assert config["mesh_size"] == 33
    
    # Caching behavior
    config2 = loader.get_config()
    assert config is config2  # Should return cached instance
    
    # Single parameter access
    assert loader.get_param("dt_days") == 50.0
    assert loader.get_param("nonexistent", default=42) == 42


@pytest.mark.unit
def test_run_summary_loading(mock_run_directory):
    """Test run summary metadata loading."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    summary = loader.get_run_summary()
    assert summary["success"] is True
    assert summary["total_steps"] == 4
    assert summary["avg_subiters"] == 8.5
    assert summary["final_residual"] == 3.2e-7
    
    # Caching
    summary2 = loader.get_run_summary()
    assert summary is summary2


@pytest.mark.unit
def test_missing_run_summary(tmp_path):
    """Test graceful handling of missing run_summary.json."""
    comm = MPI.COMM_WORLD
    run_dir = tmp_path / "incomplete_run"
    run_dir.mkdir()
    
    # Create minimal config to satisfy initialization
    with open(run_dir / "config.json", "w") as f:
        json.dump({"dt_days": 50.0}, f)
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    summary = loader.get_run_summary()
    
    assert summary == {}  # Should return empty dict, not error


@pytest.mark.unit
def test_steps_metrics_loading(mock_run_directory):
    """Test per-timestep metrics loading."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    steps_df = loader.get_steps_metrics()
    
    assert isinstance(steps_df, pd.DataFrame)
    assert len(steps_df) == 4
    assert list(steps_df["step"]) == [0, 1, 2, 3]
    assert list(steps_df["time_days"]) == [0.0, 50.0, 100.0, 150.0]
    assert all(steps_df["dt_days"] == 50.0)
    
    # Check columns
    expected_cols = {"step", "time_days", "dt_days", "num_subiters", "memory_mb"}
    assert expected_cols.issubset(set(steps_df.columns))


@pytest.mark.unit
def test_subiterations_metrics_loading(mock_run_directory):
    """Test per-subiteration metrics loading."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    subiters_df = loader.get_subiterations_metrics()
    
    assert isinstance(subiters_df, pd.DataFrame)
    assert len(subiters_df) == 10 + 8 + 9 + 7  # Sum of num_subiters
    
    # Check structure
    expected_cols = {
        "step", "iter", "time_days", "proj_res", "rel_change",
        "mech_iters", "stim_iters", "dens_iters", "dir_iters",
        "accepted", "restart", "condH"
    }
    assert expected_cols.issubset(set(subiters_df.columns))
    
    # Verify data integrity
    step_0_data = subiters_df[subiters_df["step"] == 0]
    assert len(step_0_data) == 10
    assert all(step_0_data["time_days"] == 0.0)


@pytest.mark.unit
def test_metrics_at_time(mock_run_directory):
    """Test aggregated metrics extraction at specific time."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    # Get metrics at t=100 days (step 2)
    metrics = loader.get_metrics_at_time(100.0)
    
    assert metrics["step"] == 2
    assert metrics["time_days"] == 100.0
    assert metrics["num_subiters"] == 9  # From aggregation
    assert "total_mech_iters" in metrics
    assert "total_stim_iters" in metrics
    assert "aa_acceptances" in metrics
    assert "final_proj_res" in metrics
    
    # Test closest matching (request 105 days, should get 100)
    metrics_close = loader.get_metrics_at_time(105.0)
    assert metrics_close["step"] == 2
    assert metrics_close["time_days"] == 100.0


@pytest.mark.unit
def test_available_times(mock_run_directory):
    """Test extraction of available checkpoint times."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    times = loader.get_available_times()
    
    assert isinstance(times, np.ndarray)
    np.testing.assert_array_equal(times, [0.0, 50.0, 100.0, 150.0])


@pytest.mark.mpi
def test_field_loading(mock_run_with_fields):
    """Test field snapshot loading from NPZ files."""
    run_dir, domain, spaces = mock_run_with_fields
    V, Q, T = spaces
    comm = MPI.COMM_WORLD
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    # Load all fields
    fields = loader.get_fields_at_time(0.0, fields=["u", "rho", "S", "A"])
    
    assert "u" in fields
    assert "rho" in fields
    assert "S" in fields
    assert "A" in fields
    
    # Check shapes (broadcast values, so all ranks have full array)
    assert isinstance(fields["u"], np.ndarray)
    assert isinstance(fields["rho"], np.ndarray)
    assert len(fields["u"]) > 0
    assert len(fields["rho"]) > 0


@pytest.mark.mpi
def test_field_loading_caching(mock_run_with_fields):
    """Test field loading caching behavior."""
    run_dir, domain, spaces = mock_run_with_fields
    comm = MPI.COMM_WORLD
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    # Load same field twice
    fields1 = loader.get_fields_at_time(0.0, fields=["rho"])
    fields2 = loader.get_fields_at_time(0.0, fields=["rho"])
    
    # Should return cached arrays
    assert fields1["rho"] is fields2["rho"]


@pytest.mark.mpi
def test_load_field_to_function(mock_run_with_fields):
    """Test loading field snapshot directly into DOLFINx function."""
    run_dir, domain, spaces = mock_run_with_fields
    V, Q, T = spaces
    comm = MPI.COMM_WORLD
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    # Create target functions
    rho_target = fem.Function(Q, name="rho_loaded")
    
    # Load into function (uses coordinate matching)
    loader.load_field_to_function(rho_target, "rho", time_days=None)
    
    # Verify values were loaded
    assert np.any(rho_target.x.array != 0.0)
    assert np.all(np.isfinite(rho_target.x.array))
    
    # Should be within physically reasonable bounds (original was 0.5 + 0.3*rand)
    assert np.all(rho_target.x.array >= 0.0)
    assert np.all(rho_target.x.array <= 1.0)


@pytest.mark.mpi
def test_field_statistics(mock_run_with_fields):
    """Test field statistics computation."""
    run_dir, domain, spaces = mock_run_with_fields
    comm = MPI.COMM_WORLD
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    # Compute statistics for density field
    stats = loader.get_field_statistics("rho", time_days=None)
    
    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats
    assert "std" in stats
    assert "median" in stats
    
    # Check physical reasonableness (original: 0.5 + 0.3*rand)
    assert 0.0 <= stats["min"] <= 1.0
    assert 0.0 <= stats["max"] <= 1.0
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["std"] >= 0.0


# =============================================================================
# SweepLoader Tests
# =============================================================================

@pytest.mark.unit
def test_sweep_loader_initialization(mock_sweep_directory):
    """Test sweep loader initialization."""
    comm = MPI.COMM_WORLD
    
    # Valid directory
    sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
    assert sweep.base_dir == Path(mock_sweep_directory)
    assert sweep.comm == comm
    
    # Invalid directory
    with pytest.raises(FileNotFoundError, match="Sweep directory not found"):
        SweepLoader("/nonexistent/sweep", comm, verbose=False)


@pytest.mark.unit
def test_sweep_summary_loading(mock_sweep_directory):
    """Test sweep summary loading."""
    comm = MPI.COMM_WORLD
    sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
    
    summary = sweep.get_summary()
    
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 4
    assert set(summary.columns) >= {"output_dir", "dt_days", "accel_type"}
    
    # Check parameter values
    assert set(summary["dt_days"]) == {25.0, 50.0}
    assert set(summary["accel_type"]) == {"picard", "anderson"}


@pytest.mark.unit
def test_sweep_filter_runs(mock_sweep_directory):
    """Test filtering runs by parameter values."""
    comm = MPI.COMM_WORLD
    sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
    
    # Filter by accel_type
    picard_runs = sweep.filter_runs(accel_type="picard")
    assert len(picard_runs) == 2
    assert all(picard_runs["accel_type"] == "picard")
    
    # Filter by dt_days
    dt25_runs = sweep.filter_runs(dt_days=25.0)
    assert len(dt25_runs) == 2
    assert all(dt25_runs["dt_days"] == 25.0)
    
    # Multiple filters
    specific = sweep.filter_runs(accel_type="anderson", dt_days=50.0)
    assert len(specific) == 1
    assert specific.iloc[0]["output_dir"] == "dddd4444"
    
    # No matches
    empty = sweep.filter_runs(dt_days=999.0)
    assert len(empty) == 0


@pytest.mark.unit
def test_sweep_get_loader(mock_sweep_directory):
    """Test getting individual run loaders."""
    comm = MPI.COMM_WORLD
    sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
    
    # Get loader by hash
    loader = sweep.get_loader("aaaa1111")
    
    assert isinstance(loader, SimulationLoader)
    assert loader.output_dir == Path(mock_sweep_directory) / "aaaa1111"
    
    # Caching behavior
    loader2 = sweep.get_loader("aaaa1111")
    assert loader is loader2


@pytest.mark.unit
def test_sweep_get_all_loaders(mock_sweep_directory):
    """Test getting all run loaders."""
    comm = MPI.COMM_WORLD
    sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
    
    loaders = sweep.get_all_loaders()
    
    assert len(loaders) == 4
    assert all(isinstance(loader, SimulationLoader) for loader in loaders)
    
    # Check that all hashes are covered
    hashes = {loader.output_dir.name for loader in loaders}
    expected_hashes = {"aaaa1111", "bbbb2222", "cccc3333", "dddd4444"}
    assert hashes == expected_hashes


@pytest.mark.unit
def test_sweep_fallback_without_summary(tmp_path):
    """Test sweep loading fallback when sweep_summary.csv is missing."""
    comm = MPI.COMM_WORLD
    sweep_dir = tmp_path / "sweep_no_summary"
    sweep_dir.mkdir()
    
    # Create run subdirectories without summary CSV
    for hash_val in ["run1111", "run2222"]:
        (sweep_dir / hash_val).mkdir()
    
    sweep = SweepLoader(sweep_dir, comm, verbose=False)
    summary = sweep.get_summary()
    
    # Should still return summary with output_dir column
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 2
    assert "output_dir" in summary.columns
    assert set(summary["output_dir"]) == {"run1111", "run2222"}


# =============================================================================
# Integration and cross-consistency tests
# =============================================================================

@pytest.mark.integration
def test_loader_metrics_consistency(mock_run_directory):
    """Test consistency between steps and subiterations metrics."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    steps_df = loader.get_steps_metrics()
    subiters_df = loader.get_subiterations_metrics()
    
    # Check that subiteration counts match
    for _, step_row in steps_df.iterrows():
        step_num = int(step_row["step"])
        expected_count = int(step_row["num_subiters"])
        
        actual_count = len(subiters_df[subiters_df["step"] == step_num])
        assert actual_count == expected_count, \
            f"Step {step_num}: expected {expected_count} subiters, got {actual_count}"


@pytest.mark.integration
def test_sweep_and_loader_parameter_consistency(mock_sweep_directory):
    """Test that sweep summary matches individual loader configs."""
    comm = MPI.COMM_WORLD
    sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
    summary = sweep.get_summary()
    
    for _, row in summary.iterrows():
        loader = sweep.get_loader(row["output_dir"])
        config = loader.get_config()
        
        # Check parameter consistency
        assert config["dt_days"] == row["dt_days"]
        assert config["accel_type"] == row["accel_type"]


@pytest.mark.mpi
def test_mpi_rank_consistency(mock_run_directory):
    """Test that all MPI ranks get consistent data."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    # Load data on all ranks
    config = loader.get_config()
    steps = loader.get_steps_metrics()
    subiters = loader.get_subiterations_metrics()
    
    # Verify consistency across ranks
    all_configs = comm.gather(config["dt_days"], root=0)
    all_step_counts = comm.gather(len(steps), root=0)
    all_subiter_counts = comm.gather(len(subiters), root=0)
    
    if comm.rank == 0:
        assert all(c == all_configs[0] for c in all_configs)
        assert all(c == all_step_counts[0] for c in all_step_counts)
        assert all(c == all_subiter_counts[0] for c in all_subiter_counts)


# =============================================================================
# Error handling and edge cases
# =============================================================================

@pytest.mark.unit
def test_missing_config_file(tmp_path):
    """Test error when config.json is missing."""
    comm = MPI.COMM_WORLD
    run_dir = tmp_path / "no_config"
    run_dir.mkdir()
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        loader.get_config()


@pytest.mark.unit
def test_missing_steps_csv(tmp_path):
    """Test error when steps.csv is missing."""
    comm = MPI.COMM_WORLD
    run_dir = tmp_path / "no_steps"
    run_dir.mkdir()
    
    with open(run_dir / "config.json", "w") as f:
        json.dump({"dt_days": 50.0}, f)
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    with pytest.raises(FileNotFoundError, match="Steps metrics not found"):
        loader.get_steps_metrics()


@pytest.mark.unit
def test_missing_subiterations_csv(tmp_path):
    """Test error when subiterations.csv is missing."""
    comm = MPI.COMM_WORLD
    run_dir = tmp_path / "no_subiters"
    run_dir.mkdir()
    
    with open(run_dir / "config.json", "w") as f:
        json.dump({"dt_days": 50.0}, f)
    
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    with pytest.raises(FileNotFoundError, match="Subiterations metrics not found"):
        loader.get_subiterations_metrics()


@pytest.mark.unit
def test_missing_field_npz(mock_run_directory):
    """Test error when field NPZ file is missing."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    with pytest.raises(FileNotFoundError, match="Field snapshot not found"):
        loader.get_fields_at_time(0.0, fields=["u"])


@pytest.mark.unit
def test_invalid_time_raises_error(mock_run_directory):
    """Test error when requesting non-checkpoint time."""
    comm = MPI.COMM_WORLD
    loader = SimulationLoader(mock_run_directory, comm, verbose=False)
    
    # Create dummy NPZ to satisfy file existence check
    if comm.rank == 0:
        np.savez(mock_run_directory / "rho.npz", values=np.array([0.5]))
    comm.Barrier()
    
    with pytest.raises(ValueError, match="Time.*not found in checkpoints"):
        loader.get_fields_at_time(75.0, fields=["rho"])


@pytest.mark.smoke
def test_smoke_postprocessor_workflow(mock_run_with_fields):
    """Smoke test: Full workflow from sweep to field analysis."""
    run_dir, domain, spaces = mock_run_with_fields
    comm = MPI.COMM_WORLD
    
    # Initialize loader
    loader = SimulationLoader(run_dir, comm, verbose=False)
    
    # Load config and metrics
    config = loader.get_config()
    assert config["dt_days"] == 50.0
    
    steps = loader.get_steps_metrics()
    assert len(steps) > 0
    
    subiters = loader.get_subiterations_metrics()
    assert len(subiters) > 0
    
    # Load field statistics
    stats = loader.get_field_statistics("rho")
    assert "mean" in stats
    assert "std" in stats
    
    # Get metrics at specific time
    metrics = loader.get_metrics_at_time(50.0)
    assert "num_subiters" in metrics
    assert "total_mech_iters" in metrics
