"""Tests for postprocessor.py (SimulationLoader and SweepLoader).

Simplified tests that avoid adios4dolfinx I/O operations which can cause
MPI deadlocks in pytest environment.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpi4py import MPI

from postprocessor import SimulationLoader, SweepLoader


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def shared_tmp_path(tmp_path):
    """MPI-safe tmp_path: broadcast rank 0's path to all ranks."""
    comm = MPI.COMM_WORLD
    path_str = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    return Path(path_str)


@pytest.fixture
def mock_run_directory(shared_tmp_path):
    """Create mock simulation run directory with config, summary, and metrics CSVs."""
    comm = MPI.COMM_WORLD
    run_dir = shared_tmp_path / "test_run_12345678"
    
    if comm.rank == 0:
        run_dir.mkdir(exist_ok=True)
        
        config = {
            "dt_days": 50.0,
            "total_time_days": 200.0,
            "accel_type": "anderson",
            "coupling_tol": 1e-6,
            "mesh_size": 3,
            "verbose": True,
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        run_summary = {
            "success": True,
            "total_steps": 4,
            "total_time_seconds": 123.45,
            "avg_subiters": 8.5,
            "final_residual": 3.2e-7,
        }
        with open(run_dir / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)
        
        steps_data = {
            "step": [0, 1, 2, 3],
            "time_days": [0.0, 50.0, 100.0, 150.0],
            "dt_days": [50.0, 50.0, 50.0, 50.0],
            "num_subiters": [10, 8, 9, 7],
            "memory_mb": [1234.5, 1245.3, 1256.1, 1267.8],
        }
        pd.DataFrame(steps_data).to_csv(run_dir / "steps.csv", index=False)
        
        subiters_rows = []
        for step in range(4):
            for iter_num in range(steps_data["num_subiters"][step]):
                subiters_rows.append({
                    "step": step,
                    "iter": iter_num,
                    "time_days": steps_data["time_days"][step],
                    "proj_res": 1e-4 * 0.5**iter_num,
                    "aa_step_res": 1e-3 * 0.6**iter_num,
                    "mech_iters": 5 + iter_num % 3,
                    "fab_iters": 4 + iter_num % 3,
                    "stim_iters": 4 + iter_num % 2,
                    "dens_iters": 3 + iter_num % 2,
                    "accepted": 1 if iter_num > 0 else 0,
                    "restart": 0,
                    "condH": 2.5 + 0.1 * iter_num,
                    "aa_hist": iter_num % 5,
                    "mech_time": 0.05,
                    "fab_time": 0.04,
                    "stim_time": 0.03,
                    "dens_time": 0.02,
                    "memory_mb": 1234.5 + step * 10 + iter_num * 0.5,
                })
        pd.DataFrame(subiters_rows).to_csv(run_dir / "subiterations.csv", index=False)
    
    comm.Barrier()
    return run_dir


@pytest.fixture
def mock_sweep_directory(shared_tmp_path):
    """Create mock sweep directory with multiple runs and sweep_summary.csv."""
    comm = MPI.COMM_WORLD
    sweep_dir = shared_tmp_path / "test_sweep"
    
    if comm.rank == 0:
        sweep_dir.mkdir(exist_ok=True)
        
        runs = [
            {"hash": "aaaa1111", "dt_days": 25.0, "accel_type": "picard"},
            {"hash": "bbbb2222", "dt_days": 50.0, "accel_type": "picard"},
            {"hash": "cccc3333", "dt_days": 25.0, "accel_type": "anderson"},
            {"hash": "dddd4444", "dt_days": 50.0, "accel_type": "anderson"},
        ]
        
        for run in runs:
            run_dir = sweep_dir / run["hash"]
            run_dir.mkdir(exist_ok=True)
            
            config = {"dt_days": run["dt_days"], "accel_type": run["accel_type"], "coupling_tol": 1e-6}
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f)
            
            steps = pd.DataFrame({
                "step": [0, 1],
                "time_days": [0.0, run["dt_days"]],
                "dt_days": [run["dt_days"], run["dt_days"]],
            })
            steps.to_csv(run_dir / "steps.csv", index=False)
            
            subiters = pd.DataFrame({
                "step": [0, 0, 1, 1],
                "iter": [0, 1, 0, 1],
                "time_days": [0.0, 0.0, run["dt_days"], run["dt_days"]],
                "proj_res": [1e-4, 1e-7, 1e-4, 1e-7],
            })
            subiters.to_csv(run_dir / "subiterations.csv", index=False)
        
        summary = pd.DataFrame([
            {"output_dir": r["hash"], "dt_days": r["dt_days"], "accel_type": r["accel_type"]}
            for r in runs
        ])
        summary.to_csv(sweep_dir / "sweep_summary.csv", index=False)
    
    comm.Barrier()
    return sweep_dir


# =============================================================================
# SimulationLoader Tests
# =============================================================================

class TestSimulationLoaderBasic:
    """Basic tests for SimulationLoader (no checkpoint I/O)."""
    
    def test_initialization(self, mock_run_directory):
        """Test initialization and directory validation."""
        comm = MPI.COMM_WORLD
        
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        assert loader.output_dir == Path(mock_run_directory)
        assert loader.comm == comm
        
        with pytest.raises(FileNotFoundError, match="Simulation directory not found"):
            SimulationLoader("/nonexistent/path", comm, verbose=False)

    def test_config_loading(self, mock_run_directory):
        """Test configuration loading and parameter access."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        config = loader.get_config()
        assert isinstance(config, dict)
        assert config["dt_days"] == 50.0
        assert config["accel_type"] == "anderson"
        assert config["mesh_size"] == 3
        
        # Test caching
        config2 = loader.get_config()
        assert config is config2
        
        # Test single param access
        assert loader.get_param("dt_days") == 50.0
        assert loader.get_param("nonexistent", default=42) == 42

    def test_run_summary_loading(self, mock_run_directory):
        """Test run summary metadata loading."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        summary = loader.get_run_summary()
        assert summary["success"] is True
        assert summary["total_steps"] == 4
        assert summary["avg_subiters"] == 8.5
        assert summary["final_residual"] == 3.2e-7
        
        # Test caching
        summary2 = loader.get_run_summary()
        assert summary is summary2

    def test_missing_run_summary(self, shared_tmp_path):
        """Test that missing run_summary.json returns empty dict."""
        comm = MPI.COMM_WORLD
        run_dir = shared_tmp_path / "incomplete_run"
        
        if comm.rank == 0:
            run_dir.mkdir(exist_ok=True)
            with open(run_dir / "config.json", "w") as f:
                json.dump({"dt_days": 50.0}, f)
        
        comm.Barrier()
        
        loader = SimulationLoader(run_dir, comm, verbose=False)
        summary = loader.get_run_summary()
        
        assert summary == {}

    def test_steps_metrics_loading(self, mock_run_directory):
        """Test per-timestep metrics loading."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        steps_df = loader.get_steps_metrics()
        
        assert isinstance(steps_df, pd.DataFrame)
        assert len(steps_df) == 4
        assert list(steps_df["step"]) == [0, 1, 2, 3]
        assert list(steps_df["time_days"]) == [0.0, 50.0, 100.0, 150.0]
        assert all(steps_df["dt_days"] == 50.0)
        
        expected_cols = {"step", "time_days", "dt_days", "num_subiters", "memory_mb"}
        assert expected_cols.issubset(set(steps_df.columns))

    def test_subiterations_metrics_loading(self, mock_run_directory):
        """Test per-subiteration metrics loading."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        subiters_df = loader.get_subiterations_metrics()
        
        assert isinstance(subiters_df, pd.DataFrame)
        assert len(subiters_df) == 10 + 8 + 9 + 7  # Sum of num_subiters
        
        expected_cols = {
            "step", "iter", "time_days", "proj_res", "aa_step_res",
            "mech_iters", "fab_iters", "stim_iters", "dens_iters",
            "accepted", "restart", "condH"
        }
        assert expected_cols.issubset(set(subiters_df.columns))
        
        step_0_data = subiters_df[subiters_df["step"] == 0]
        assert len(step_0_data) == 10
        assert all(step_0_data["time_days"] == 0.0)

    def test_metrics_at_time(self, mock_run_directory):
        """Test aggregated metrics extraction at specific time."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        metrics = loader.get_metrics_at_time(100.0)
        
        assert metrics["step"] == 2
        assert metrics["time_days"] == 100.0
        assert metrics["num_subiters"] == 9
        assert "total_mech_iters" in metrics
        assert "total_stim_iters" in metrics
        assert "aa_acceptances" in metrics
        assert "final_proj_res" in metrics
        
        # Test closest time matching
        metrics_close = loader.get_metrics_at_time(105.0)
        assert metrics_close["step"] == 2
        assert metrics_close["time_days"] == 100.0

    def test_available_times(self, mock_run_directory):
        """Test extraction of available checkpoint times."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        times = loader.get_available_times()
        
        assert isinstance(times, np.ndarray)
        np.testing.assert_array_equal(times, [0.0, 50.0, 100.0, 150.0])


class TestSimulationLoaderErrors:
    """Error handling tests for SimulationLoader."""
    
    def test_missing_config_file(self, shared_tmp_path):
        """Test error when config.json is missing."""
        comm = MPI.COMM_WORLD
        run_dir = shared_tmp_path / "no_config"
        
        if comm.rank == 0:
            run_dir.mkdir(exist_ok=True)
        comm.Barrier()
        
        loader = SimulationLoader(run_dir, comm, verbose=False)
        
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            loader.get_config()

    def test_missing_steps_csv(self, shared_tmp_path):
        """Test error when steps.csv is missing."""
        comm = MPI.COMM_WORLD
        run_dir = shared_tmp_path / "no_steps"
        
        if comm.rank == 0:
            run_dir.mkdir(exist_ok=True)
            with open(run_dir / "config.json", "w") as f:
                json.dump({"dt_days": 50.0}, f)
        
        comm.Barrier()
        
        loader = SimulationLoader(run_dir, comm, verbose=False)
        
        with pytest.raises(FileNotFoundError, match="Steps metrics not found"):
            loader.get_steps_metrics()

    def test_missing_subiterations_csv(self, shared_tmp_path):
        """Test error when subiterations.csv is missing."""
        comm = MPI.COMM_WORLD
        run_dir = shared_tmp_path / "no_subiters"
        
        if comm.rank == 0:
            run_dir.mkdir(exist_ok=True)
            with open(run_dir / "config.json", "w") as f:
                json.dump({"dt_days": 50.0}, f)
        
        comm.Barrier()
        
        loader = SimulationLoader(run_dir, comm, verbose=False)
        
        with pytest.raises(FileNotFoundError, match="Subiterations metrics not found"):
            loader.get_subiterations_metrics()

    def test_missing_field_checkpoint(self, mock_run_directory):
        """Test error when field checkpoint is missing."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            loader.get_fields_at_time(0.0, fields=["u"])


# =============================================================================
# SweepLoader Tests
# =============================================================================

class TestSweepLoader:
    """Tests for SweepLoader."""
    
    def test_initialization(self, mock_sweep_directory):
        """Test sweep loader initialization."""
        comm = MPI.COMM_WORLD
        
        # Valid directory
        sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
        assert sweep.base_dir == Path(mock_sweep_directory)
        assert sweep.comm == comm
        
        # Invalid directory
        with pytest.raises(FileNotFoundError, match="Sweep directory not found"):
            SweepLoader("/nonexistent/sweep", comm, verbose=False)

    def test_summary_loading(self, mock_sweep_directory):
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

    def test_filter_runs(self, mock_sweep_directory):
        """Test filtering runs by parameter values."""
        comm = MPI.COMM_WORLD
        sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
        
        picard_runs = sweep.filter_runs(accel_type="picard")
        assert len(picard_runs) == 2
        assert all(picard_runs["accel_type"] == "picard")
        
        dt25_runs = sweep.filter_runs(dt_days=25.0)
        assert len(dt25_runs) == 2
        assert all(dt25_runs["dt_days"] == 25.0)
        
        specific = sweep.filter_runs(accel_type="anderson", dt_days=50.0)
        assert len(specific) == 1
        assert specific.iloc[0]["output_dir"] == "dddd4444"
        
        empty = sweep.filter_runs(dt_days=999.0)
        assert len(empty) == 0

    def test_get_loader(self, mock_sweep_directory):
        """Test getting individual run loaders."""
        comm = MPI.COMM_WORLD
        sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
        
        loader = sweep.get_loader("aaaa1111")
        
        assert isinstance(loader, SimulationLoader)
        assert loader.output_dir == Path(mock_sweep_directory) / "aaaa1111"
        
        # Test caching
        loader2 = sweep.get_loader("aaaa1111")
        assert loader is loader2

    def test_get_all_loaders(self, mock_sweep_directory):
        """Test getting all run loaders."""
        comm = MPI.COMM_WORLD
        sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
        
        loaders = sweep.get_all_loaders()
        
        assert len(loaders) == 4
        assert all(isinstance(loader, SimulationLoader) for loader in loaders)
        
        hashes = {loader.output_dir.name for loader in loaders}
        expected_hashes = {"aaaa1111", "bbbb2222", "cccc3333", "dddd4444"}
        assert hashes == expected_hashes

    def test_missing_sweep_summary(self, shared_tmp_path):
        """Test that missing sweep_summary.csv scans directories."""
        comm = MPI.COMM_WORLD
        sweep_dir = shared_tmp_path / "sweep_no_summary"
        
        if comm.rank == 0:
            sweep_dir.mkdir(exist_ok=True)
            for hash_val in ["run1111", "run2222"]:
                (sweep_dir / hash_val).mkdir(exist_ok=True)
        
        comm.Barrier()
        
        sweep = SweepLoader(sweep_dir, comm, verbose=False)
        summary = sweep.get_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "output_dir" in summary.columns
        assert set(summary["output_dir"]) == {"run1111", "run2222"}


# =============================================================================
# Integration and cross-consistency tests
# =============================================================================

class TestLoaderConsistency:
    """Tests for data consistency between loaders."""
    
    def test_metrics_consistency(self, mock_run_directory):
        """Test consistency between steps and subiterations metrics."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        steps_df = loader.get_steps_metrics()
        subiters_df = loader.get_subiterations_metrics()
        
        for _, step_row in steps_df.iterrows():
            step_num = int(step_row["step"])
            expected_count = int(step_row["num_subiters"])
            actual_count = len(subiters_df[subiters_df["step"] == step_num])
            assert actual_count == expected_count

    def test_sweep_and_loader_parameter_consistency(self, mock_sweep_directory):
        """Test that sweep summary matches individual loader configs."""
        comm = MPI.COMM_WORLD
        sweep = SweepLoader(mock_sweep_directory, comm, verbose=False)
        summary = sweep.get_summary()
        
        for _, row in summary.iterrows():
            loader = sweep.get_loader(row["output_dir"])
            config = loader.get_config()
            assert config["dt_days"] == row["dt_days"]
            assert config["accel_type"] == row["accel_type"]

    def test_mpi_rank_consistency(self, mock_run_directory):
        """Test that all MPI ranks get consistent data."""
        comm = MPI.COMM_WORLD
        loader = SimulationLoader(mock_run_directory, comm, verbose=False)
        
        config = loader.get_config()
        steps = loader.get_steps_metrics()
        subiters = loader.get_subiterations_metrics()
        
        all_configs = comm.gather(config["dt_days"], root=0)
        all_step_counts = comm.gather(len(steps), root=0)
        all_subiter_counts = comm.gather(len(subiters), root=0)
        
        if comm.rank == 0:
            assert all(c == all_configs[0] for c in all_configs)
            assert all(c == all_step_counts[0] for c in all_step_counts)
            assert all(c == all_subiter_counts[0] for c in all_subiter_counts)
