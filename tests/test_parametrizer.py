"""Tests for parametrizer module."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
from mpi4py import MPI

from parametrizer import ParameterSweep, Parametrizer


# Test fixtures
@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def comm():
    """MPI communicator."""
    return MPI.COMM_WORLD


# Dummy callable for testing
def dummy_callable(
    param_point: Dict[str, Any],
    output_path: Path,
    comm: MPI.Comm
) -> None:
    """Simple test callable that creates output directory."""
    if comm.rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)


# ParameterSweep Tests
class TestParameterSweep:
    """Tests for ParameterSweep class."""
    
    def test_init_basic(self, temp_output_dir):
        """Test basic initialization."""
        sweep = ParameterSweep(
            params={"N": [8, 16], "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir
        )
        assert sweep.params == {"N": [8, 16], "dt": [0.5, 1.0]}
        assert sweep.base_output_dir == temp_output_dir
    
    def test_init_auto_listify(self, temp_output_dir):
        """Test automatic conversion of scalar values to lists."""
        sweep = ParameterSweep(
            params={"N": 8, "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir
        )
        assert sweep.params == {"N": [8], "dt": [0.5, 1.0]}
    
    def test_init_empty_params(self, temp_output_dir):
        """Test that empty params raises error."""
        with pytest.raises(ValueError, match="params dictionary cannot be empty"):
            ParameterSweep(
                params={},
                base_output_dir=temp_output_dir
            )
    
    def test_generate_points_single_param(self, temp_output_dir):
        """Test point generation with single parameter."""
        sweep = ParameterSweep(
            params={"N": [8, 16, 24]},
            base_output_dir=temp_output_dir
        )
        points = sweep.generate_points()
        
        assert len(points) == 3
        assert points == [{"N": 8}, {"N": 16}, {"N": 24}]
    
    def test_generate_points_cartesian_product(self, temp_output_dir):
        """Test point generation with Cartesian product."""
        sweep = ParameterSweep(
            params={"N": [8, 16], "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir
        )
        points = sweep.generate_points()
        
        assert len(points) == 4
        expected = [
            {"N": 8, "dt": 0.5},
            {"N": 8, "dt": 1.0},
            {"N": 16, "dt": 0.5},
            {"N": 16, "dt": 1.0},
        ]
        assert points == expected
    
    def test_generate_points_three_params(self, temp_output_dir):
        """Test point generation with three parameters."""
        sweep = ParameterSweep(
            params={"N": [8, 16], "dt": [0.5, 1.0], "tol": [1e-5, 1e-6]},
            base_output_dir=temp_output_dir
        )
        points = sweep.generate_points()
        
        assert len(points) == 8  # 2 * 2 * 2
        # Verify all combinations exist
        assert {"N": 8, "dt": 0.5, "tol": 1e-5} in points
        assert {"N": 16, "dt": 1.0, "tol": 1e-6} in points
    
    def test_format_output_path_with_hash(self, temp_output_dir):
        """Test output path formatting with hash."""
        sweep = ParameterSweep(
            params={"N": [8], "dt": [0.5]},
            base_output_dir=temp_output_dir
        )
        
        path = sweep.format_output_path({"N": 8, "dt": 0.5})
        assert path.parent == temp_output_dir
        assert len(path.name) == 8
    
    def test_hash_uniqueness(self, temp_output_dir):
        """Test that different parameters produce different hashes."""
        sweep = ParameterSweep(
            params={"N": [8], "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir
        )
        
        path1 = sweep.format_output_path({"N": 8, "dt": 0.5})
        path2 = sweep.format_output_path({"N": 8, "dt": 1.0})
        assert path1 != path2
    
    def test_hash_stability(self, temp_output_dir):
        """Test that same parameters always produce same hash."""
        sweep = ParameterSweep(
            params={"N": [8], "dt": [0.5]},
            base_output_dir=temp_output_dir
        )
        
        path1 = sweep.format_output_path({"N": 8, "dt": 0.5})
        path2 = sweep.format_output_path({"N": 8, "dt": 0.5})
        assert path1 == path2
    
    def test_total_runs(self, temp_output_dir):
        """Test total runs calculation."""
        sweep = ParameterSweep(
            params={"N": [8, 16, 24], "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir
        )
        assert sweep.total_runs() == 6
    
    def test_total_runs_single_value(self, temp_output_dir):
        """Test total runs with single values."""
        sweep = ParameterSweep(
            params={"N": [8], "dt": [1.0]},
            base_output_dir=temp_output_dir
        )
        assert sweep.total_runs() == 1
    
    def test_metadata_storage(self, temp_output_dir):
        """Test metadata is stored correctly."""
        metadata = {"study": "convergence", "author": "test"}
        sweep = ParameterSweep(
            params={"N": [8]},
            base_output_dir=temp_output_dir,
            metadata=metadata
        )
        assert sweep.metadata == metadata


# Parametrizer Tests
class TestParametrizer:
    """Tests for Parametrizer class."""
    
    def test_init(self, temp_output_dir, comm):
        """Test parametrizer initialization."""
        sweep = ParameterSweep(
            params={"N": [8]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        assert parametrizer.sweep == sweep
        assert parametrizer.callable_func == dummy_callable
        assert parametrizer.comm == comm
    
    def test_run_single_point(self, temp_output_dir, comm):
        """Test single run execution."""
        sweep = ParameterSweep(
            params={"N": [8]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        parametrizer.run()
        
        if comm.rank == 0:
            csv_file = temp_output_dir / "sweep_summary.csv"
            assert csv_file.exists()
            
            import csv
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 1
            assert rows[0]["N"] == "8"
    
    def test_run_multiple_points(self, temp_output_dir, comm):
        """Test multiple runs execution."""
        sweep = ParameterSweep(
            params={"N": [8, 16, 24]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        parametrizer.run()
        
        if comm.rank == 0:
            csv_file = temp_output_dir / "sweep_summary.csv"
            assert csv_file.exists()
            
            import csv
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 3
            assert rows[0]["N"] == "8"
            assert rows[1]["N"] == "16"
            assert rows[2]["N"] == "24"
    
    def test_run_cartesian_product(self, temp_output_dir, comm):
        """Test run with Cartesian product of parameters."""
        sweep = ParameterSweep(
            params={"N": [8, 16], "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        parametrizer.run()
        
        if comm.rank == 0:
            csv_file = temp_output_dir / "sweep_summary.csv"
            assert csv_file.exists()
            
            import csv
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 4
            # Check all combinations executed
            param_combos = [(r["N"], r["dt"]) for r in rows]
            assert ("8", "0.5") in param_combos
            assert ("16", "1.0") in param_combos
    
    def test_run_saves_results(self, temp_output_dir, comm):
        """Test that run saves CSV and JSON summary."""
        sweep = ParameterSweep(
            params={"N": [8]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        parametrizer.run()
        
        if comm.rank == 0:
            csv_file = temp_output_dir / "sweep_summary.csv"
            json_file = temp_output_dir / "sweep_summary.json"
            
            assert csv_file.exists()
            assert json_file.exists()
    
    def test_run_single_method(self, temp_output_dir, comm):
        """Test run_single method."""
        sweep = ParameterSweep(
            params={"N": [8, 16]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        parametrizer.run_single({"N": 24})
        
        # Just verify it doesn't crash
        assert True
    
    def test_run_single_custom_output_path(self, temp_output_dir, comm):
        """Test run_single with custom output path."""
        sweep = ParameterSweep(
            params={"N": [8]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        
        custom_path = temp_output_dir / "custom_run"
        parametrizer.run_single({"N": 8}, output_path=custom_path)
        
        if comm.rank == 0:
            assert custom_path.exists()
    
    def test_callable_receives_correct_args(self, temp_output_dir, comm):
        """Test callable receives correct arguments."""
        received_args = {}
        
        def test_callable(param_point, output_path, comm):
            received_args["param_point"] = param_point
            received_args["output_path"] = output_path
            received_args["comm"] = comm
        
        sweep = ParameterSweep(
            params={"N": [8]},
            base_output_dir=temp_output_dir
        )
        parametrizer = Parametrizer(sweep, test_callable, comm, verbose=False)
        parametrizer.run()
        
        assert received_args["param_point"] == {"N": 8}
        # Hash-based naming, just check it exists
        assert received_args["output_path"].parent == temp_output_dir
        assert len(received_args["output_path"].name) == 8
        assert received_args["comm"] == comm


# Integration Tests
class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow(self, temp_output_dir, comm):
        """Test complete parameter sweep workflow."""
        sweep = ParameterSweep(
            params={"N": [8, 16], "dt": [0.5, 1.0]},
            base_output_dir=temp_output_dir,
            metadata={"study": "integration_test"}
        )
        
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        parametrizer.run()
        
        if comm.rank == 0:
            assert (temp_output_dir / "sweep_summary.csv").exists()
            assert (temp_output_dir / "sweep_summary.json").exists()
            
            import csv
            with open(temp_output_dir / "sweep_summary.csv") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 4
            assert any(row["N"] == "8" and row["dt"] == "0.5" for row in rows)
            assert any(row["N"] == "16" and row["dt"] == "1.0" for row in rows)
            
            with open(temp_output_dir / "sweep_summary.json") as f:
                data = json.load(f)
            
            assert data["metadata"]["study"] == "integration_test"
            assert data["total_runs"] == 4
    
    def test_mixed_param_types(self, temp_output_dir, comm):
        """Test sweep with mixed parameter types (int, float, string)."""
        sweep = ParameterSweep(
            params={
                "N": [8, 16],
                "dt": [0.5, 1.0],
                "solver": ["picard", "anderson"]
            },
            base_output_dir=temp_output_dir
        )
        
        parametrizer = Parametrizer(sweep, dummy_callable, comm, verbose=False)
        parametrizer.run()
        
        if comm.rank == 0:
            import csv
            with open(temp_output_dir / "sweep_summary.csv") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 8  # 2 * 2 * 2
            # Verify mixed types handled correctly
            assert any(r["solver"] == "picard" for r in rows)
            assert any(r["solver"] == "anderson" for r in rows)
