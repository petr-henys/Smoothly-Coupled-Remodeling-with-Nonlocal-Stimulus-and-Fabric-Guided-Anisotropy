"""Test convergence analysis functionality."""

import tempfile
import shutil
from pathlib import Path
from mpi4py import MPI
import numpy as np
import pytest

from dolfinx import mesh, fem
import basix.ufl
from analysis.utils import (
    save_function_npz,
    load_field_from_npz,
    compute_richardson_triplets_qoi,
    GCI_SAFETY_FACTOR
)


@pytest.mark.mpi
def test_save_load_npz_roundtrip():
    """Test that NPZ save/load preserves function values."""
    comm = MPI.COMM_WORLD
    
    # Create test mesh and function
    domain = mesh.create_unit_cube(comm, 8, 8, 8, ghost_mode=mesh.GhostMode.shared_facet)
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, P1)
    
    # Create test function with known pattern
    u_orig = fem.Function(V, name="test_function")
    u_orig.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]) * x[2])
    u_orig.x.scatter_forward()
    
    # Create temporary directory for test
    temp_dir = Path(tempfile.mkdtemp()) if comm.rank == 0 else None
    temp_dir = comm.bcast(temp_dir, root=0)
    
    try:
        # Save to NPZ
        npz_path = temp_dir / "test_function.npz"
        save_function_npz(u_orig, npz_path, comm)
        
        # Load back into new function
        u_loaded = fem.Function(V, name="loaded_function")
        domain_loaded, u_loaded = load_field_from_npz(temp_dir, comm, 8, "test_function", "scalar")
        
        # Compare values
        diff = u_orig.x.array - u_loaded.x.array
        max_diff = comm.allreduce(np.max(np.abs(diff)), op=MPI.MAX)
        
        assert max_diff < 1e-14, f"Roundtrip error too large: {max_diff}"
        
        if comm.rank == 0:
            print(f"✓ NPZ roundtrip test passed (max_diff = {max_diff:.2e})")
    
    finally:
        if comm.rank == 0:
            shutil.rmtree(temp_dir)


@pytest.mark.unit
def test_richardson_extrapolation():
    """Test Richardson extrapolation with known convergence."""
    # Create synthetic data with known convergence order p=2
    h_values = [0.1, 0.05, 0.025, 0.0125]
    exact_value = 1.0
    p_true = 2.0
    
    # Generate QoI values with p=2 convergence: Q(h) = Q_exact + C*h^p
    C = 0.1
    qoi_values = [exact_value + C * h**p_true for h in h_values]
    
    # Apply Richardson extrapolation
    richardson_data = compute_richardson_triplets_qoi(
        h_values, qoi_values, safety_factor=GCI_SAFETY_FACTOR
    )
    
    assert len(richardson_data) == 2, "Should have 2 Richardson triplets"
    
    # Check estimated convergence orders
    p_estimates = [r["p"] for r in richardson_data]
    for p_est in p_estimates:
        assert abs(p_est - p_true) < 0.1, f"Convergence order estimate {p_est} too far from true value {p_true}"
    
    # Check extrapolated values
    q_ext_estimates = [r["Q_ext"] for r in richardson_data]
    for q_ext in q_ext_estimates:
        if np.isfinite(q_ext):
            assert abs(q_ext - exact_value) < 0.05, f"Extrapolated value {q_ext} too far from exact {exact_value}"
    
    print(f"✓ Richardson test passed: p_estimates = {p_estimates}, Q_ext = {q_ext_estimates}")


@pytest.mark.unit  
def test_gci_computation():
    """Test Grid Convergence Index computation."""
    # Test data with good convergence
    h_values = [0.08, 0.04, 0.02] 
    qoi_values = [1.1, 1.025, 1.00625]  # Converging to 1.0 with p≈2
    
    richardson_data = compute_richardson_triplets_qoi(h_values, qoi_values)
    
    assert len(richardson_data) == 1, "Should have 1 Richardson triplet"
    
    result = richardson_data[0]
    
    # Check GCI values are reasonable (should be small for good convergence)
    assert 0 < result["GCI32_percent"] < 10, f"GCI32 should be reasonable: {result['GCI32_percent']}"
    assert 0 < result["GCI21_percent"] < 10, f"GCI21 should be reasonable: {result['GCI21_percent']}"
    
    # Check beta (should be close to 1 for consistent convergence)
    if np.isfinite(result["beta"]):
        assert 0.5 < result["beta"] < 2.0, f"Beta should be close to 1: {result['beta']}"
    
    print(f"✓ GCI test passed: GCI32={result['GCI32_percent']:.2f}%, GCI21={result['GCI21_percent']:.2f}%, beta={result['beta']:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])