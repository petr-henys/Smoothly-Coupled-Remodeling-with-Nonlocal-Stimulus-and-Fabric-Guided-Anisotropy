"""Tests for CheckpointStorage: mesh/function write/read round-trip with adios4dolfinx."""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from pathlib import Path

from simulation.checkpoint import CheckpointStorage, load_checkpoint_mesh, load_checkpoint_function
from simulation.config import Config
from simulation.params import MaterialParams, OutputParams

@pytest.fixture
def checkpoint_cfg(shared_tmpdir, unit_cube, facet_tags):
    """Configuration for checkpoint tests."""
    return Config(
        domain=unit_cube, 
        facet_tags=facet_tags,
        material=MaterialParams(),
        output=OutputParams(results_dir=str(shared_tmpdir / "test_ckpt"))
    )

class TestCheckpointStorage:
    """Test MPI-independent checkpointing."""

    def test_write_mesh(self, checkpoint_cfg):
        """Test writing mesh to checkpoint."""
        storage = CheckpointStorage(checkpoint_cfg)
        storage.write_mesh()
        storage.close()
        
        assert Path(storage.checkpoint_path).exists()
        # It's a directory for ADIOS2 BP4 (usually)
        assert Path(storage.checkpoint_path).is_dir() or Path(storage.checkpoint_path).is_file()

    def test_write_read_scalar(self, checkpoint_cfg, spaces):
        """Test writing and reading a scalar function."""
        storage = CheckpointStorage(checkpoint_cfg)
        storage.write_mesh()  # Must write mesh first
        
        # Create a scalar field
        V = spaces.Q
        rho = fem.Function(V, name="rho")
        rho.x.array[:] = 0.5
        rho.x.scatter_forward()
        
        t = 10.0
        storage.write_function(rho, t)
        storage.close()
        
        # Verify file exists
        assert Path(storage.checkpoint_path).exists()
        
        # Read back
        comm = checkpoint_cfg.domain.comm
        mesh_new = load_checkpoint_mesh(storage.checkpoint_path, comm)
        V_new = fem.functionspace(mesh_new, ("Lagrange", 1))
        
        rho_new = load_checkpoint_function(storage.checkpoint_path, "rho", V_new, t)
        
        # Check max value
        max_val = comm.allreduce(rho_new.x.array.max(), op=MPI.MAX)
        assert np.isclose(max_val, 0.5)

    def test_write_read_vector(self, checkpoint_cfg, spaces):
        """Test writing and reading a vector function."""
        storage = CheckpointStorage(checkpoint_cfg)
        storage.write_mesh()
        
        # Create vector field
        V = spaces.V
        u = fem.Function(V, name="u")
        # Set u = (1, 0, 0)
        u_vals = u.x.array.reshape((-1, 3))
        u_vals[:, 0] = 1.0
        u.x.array[:] = u_vals.flatten()
        u.x.scatter_forward()
        
        t = 5.0
        storage.write_function(u, t)
        storage.close()
        
        # Read back
        comm = checkpoint_cfg.domain.comm
        mesh_new = load_checkpoint_mesh(storage.checkpoint_path, comm)
        V_new = fem.functionspace(mesh_new, ("Lagrange", 1, (3,)))
        
        u_new = load_checkpoint_function(storage.checkpoint_path, "u", V_new, t)
        
        # Check max X component
        u_new_vals = u_new.x.array.reshape((-1, 3))
        max_x = comm.allreduce(u_new_vals[:, 0].max(), op=MPI.MAX)
        assert np.isclose(max_x, 1.0)

