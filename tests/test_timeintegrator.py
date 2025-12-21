"""
Tests for TimeIntegrator.
"""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import basix.ufl

from simulation.timeintegrator import TimeIntegrator
from simulation.params import TimeParams

@pytest.fixture
def ti_setup():
    """Setup for TimeIntegrator tests."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 4, 4, 4)
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, P1)
    
    f1 = fem.Function(V, name="f1")
    f2 = fem.Function(V, name="f2")
    
    state_fields = {"f1": f1, "f2": f2}
    
    time_params = TimeParams(dt_min=0.1, dt_max=10.0)
    ti = TimeIntegrator(comm, state_fields, time_params=time_params)
    return ti, state_fields

class TestTimeIntegrator:
    """Tests for TimeIntegrator class."""

    def test_initialization(self, ti_setup):
        """Test initialization and field setup."""
        ti, fields = ti_setup
        assert ti._N_total > 0
        assert len(ti._fields) == 2
        assert ti.step_count == 0

    def test_predict_first_step(self, ti_setup):
        """Test prediction for the first step (Forward Euler)."""
        ti, fields = ti_setup
        
        # Set initial values
        fields["f1"].x.array[:] = 1.0
        fields["f2"].x.array[:] = 2.0
        
        # Set rates (manually, as if computed by previous step)
        # For first step, predict uses rate_last which is 0 initialized?
        # Wait, predict uses rate_last. rate_last is updated in commit_step.
        # So initially rate_last is 0.
        
        dt = 1.0
        pred = ti.predict(dt)
        
        # Should be same as current values since rate is 0
        assert np.allclose(pred["f1"], 1.0)
        assert np.allclose(pred["f2"], 2.0)

    def test_commit_and_predict_second_step(self, ti_setup):
        """Test commit_step and subsequent prediction."""
        ti, fields = ti_setup
        
        # Step 0: t=0, f=0
        fields["f1"].x.array[:] = 0.0
        old_fields = {"f1": fields["f1"].copy(), "f2": fields["f2"].copy()}
        
        # Step 1: t=1, f=1 (rate = 1)
        fields["f1"].x.array[:] = 1.0
        
        dt = 1.0
        ti.commit_step(dt, fields, old_fields)
        
        assert ti.step_count == 1
        
        # Check owned DOFs only (ghosts are not updated by commit_step)
        n_owned = ti._fields["f1"].n_owned
        assert np.allclose(ti._fields["f1"].rate_last.x.array[:n_owned], 1.0)
        
        # Predict for Step 2 (AB1 / Forward Euler using rate_last)
        # pred = val + dt * rate_last = 1.0 + 1.0 * 1.0 = 2.0
        pred = ti.predict(dt)
        assert np.allclose(pred["f1"], 2.0)

    def test_wrms_error(self, ti_setup):
        """Test WRMS error computation."""
        ti, fields = ti_setup
        
        # Set fields to 1.0
        fields["f1"].x.array[:] = 1.0
        fields["f2"].x.array[:] = 1.0
        
        # Prediction is 1.1 (error 0.1)
        pred = {
            "f1": np.full_like(fields["f1"].x.array, 1.1),
            "f2": np.full_like(fields["f2"].x.array, 1.1)
        }
        
        # rtol=1e-3, atol=1e-4
        # scale = 1e-3 * 1.0 + 1e-4 = 0.0011
        # diff = 0.1
        # term = (0.1 / 0.0011)^2 approx (90.9)^2 approx 8264
        
        error = ti.compute_wrms_error(pred)
        assert error > 1.0 # Should be large

    def test_suggest_dt_divergence(self, ti_setup):
        """Test DT reduction on divergence."""
        ti, _ = ti_setup
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=False, error_norm=0.0)
        assert not accepted
        assert dt_new < 1.0
        assert reason == "diverged"

    def test_suggest_dt_large_error(self, ti_setup):
        """Test DT reduction on large error."""
        ti, _ = ti_setup
        ti.step_count = 1 # Not first step
        
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=True, error_norm=2.0)
        assert not accepted
        assert dt_new < 1.0
        assert "error" in reason

    def test_suggest_dt_acceptance(self, ti_setup):
        """Test DT acceptance and growth."""
        ti, _ = ti_setup
        ti.step_count = 1
        
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=True, error_norm=0.5)
        assert accepted
        assert dt_new >= 1.0 # Should grow or stay same
        assert reason == "accepted"

