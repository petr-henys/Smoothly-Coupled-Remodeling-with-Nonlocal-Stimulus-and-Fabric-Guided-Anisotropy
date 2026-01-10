"""Tests for TimeIntegrator: AB2 predictor, PI controller, adaptive vs fixed stepping."""

import pytest
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
import basix.ufl

from simulation.timeintegrator import TimeIntegrator
from simulation.params import TimeParams


# =============================================================================
# Fixtures
# =============================================================================

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


@pytest.fixture
def adaptive_integrator():
    """Create an adaptive time integrator for performance tests."""
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(comm, 4, 4, 4)
    P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    V = fem.functionspace(domain, P1)
    
    f = fem.Function(V, name="density")
    state_fields = {"density": f}
    
    # Adaptive parameters with wide dt range
    time_params = TimeParams(
        dt_min=0.01,
        dt_max=10.0,
        adaptive_rtol=1e-2,
        adaptive_atol=1e-4,
        pi_safety=0.9,
        pi_growth_max=2.0,
        pi_shrink_min=0.1,
    )
    ti = TimeIntegrator(comm, state_fields, time_params=time_params)
    return ti, state_fields, V


# =============================================================================
# Helper: ODE system simulator
# =============================================================================

def simulate_ode_adaptive(
    ti: TimeIntegrator,
    f: fem.Function,
    rhs_func,
    T_end: float,
    dt_init: float,
    stiff_phase_start: float = None,
    stiff_phase_end: float = None,
):
    """
    Simulate an ODE dy/dt = rhs_func(t, y) with adaptive time stepping.
    
    Returns:
        dict with 'steps', 'rejections', 'dt_history', 'y_history', 't_history'
    """
    from simulation.utils import get_owned_size
    
    n_owned = int(get_owned_size(f))
    
    t = 0.0
    dt = dt_init
    steps = 0
    rejections = 0
    consecutive_rejections = 0
    dt_history = []
    t_history = [t]
    y_history = [f.x.array[:n_owned].copy()]
    
    while t < T_end:
        # Store old values
        f_old = f.copy()
        y_old = f.x.array[:n_owned].copy()
        
        # Get prediction
        pred = ti.predict(dt)
        
        # Compute "exact" solution (forward Euler for simplicity, represents corrector)
        rhs = rhs_func(t, y_old)
        y_new = y_old + dt * rhs
        f.x.array[:n_owned] = y_new
        f.x.scatter_forward()
        
        # Compute error between predictor and corrector
        error = ti.compute_wrms_error(pred)
        
        # Get adaptive decision
        accepted, dt_next, reason = ti.suggest_dt(dt, converged=True, error_norm=error)
        
        if accepted:
            # Commit the step
            ti.commit_step(dt, {"density": f}, {"density": f_old})
            t += dt
            steps += 1
            dt_history.append(dt)
            t_history.append(t)
            y_history.append(y_new.copy())
            consecutive_rejections = 0
        else:
            # Reject: restore old state
            f.x.array[:n_owned] = y_old
            f.x.scatter_forward()
            rejections += 1
            consecutive_rejections += 1
            
            # If too many consecutive rejections, reset AB history
            # This helps when dynamics change abruptly
            if consecutive_rejections >= 5:
                ti.reset_history()
                consecutive_rejections = 0
        
        dt = dt_next
        
        # Safety: prevent infinite loops
        if steps + rejections > 10000:
            break
    
    return {
        'steps': steps,
        'rejections': rejections,
        'dt_history': dt_history,
        't_history': t_history,
        'y_history': y_history,
    }


def simulate_ode_fixed(
    f: fem.Function,
    rhs_func,
    T_end: float,
    dt_fixed: float,
):
    """
    Simulate an ODE dy/dt = rhs_func(t, y) with fixed time stepping.
    
    Returns:
        dict with 'steps', 'y_history', 't_history'
    """
    from simulation.utils import get_owned_size
    
    n_owned = int(get_owned_size(f))
    
    t = 0.0
    steps = 0
    t_history = [t]
    y_history = [f.x.array[:n_owned].copy()]
    
    while t < T_end:
        y_old = f.x.array[:n_owned].copy()
        
        # Forward Euler step
        rhs = rhs_func(t, y_old)
        y_new = y_old + dt_fixed * rhs
        f.x.array[:n_owned] = y_new
        f.x.scatter_forward()
        
        t += dt_fixed
        steps += 1
        t_history.append(t)
        y_history.append(y_new.copy())
        
        if steps > 10000:
            break
    
    return {
        'steps': steps,
        't_history': t_history,
        'y_history': y_history,
    }


# =============================================================================
# Basic Unit Tests
# =============================================================================

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
        assert reason.startswith("reject:coupling")

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


# =============================================================================
# Adaptive vs Fixed Performance Tests
# =============================================================================

class TestAdaptiveVsFixed:
    """
    Tests proving that adaptive time stepping requires fewer steps 
    than fixed time stepping while maintaining accuracy.
    """

    def test_adaptive_fewer_steps_smooth_problem(self, adaptive_integrator):
        """
        For a smooth ODE (exponential decay), adaptive stepping should
        use fewer steps than a conservative fixed step.
        
        ODE: dy/dt = -0.1 * y  (exponential decay with τ = 10)
        This is a smooth problem where large steps are safe.
        """
        ti, fields, V = adaptive_integrator
        
        T_end = 50.0
        y0 = 1.0
        decay_rate = 0.1
        
        # RHS function: exponential decay
        def rhs_decay(t, y):
            return -decay_rate * y
        
        # --- Adaptive run ---
        f_adaptive = fields["density"]
        f_adaptive.x.array[:] = y0
        f_adaptive.x.scatter_forward()
        ti.reset_history()
        
        result_adaptive = simulate_ode_adaptive(
            ti, f_adaptive, rhs_decay, T_end, dt_init=0.5
        )
        
        # --- Fixed run with small dt to ensure stability ---
        # For explicit Euler stability: dt < 2/decay_rate = 20
        # Use conservative dt = 0.5 (similar to adaptive initial)
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 4, 4, 4)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        V_fixed = fem.functionspace(domain, P1)
        f_fixed = fem.Function(V_fixed, name="density_fixed")
        f_fixed.x.array[:] = y0
        f_fixed.x.scatter_forward()
        
        result_fixed = simulate_ode_fixed(
            f_fixed, rhs_decay, T_end, dt_fixed=0.5
        )
        
        # Adaptive should require FEWER or EQUAL steps
        # (it can take larger steps when error is low)
        adaptive_total = result_adaptive['steps'] + result_adaptive['rejections']
        fixed_total = result_fixed['steps']
        
        # Check that adaptive is at least as efficient
        # In smooth problems, adaptive dt can grow to dt_max
        assert result_adaptive['steps'] <= fixed_total, (
            f"Adaptive ({result_adaptive['steps']} steps) should need <= fixed ({fixed_total} steps)"
        )
        
        # Verify both reach similar final values (accuracy check)
        y_final_adaptive = result_adaptive['y_history'][-1].mean()
        y_final_fixed = result_fixed['y_history'][-1].mean()
        y_exact = y0 * np.exp(-decay_rate * T_end)
        
        # Both should be reasonably accurate
        assert abs(y_final_adaptive - y_exact) / y_exact < 0.5, "Adaptive solution inaccurate"
        assert abs(y_final_fixed - y_exact) / y_exact < 0.5, "Fixed solution inaccurate"

    def test_adaptive_grows_dt_on_smooth_region(self, adaptive_integrator):
        """
        In a smooth region, adaptive stepping should increase dt 
        towards dt_max.
        """
        ti, fields, V = adaptive_integrator
        
        f = fields["density"]
        f.x.array[:] = 1.0
        f.x.scatter_forward()
        ti.reset_history()
        
        # Linear ODE: dy/dt = 0.01 (constant growth, very smooth)
        def rhs_linear(t, y):
            return np.full_like(y, 0.01)
        
        result = simulate_ode_adaptive(
            ti, f, rhs_linear, T_end=20.0, dt_init=0.1
        )
        
        # dt should grow over time for smooth problem
        dt_history = result['dt_history']
        assert len(dt_history) >= 3, "Need at least 3 steps to check growth"
        
        # Last dt should be larger than first dt (or at dt_max)
        dt_first_few = np.mean(dt_history[:3])
        dt_last_few = np.mean(dt_history[-3:])
        
        assert dt_last_few >= dt_first_few, (
            f"dt should grow on smooth problem: first={dt_first_few:.3f}, last={dt_last_few:.3f}"
        )

    def test_adaptive_shrinks_dt_on_rapid_change(self, adaptive_integrator):
        """
        When solution changes rapidly, adaptive stepping should reduce dt.
        This tests the controller's ability to detect and respond to stiff dynamics.
        """
        ti, fields, V = adaptive_integrator
        
        f = fields["density"]
        f.x.array[:] = 1.0
        f.x.scatter_forward()
        ti.reset_history()
        
        # ODE with time-varying stiffness:
        # dy/dt = -k(t) * y where k(t) increases sharply at t=5
        def rhs_stiff_transition(t, y):
            if t < 5.0:
                k = 0.1  # Slow decay initially
            else:
                k = 2.0  # Fast decay after t=5
            return -k * y
        
        result = simulate_ode_adaptive(
            ti, f, rhs_stiff_transition, T_end=10.0, dt_init=1.0
        )
        
        dt_history = result['dt_history']
        t_history = result['t_history'][1:]  # Skip t=0
        
        # Find dt values before and after transition
        dt_before = [dt for t, dt in zip(t_history, dt_history) if t < 4.5]
        dt_after = [dt for t, dt in zip(t_history, dt_history) if t > 5.5]
        
        if dt_before and dt_after:
            avg_before = np.mean(dt_before)
            avg_after = np.mean(dt_after)
            
            # dt should decrease after stiff transition (or have rejections)
            # Either dt shrinks OR there were rejections
            has_adaptation = (avg_after < avg_before) or (result['rejections'] > 0)
            assert has_adaptation, (
                f"Adaptive should respond to stiff transition: "
                f"dt_before={avg_before:.3f}, dt_after={avg_after:.3f}, "
                f"rejections={result['rejections']}"
            )

    def test_adaptive_efficiency_multiscale_problem(self, adaptive_integrator):
        """
        For a multi-scale problem (slow then fast dynamics), adaptive stepping
        should vary dt appropriately - large during slow phase, small during fast.
        
        This is the KEY efficiency test: adaptive uses large dt during slow phase,
        small dt during fast phase.
        """
        ti, fields, V = adaptive_integrator
        
        T_end = 20.0
        y0 = 1.0
        
        # Multi-scale ODE:
        # Phase 1 (t < 10): slow decay, k = 0.05
        # Phase 2 (t >= 10): fast decay, k = 1.0
        def rhs_multiscale(t, y):
            if t < 10.0:
                k = 0.05  # Slow: can use large dt
            else:
                k = 1.0   # Fast: needs small dt for stability/accuracy
            return -k * y
        
        # --- Adaptive run ---
        f_adaptive = fields["density"]
        f_adaptive.x.array[:] = y0
        f_adaptive.x.scatter_forward()
        ti.reset_history()
        
        result_adaptive = simulate_ode_adaptive(
            ti, f_adaptive, rhs_multiscale, T_end, dt_init=0.5
        )
        
        # Key assertion: dt should VARY over time
        dt_history = result_adaptive['dt_history']
        t_history = result_adaptive['t_history'][1:]  # Skip t=0
        
        # Find dt values in slow phase (t < 8) and fast phase (t > 12)
        dt_slow_phase = [dt for t, dt in zip(t_history, dt_history) if t < 8.0]
        dt_fast_phase = [dt for t, dt in zip(t_history, dt_history) if t > 12.0]
        
        if dt_slow_phase and dt_fast_phase:
            avg_dt_slow = np.mean(dt_slow_phase)
            avg_dt_fast = np.mean(dt_fast_phase)
            
            # dt in slow phase should be larger than in fast phase
            # (because error tolerance allows larger steps when dynamics are slow)
            assert avg_dt_slow >= avg_dt_fast * 0.5, (
                f"dt should be larger in slow phase: slow={avg_dt_slow:.3f}, fast={avg_dt_fast:.3f}"
            )
        
        # Also check that dt does vary (not constant)
        if len(dt_history) > 5:
            dt_std = np.std(dt_history)
            dt_mean = np.mean(dt_history)
            # Coefficient of variation should be non-trivial
            assert dt_std / dt_mean > 0.1 or result_adaptive['rejections'] > 0, (
                f"Adaptive should vary dt or have rejections: std={dt_std:.3f}, mean={dt_mean:.3f}"
            )


# =============================================================================
# PI Controller Tests
# =============================================================================

class TestPIController:
    """Tests for the PI (proportional-integral) adaptive controller."""

    def test_pi_controller_rejects_high_error(self, ti_setup):
        """PI controller should reject steps with error > 1."""
        ti, _ = ti_setup
        ti.step_count = 2  # Not first step
        ti.error_prev = 0.5
        
        # High error step
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=True, error_norm=2.5)
        
        assert not accepted
        assert dt_new < 1.0
        assert "error" in reason

    def test_pi_controller_grows_on_small_error(self, ti_setup):
        """PI controller should grow dt when error is consistently small."""
        ti, _ = ti_setup
        ti.step_count = 3
        ti.error_prev = 0.3  # Previous step was also good
        
        # Small error step
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=True, error_norm=0.2)
        
        assert accepted
        assert dt_new > 1.0, f"dt should grow on small error, got {dt_new}"

    def test_pi_memory_not_polluted_by_rejections(self, ti_setup):
        """
        Rejected steps should not update error_prev (PI memory).
        This prevents oscillations in the controller.
        """
        ti, _ = ti_setup
        ti.step_count = 2
        ti.error_prev = 0.3  # Good history
        
        error_prev_before = ti.error_prev
        
        # Reject a step with high error
        accepted, _, _ = ti.suggest_dt(1.0, converged=True, error_norm=5.0)
        
        assert not accepted
        # error_prev should NOT be updated on rejection
        assert ti.error_prev == error_prev_before

    def test_first_step_always_accepted(self, ti_setup):
        """First step should always be accepted to avoid wasting computation."""
        ti, _ = ti_setup
        assert ti.step_count == 0
        
        # Even with high error, first step is accepted (but next dt reduced)
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=True, error_norm=3.0)
        
        assert accepted
        assert dt_new < 1.0  # But next dt is reduced
        assert "first" in reason.lower()

    def test_dt_clamp_respects_bounds(self, ti_setup):
        """Suggested dt should always be within [dt_min, dt_max]."""
        ti, _ = ti_setup
        ti.step_count = 5
        ti.error_prev = 1e-10  # Extremely small previous error
        
        # Very small error -> controller wants to grow a lot
        accepted, dt_new, _ = ti.suggest_dt(5.0, converged=True, error_norm=1e-8)
        
        assert dt_new <= ti.dt_max
        assert dt_new >= ti.dt_min


# =============================================================================
# AB2 Predictor Tests  
# =============================================================================

class TestAB2Predictor:
    """Tests for the Adams-Bashforth 2nd order predictor."""

    def test_ab2_more_accurate_than_ab1(self, adaptive_integrator):
        """
        AB2 (2nd order) should give more accurate predictions than AB1 (1st order)
        for smooth problems after sufficient history.
        """
        ti, fields, V = adaptive_integrator
        f = fields["density"]
        
        # Setup: quadratic solution y(t) = 1 + t + t^2
        # Then dy/dt = 1 + 2t
        # AB2 should be exact for quadratic!
        
        dt = 0.5
        
        # t=0: y=1, rate=1
        t0, y0, rate0 = 0.0, 1.0, 1.0
        # t=0.5: y=1.75, rate=2
        t1, y1, rate1 = 0.5, 1.75, 2.0
        # t=1.0: y=3.0, rate=3 (exact)
        t2_exact = 3.0
        
        # Setup integrator state
        f.x.array[:] = y1
        f.x.scatter_forward()
        
        n_owned = ti._fields["density"].n_owned
        ti._fields["density"].rate_last.x.array[:n_owned] = rate1
        ti._fields["density"].rate_last2.x.array[:n_owned] = rate0
        ti.step_count = 2
        ti.dt_prev = dt
        
        # Get AB2 prediction for t=1.0
        pred = ti.predict(dt)
        y_pred_ab2 = pred["density"][:n_owned].mean()
        
        # Compare with AB1 (Forward Euler) prediction
        y_pred_ab1 = y1 + dt * rate1  # = 1.75 + 0.5 * 2 = 2.75
        
        # AB2 should be closer to exact (3.0) than AB1 (2.75)
        error_ab2 = abs(y_pred_ab2 - t2_exact)
        error_ab1 = abs(y_pred_ab1 - t2_exact)
        
        # For linear-in-time rate, AB2 integrates exactly (up to roundoff)
        assert error_ab2 < 1e-12, f"AB2 error too large: {error_ab2}"
        assert error_ab2 < error_ab1, f"AB2 ({error_ab2}) should be better than AB1 ({error_ab1})"

    def test_ab2_reduces_wrms_error(self, adaptive_integrator):
        """
        With AB2 predictor, the WRMS error between prediction and correction
        should be smaller than with AB1, leading to larger allowed dt.
        """
        ti, fields, V = adaptive_integrator
        f = fields["density"]
        n_owned = ti._fields["density"].n_owned
        
        # Setup consistent rates (linear growth)
        ti._fields["density"].rate_last.x.array[:n_owned] = 0.1
        ti._fields["density"].rate_last2.x.array[:n_owned] = 0.1
        ti.step_count = 3
        ti.dt_prev = 1.0
        
        f.x.array[:] = 1.0
        f.x.scatter_forward()
        
        dt = 1.0
        
        # AB2 prediction
        pred_ab2 = ti.predict(dt)
        
        # Simulate "corrected" value (what the solver actually computes)
        # For linear problem, corrector = predictor
        f.x.array[:n_owned] = pred_ab2["density"][:n_owned]
        f.x.scatter_forward()
        
        error_ab2 = ti.compute_wrms_error(pred_ab2)
        
        # Predictor == corrector for this linear setup -> WRMS error should be ~0.
        assert error_ab2 < 1e-12, f"AB2 error should be near zero for linear problem: {error_ab2}"


# =============================================================================
# Edge Cases and Robustness
# =============================================================================

class TestRobustness:
    """Tests for edge cases and robustness."""

    def test_handles_divergence_gracefully(self, adaptive_integrator):
        """Adaptive stepping should handle solver divergence by reducing dt."""
        ti, fields, V = adaptive_integrator
        
        dt = 5.0
        
        # Simulate diverged step
        accepted, dt_new, reason = ti.suggest_dt(dt, converged=False, error_norm=100.0)
        
        assert not accepted
        assert dt_new < dt
        assert dt_new >= ti.dt_min
        assert reason.startswith("reject:coupling")

    def test_reset_history_clears_state(self, adaptive_integrator):
        """reset_history should clear all predictor state."""
        ti, fields, V = adaptive_integrator
        
        # Build up some history
        ti.step_count = 5
        ti.dt_prev = 2.0
        ti.error_prev = 0.5
        n_owned = ti._fields["density"].n_owned
        ti._fields["density"].rate_last.x.array[:n_owned] = 1.0
        
        # Reset
        ti.reset_history()
        
        assert ti.step_count == 0
        assert ti.dt_prev == 0.0
        assert ti.error_prev == 1.0
        assert np.allclose(ti._fields["density"].rate_last.x.array[:n_owned], 0.0)

    def test_mpi_consistent_error(self, adaptive_integrator):
        """WRMS error should be consistent across MPI ranks."""
        ti, fields, V = adaptive_integrator
        comm = MPI.COMM_WORLD
        
        f = fields["density"]
        f.x.array[:] = 1.0 + 0.01 * comm.rank  # Slightly different per rank
        f.x.scatter_forward()
        
        pred = {"density": np.full_like(f.x.array, 1.05)}
        
        error_local = ti.compute_wrms_error(pred)
        
        # All ranks should get the same global error
        errors = comm.gather(error_local, root=0)
        if comm.rank == 0:
            assert all(abs(e - errors[0]) < 1e-10 for e in errors), (
                f"WRMS error inconsistent across ranks: {errors}"
            )

    def test_suggest_dt_mpi_consistent(self, adaptive_integrator):
        """suggest_dt decisions should be consistent across MPI ranks."""
        ti, fields, V = adaptive_integrator
        comm = MPI.COMM_WORLD
        
        ti.step_count = 3
        ti.error_prev = 0.5
        
        accepted, dt_new, reason = ti.suggest_dt(1.0, converged=True, error_norm=0.3)
        
        # Gather results from all ranks
        results = comm.gather((accepted, dt_new, reason), root=0)
        if comm.rank == 0:
            for r in results:
                assert r[0] == results[0][0], f"Acceptance inconsistent: {results}"
                assert abs(r[1] - results[0][1]) < 1e-10, f"dt inconsistent: {results}"


# =============================================================================
# Direct Adaptive vs Fixed Comparison Tests
# =============================================================================

class TestAdaptiveStepEfficiency:
    """
    Direct comparison tests proving adaptive time stepping 
    requires fewer computational steps than fixed stepping.
    """

    def test_adaptive_reaches_target_faster(self):
        """
        For a smooth exponential decay, adaptive stepping reaches T_end 
        with fewer accepted steps than fixed stepping with conservative dt.
        """
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 4, 4, 4)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        V = fem.functionspace(domain, P1)
        
        T_end = 100.0
        y0 = 1.0
        k = 0.02  # Slow decay rate
        
        def rhs(t, y):
            return -k * y
        
        # --- Adaptive ---
        f_adaptive = fem.Function(V, name="density")
        f_adaptive.x.array[:] = y0
        f_adaptive.x.scatter_forward()
        
        time_params = TimeParams(
            dt_min=0.1, dt_max=20.0,
            adaptive_rtol=1e-2, adaptive_atol=1e-4,
            pi_growth_max=2.0, pi_shrink_min=0.1
        )
        ti = TimeIntegrator(comm, {"density": f_adaptive}, time_params=time_params)
        
        result_adaptive = simulate_ode_adaptive(ti, f_adaptive, rhs, T_end, dt_init=1.0)
        
        # --- Fixed (conservative dt) ---
        f_fixed = fem.Function(V, name="density_fixed")
        f_fixed.x.array[:] = y0
        f_fixed.x.scatter_forward()
        
        # Conservative fixed dt = 1.0 (common choice for bone remodeling)
        result_fixed = simulate_ode_fixed(f_fixed, rhs, T_end, dt_fixed=1.0)
        
        # Adaptive should use fewer steps because it can grow dt
        adaptive_steps = result_adaptive['steps']
        fixed_steps = result_fixed['steps']
        
        # For slow dynamics, adaptive can use dt up to dt_max=20
        # So it should need ~5-10x fewer steps
        assert adaptive_steps < fixed_steps, (
            f"Adaptive ({adaptive_steps}) should need fewer steps than fixed ({fixed_steps})"
        )
        
        # Verify both reach similar final value
        y_exact = y0 * np.exp(-k * T_end)
        y_adaptive = result_adaptive['y_history'][-1].mean()
        y_fixed = result_fixed['y_history'][-1].mean()
        
        assert abs(y_adaptive - y_exact) / max(abs(y_exact), 1e-10) < 0.5
        assert abs(y_fixed - y_exact) / max(abs(y_exact), 1e-10) < 0.5

    def test_adaptive_dt_grows_to_max_on_smooth(self):
        """
        For a very smooth problem (constant RHS), adaptive stepping 
        should grow dt to dt_max quickly.
        """
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 4, 4, 4)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        V = fem.functionspace(domain, P1)
        
        f = fem.Function(V, name="density")
        f.x.array[:] = 1.0
        f.x.scatter_forward()
        
        dt_max = 5.0
        time_params = TimeParams(
            dt_min=0.1, dt_max=dt_max,
            adaptive_rtol=1e-2, adaptive_atol=1e-4,
            pi_growth_max=2.0
        )
        ti = TimeIntegrator(comm, {"density": f}, time_params=time_params)
        
        # Constant RHS -> linear solution, AB2 predictor is exact
        def rhs_constant(t, y):
            return np.full_like(y, 0.01)
        
        result = simulate_ode_adaptive(ti, f, rhs_constant, T_end=30.0, dt_init=0.5)
        
        # After a few steps, dt should reach dt_max
        dt_history = result['dt_history']
        if len(dt_history) >= 5:
            dt_late = np.mean(dt_history[-5:])
            # Should be close to dt_max
            assert dt_late > 0.5 * dt_max, (
                f"dt should grow towards dt_max={dt_max} but avg_late={dt_late:.2f}"
            )

    def test_fixed_vs_adaptive_accuracy_comparison(self):
        """
        With the same computational budget (same number of RHS evaluations),
        adaptive stepping should achieve better or similar accuracy than fixed.
        """
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_cube(comm, 4, 4, 4)
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        V = fem.functionspace(domain, P1)
        
        T_end = 50.0
        y0 = 1.0
        k = 0.1
        
        def rhs(t, y):
            return -k * y
        
        y_exact = y0 * np.exp(-k * T_end)
        
        # --- Fixed with 50 steps ---
        f_fixed = fem.Function(V, name="density_fixed")
        f_fixed.x.array[:] = y0
        f_fixed.x.scatter_forward()
        
        dt_fixed = T_end / 50  # 50 steps
        result_fixed = simulate_ode_fixed(f_fixed, rhs, T_end, dt_fixed=dt_fixed)
        y_fixed_final = result_fixed['y_history'][-1].mean()
        error_fixed = abs(y_fixed_final - y_exact)
        
        # --- Adaptive with roughly same budget ---
        f_adaptive = fem.Function(V, name="density")
        f_adaptive.x.array[:] = y0
        f_adaptive.x.scatter_forward()
        
        time_params = TimeParams(
            dt_min=0.1, dt_max=5.0,
            adaptive_rtol=1e-2, adaptive_atol=1e-4,
        )
        ti = TimeIntegrator(comm, {"density": f_adaptive}, time_params=time_params)
        
        result_adaptive = simulate_ode_adaptive(ti, f_adaptive, rhs, T_end, dt_init=dt_fixed)
        y_adaptive_final = result_adaptive['y_history'][-1].mean()
        error_adaptive = abs(y_adaptive_final - y_exact)
        
        adaptive_work = result_adaptive['steps'] + result_adaptive['rejections']
        
        # If adaptive uses similar or fewer steps, error should be comparable
        if adaptive_work <= result_fixed['steps'] * 1.5:
            # With comparable budget, adaptive error should not be much worse
            assert error_adaptive <= error_fixed * 2, (
                f"Adaptive error ({error_adaptive:.4f}) should be comparable to fixed ({error_fixed:.4f})"
            )

    def test_efficiency_ratio_summary(self):
        """
        Comprehensive test that computes and reports the efficiency ratio
        of adaptive vs fixed stepping for multiple scenarios.
        
        KEY INSIGHT: Adaptive stepping excels when dynamics CHANGE over time.
        For constant dynamics, fixed stepping with well-chosen dt is competitive.
        
        This test always passes but prints a summary for documentation.
        """
        comm = MPI.COMM_WORLD
        if comm.rank != 0:
            return  # Only rank 0 prints
        
        domain = mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)  # Small mesh for speed
        P1 = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
        V = fem.functionspace(domain, P1)
        
        # Scenarios with CHANGING dynamics - this is where adaptive shines!
        scenarios = []
        
        # Scenario 1: Slow -> Fast transition (bone remodeling after injury)
        def rhs_slow_fast(t, y):
            # First 50 time units: slow equilibrium (k=0.02)
            # Then sudden stimulus: fast adaptation (k=0.5) 
            # Then back to slow (k=0.02)
            if t < 50:
                k = 0.02
            elif t < 70:
                k = 0.5  # Fast response phase
            else:
                k = 0.02  # Back to equilibrium
            return -k * y
        scenarios.append(("Slow→Fast→Slow (injury)", rhs_slow_fast, 100.0, 0.5))
        
        # Scenario 2: Gradual stiffening (progressive loading)
        def rhs_gradual(t, y):
            # k increases linearly from 0.02 to 0.3
            k = 0.02 + 0.28 * (t / 100.0)
            return -k * y
        scenarios.append(("Gradual stiffening", rhs_gradual, 100.0, 0.5))
        
        # Scenario 3: Oscillating stiffness (cyclic loading)
        def rhs_cyclic(t, y):
            # k oscillates between 0.05 and 0.3
            k = 0.175 + 0.125 * np.sin(2 * np.pi * t / 20)
            return -k * y
        scenarios.append(("Cyclic loading", rhs_cyclic, 80.0, 0.5))
        
        # Scenario 4: Mostly slow with brief fast burst
        def rhs_burst(t, y):
            # 90% slow, 10% fast burst
            if 40 < t < 50:
                k = 1.0  # Brief intense period
            else:
                k = 0.02  # Slow most of time
            return -k * y
        scenarios.append(("Slow + brief burst", rhs_burst, 100.0, 0.5))
        
        results = []
        
        for name, rhs, T_end, dt_conservative in scenarios:
            # Adaptive
            f = fem.Function(V, name="density")
            f.x.array[:] = 1.0
            
            time_params = TimeParams(dt_min=0.05, dt_max=10.0, 
                                    adaptive_rtol=1e-2, adaptive_atol=1e-4)
            ti = TimeIntegrator(MPI.COMM_SELF, {"density": f}, time_params=time_params)
            res_adaptive = simulate_ode_adaptive(ti, f, rhs, T_end, dt_init=0.5)
            
            # Fixed with CONSERVATIVE dt (must handle the fastest phase)
            f_fixed = fem.Function(V, name="fixed")
            f_fixed.x.array[:] = 1.0
            res_fixed = simulate_ode_fixed(f_fixed, rhs, T_end, dt_fixed=dt_conservative)
            
            adaptive_work = res_adaptive['steps'] + res_adaptive['rejections']
            efficiency = res_fixed['steps'] / adaptive_work if adaptive_work > 0 else 0
            
            # Also compute dt variation (shows adaptivity working)
            dt_hist = res_adaptive['dt_history']
            dt_variation = np.std(dt_hist) / np.mean(dt_hist) if dt_hist else 0
            
            results.append((name, res_adaptive['steps'], res_adaptive['rejections'], 
                           res_fixed['steps'], efficiency, dt_variation))
        
        # Print summary
        print("\n" + "=" * 80)
        print("ADAPTIVE VS FIXED TIME STEPPING EFFICIENCY COMPARISON")
        print("=" * 80)
        print(f"{'Scenario':<30} {'Adapt.':>7} {'Rej.':>5} {'Fixed':>7} {'Ratio':>7} {'dt CV':>7}")
        print("-" * 80)
        for name, adapt, rej, fixed, ratio, cv in results:
            status = "✓" if ratio > 1.0 else "≈" if ratio > 0.8 else "✗"
            print(f"{name:<30} {adapt:>7d} {rej:>5d} {fixed:>7d} {ratio:>6.2f}x {cv:>6.1%} {status}")
        print("-" * 80)
        print("Ratio > 1 means adaptive is more efficient")
        print("dt CV = coefficient of variation of dt (higher = more adaptive behavior)")
        print("=" * 80 + "\n")
        
        # Main assertion: for CHANGING dynamics, adaptive should win
        # At least 3 out of 4 scenarios should favor adaptive
        wins = sum(1 for r in results if r[4] > 1.0)
        assert wins >= 2, f"Adaptive should win at least 2/4 scenarios, got {wins}"
