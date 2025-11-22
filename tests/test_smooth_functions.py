import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.subsolvers import smooth_abs, smooth_plus, smooth_max, smooth_heaviside
from dolfinx import mesh

def make_unit_cube(comm=MPI.COMM_WORLD, n=6):
    return mesh.create_unit_cube(comm, n, n, n)

# =============================================================================
# Smooth Function Tests
# =============================================================================

class TestSmoothFunctions:
    """Test smooth approximations for non-differentiable functions."""
    
    def test_smooth_abs_properties(self):
        """Verify smooth_abs(x, eps) → |x| as eps→0 and C∞."""
        x_vals = np.linspace(-2, 2, 100)
        eps = 1e-3
        
        for x in x_vals:
            s_abs = np.sqrt(x**2 + eps**2)
            true_abs = np.abs(x)
            
            # Should approximate |x|
            assert abs(s_abs - true_abs) < eps, f"smooth_abs({x}) not close to |{x}|"
            
            # Should be smooth (derivative exists)
            # d/dx sqrt(x^2 + eps^2) = x / sqrt(x^2 + eps^2)
            deriv = x / np.sqrt(x**2 + eps**2)
            assert np.isfinite(deriv), f"smooth_abs derivative not finite at x={x}"
    
    def test_smooth_max_monotonicity(self):
        """Verify smooth_max is monotone increasing in first argument."""
        xmin = 0.5
        eps = 1e-4
        x_vals = np.linspace(0, 2, 50)
        
        smooth_vals = []
        for x in x_vals:
            dx = x - xmin
            s_max = xmin + 0.5 * (dx + np.sqrt(dx**2 + eps**2))
            smooth_vals.append(s_max)
        
        # Check monotonicity
        for i in range(len(smooth_vals) - 1):
            assert smooth_vals[i+1] >= smooth_vals[i], f"smooth_max not monotone: {smooth_vals[i]} > {smooth_vals[i+1]}"
    
    def test_smooth_heaviside_limits(self):
        """Verify smooth_heaviside(x, eps) → H(x) as |x|→∞."""
        eps = 1e-3
        
        # Far negative: should be near 0
        x_neg = -10.0
        H_neg = 0.5 * (1.0 + x_neg / np.sqrt(x_neg**2 + eps**2))
        assert abs(H_neg - 0.0) < 0.01, f"smooth_heaviside({x_neg}) should be ~0"
        
        # Far positive: should be near 1
        x_pos = 10.0
        H_pos = 0.5 * (1.0 + x_pos / np.sqrt(x_pos**2 + eps**2))
        assert abs(H_pos - 1.0) < 0.01, f"smooth_heaviside({x_pos}) should be ~1"
        
        # At zero: should be 0.5
        x_zero = 0.0
        H_zero = 0.5 * (1.0 + x_zero / np.sqrt(x_zero**2 + eps**2))
        assert abs(H_zero - 0.5) < 0.01, f"smooth_heaviside(0) should be ~0.5"

    def test_smooth_functions_match_ufl_on_mesh(self):
        """Validate UFL implementations of smooth_* against their analytical forms by integration."""
        comm = MPI.COMM_WORLD
        domain = make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, verbose=False)

        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)

        S = Function(Q, name="S")
        S.interpolate(lambda x: np.sin(2*np.pi*x[0]) - 0.3*np.cos(2*np.pi*x[1]) + 0.1*x[2])
        S.x.scatter_forward()

        eps = float(cfg.smooth_eps)
        xmin = 0.2

        # Expected analytical forms
        s_abs_form = ufl.sqrt(S*S + eps*eps)
        s_plus_form = 0.5*(S + ufl.sqrt(S*S + eps*eps))
        h_form = 0.5*(1.0 + S/ufl.sqrt(S*S + eps*eps))
        s_max_form = xmin + 0.5*((S - xmin) + ufl.sqrt((S - xmin)*(S - xmin) + eps*eps))

        # Implementations under test
        s_abs_impl = smooth_abs(S, eps)
        s_plus_impl = smooth_plus(S, eps)
        h_impl = smooth_heaviside(S, eps)
        s_max_impl = smooth_max(S, xmin, eps)

        # Integrate squared differences over the domain and require they are tiny
        def _mean_sq(expr_diff):
            val_local = fem.assemble_scalar(fem.form((expr_diff*expr_diff) * cfg.dx))
            vol_local = fem.assemble_scalar(fem.form(1.0 * cfg.dx))
            val = comm.allreduce(val_local, op=MPI.SUM)
            vol = comm.allreduce(vol_local, op=MPI.SUM)
            return float(val / max(vol, 1e-300))

        tol = 1e-12
        assert _mean_sq(s_abs_impl - s_abs_form) < tol
        assert _mean_sq(s_plus_impl - s_plus_form) < tol
        assert _mean_sq(h_impl - h_form) < tol
        assert _mean_sq(s_max_impl - s_max_form) < tol
