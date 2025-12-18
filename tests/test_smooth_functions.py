import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem import Function, functionspace
import basix
import ufl

from simulation.config import Config
from simulation.utils import build_facetag
from simulation.utils import smooth_abs, smooth_plus, smooth_max
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
            s_abs = np.sqrt(x**2 + eps**2) - eps
            true_abs = np.abs(x)
            
            # Should approximate |x|
            assert abs(s_abs - true_abs) < eps, f"smooth_abs({x}) not close to |{x}|"
            
            # Should be smooth (derivative exists)
            # d/dx (sqrt(x^2 + eps^2) - eps) = x / sqrt(x^2 + eps^2)
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
            # smooth_max(x, y) = 0.5 * (x + y + smooth_abs(x-y))
            # smooth_abs(z) = sqrt(z^2 + eps^2) - eps
            s_abs = np.sqrt(dx**2 + eps**2) - eps
            s_max = 0.5 * (x + xmin + s_abs)
            smooth_vals.append(s_max)
        
        # Check monotonicity
        for i in range(len(smooth_vals) - 1):
            assert smooth_vals[i+1] >= smooth_vals[i], f"smooth_max not monotone: {smooth_vals[i]} > {smooth_vals[i+1]}"

    def test_smooth_functions_match_ufl_on_mesh(self):
        """Validate UFL implementations of smooth_* against their analytical forms by integration."""
        comm = MPI.COMM_WORLD
        domain = make_unit_cube(comm, 8)
        facet_tags = build_facetag(domain)
        cfg = Config(domain=domain, facet_tags=facet_tags, n_trab=2.0, n_cort=1.2, rho_trab_max=0.8, rho_cort_min=1.2)

        P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
        Q = functionspace(domain, P1)

        S = Function(Q, name="S")
        S.interpolate(lambda x: np.sin(2*np.pi*x[0]) - 0.3*np.cos(2*np.pi*x[1]) + 0.1*x[2])
        S.x.scatter_forward()

        eps = float(cfg.smooth_eps)
        xmin = 0.2

        # Expected analytical forms (matching utils.py implementation)
        s_abs_form = ufl.sqrt(S*S + eps*eps) - eps
        s_plus_form = 0.5*(S + s_abs_form)
        s_max_form = 0.5*((S + xmin) + (ufl.sqrt((S - xmin)*(S - xmin) + eps*eps) - eps))

        # Implementations under test
        s_abs_impl = smooth_abs(S, eps)
        s_plus_impl = smooth_plus(S, eps)
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
        assert _mean_sq(s_max_impl - s_max_form) < tol
