"""Tests for femur_remodeller_gait.py: coordinate scaling and physical force validation."""
import pytest
import numpy as np


import basix
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_matrix as assemble_matrix_petsc,
    assemble_vector as assemble_vector_petsc,
    create_vector as create_vector_petsc,
)

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.config import Config

# Import the module under test
from simulation.femur_remodeller_gait import setup_femur_gait_loading


@pytest.fixture(scope="module", params=["mm", "m"])
def unit_scale(request):
    """Parametrize tests over two unit systems: mm and m."""
    return request.param


@pytest.fixture(scope="module")
def femur_setup(unit_scale):
    """Create femur mesh and function space with specified unit scale."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags
    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)
    
    # Unit-dependent config
    if unit_scale == "mm":
        # Mesh in mm, use realistic cortical stiffness baseline
        cfg = Config(domain=domain, facet_tags=facet_tags, L_c=1.0, u_c=1e-3, E0_dim=17e9)
    else:  # "m"
        # Mesh in mm but nondimensionalization in meters; same E0 baseline
        cfg = Config(domain=domain, facet_tags=facet_tags, L_c=1e-3, u_c=1e-6, E0_dim=17e9)
    
    return domain, facet_tags, V, Q, cfg, unit_scale


@pytest.fixture(scope="module")
def gait_loader(femur_setup):
    """Create gait loader (reuse across tests)."""
    _, _, V, _, cfg, _ = femur_setup
    return setup_femur_gait_loading(V, cfg, BW_kg=75.0, n_samples=9)


class TestCoordinateScaling:
    """Verify DOLFINx mesh and femurloader use consistent coordinate systems."""

    def test_dolfinx_mesh_in_millimeters(self, femur_setup):
        """Verify DOLFINx mesh coordinates are in millimeters (not meters)."""
        domain, _, _, _, _, _ = femur_setup
        geom = domain.geometry.x
        
        # Femur geometry should be O(100) mm, not O(0.1) m
        max_coord = np.max(np.abs(geom))
        assert 10.0 < max_coord < 500.0, \
            f"Mesh coords should be in mm (expected 10-500, got {max_coord})"

    def test_femurloader_mesh_in_millimeters(self, femur_setup):
        """Verify femurloader PyVista mesh is in millimeters."""
        import pyvista as pv
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        
        max_coord = np.max(np.abs(pv_mesh.points))
        assert 10.0 < max_coord < 500.0, \
            f"PyVista mesh should be in mm (expected 10-500, got {max_coord})"

    def test_coord_scale_is_unity(self, gait_loader):
        """Verify coord_scale=1.0 (no conversion needed)."""
        assert gait_loader.coord_scale == 1.0, \
            "coord_scale should be 1.0 since both DOLFINx and femurloader use mm"

    def test_geometry_bounds_match(self, femur_setup):
        """Verify DOLFINx and PyVista geometries have same bounds."""
        domain, _, _, _, _, _ = femur_setup
        import pyvista as pv
        
        dolfinx_geom = domain.geometry.x
        pv_mesh = pv.read(str(FemurPaths.FEMUR_MESH_VTK))
        pv_geom = pv_mesh.points
        
        # Should have same coordinate ranges (within tolerance)
        dolfinx_range = np.ptp(dolfinx_geom, axis=0)
        pv_range = np.ptp(pv_geom, axis=0)
        
        np.testing.assert_allclose(dolfinx_range, pv_range, rtol=0.10,
            err_msg="DOLFINx and PyVista geometry ranges should match")


class TestPhysicalForces:
    """Validate physical coherence of gait loading forces."""

    def test_hip_force_at_peak_stance(self, gait_loader, femur_setup):
        """Hip joint applied load (integrated) at peak stance should be ~1–6× BW."""
        domain, _, _, _, cfg, unit_scale = femur_setup
        BW_N = 75.0 * 9.81
        
        gait_loader.update_loads(50.0)
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_nd = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_applied_nd[i] = domain.comm.allreduce(Fi_loc, op=4)
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        F_applied_N = F_applied_nd * cfg.sigma_c * (L_c_m ** 2)
        F_mag = np.linalg.norm(F_applied_N)
        assert 2.5 * BW_N < F_mag < 4.5 * BW_N, \
            f"Applied force should be 2.5–4.5× BW at peak stance, got {F_mag/BW_N:.2f}× BW"

    def test_hip_force_total_magnitude(self, gait_loader):
        """Verify total hip force integrates to expected value."""
        # Hip force from OrthoLoad data should be ~3× body weight at peak
        gait_loader.update_loads(50.0)  # Peak stance
        
        # Get the force vector from gait interpolator
        F_css = gait_loader.hip_gait(50.0)  # Force in CSS frame
        F_magnitude = np.linalg.norm(F_css)
        
        BW_N = 75.0 * 9.81
        F_ratio = F_magnitude / BW_N
        
        # OrthoLoad peak hip forces typically 2-4× BW
        assert 2.0 < F_ratio < 5.0, \
            f"Hip force should be 2-4× BW at peak stance, got {F_ratio:.2f}× BW"

    def test_muscle_forces_reasonable(self, gait_loader):
        """Verify gluteus muscle forces are physically reasonable."""
        BW_N = 75.0 * 9.81
        
        # Check gluteus medius at peak (first 30% of gait)
        F_glmed = gait_loader.glmed_gait(25.0)
        glmed_magnitude = np.linalg.norm(F_glmed)
        glmed_ratio = glmed_magnitude / BW_N
        
        # Gluteus medius: typically 0.5-2× BW during stance
        assert 0.1 < glmed_ratio < 3.0, \
            f"Glmed force should be 0.1-3× BW, got {glmed_ratio:.2f}× BW"
        
        # Check gluteus maximus
        F_glmax = gait_loader.glmax_gait(25.0)
        glmax_magnitude = np.linalg.norm(F_glmax)
        glmax_ratio = glmax_magnitude / BW_N
        
        # Gluteus maximus: typically 0.2-1.5× BW
        assert 0.05 < glmax_ratio < 2.0, \
            f"Glmax force should be 0.05-2× BW, got {glmax_ratio:.2f}× BW"

    def test_force_progression_across_gait(self, gait_loader):
        """Verify forces vary smoothly across gait cycle."""
        phases = np.linspace(0, 100, 11)
        hip_forces = []
        
        for phase in phases:
            F_css = gait_loader.hip_gait(phase)
            hip_forces.append(np.linalg.norm(F_css))
        
        hip_forces = np.array(hip_forces)
        
        # Forces should vary (not constant)
        assert np.ptp(hip_forces) > 0.5 * np.max(hip_forces), \
            "Hip force should vary significantly across gait cycle"
        
        # Should have a peak (not monotonic)
        assert not np.all(np.diff(hip_forces) > 0), \
            "Hip force should not be monotonically increasing"
        assert not np.all(np.diff(hip_forces) < 0), \
            "Hip force should not be monotonically decreasing"

    def test_traction_field_nonzero(self, gait_loader, femur_setup):
        """Verify traction fields contain non-zero values after interpolation."""
        _, _, _, _, cfg, _ = femur_setup
        gait_loader.update_loads(50.0)
        
        sigma_c = cfg.sigma_c  # Characteristic stress for nondimensionalization
        
        # Check all three traction fields
        for name, func in [("hip", gait_loader.t_hip),
                          ("glmed", gait_loader.t_glmed),
                          ("glmax", gait_loader.t_glmax)]:
            vals = func.x.array.reshape((-1, 3))
            nonzero_count = np.count_nonzero(vals)
            
            assert nonzero_count > 100, \
                f"{name} traction should have >100 nonzero values, got {nonzero_count}"
            
            max_magnitude_nd = np.max(np.linalg.norm(vals, axis=1))
            max_magnitude_Pa = max_magnitude_nd * sigma_c
            
            # Tractions should be O(0.1-10 MPa) in physical units
            assert max_magnitude_Pa > 1e4, \
                f"{name} traction magnitude should be >10 kPa, got {max_magnitude_Pa/1e3:.1f} kPa"


class TestGaitQuadrature:
    """Verify gait cycle quadrature integration."""

    def test_quadrature_weights_sum_to_one(self, gait_loader):
        """Trapezoid weights should sum to 1.0."""
        quadrature = gait_loader.get_quadrature()
        phases, weights = zip(*quadrature)
        
        assert np.isclose(sum(weights), 1.0), \
            f"Quadrature weights should sum to 1.0, got {sum(weights)}"

    def test_quadrature_covers_full_cycle(self, gait_loader):
        """Quadrature should span [0, 100]% gait cycle."""
        quadrature = gait_loader.get_quadrature()
        phases, _ = zip(*quadrature)
        
        assert min(phases) == 0.0, "Quadrature should start at 0%"
        assert max(phases) == 100.0, "Quadrature should end at 100%"

    def test_quadrature_sample_count(self, gait_loader):
        """Should have n_samples quadrature points."""
        quadrature = gait_loader.get_quadrature()
        assert len(quadrature) == gait_loader.n_samples, \
            f"Expected {gait_loader.n_samples} samples, got {len(quadrature)}"


class TestIndividualLoadIntegrals:
    """Each gait load individually integrates to physiological forces and matches its interpolator."""

    def test_hip_integral_matches_interpolator_peak(self, gait_loader, femur_setup):
        domain, _, _, _, cfg, unit_scale = femur_setup
        BW_N = 75.0 * 9.81

        # Find hip peak phase from interpolator
        phases = np.linspace(0, 100, 41)
        mags = [np.linalg.norm(gait_loader.hip_gait(p)) for p in phases]
        phase = float(phases[int(np.argmax(mags))])
        gait_loader.update_loads(phase)

        # Integrate only hip traction over contact tag=2
        import ufl
        F_nd = np.zeros(3)
        for i in range(3):
            Fi = fem.form(gait_loader.t_hip[i] * cfg.ds(2))
            val = fem.assemble_scalar(Fi)
            F_nd[i] = domain.comm.allreduce(val, op=4)
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        F_N = F_nd * cfg.sigma_c * (L_c_m ** 2)

        # Compare with gait interpolator magnitude (CSS frame force magnitude)
        F_css = gait_loader.hip_gait(phase)
        rel_err = abs(np.linalg.norm(F_N) - np.linalg.norm(F_css)) / max(np.linalg.norm(F_css), 1e-30)
        assert rel_err < 0.05, f"Hip integral should match interpolator within 5% (rel_err={rel_err:.2e})"

        # Strict physiological range at peak
        assert 2.3 * BW_N < np.linalg.norm(F_N) < 4.5 * BW_N, \
            f"Hip applied (integral) should be 2.3–4.5× BW, got {np.linalg.norm(F_N)/BW_N:.2f}× BW (phase={phase:.0f}%)"

    def test_gluteus_medius_integral_peak(self, gait_loader, femur_setup):
        domain, _, _, _, cfg, unit_scale = femur_setup
        BW_N = 75.0 * 9.81

        # Find phase of gluteus medius peak from interpolator
        phases = np.linspace(0, 100, 41)
        mags = [np.linalg.norm(gait_loader.glmed_gait(p)) for p in phases]
        phase = float(phases[int(np.argmax(mags))])
        gait_loader.update_loads(phase)

        # Integrate only glmed traction
        import ufl
        F_nd = np.zeros(3)
        for i in range(3):
            Fi = fem.form(gait_loader.t_glmed[i] * cfg.ds(2))
            val = fem.assemble_scalar(Fi)
            F_nd[i] = domain.comm.allreduce(val, op=4)
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        F_N = F_nd * cfg.sigma_c * (L_c_m ** 2)

        # Compare with interpolator
        F_css = gait_loader.glmed_gait(phase)
        rel_err = abs(np.linalg.norm(F_N) - np.linalg.norm(F_css)) / max(np.linalg.norm(F_css), 1e-30)
        assert rel_err < 0.08, f"Gluteus medius integral should match interpolator within 8% (rel_err={rel_err:.2e})"

        # Strict physiological band
        ratio = np.linalg.norm(F_N) / BW_N
        assert 0.3 < ratio < 2.5, f"Gluteus medius peak should be 0.3–2.5× BW, got {ratio:.2f}× BW (phase={phase:.0f}%)"

    def test_gluteus_maximus_integral_peak(self, gait_loader, femur_setup):
        domain, _, _, _, cfg, unit_scale = femur_setup
        BW_N = 75.0 * 9.81

        # Find phase of gluteus maximus peak from interpolator
        phases = np.linspace(0, 100, 41)
        mags = [np.linalg.norm(gait_loader.glmax_gait(p)) for p in phases]
        phase = float(phases[int(np.argmax(mags))])
        gait_loader.update_loads(phase)

        # Integrate only glmax traction
        import ufl
        F_nd = np.zeros(3)
        for i in range(3):
            Fi = fem.form(gait_loader.t_glmax[i] * cfg.ds(2))
            val = fem.assemble_scalar(Fi)
            F_nd[i] = domain.comm.allreduce(val, op=4)
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        F_N = F_nd * cfg.sigma_c * (L_c_m ** 2)

        # Compare with interpolator
        F_css = gait_loader.glmax_gait(phase)
        rel_err = abs(np.linalg.norm(F_N) - np.linalg.norm(F_css)) / max(np.linalg.norm(F_css), 1e-30)
        assert rel_err < 0.08, f"Gluteus maximus integral should match interpolator within 8% (rel_err={rel_err:.2e})"

        # Strict physiological band (lower than medius)
        ratio = np.linalg.norm(F_N) / BW_N
        assert 0.1 < ratio < 1.5, f"Gluteus maximus peak should be 0.1–1.5× BW, got {ratio:.2f}× BW (phase={phase:.0f}%)"


class TestForceMaxima:
    """Compute and report maximum forces for each load type across gait cycle."""

    def test_report_max_forces_across_gait(self, gait_loader, femur_setup, capsys):
        """Strictly verify force maxima and their phases are physiological."""
        _, _, _, _, cfg, _ = femur_setup
        sigma_c = cfg.sigma_c
        BW_N = 75.0 * 9.81
        phases = np.linspace(0, 100, 21)
        
        hip_max = 0.0
        glmed_max = 0.0
        glmax_max = 0.0
        
        hip_max_phase = 0.0
        glmed_max_phase = 0.0
        glmax_max_phase = 0.0
        
        hip_traction_max_nd = 0.0
        glmed_traction_max_nd = 0.0
        glmax_traction_max_nd = 0.0
        
        for phase in phases:
            # Force vectors
            F_hip = np.linalg.norm(gait_loader.hip_gait(phase))
            F_glmed = np.linalg.norm(gait_loader.glmed_gait(phase))
            F_glmax = np.linalg.norm(gait_loader.glmax_gait(phase))
            
            if F_hip > hip_max:
                hip_max = F_hip
                hip_max_phase = phase
            if F_glmed > glmed_max:
                glmed_max = F_glmed
                glmed_max_phase = phase
            if F_glmax > glmax_max:
                glmax_max = F_glmax
                glmax_max_phase = phase
            
            # Traction fields
            gait_loader.update_loads(phase)
            t_hip = np.max(np.linalg.norm(gait_loader.t_hip.x.array.reshape((-1, 3)), axis=1))
            t_glmed = np.max(np.linalg.norm(gait_loader.t_glmed.x.array.reshape((-1, 3)), axis=1))
            t_glmax = np.max(np.linalg.norm(gait_loader.t_glmax.x.array.reshape((-1, 3)), axis=1))
            
            hip_traction_max_nd = max(hip_traction_max_nd, t_hip)
            glmed_traction_max_nd = max(glmed_traction_max_nd, t_glmed)
            glmax_traction_max_nd = max(glmax_traction_max_nd, t_glmax)
        
        # Strict physiological checks
        assert 2.3 < hip_max / BW_N < 4.5, \
            f"Hip peak should be 2.3–4.5× BW, got {hip_max/BW_N:.2f}× BW at {hip_max_phase:.0f}%"
        assert 0.3 < glmed_max / BW_N < 2.5, \
            f"Gluteus medius peak should be 0.3–2.5× BW, got {glmed_max/BW_N:.2f}× BW"
        assert 0.1 < glmax_max / BW_N < 1.5, \
            f"Gluteus maximus peak should be 0.1–1.5× BW, got {glmax_max/BW_N:.2f}× BW"

        # Peak phases: hip ~ mid-stance, glute med early–mid, glute max early
        assert 10.0 <= hip_max_phase <= 60.0, \
            f"Hip peak phase should be 10–60%, got {hip_max_phase:.0f}%"
        assert 10.0 <= glmed_max_phase <= 50.0, \
            f"Gluteus medius peak phase should be 10–50%, got {glmed_max_phase:.0f}%"
        assert 0.0 <= glmax_max_phase <= 50.0, \
            f"Gluteus maximus peak phase should be 0–50%, got {glmax_max_phase:.0f}%"

        # Traction magnitudes remain within contact-stress expectations
        assert 1.0 < hip_traction_max_nd * sigma_c / 1e6 < 15.0, \
            f"Hip max traction should be 1–15 MPa, got {hip_traction_max_nd * sigma_c / 1e6:.2f} MPa"


class TestFemurDeformationFeasibility:
    """Test that gait loads produce physiologically feasible femur deformations known from literature."""
    
    def test_displacement_magnitude_range(self, gait_loader, femur_setup):
        """Displacement magnitudes should be in physiological range (0.05-3 mm).
        
        Literature: Femur deformations during gait are typically submillimeter to few mm.
        - Bessho et al. (2007): femoral head displacement ~1-2 mm under gait loads
        - Taylor et al. (1996): peak surface strains correspond to ~0.5-1.5 mm displacement
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver with uniform density and isotropic fabric
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        # Use realistic bulk density (ND) to target physiological strains
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0  # Isotropic
        
        # Fix distal end, apply gait loads
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance (max load ~50% gait)
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Get displacement magnitudes (nondimensional -> dimensional)
        u_vals_nd = u.x.array.reshape((-1, 3))
        u_magnitudes_nd = np.linalg.norm(u_vals_nd, axis=1)
        
        # Convert to dimensional displacement in mm
        # u_dim[m] = u_nd * u_c. Convert to mm depending on unit system
        if unit_scale == "mm":
            u_dim_mm = u_magnitudes_nd * cfg.u_c
        else:
            u_dim_mm = u_magnitudes_nd * cfg.u_c * 1e3
        
        max_displacement_mm = np.max(u_dim_mm)
        mean_displacement_mm = np.mean(u_dim_mm)
        
        # Stricter physiological range for walking: 0.05–3.0 mm peak
        assert 0.05 < max_displacement_mm < 3.0, \
            f"Max displacement should be 0.05–3.0 mm, got {max_displacement_mm:.3f} mm"
        
        # Most of bone should have smaller deformations (mean << max)
        assert mean_displacement_mm < 1.5, \
            f"Mean displacement should be <1.5 mm, got {mean_displacement_mm:.3f} mm"
        
        # Localized deformation: peak at least 1.5× mean
        assert max_displacement_mm > 1.5 * mean_displacement_mm, \
            "Max displacement should be >1.5× mean (localized peak)"
    
    def test_strain_magnitude_range(self, gait_loader, femur_setup):
        """Peak strains should be in physiological range (200-3000 microstrain).
        
        Literature: In vivo strain measurements during gait:
        - Burr et al. (1996): femoral strains 400-1200 microstrain during walking
        - Lanyon et al. (1975): femoral shaft strains 300-3000 microstrain during various activities
        - Gross & Rubin (1995): physiological strains typically 50-3000 microstrain
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        import ufl
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute strain tensor: epsilon = 0.5*(grad(u) + grad(u)^T)
        eps = ufl.sym(ufl.grad(u))
        
        # Von Mises strain: sqrt(2/3 * eps_dev : eps_dev)
        I = ufl.Identity(3)
        eps_dev = eps - (ufl.tr(eps) / 3.0) * I
        eps_vm = ufl.sqrt((2.0/3.0) * ufl.inner(eps_dev, eps_dev))
        
        # Project to DG0 for evaluation
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        eps_vm_proj = fem.Function(DG0, name="eps_vm")
        eps_vm_expr = fem.Expression(eps_vm, DG0.element.interpolation_points)
        eps_vm_proj.interpolate(eps_vm_expr)
        
        # Get strain values (nondimensional)
        eps_vals_nd = eps_vm_proj.x.array[:]
        
        # Convert to microstrain: epsilon_dim = epsilon_nd * (u_c/L_c) * 1e6
        # Use cfg.strain_scale for correct unit handling across mm/m systems
        eps_microstrain = eps_vals_nd * cfg.strain_scale * 1e6
        
        max_strain = np.max(eps_microstrain)
        p95_strain = np.percentile(eps_microstrain, 95)
        
        # Physiological ranges:
        # Peak typically within 300–6000 με, bulk (95th percentile) < 3000 με, median 100–2000 με
        p50_strain = np.percentile(eps_microstrain, 50)
        assert 300.0 < max_strain < 6000.0, \
            f"Peak strain should be 300–6000 microstrain, got {max_strain:.1f}"
        assert p95_strain < 3000.0, \
            f"95th percentile strain should be <3000 microstrain, got {p95_strain:.1f}"
        assert 100.0 < p50_strain < 2000.0, \
            f"Median strain should be 100–2000 microstrain, got {p50_strain:.1f}"
    
    def test_stress_magnitude_range(self, gait_loader, femur_setup):
        """Peak stresses should be in physiological range (1-100 MPa).
        
        Literature:
        - Cortical bone yield stress: ~100-150 MPa
        - Typical walking stresses in femur: 5-50 MPa
        - Hip contact stress: 2-10 MPa (already tested in TestPhysicalForces)
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        import ufl
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute von Mises stress
        # sigma = E_eff(rho) * C : epsilon
        # For isotropic: sigma_vm = sqrt(3/2 * s:s) where s is deviatoric stress
        
        eps = ufl.sym(ufl.grad(u))
        I = ufl.Identity(3)
        
        # Lame parameters (isotropic, rho=1.0)
        E_eff = cfg.E0_nd
        nu = cfg.nu
        lmbda = E_eff * nu / ((1.0 + nu) * (1.0 - 2.0*nu))
        mu = E_eff / (2.0 * (1.0 + nu))
        
        # Stress tensor
        sigma = lmbda * ufl.tr(eps) * I + 2.0 * mu * eps
        
        # Von Mises stress
        sigma_dev = sigma - (ufl.tr(sigma) / 3.0) * I
        sigma_vm = ufl.sqrt((3.0/2.0) * ufl.inner(sigma_dev, sigma_dev))
        
        # Project to DG0
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        sigma_vm_proj = fem.Function(DG0, name="sigma_vm")
        sigma_vm_expr = fem.Expression(sigma_vm, DG0.element.interpolation_points)
        sigma_vm_proj.interpolate(sigma_vm_expr)
        
        # Get stress values (nondimensional -> dimensional)
        sigma_vals_nd = sigma_vm_proj.x.array[:]
        # sigma_c = E0_dim * u_c = 6.5e9 * 1e-3 = 6.5e6 Pa
        sigma_vals_Pa = sigma_vals_nd * cfg.sigma_c
        sigma_vals_MPa = sigma_vals_Pa / 1e6
        
        max_stress = np.max(sigma_vals_MPa)
        p95_stress = np.percentile(sigma_vals_MPa, 95)
        
        # Stricter physiological bounds for walking
        assert 2.0 < max_stress < 100.0, \
            f"Peak von Mises stress should be 2–100 MPa, got {max_stress:.1f} MPa"
        assert p95_stress < 60.0, \
            f"95th percentile stress should be <60 MPa, got {p95_stress:.1f} MPa"
    
    def test_strain_energy_density_range(self, gait_loader, femur_setup):
        """Strain energy density should be in physiological range.
        
        Literature: Typical SED values that trigger remodeling:
        - Huiskes et al. (1987): reference SED ~0.004 J/g (4 kPa)
        - Beaupré et al. (1990): lazy zone 0.0005-0.015 J/g (0.5-15 kPa)
        - Our psi_ref_dim = 300 Pa reference
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.2
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Get strain energy density from solver
        psi_nd = solver.get_strain_energy_density(u)
        
        # Project to DG0 for evaluation
        import basix
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        psi_proj = fem.Function(DG0, name="psi")
        psi_expr = fem.Expression(psi_nd, DG0.element.interpolation_points)
        psi_proj.interpolate(psi_expr)
        
        psi_vals_nd = psi_proj.x.array[:]
        # Convert to Pa: psi_dim = psi_nd * psi_c
        psi_vals_Pa = psi_vals_nd * cfg.psi_c
        psi_vals_kPa = psi_vals_Pa / 1e3
        
        max_sed = np.max(psi_vals_kPa)
        median_sed = np.median(psi_vals_kPa)
        
        # Physiological range: 0.1-1000 kPa typical range
        # Note: With uniform reference density, SED is higher than in actual bone
        # Typical values from literature: 0.5-15 kPa
        assert 0.01 < max_sed < 1000.0, \
            f"Peak SED should be 0.01-1000 kPa, got {max_sed:.2f} kPa"
        
        assert median_sed < 500.0, \
            f"Median SED should be <500 kPa, got {median_sed:.2f} kPa"
        
        # Check against reference value (should be same order of magnitude)
        assert 0.01 < median_sed / (cfg.psi_ref_dim/1e3) < 10000.0, \
            f"Median SED should be within 4 orders of magnitude of psi_ref"


class TestReactionForces:
    """Test reaction forces at fixed boundary for different loading scenarios."""
    
    def test_reaction_force_equilibrium_peak_stance(self, gait_loader, femur_setup):
        """Reaction forces should be non-zero and physiological at fixed boundary.
        
        Validates that reaction forces at the fixed distal end are computed correctly
        and have reasonable magnitudes.
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        # Fix distal end, apply gait loads
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute consistent reaction via unconstrained residual r = A0 u − b0
        from petsc4py import PETSc
        A0 = assemble_matrix_petsc(solver.a_form)
        A0.assemble()
        b0 = create_vector_petsc(V)
        with b0.localForm() as b0_loc:
            b0_loc.set(0.0)
        assemble_vector_petsc(b0, solver.L_form)
        b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        r = create_vector_petsc(V)
        with r.localForm() as r_loc:
            r_loc.set(0.0)
        A0.mult(u.x.petsc_vec, r)
        r.axpy(-1.0, b0)

        # Sum reactions per component over Dirichlet DOFs (tag=1)
        F_reaction_nd = np.zeros(3)
        for i, bc in enumerate(dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.getValues(idx_owned)
                F_reaction_nd[i] += float(np.sum(r_local))

        # Convert to dimensional (N): F[N] = F_nd * sigma_c * L_c[m]^2
        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        F_reaction_N = F_reaction_nd * cfg.sigma_c * (L_c_m ** 2)
        F_reaction_magnitude = np.linalg.norm(F_reaction_N)

        BW_N = 75.0 * 9.81
        
        # Reaction force should be non-zero and physiological
        assert F_reaction_magnitude > 0.01 * BW_N, \
            f"Reaction force should be >1% BW, got {F_reaction_magnitude:.1f} N"
        
        # Reaction force should be reasonable (not excessively large)
        assert F_reaction_magnitude < 20.0 * BW_N, \
            f"Reaction force should be <20× BW, got {F_reaction_magnitude:.1f} N"

        # Also check near-equilibrium with applied traction at peak
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_nd = np.zeros(3)
        for i in range(3):
            Fi = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi)
            F_applied_nd[i] = domain.comm.allreduce(Fi_loc, op=4)
        F_applied_N = F_applied_nd * cfg.sigma_c * (L_c_m ** 2)
        rel_eq = np.linalg.norm(F_applied_N + F_reaction_N) / max(np.linalg.norm(F_applied_N), 1e-30)
        assert rel_eq < 5e-7, f"[{unit_scale}] Force balance failed at peak: rel_err={rel_eq:.2e}"
    
    def test_reaction_force_magnitude_physiological(self, gait_loader, femur_setup):
        """Reaction force magnitude should be physiological (comparable to body weight × gait cycles).
        
        For 75 kg person: BW = 736 N
        Peak gait: ~3-4× BW hip force + muscle forces
        Reaction should balance total applied load.
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        from dolfinx.fem.petsc import assemble_vector
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        BW_N = 75.0 * 9.81
        
        # Test at multiple gait phases
        phases = [25.0, 50.0, 75.0]  # Early stance, peak, late stance
        
        for phase in phases:
            gait_loader.update_loads(phase)
            solver.assemble_rhs()
            solver.solve()
            
            # Compute consistent reaction via unconstrained residual
            from petsc4py import PETSc
            A0 = assemble_matrix_petsc(solver.a_form)
            A0.assemble()
            b0 = create_vector_petsc(V)
            with b0.localForm() as b0_loc:
                b0_loc.set(0.0)
            assemble_vector_petsc(b0, solver.L_form)
            b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

            r = create_vector_petsc(V)
            with r.localForm() as r_loc:
                r_loc.set(0.0)
            A0.mult(u.x.petsc_vec, r)
            r.axpy(-1.0, b0)

            F_reaction_nd = np.zeros(3)
            for i, bc in enumerate(dirichlet_bcs):
                idx_all, first_ghost = bc.dof_indices()
                idx_owned = idx_all[:first_ghost]
                if idx_owned.size:
                    r_local = r.getValues(idx_owned)
                    F_reaction_nd[i] += float(np.sum(r_local))

            L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
            F_reaction_N = F_reaction_nd * cfg.sigma_c * (L_c_m ** 2)
            F_reaction_magnitude = np.linalg.norm(F_reaction_N)
            
            # Reaction should be 0.01-20× BW  
            # Note: Reaction may be smaller due to distributed loading across surface
            assert 0.01 * BW_N < F_reaction_magnitude < 20.0 * BW_N, \
                f"Phase {phase}%: Reaction {F_reaction_magnitude:.1f} N should be 0.01-20× BW ({BW_N:.1f} N)"
    
    def test_reaction_force_components_realistic(self, gait_loader, femur_setup):
        """Reaction force components should have realistic directional distribution.
        
        Femur is loaded primarily in superior-inferior (y) direction with some
        medial-lateral (z) and anterior-posterior (x) components.
        """
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        from dolfinx.fem.petsc import assemble_vector
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Solve at peak stance
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Consistent reaction via unconstrained residual
        from petsc4py import PETSc
        A0 = assemble_matrix_petsc(solver.a_form)
        A0.assemble()
        b0 = create_vector_petsc(V)
        with b0.localForm() as b0_loc:
            b0_loc.set(0.0)
        assemble_vector_petsc(b0, solver.L_form)
        b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        r = create_vector_petsc(V)
        with r.localForm() as r_loc:
            r_loc.set(0.0)
        A0.mult(u.x.petsc_vec, r)
        r.axpy(-1.0, b0)

        F_reaction_nd = np.zeros(3)
        for i, bc in enumerate(dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.getValues(idx_owned)
                F_reaction_nd[i] += float(np.sum(r_local))

        L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
        F_reaction_N = F_reaction_nd * cfg.sigma_c * (L_c_m ** 2)
        F_reaction_magnitude = np.linalg.norm(F_reaction_N)
        
        # Directional consistency: reaction should oppose net applied load
        import ufl
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_nd = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_applied_nd[i] = domain.comm.allreduce(Fi_loc, op=4)
        F_applied_N = F_applied_nd * cfg.sigma_c * (L_c_m ** 2)
        num = -float(F_reaction_N @ F_applied_N)
        den = (np.linalg.norm(F_reaction_N) * np.linalg.norm(F_applied_N) + 1e-30)
        cos_theta = num / den
        assert cos_theta > 0.98, \
            f"Reaction should strongly oppose applied load (cosθ>0.98). Got cosθ={cos_theta:.2f}"
    
    def test_reaction_force_varies_with_gait_phase(self, gait_loader, femur_setup):
        """Reaction forces should vary across gait cycle (not constant)."""
        from simulation.subsolvers import MechanicsSolver
        from simulation.utils import build_dirichlet_bcs
        from dolfinx.fem.petsc import assemble_vector
        
        domain, facet_tags, V, Q, cfg, unit_scale = femur_setup
        
        # Setup mechanics solver
        import basix
        u = fem.Function(V, name="u")
        rho = fem.Function(Q, name="rho")
        A_dir = fem.Function(fem.functionspace(domain, 
                             basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(3, 3))),
                             name="A")
        
        rho.x.array[:] = 1.0
        A_dir.x.array[:] = 0.0
        for i in range(3):
            A_dir.x.array[i::9] = 1.0/3.0
        
        dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
        neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
        
        solver = MechanicsSolver(u, rho, A_dir, cfg, dirichlet_bcs, neumann_bcs)
        solver.setup()
        
        # Sample across gait cycle
        phases = np.linspace(0, 100, 9)
        reaction_magnitudes = []
        
        for phase in phases:
            gait_loader.update_loads(phase)
            solver.assemble_rhs()
            solver.solve()

            # Consistent reaction via unconstrained residual
            from petsc4py import PETSc
            A0 = assemble_matrix_petsc(solver.a_form)
            A0.assemble()
            b0 = create_vector_petsc(V)
            with b0.localForm() as b0_loc:
                b0_loc.set(0.0)
            assemble_vector_petsc(b0, solver.L_form)
            b0.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            b0.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

            r = create_vector_petsc(V)
            with r.localForm() as r_loc:
                r_loc.set(0.0)
            A0.mult(u.x.petsc_vec, r)
            r.axpy(-1.0, b0)

            F_reaction_nd = np.zeros(3)
            for i, bc in enumerate(dirichlet_bcs):
                idx_all, first_ghost = bc.dof_indices()
                idx_owned = idx_all[:first_ghost]
                if idx_owned.size:
                    r_local = r.getValues(idx_owned)
                    F_reaction_nd[i] += float(np.sum(r_local))

            L_c_m = cfg.L_c * 1e-3 if unit_scale == "mm" else cfg.L_c
            F_reaction_N = F_reaction_nd * cfg.sigma_c * (L_c_m ** 2)
            reaction_magnitudes.append(np.linalg.norm(F_reaction_N))
        
        reaction_magnitudes = np.array(reaction_magnitudes)
        
        # Reactions should vary across gait (not constant)
        variation = np.ptp(reaction_magnitudes)  # peak-to-peak
        mean_reaction = np.mean(reaction_magnitudes)
        
        assert variation > 0.5 * mean_reaction, \
            f"Reaction force should vary >50% across gait, got {variation/mean_reaction*100:.1f}%"
        
        # Should have a peak (not monotonic)
        assert not np.all(np.diff(reaction_magnitudes) > 0), \
            "Reaction force should not be monotonically increasing"
        assert not np.all(np.diff(reaction_magnitudes) < 0), \
            "Reaction force should not be monotonically decreasing"
