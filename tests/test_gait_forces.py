"""Tests for gait cycle force validation and quadrature.

Tests physical coherence of forces during gait loading:
- Hip joint force magnitudes and directions
- Gluteus muscle forces (medius and maximus)
- Force progression across gait cycle
- Traction field non-zero verification
- Gait quadrature integration properties
- Individual load integral validation
- Force maxima identification and physiological ranges

Related test files:
- `test_gait_energy.py`: Strain energy accumulation
- `test_femur_mechanics.py`: Deformation and reaction forces
- `test_gait_geometry.py`: Coordinate system validation
"""

import numpy as np
import pytest
import basix
from mpi4py import MPI
from dolfinx import fem

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.femur_gait import setup_femur_gait_loading
from simulation.config import Config
from simulation.utils import build_dirichlet_bcs


@pytest.fixture(scope="module")
def femur_mechanics_setup():
    """Create femur mesh, function spaces, and config."""
    mdl = FEBio2Dolfinx(FemurPaths.FEMUR_MESH_FEB)
    domain = mdl.mesh_dolfinx
    facet_tags = mdl.meshtags

    P1_vec = basix.ufl.element("Lagrange", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
    P1_scalar = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
    V = fem.functionspace(domain, P1_vec)
    Q = fem.functionspace(domain, P1_scalar)

    cfg = Config(domain=domain, facet_tags=facet_tags, verbose=True)
    gait_loader = setup_femur_gait_loading(V, mass_tonnes=0.075, n_samples=9)

    return domain, facet_tags, V, Q, cfg, gait_loader


@pytest.fixture(scope="module")
def femur_geometry_setup(femur_mechanics_setup):
    """Geometry setup for force tests."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup
    cfg.E0 = 17e3
    unit_scale = 1.0
    return domain, facet_tags, V, Q, cfg, unit_scale


@pytest.fixture
def gait_loader(femur_mechanics_setup):
    """Return gait loader for force tests."""
    domain, facet_tags, V, Q, cfg, _ = femur_mechanics_setup
    from simulation.femur_gait import setup_femur_gait_loading
    return setup_femur_gait_loading(V, mass_tonnes=0.075, n_samples=9)


@pytest.mark.slow
class TestPhysicalForces:
    """Validate physical coherence of gait loading forces."""

    def test_hip_force_at_peak_stance(self, gait_loader, femur_geometry_setup):
        """Hip joint applied load (integrated) at peak stance should be ~2.5–4.5× BW."""
        domain, _, _, _, cfg, unit_scale = femur_geometry_setup
        BW_N = 75.0 * 9.81

        gait_loader.update_loads(50.0)
        import ufl

        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_N = np.zeros(3)
        for i in range(3):
            Fi_form = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi_form)
            F_applied_N[i] = domain.comm.allreduce(Fi_loc, op=MPI.SUM)
        F_mag = np.linalg.norm(F_applied_N)
        assert 2.5 * BW_N < F_mag < 4.5 * BW_N, f"Applied force should be 2.5–4.5× BW at peak stance, got {F_mag/BW_N:.2f}× BW"

    def test_hip_force_total_magnitude(self, gait_loader):
        """Verify total hip force integrates to expected value."""
        F_css = gait_loader.hip_gait(50.0)
        F_magnitude = np.linalg.norm(F_css)

        BW_N = 75.0 * 9.81
        F_ratio = F_magnitude / BW_N

        assert 2.0 < F_ratio < 5.0, f"Hip force should be 2-4× BW at peak stance, got {F_ratio:.2f}× BW"

    def test_muscle_forces_reasonable(self, gait_loader):
        """Verify gluteus muscle forces are physically reasonable."""
        BW_N = 75.0 * 9.81

        F_glmed = gait_loader.glmed_gait(25.0)
        glmed_magnitude = np.linalg.norm(F_glmed)
        glmed_ratio = glmed_magnitude / BW_N
        assert 0.1 < glmed_ratio < 3.0, f"Glmed force should be 0.1-3× BW, got {glmed_ratio:.2f}× BW"

        F_glmax = gait_loader.glmax_gait(25.0)
        glmax_magnitude = np.linalg.norm(F_glmax)
        glmax_ratio = glmax_magnitude / BW_N
        assert 0.05 < glmax_ratio < 2.0, f"Glmax force should be 0.05-2× BW, got {glmax_ratio:.2f}× BW"

    def test_force_progression_across_gait(self, gait_loader):
        """Verify forces vary smoothly across gait cycle."""
        phases = np.linspace(0, 100, 11)
        hip_forces = []

        for phase in phases:
            F_css = gait_loader.hip_gait(phase)
            hip_forces.append(np.linalg.norm(F_css))

        hip_forces = np.array(hip_forces)

        assert np.ptp(hip_forces) > 0.5 * np.max(hip_forces), "Hip force should vary significantly across gait cycle"

        assert not np.all(np.diff(hip_forces) > 0), "Hip force should not be monotonically increasing"
        assert not np.all(np.diff(hip_forces) < 0), "Hip force should not be monotonically decreasing"

    def test_traction_field_nonzero(self, gait_loader, femur_geometry_setup):
        """Verify traction fields contain non-zero values after interpolation."""
        _, _, _, _, cfg, _ = femur_geometry_setup
        gait_loader.update_loads(50.0)
        comm = gait_loader.t_hip.function_space.mesh.comm

        for name, func in [
            ("hip", gait_loader.t_hip),
            ("glmed", gait_loader.t_glmed),
            ("glmax", gait_loader.t_glmax),
        ]:
            # Check owned dofs only to avoid double counting
            V = func.function_space
            n_local = V.dofmap.index_map.size_local
            bs = V.dofmap.index_map_bs
            vals = func.x.array[:n_local*bs].reshape((-1, 3))
            
            local_nonzero = np.count_nonzero(vals)
            nonzero_count = comm.allreduce(local_nonzero, op=MPI.SUM)

            assert nonzero_count > 100, f"{name} traction should have >100 nonzero values, got {nonzero_count}"

            local_max = np.max(np.linalg.norm(vals, axis=1)) if vals.size > 0 else 0.0
            max_magnitude_MPa = comm.allreduce(local_max, op=MPI.MAX)
            assert max_magnitude_MPa > 1e-6, f"{name} traction magnitude should be >1e-6 MPa, got {max_magnitude_MPa:.3e} MPa"


class TestGaitQuadrature:
    """Verify gait cycle quadrature integration."""

    def test_quadrature_properties(self, gait_loader):
        """Verify quadrature weights, coverage, and sample count."""
        quadrature = gait_loader.get_quadrature()
        phases, weights = zip(*quadrature)

        assert np.isclose(sum(weights), 1.0), f"Quadrature weights should sum to 1.0, got {sum(weights)}"

        assert min(phases) == 0.0, "Quadrature should start at 0%"
        assert max(phases) == 100.0, "Quadrature should end at 100%"

        assert len(quadrature) == gait_loader.n_samples, f"Expected {gait_loader.n_samples} samples, got {len(quadrature)}"


@pytest.mark.slow
class TestIndividualLoadIntegrals:
    """Each gait load individually integrates to physiological forces and matches its interpolator."""

    @pytest.mark.parametrize(
        "load_name,traction_attr,gait_method,tol,bw_min,bw_max",
        [
            ("hip", "t_hip", "hip_gait", 0.05, 2.3, 4.5),
            ("gluteus_medius", "t_glmed", "glmed_gait", 0.08, 0.3, 2.5),
            ("gluteus_maximus", "t_glmax", "glmax_gait", 0.08, 0.1, 1.5),
        ],
    )
    def test_load_integral_matches_interpolator_peak(
        self, gait_loader, femur_geometry_setup, load_name, traction_attr, gait_method, tol, bw_min, bw_max
    ):
        """Verify load integral matches interpolator and is in physiological range."""
        domain, _, _, _, cfg, unit_scale = femur_geometry_setup
        BW_N = 75.0 * 9.81
        comm = domain.comm
        rank = comm.Get_rank()

        phases = np.linspace(0, 100, 41)
        
        # Determine peak phase and expected force on rank 0
        phase = 0.0
        F_css = np.zeros(3)
        
        if rank == 0:
            gait_fn = getattr(gait_loader, gait_method)
            mags = [np.linalg.norm(gait_fn(p)) for p in phases]
            phase = float(phases[int(np.argmax(mags))])
            F_css = gait_fn(phase)
        
        # Broadcast phase and F_css to all ranks
        phase = comm.bcast(phase, root=0)
        F_css = comm.bcast(F_css, root=0)

        gait_loader.update_loads(phase)

        import ufl

        traction = getattr(gait_loader, traction_attr)
        F_N = np.zeros(3)
        for i in range(3):
            Fi = fem.form(traction[i] * cfg.ds(2))
            val = fem.assemble_scalar(Fi)
            F_N[i] = comm.allreduce(val, op=MPI.SUM)

        rel_err = abs(np.linalg.norm(F_N) - np.linalg.norm(F_css)) / max(np.linalg.norm(F_css), 1e-30)
        assert rel_err < tol, f"{load_name} integral error {rel_err:.2e} exceeds tolerance {tol}"

        ratio = np.linalg.norm(F_N) / BW_N
        assert bw_min < ratio < bw_max, f"{load_name} peak should be {bw_min}–{bw_max}× BW, got {ratio:.2f}× BW at {phase:.0f}%"


@pytest.mark.slow
class TestForceMaxima:
    """Compute and report maximum forces for each load type across gait cycle."""

    def test_report_max_forces_across_gait(self, gait_loader, femur_geometry_setup, capsys):
        """Strictly verify force maxima and their phases are physiological."""
        _, _, _, _, cfg, _ = femur_geometry_setup
        BW_N = 75.0 * 9.81
        phases = np.linspace(0, 100, 21)
        comm = gait_loader.t_hip.function_space.mesh.comm
        rank = comm.Get_rank()

        hip_max = 0.0
        glmed_max = 0.0
        glmax_max = 0.0

        hip_max_phase = 0.0
        glmed_max_phase = 0.0
        glmax_max_phase = 0.0

        hip_traction_max_MPa = 0.0
        glmed_traction_max_MPa = 0.0
        glmax_traction_max_MPa = 0.0

        # Helper to get global max
        def get_global_max(func):
            V = func.function_space
            n_local = V.dofmap.index_map.size_local
            bs = V.dofmap.index_map_bs
            vals = func.x.array[:n_local*bs].reshape((-1, 3))
            local_max = np.max(np.linalg.norm(vals, axis=1)) if vals.size > 0 else 0.0
            return comm.allreduce(local_max, op=MPI.MAX)

        for phase in phases:
            if rank == 0:
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

            gait_loader.update_loads(phase)
            t_hip = get_global_max(gait_loader.t_hip)
            t_glmed = get_global_max(gait_loader.t_glmed)
            t_glmax = get_global_max(gait_loader.t_glmax)

            hip_traction_max_MPa = max(hip_traction_max_MPa, t_hip)
            glmed_traction_max_MPa = max(glmed_traction_max_MPa, t_glmed)
            glmax_traction_max_MPa = max(glmax_traction_max_MPa, t_glmax)

        # Broadcast max forces and phases from rank 0
        hip_max = comm.bcast(hip_max, root=0)
        glmed_max = comm.bcast(glmed_max, root=0)
        glmax_max = comm.bcast(glmax_max, root=0)
        
        hip_max_phase = comm.bcast(hip_max_phase, root=0)
        glmed_max_phase = comm.bcast(glmed_max_phase, root=0)
        glmax_max_phase = comm.bcast(glmax_max_phase, root=0)

        assert 2.3 < hip_max / BW_N < 4.5, f"Hip peak should be 2.3–4.5× BW, got {hip_max/BW_N:.2f}× BW at {hip_max_phase:.0f}%"
        assert 0.3 < glmed_max / BW_N < 2.5, f"Gluteus medius peak should be 0.3–2.5× BW, got {glmed_max/BW_N:.2f}× BW"
        assert 0.1 < glmax_max / BW_N < 1.5, f"Gluteus maximus peak should be 0.1–1.5× BW, got {glmax_max/BW_N:.2f}× BW"

        assert 10.0 <= hip_max_phase <= 60.0, f"Hip peak phase should be 10–60%, got {hip_max_phase:.0f}%"
        assert 10.0 <= glmed_max_phase <= 50.0, f"Gluteus medius peak phase should be 10–50%, got {glmed_max_phase:.0f}%"
        assert 0.0 <= glmax_max_phase <= 50.0, f"Gluteus maximus peak phase should be 0–50%, got {glmax_max_phase:.0f}%"

        assert 1e-6 < hip_traction_max_MPa < 20.0, f"Hip max traction should be 1e-6–20 MPa, got {hip_traction_max_MPa:.3e} MPa"
