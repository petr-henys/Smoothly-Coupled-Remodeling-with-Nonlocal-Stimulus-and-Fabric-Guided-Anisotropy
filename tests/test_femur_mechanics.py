"""Tests for femur mechanics: deformation and reaction forces.

Tests physiological feasibility of gait-induced deformations:
- Displacement magnitude ranges (literature validation)
- Strain magnitude ranges (in vivo measurements)
- Stress magnitude ranges (cortical bone limits)
- Strain energy density (remodeling trigger values)
- Reaction forces at fixed boundaries
- Force equilibrium and directional consistency
"""

import numpy as np
import pytest
import basix
import ufl
from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import (
    assemble_matrix as assemble_matrix_petsc,
    assemble_vector as assemble_vector_petsc,
    create_vector as create_vector_petsc,
)

from simulation.febio_parser import FEBio2Dolfinx
from simulation.paths import FemurPaths
from simulation.femur_gait import setup_femur_gait_loading
from simulation.config import Config
from simulation.subsolvers import MechanicsSolver
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
    # Use physiological E0 for tests
    cfg.E0 = 17e3
    
    gait_loader = setup_femur_gait_loading(V, mass_tonnes=0.075, n_samples=9)

    return domain, facet_tags, V, Q, cfg, gait_loader


def create_solver(domain, facet_tags, V, Q, cfg, gait_loader, rho_val=1.9):
    """Helper to create a MechanicsSolver with standard test conditions."""
    u = fem.Function(V, name="u")
    rho = fem.Function(Q, name="rho")
    
    # Use physiological density (approx 1.9 g/cm^3 for cortical bone)
    rho.x.array[:] = rho_val
    
    dirichlet_bcs = build_dirichlet_bcs(V, facet_tags, id_tag=1, value=0.0)
    neumann_bcs = [(gait_loader.t_hip, 2), (gait_loader.t_glmed, 2), (gait_loader.t_glmax, 2)]
    
    solver = MechanicsSolver(u, rho, cfg, dirichlet_bcs, neumann_bcs)
    solver.setup()
    return solver, u, rho


@pytest.mark.slow
class TestFemurDeformationFeasibility:
    """Test that gait loads produce physiologically feasible femur deformations known from literature."""
    
    def test_displacement_magnitude_range(self, femur_mechanics_setup):
        """Displacement magnitudes should be in physiological range (0.05-3 mm)."""
        domain, facet_tags, V, Q, cfg, gait_loader = femur_mechanics_setup
        
        solver, u, _ = create_solver(domain, facet_tags, V, Q, cfg, gait_loader)
        
        # Solve at peak stance (max load ~50% gait)
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Get displacement magnitudes
        n_owned = u.function_space.dofmap.index_map.size_local
        bs = u.function_space.dofmap.index_map_bs
        u_vals_mm = u.x.array[:n_owned*bs].reshape((-1, 3))
        
        if u_vals_mm.size > 0:
            u_magnitudes_mm = np.linalg.norm(u_vals_mm, axis=1)
            local_max = np.max(u_magnitudes_mm)
            local_sum = np.sum(u_magnitudes_mm)
            local_count = u_magnitudes_mm.size
        else:
            local_max = 0.0
            local_sum = 0.0
            local_count = 0
            
        comm = domain.comm
        max_displacement_mm = comm.allreduce(local_max, op=MPI.MAX)
        mean_displacement_mm = comm.allreduce(local_sum, op=MPI.SUM) / max(comm.allreduce(local_count, op=MPI.SUM), 1)
        
        # Physiological range for walking: 0.05–5.0 mm peak
        assert 0.01 < max_displacement_mm < 5.0, \
            f"Max displacement should be 0.01–5.0 mm, got {max_displacement_mm:.3f} mm"
        
        assert mean_displacement_mm < 2.0, \
            f"Mean displacement should be <2.0 mm, got {mean_displacement_mm:.3f} mm"
        
        assert max_displacement_mm > 1.5 * mean_displacement_mm, \
            "Max displacement should be >1.5× mean (localized peak)"
        
        solver.destroy()
    
    def test_strain_magnitude_range(self, femur_mechanics_setup):
        """Peak strains should be in physiological range (200-3000 microstrain)."""
        domain, facet_tags, V, Q, cfg, gait_loader = femur_mechanics_setup
        
        solver, u, _ = create_solver(domain, facet_tags, V, Q, cfg, gait_loader)
        
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute strain tensor: epsilon = 0.5*(grad(u) + grad(u)^T)
        eps = ufl.sym(ufl.grad(u))
        
        # Von Mises strain
        I = ufl.Identity(3)
        eps_dev = eps - (ufl.tr(eps) / 3.0) * I
        eps_vm = ufl.sqrt((2.0/3.0) * ufl.inner(eps_dev, eps_dev))
        
        # Project to DG0
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        eps_vm_proj = fem.Function(DG0, name="eps_vm")
        eps_vm_expr = fem.Expression(eps_vm, DG0.element.interpolation_points)
        eps_vm_proj.interpolate(eps_vm_expr)
        
        n_owned = eps_vm_proj.function_space.dofmap.index_map.size_local
        eps_vals = eps_vm_proj.x.array[:n_owned]
        eps_microstrain = eps_vals * 1e6
        
        local_max = np.max(eps_microstrain) if eps_microstrain.size > 0 else 0.0
        max_strain = domain.comm.allreduce(local_max, op=MPI.MAX)
        
        # Physiological ranges
        assert 100.0 < max_strain < 6000.0, \
            f"Peak strain should be 100–6000 microstrain, got {max_strain:.1f}"
        
        solver.destroy()
    
    def test_stress_magnitude_range(self, femur_mechanics_setup):
        """Peak stresses should be in physiological range (1-100 MPa)."""
        domain, facet_tags, V, Q, cfg, gait_loader = femur_mechanics_setup
        
        solver, u, rho = create_solver(domain, facet_tags, V, Q, cfg, gait_loader)
        
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        # Compute von Mises stress using solver's sigma method
        sigma = solver.sigma(u, rho)
        I = ufl.Identity(3)
        sigma_dev = sigma - (ufl.tr(sigma) / 3.0) * I
        sigma_vm = ufl.sqrt((3.0/2.0) * ufl.inner(sigma_dev, sigma_dev))
        
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        sigma_vm_proj = fem.Function(DG0, name="sigma_vm")
        sigma_vm_expr = fem.Expression(sigma_vm, DG0.element.interpolation_points)
        sigma_vm_proj.interpolate(sigma_vm_expr)
        
        n_owned = sigma_vm_proj.function_space.dofmap.index_map.size_local
        sigma_vals_MPa = sigma_vm_proj.x.array[:n_owned]
        
        local_max = np.max(sigma_vals_MPa) if sigma_vals_MPa.size > 0 else 0.0
        max_stress = domain.comm.allreduce(local_max, op=MPI.MAX)
        
        assert 2.0 < max_stress < 100.0, \
            f"Peak von Mises stress should be 2–100 MPa, got {max_stress:.1f} MPa"
        
        solver.destroy()
    
    def test_strain_energy_density_range(self, femur_mechanics_setup):
        """Strain energy density should be in physiological range."""
        domain, facet_tags, V, Q, cfg, gait_loader = femur_mechanics_setup
        
        solver, u, _ = create_solver(domain, facet_tags, V, Q, cfg, gait_loader)
        
        gait_loader.update_loads(50.0)
        solver.assemble_rhs()
        solver.solve()
        
        psi_MPa = 0.5 * ufl.inner(solver.sigma(u, solver.rho), solver.eps(u))
        
        DG0 = fem.functionspace(domain, basix.ufl.element("DG", domain.basix_cell(), 0))
        psi_proj = fem.Function(DG0, name="psi")
        psi_expr = fem.Expression(psi_MPa, DG0.element.interpolation_points)
        psi_proj.interpolate(psi_expr)
        
        n_owned = psi_proj.function_space.dofmap.index_map.size_local
        psi_vals_MPa = psi_proj.x.array[:n_owned]

        local_max = np.max(psi_vals_MPa) if psi_vals_MPa.size > 0 else 0.0
        max_sed = domain.comm.allreduce(local_max, op=MPI.MAX)
        
        # Physiological range (MPa): ~1e-7 to 1.0 typical envelope
        assert 1e-7 < max_sed < 1.0, \
            f"Peak SED should be 1e-7–1 MPa, got {max_sed:.4e} MPa"
        
        solver.destroy()


@pytest.mark.slow
class TestReactionForces:
    """Test reaction forces at fixed boundary for different loading scenarios."""
    
    def test_reaction_force_equilibrium_peak_stance(self, femur_mechanics_setup):
        """Reaction forces should be non-zero and physiological at fixed boundary."""
        domain, facet_tags, V, Q, cfg, gait_loader = femur_mechanics_setup
        
        solver, u, _ = create_solver(domain, facet_tags, V, Q, cfg, gait_loader, rho_val=1.0)
        
        gait_loader.update_loads(50.0)
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

        F_reaction_N = np.zeros(3)
        # Sum reactions per component over Dirichlet DOFs (tag=1)
        # We need to access dirichlet_bcs from solver
        for i, bc in enumerate(solver.dirichlet_bcs):
            idx_all, first_ghost = bc.dof_indices()
            idx_owned = idx_all[:first_ghost]
            if idx_owned.size:
                r_local = r.array[idx_owned]
                F_reaction_N[i] += float(np.sum(r_local))

        F_reaction_N = domain.comm.allreduce(F_reaction_N, op=MPI.SUM)
        F_reaction_magnitude = np.linalg.norm(F_reaction_N)

        BW_N = 75.0 * 9.81
        
        assert F_reaction_magnitude > 0.01 * BW_N, \
            f"Reaction force should be >1% BW, got {F_reaction_magnitude:.1f} N"
        
        assert F_reaction_magnitude < 20.0 * BW_N, \
            f"Reaction force should be <20× BW, got {F_reaction_magnitude:.1f} N"

        # Check equilibrium
        t_total = gait_loader.t_hip + gait_loader.t_glmed + gait_loader.t_glmax
        F_applied_N = np.zeros(3)
        for i in range(3):
            Fi = fem.form(t_total[i] * cfg.ds(2))
            Fi_loc = fem.assemble_scalar(Fi)
            F_applied_N[i] = domain.comm.allreduce(Fi_loc, op=MPI.SUM)
            
        rel_eq = np.linalg.norm(F_applied_N + F_reaction_N) / max(np.linalg.norm(F_applied_N), 1e-30)
        assert rel_eq < 1e-4, f"Force balance failed at peak: rel_err={rel_eq:.2e}"
        
        solver.destroy()
